"""
BERT Classifier Training Script
================================

Trains DistilBERT model to classify queries into 3 categories:
- simple: Direct lookups (top-k=3)
- aggregation: Count/sum/average (top-k=100)
- ultra_complex: Deep analysis (batch processing)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW 
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryDataset(Dataset):
    """
    Custom Dataset for query classification
    
    Converts text queries into BERT-compatible format
    """
    
    def __init__(self, queries, labels, tokenizer, max_length=64):
        """
        Args:
            queries: List of query strings
            labels: List of label indices (0, 1, 2)
            tokenizer: DistilBERT tokenizer
            max_length: Max tokens per query
        """
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        """
        Tokenize query and return tensors
        
        Returns:
            {
                'input_ids': Tensor,      # Token IDs
                'attention_mask': Tensor, # Which tokens to attend to
                'label': Tensor          # Class label
            }
        """
        query = self.queries[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTTrainer:
    """
    Handles training of DistilBERT classifier
    """
    
    def __init__(
        self,
        model_name='distilbert-base-uncased',
        num_labels=3,
        learning_rate=1e-5,
        epochs=5,
        batch_size=8
    ):
        """Initialize trainer"""
        logger.info("=" * 70)
        logger.info("BERT CLASSIFIER TRAINER")
        logger.info("=" * 70)
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Device setup (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load model
        logger.info(f"Loading pre-trained model: {model_name}")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        logger.info("✅ Trainer initialized!")
    
    def load_data(self, data_path):
        """
        Load and prepare training data
        
        Args:
            data_path: Path to training_data.json
        
        Returns:
            train_loader, val_loader, label_to_id, id_to_label
        """
        logger.info(f"\nLoading data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Total examples: {len(data)}")
        
        # Extract queries and labels
        queries = [item['query'] for item in data]
        labels_str = [item['label'] for item in data]
        
        # Create label mappings
        unique_labels = sorted(set(labels_str))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        logger.info(f"Label mappings: {label_to_id}")
        
        # Convert labels to indices
        labels = [label_to_id[label] for label in labels_str]
        
        # Count per class
        for label_name, label_id in label_to_id.items():
            count = labels.count(label_id)
            logger.info(f"  - {label_name}: {count} examples")
        
        # Train/validation split (80/20)
        train_queries, val_queries, train_labels, val_labels = train_test_split(
            queries, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"\nTrain: {len(train_queries)}, Validation: {len(val_queries)}")
        
        # Create datasets
        train_dataset = QueryDataset(train_queries, train_labels, self.tokenizer)
        val_dataset = QueryDataset(val_queries, val_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, label_to_id, id_to_label
    
    def train(self, train_loader, val_loader):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history
        """
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        logger.info("\n" + "=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            logger.info("-" * 70)
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return history
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, id_to_label):
        """
        Detailed evaluation with classification report
        
        Args:
            val_loader: Validation data loader
            id_to_label: Mapping from ID to label name
        """
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION")
        logger.info("=" * 70)
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Convert IDs to labels
        pred_labels = [id_to_label[pred] for pred in predictions]
        true_label_names = [id_to_label[label] for label in true_labels]
        
        # Print classification report
        print("\n" + classification_report(
            true_label_names,
            pred_labels,
            target_names=list(id_to_label.values())
        ))
        
        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def save_model(self, output_dir):
        """
        Save trained model and tokenizer
        
        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving model to: {output_path}")
        
        # Save model
        model_path = output_path / "query_classifier.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"✅ Model saved: {model_path}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"✅ Tokenizer saved")
        
        # Save config
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': 64
        }
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Config saved")


def main():

    set_seed(42)

    """Main training pipeline"""
    print("=" * 70)
    print("BERT QUERY CLASSIFIER - TRAINING")
    print("=" * 70)
    
    # Paths
    data_path = Path(__file__).parent / "model" / "training_data.json"
    output_dir = Path(__file__).parent / "model"
    
    # Initialize trainer
    trainer = BERTTrainer(
        model_name='distilbert-base-uncased',
        num_labels=3,
        learning_rate=1e-5,
        epochs=5,
        batch_size=8
    )
    
    # Load data
    train_loader, val_loader, label_to_id, id_to_label = trainer.load_data(data_path)
    
    # Save label mappings
    label_path = output_dir / "label_encoder.json"
    with open(label_path, 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label
        }, f, indent=2)
    logger.info(f"✅ Label mappings saved: {label_path}")
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate
    accuracy = trainer.evaluate(val_loader, id_to_label)
    
    # Save model
    trainer.save_model(output_dir)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Model saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Test with: python src/classification/query_classifier.py")
    print("2. Integrate with pipelines")


if __name__ == "__main__":
    main()