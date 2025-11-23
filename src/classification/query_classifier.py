"""
Query Classifier - BERT-based Inference
========================================

Loads trained DistilBERT model and classifies queries into:
- simple: Direct lookups (top-k=3)
- aggregation: Count/sum/average (top-k=100)
- ultra_complex: Deep analysis (batch processing)

Usage:
    from src.classification.query_classifier import QueryClassifier
    
    classifier = QueryClassifier()
    result = classifier.classify("How many employees have H-1B?")
    print(result)
    # {'label': 'aggregation', 'confidence': 0.94, 'all_scores': {...}}
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    BERT-based query classifier
    
    Classifies queries into 3 categories:
    1. simple - Direct lookups
    2. aggregation - Count/sum/average
    3. ultra_complex - Predictions/deep analysis
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize classifier
        
        Args:
            model_dir: Path to model directory (default: auto-detect)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "model"
        else:
            model_dir = Path(model_dir)
        
        logger.info("=" * 70)
        logger.info("QUERY CLASSIFIER - INITIALIZING")
        logger.info("=" * 70)
        
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load config
        config_path = model_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Model: {self.config['model_name']}")
        
        # Load label mappings
        label_path = model_dir / "label_encoder.json"
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            self.label_to_id = label_data['label_to_id']
            # Convert string keys to int for id_to_label
            self.id_to_label = {int(k): v for k, v in label_data['id_to_label'].items()}
        
        logger.info(f"Labels: {list(self.label_to_id.keys())}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        
        # Load model
        logger.info("Loading trained model...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=self.config['num_labels']
        )
        
        # Load trained weights
        model_path = model_dir / "query_classifier.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info("✅ Classifier ready!")
        logger.info("=" * 70)
    
    def classify(self, query: str) -> dict:
        """
        Classify a query
        
        Args:
            query: User's question
        
        Returns:
            {
                'query': str,           # Original query
                'label': str,           # Predicted label
                'confidence': float,    # Confidence score (0-1)
                'all_scores': dict      # Scores for all labels
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get prediction
        predicted_id = torch.argmax(probabilities).item()
        predicted_label = self.id_to_label[predicted_id]
        confidence = probabilities[predicted_id].item()
        
        # All scores
        all_scores = {
            self.id_to_label[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }
        
        result = {
            'query': query,
            'label': predicted_label,
            'confidence': confidence,
            'all_scores': all_scores
        }
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        
        return result
    
    def classify_batch(self, queries: list) -> list:
        """
        Classify multiple queries at once (faster)
        
        Args:
            queries: List of query strings
        
        Returns:
            List of classification results
        """
        results = []
        
        for query in queries:
            result = self.classify(query)
            results.append(result)
        
        return results


def main():
    """Test classifier"""
    print("=" * 70)
    print("QUERY CLASSIFIER - TESTING")
    print("=" * 70)
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        # Simple
        ("What is the copay for primary care?", "simple"),
        ("Does employee 1503 have H1B visa?", "simple"),
        ("Explain dental benefits", "simple"),
        
        # Aggregation
        ("How many employees have H-1B visas?", "aggregation"),
        ("Calculate average salary", "aggregation"),
        ("Compare PPO 1000 and PPO 2500", "aggregation"),
        ("Find out total employees with H1-B visas", "aggregation"),
        
        # Ultra-complex
        ("Predict which employees need raises", "ultra_complex"),
        ("Analyze all employees for flight risk", "ultra_complex"),
        ("Recommend compensation strategy", "ultra_complex")
    ]
    
    print("\n" + "=" * 70)
    print("TESTING CLASSIFICATION")
    print("=" * 70)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print("-" * 70)
        
        result = classifier.classify(query)
        
        print(f"Predicted: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All scores: {result['all_scores']}")
        
        # Check if correct
        if result['label'] == expected:
            print("✅ CORRECT")
            correct += 1
        else:
            print("❌ WRONG")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")
    print("=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()