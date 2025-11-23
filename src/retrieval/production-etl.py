"""
Production ETL Pipeline - Data Processing & Embedding Generation
================================================================

Loads HR data from JSON files, cleans it, chunks intelligently,
generates embeddings, and stores in ChromaDB.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import Config
import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from tqdm import tqdm


class ProductionETL:
    """
    ETL Pipeline for HR Data
    
    Steps:
    1. Load JSON files
    2. Clean text (handle NaT, nan, None)
    3. Chunk documents intelligently
    4. Generate embeddings
    5. Store in ChromaDB
    """
    
    def __init__(self):
        print("Initializing Production ETL Pipeline...")
        
        # Load embedding model from config
        print(f"Loading {Config.EMBEDDING_MODEL}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL,
            device=Config.EMBEDDING_DEVICE
        )
        print(f"Model loaded on {Config.EMBEDDING_DEVICE}")
        
        # Initialize semantic chunker
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
        print("Semantic chunker ready")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        print(f"ChromaDB connected: {Config.CHROMA_DB_PATH}")
    
    def clean_text(self, text: str) -> str:
        """Clean text - replace missing value indicators"""
        replacements = {
            'NaT': '[Not Available]',
            ' nan': ' [Not Specified]',
            '(nan)': '[Not Specified]',
            'Entry to US: None': 'Entry to US: [Not Recorded]',
            'None - None': '[Not Specified]',
            'salary of $nan': 'Salary [Not Disclosed]',
            'of $nan': '[Not Disclosed]'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_employee_metadata(self, summary: str, emp_id: int) -> Dict:
        """Extract structured metadata from employee summary"""
        # Get most recent visa type
        visa_matches = re.findall(r'Visa type: ([A-Za-z0-9\- ]+)', summary)
        current_visa = visa_matches[-1].strip() if visa_matches else "Unknown"
        
        # Extract employment type
        employment_match = re.search(r'working as a (\w+)', summary)
        employment_type = employment_match.group(1) if employment_match else "Unknown"
        
        # Extract assignment
        assignment_match = re.search(r'Assignment: (\w+)', summary)
        assignment = assignment_match.group(1) if assignment_match else "Unknown"
        
        # Check termination
        is_terminated = ('Termination Date:' in summary and 
                        'Termination Date: [Not Available]' not in summary)
        
        return {
            'type': 'employee',
            'employee_id': str(emp_id),
            'visa_type': current_visa,
            'employment_type': employment_type,
            'assignment': assignment,
            'is_terminated': is_terminated
        }
    
    def process_employees(self) -> List[TextNode]:
        """Load and process employee records"""
        print("\nProcessing employees...")
        
        file_path = Config.RAW_DATA_DIR / "HRWIKI.Employee and Visa sponsorship information.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            employees = json.load(f)
        
        nodes = []
        for emp in tqdm(employees, desc="Employees"):
            emp_id = emp.get('employeeid', 'unknown')
            summary = self.clean_text(emp.get('summary', ''))
            
            # Extract metadata
            metadata = self.extract_employee_metadata(summary, emp_id)
            
            # Create searchable text
            searchable_text = f"""
                Employee ID: {emp_id}
                {summary}

                Key Information:
                - Current Visa: {metadata['visa_type']}
                - Employment: {metadata['employment_type']}
                - Assignment: {metadata['assignment']}
                - Status: {'Terminated' if metadata['is_terminated'] else 'Active'}
                            """.strip()
            
            # Create node
            node = TextNode(
                text=searchable_text,
                id_=f"emp_{emp_id}",
                metadata=metadata
            )
            nodes.append(node)
        
        print(f"✅ Processed {len(nodes)} employees")
        return nodes
    
    def process_documents(self, file_path: Path, doc_type: str, 
                         plan_name: str) -> List[TextNode]:
        """Load and chunk documents intelligently"""
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        all_nodes = []
        for idx, record in enumerate(records):
            content = self.clean_text(record.get('content', ''))
            
            # Create document
            doc = Document(
                text=content,
                metadata={
                    'type': doc_type,
                    'plan_name': plan_name,
                    'source_file': file_path.name
                }
            )
            
            # Chunk large documents
            if len(content.split()) > 400:
                nodes = self.splitter.get_nodes_from_documents([doc])
                
                # Add chunk metadata
                for chunk_idx, node in enumerate(nodes):
                    node.id_ = f"{doc_type}_{plan_name}_{idx}_chunk_{chunk_idx}"
                    node.metadata.update({
                        'chunk_index': chunk_idx,
                        'total_chunks': len(nodes)
                    })
                
                all_nodes.extend(nodes)
            else:
                # Small content - no chunking needed
                node = TextNode(
                    text=content,
                    id_=f"{doc_type}_{plan_name}_{idx}",
                    metadata={
                        'type': doc_type,
                        'plan_name': plan_name,
                        'source_file': file_path.name
                    }
                )
                all_nodes.append(node)
        
        return all_nodes
    
    def add_to_chromadb(self, collection_name: str, nodes: List[TextNode]):
        """Add nodes to ChromaDB with embeddings"""
        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear existing data
        try:
            existing = collection.get()
            if existing['ids']:
                collection.delete(ids=existing['ids'])
        except:
            pass
        
        # Process in batches
        batch_size = Config.EMBEDDING_BATCH_SIZE
        for i in tqdm(range(0, len(nodes), batch_size), 
                     desc=f"Embedding {collection_name}"):
            batch = nodes[i:i + batch_size]
            
            # Generate embeddings
            texts = [node.get_content() for node in batch]
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            
            # Add to ChromaDB
            collection.add(
                ids=[node.id_ for node in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[node.metadata for node in batch]
            )
        
        print(f"Added {len(nodes)} records to {collection_name}")
    
    def run(self):
        """Execute full ETL pipeline"""
        print("\n" + "=" * 70)
        print("PRODUCTION ETL PIPELINE")
        print("=" * 70)
        
        # Process employees
        employee_nodes = self.process_employees()
        self.add_to_chromadb("employees", employee_nodes)
        
        # Define documents to process
        documents_config = [
            ("HRWIKI.1000 PLAN SBC - ITLIZE GLOBAL.json", "medical", "ppo_1000"),
            ("HRWIKI.2500 PLAN SBC - ITLIZE GLOBAL.json", "medical", "ppo_2500"),
            ("HRWIKI.Medical plan summary - Price Details 2025.json", "medical", "pricing"),
            ("HRWIKI.Delta Dental Benefit Summary.json", "dental", "standard"),
            ("HRWIKI.Itlize Global LLC - DELTA Buy-Up Plan - PPO Plus Premier - Non Par MAC Benefit Summary.json", "dental", "buyup"),
            ("HRWIKI.Delta Vision Benefit Summary.json", "vision", "standard"),
            ("HRWIKI.EmploymentAgreement.json", "employment", "standard"),
            ("HRWIKI.Possible Questions Summary.json", "faq", "common")
        ]
        
        # Map doc types to collections
        collection_mapping = {
            'medical': 'medical_plans',
            'dental': 'dental_plans',
            'vision': 'vision_plans',
            'employment': 'employment_agreements',
            'faq': 'faq'
        }
        
        # Group documents by collection
        collection_docs = {col: [] for col in Config.COLLECTIONS if col != 'employees'}
        
        print("\nProcessing documents...")
        for filename, doc_type, plan_name in documents_config:
            file_path = Config.RAW_DATA_DIR / filename
            if file_path.exists():
                nodes = self.process_documents(file_path, doc_type, plan_name)
                collection_name = collection_mapping[doc_type]
                collection_docs[collection_name].extend(nodes)
                print(f"{filename}: {len(nodes)} chunks")
        
        # Add to ChromaDB
        print("\nAdding to ChromaDB...")
        for collection_name, nodes in collection_docs.items():
            if nodes:
                self.add_to_chromadb(collection_name, nodes)
        
        # Summary
        print("\n" + "=" * 70)
        print("ETL COMPLETE")
        print("=" * 70)
        
        collections = self.client.list_collections()
        total = 0
        for col in collections:
            count = self.client.get_collection(col.name).count()
            total += count
            print(f"{col.name}: {count} records")
        
        print(f"\nTotal: {total} records")
        print(f"Model: {Config.EMBEDDING_MODEL}")
        print("=" * 70)


def main():
    etl = ProductionETL()
    etl.run()


if __name__ == "__main__":
    main()