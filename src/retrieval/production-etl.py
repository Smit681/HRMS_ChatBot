import json
from pathlib import Path
from typing import List, Dict
import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from tqdm import tqdm
import re

class ProductionETL:
    def __init__(self):
        print("🚀 Initializing Production ETL Pipeline...")
        
        # Initialize better embedding model (BGE-large)
        print("📥 Loading BGE-large embedding model (this may take a moment)...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda"  # Use your T4 GPU
        )
        print("✓ Embedding model loaded on GPU")
        
        # Initialize semantic chunker
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
        print("✓ Semantic chunker initialized")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path="data/embeddings")
        print("✓ Connected to ChromaDB")
    
    def clean_text(self, text: str) -> str:
        """Clean text data automatically"""
        # Replace missing value indicators
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
        """Extract structured metadata from employee text"""
        # Extract visa types (get most recent)
        visa_matches = re.findall(r'Visa type: ([A-Za-z0-9\- ]+)', summary)
        current_visa = visa_matches[-1].strip() if visa_matches else "Unknown"
        
        # Extract employment type
        employment_match = re.search(r'working as a (\w+)', summary)
        employment_type = employment_match.group(1) if employment_match else "Unknown"
        
        # Extract assignment
        assignment_match = re.search(r'Assignment: (\w+)', summary)
        assignment = assignment_match.group(1) if assignment_match else "Unknown"
        
        # Check termination status
        is_terminated = ('Termination Date:' in summary and 
                        'Termination Date: [Not Available]' not in summary)
        
        # Extract salary
        salary_match = re.search(r'Salary of \$(\d+\.?\d*)', summary)
        has_salary = bool(salary_match and salary_match.group(1) != 'nan')
        
        return {
            'type': 'employee',
            'employee_id': str(emp_id),
            'visa_type': current_visa,
            'employment_type': employment_type,
            'assignment': assignment,
            'is_terminated': is_terminated,
            'has_salary': has_salary
        }
    
    def load_and_process_employees(self) -> List[TextNode]:
        """Load and process employee data"""
        print("\n📋 Processing employee records...")
        
        file_path = Path("data/raw/HRWIKI.Employee and Visa sponsorship information.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            employees = json.load(f)
        
        nodes = []
        for emp in tqdm(employees, desc="Processing employees"):
            emp_id = emp.get('employeeid', 'unknown')
            summary = self.clean_text(emp.get('summary', ''))
            
            # Create enhanced searchable text
            metadata = self.extract_employee_metadata(summary, emp_id)
            
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
        
        print(f"✓ Processed {len(nodes)} employee records")
        return nodes
    
    def load_and_process_documents(self, file_path: Path, doc_type: str, 
                                   plan_name: str) -> List[TextNode]:
        """Load and intelligently chunk documents"""
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        all_nodes = []
        for idx, record in enumerate(records):
            content = self.clean_text(record.get('content', ''))
            
            # Create LlamaIndex document
            doc = Document(
                text=content,
                metadata={
                    'type': doc_type,
                    'plan_name': plan_name,
                    'source_file': file_path.name
                }
            )
            
            # Use semantic chunking for large documents
            if len(content.split()) > 400:
                nodes = self.splitter.get_nodes_from_documents([doc])
                
                # Add chunk-specific metadata
                for chunk_idx, node in enumerate(nodes):
                    node.id_ = f"{doc_type}_{plan_name}_{idx}_chunk_{chunk_idx}"
                    node.metadata.update({
                        'chunk_index': chunk_idx,
                        'total_chunks': len(nodes)
                    })
                
                all_nodes.extend(nodes)
            else:
                # Small content - no chunking
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
    
    def add_nodes_to_collection(self, collection_name: str, nodes: List[TextNode]):
        """Add nodes to ChromaDB with embeddings"""
        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Clear existing data
        try:
            existing = collection.get()
            if existing['ids']:
                collection.delete(ids=existing['ids'])
        except:
            pass
        
        # Process in batches
        batch_size = 50
        for i in tqdm(range(0, len(nodes), batch_size), 
                     desc=f"Embedding {collection_name}"):
            batch = nodes[i:i + batch_size]
            
            # Generate embeddings
            texts = [node.get_content() for node in batch]
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            
            # Prepare data for ChromaDB
            ids = [node.id_ for node in batch]
            metadatas = [node.metadata for node in batch]
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        
        print(f"✓ Added {len(nodes)} records to {collection_name}")
    
    def run_pipeline(self):
        """Execute full production ETL pipeline"""
        print("\n" + "="*70)
        print("PRODUCTION ETL PIPELINE - PHASE 2")
        print("="*70)
        
        # Process employees
        employee_nodes = self.load_and_process_employees()
        self.add_nodes_to_collection("employees", employee_nodes)
        
        # Process insurance and policy documents
        documents_config = [
            ("data/raw/HRWIKI.1000 PLAN SBC - ITLIZE GLOBAL.json", "medical", "ppo_1000"),
            ("data/raw/HRWIKI.2500 PLAN SBC - ITLIZE GLOBAL.json", "medical", "ppo_2500"),
            ("data/raw/HRWIKI.Medical plan summary - Price Details 2025.json", "medical", "pricing"),
            ("data/raw/HRWIKI.Delta Dental Benefit Summary.json", "dental", "standard"),
            ("data/raw/HRWIKI.Itlize Global LLC - DELTA Buy-Up Plan - PPO Plus Premier - Non Par MAC Benefit Summary.json", "dental", "buyup"),
            ("data/raw/HRWIKI.Delta Vision Benefit Summary.json", "vision", "standard"),
            ("data/raw/HRWIKI.EmploymentAgreement.json", "employment", "standard"),
            ("data/raw/HRWIKI.Possible Questions Summary.json", "faq", "common")
        ]
        
        # Group by collection
        collection_docs = {
            'medical_plans': [],
            'dental_plans': [],
            'vision_plans': [],
            'employment_agreements': [],
            'faq': []
        }
        
        collection_mapping = {
            'medical': 'medical_plans',
            'dental': 'dental_plans',
            'vision': 'vision_plans',
            'employment': 'employment_agreements',
            'faq': 'faq'
        }
        
        print("\n📋 Processing documents...")
        for file_path, doc_type, plan_name in documents_config:
            path = Path(file_path)
            if path.exists():
                nodes = self.load_and_process_documents(path, doc_type, plan_name)
                collection_name = collection_mapping[doc_type]
                collection_docs[collection_name].extend(nodes)
                print(f"  ✓ {path.name}: {len(nodes)} chunks")
        
        # Add to collections
        print("\n📊 Adding to ChromaDB collections...")
        for collection_name, nodes in collection_docs.items():
            if nodes:
                self.add_nodes_to_collection(collection_name, nodes)
        
        # Summary
        print("\n" + "="*70)
        print("ETL PIPELINE COMPLETE")
        print("="*70)
        
        collections = self.client.list_collections()
        total_records = 0
        for col in collections:
            collection = self.client.get_collection(col.name)
            count = collection.count()
            total_records += count
            print(f"✓ {col.name}: {count} records")
        
        print(f"\n✓ Total records: {total_records}")
        print(f"✓ Embedding model: BAAI/bge-large-en-v1.5 (1024 dimensions)")
        print(f"✓ Chunking: Semantic (smart breakpoints)")
        print("="*70)
        
        # Test query
        print("\n🧪 Testing retrieval...")
        test_col = self.client.get_collection("employees")
        
        # Generate query embedding
        query = "software developers with H-1B visa"
        query_embedding = self.embed_model.get_query_embedding(query)
        
        results = test_col.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        print(f"✓ Query: '{query}'")
        print(f"✓ Found {len(results['ids'][0])} results")
        print(f"✓ Top result: {results['ids'][0][0]}")
        print("\n🎉 Production ETL pipeline successful!")

def main():
    import time
    start = time.time()
    
    etl = ProductionETL()
    etl.run_pipeline()
    
    elapsed = time.time() - start
    print(f"\n⏱ Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()