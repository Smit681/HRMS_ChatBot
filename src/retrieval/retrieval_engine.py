"""
RETRIEVAL ENGINE - Hybrid Search System

Combines semantic search (meaning-based) with keyword search (exact match) for optimal retrieval.

WHY HYBRID SEARCH?
- Semantic alone misses exact terms (employee IDs, policy names)
- Keyword alone misses conceptual matches ("visa expiring" vs "visa renewal")
- Hybrid gets best of both worlds

EXAMPLE:
    Query: "Employee 1503 salary"
    - Semantic: Finds documents about "compensation", "pay", "earnings"
    - Keyword: Finds exact match for "1503"
    - Combined: Gets exact employee with salary context
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Hybrid retrieval system combining semantic and keyword search
    
    SEARCH STRATEGIES:
    1. Semantic Search: Find similar meaning (embeddings)
    2. Keyword Search: Find exact terms (BM25-style)
    3. Hybrid: Combine both with weighting
    
    COLLECTIONS:
    - employee_visa: Employee records and visa information
    - medical_plans: Health insurance details
    - dental_plans: Dental coverage
    - vision_plans: Vision benefits
    - employment_agreement: Policies and agreements
    - general_questions: Common Q&A
    """
    
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize retrieval engine
        
        Args:
            chroma_path: Path to ChromaDB persistent storage
            embedding_model: Sentence transformer model name
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        logger.info("Initializing Retrieval Engine...")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Set hybrid search weights
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Load collections
        self.collections = {}
        collection_names = [
            "employee_visa",
            "medical_plans",
            "dental_plans",
            "vision_plans",
            "employment_agreement",
            "general_questions"
        ]
        
        for name in collection_names:
            try:
                self.collections[name] = self.client.get_collection(name)
                count = self.collections[name].count()
                logger.info(f"Loaded collection '{name}': {count} documents")
            except Exception as e:
                logger.warning(f"Collection '{name}' not found: {e}")
        
        logger.info("✅ Retrieval Engine ready!")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collections: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        strategy: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to return
            collections: List of collection names to search (None = all)
            filters: Metadata filters {"field": "value"}
            strategy: "semantic", "keyword", or "hybrid"
        
        Returns:
            List of documents with scores and metadata
        """
        logger.info(f"Retrieving top-{top_k} documents for: {query}")
        
        # Determine which collections to search
        if collections is None:
            collections = list(self.collections.keys())
        
        # Execute appropriate search strategy
        if strategy == "semantic":
            results = self._semantic_search(query, top_k, collections, filters)
        elif strategy == "keyword":
            results = self._keyword_search(query, top_k, collections, filters)
        else:  # hybrid
            results = self._hybrid_search(query, top_k, collections, filters)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using embeddings
        
        HOW IT WORKS:
        1. Convert query to embedding vector
        2. Find nearest neighbors in vector space
        3. Rank by cosine similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                # Query ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filters
                )
                
                # Format results
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'score': 1 / (1 + results['distances'][0][i]),  # Convert distance to score
                        'collection': col_name,
                        'strategy': 'semantic'
                    })
            
            except Exception as e:
                logger.error(f"Error searching collection {col_name}: {e}")
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search (BM25-style)
        
        HOW IT WORKS:
        1. Extract keywords from query
        2. Match against document text
        3. Score by term frequency and rarity
        """
        # Extract important keywords (simple approach)
        keywords = self._extract_keywords(query)
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                # Get all documents (ChromaDB doesn't have native keyword search)
                all_docs = collection.get()
                
                # Score each document
                for i in range(len(all_docs['ids'])):
                    doc_text = all_docs['documents'][i].lower()
                    
                    # Simple keyword scoring
                    score = sum(
                        doc_text.count(kw.lower()) * (1 + 1/len(kw))  # Weight by keyword rarity
                        for kw in keywords
                    )
                    
                    if score > 0:
                        all_results.append({
                            'id': all_docs['ids'][i],
                            'text': all_docs['documents'][i],
                            'metadata': all_docs['metadatas'][i],
                            'score': score,
                            'collection': col_name,
                            'strategy': 'keyword'
                        })
            
            except Exception as e:
                logger.error(f"Error searching collection {col_name}: {e}")
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword
        
        FORMULA:
        final_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
        """
        # Get results from both strategies
        semantic_results = self._semantic_search(query, top_k * 2, collections, filters)
        keyword_results = self._keyword_search(query, top_k * 2, collections, filters)
        
        # Normalize scores to 0-1 range
        def normalize_scores(results):
            if not results:
                return results
            max_score = max(r['score'] for r in results)
            if max_score > 0:
                for r in results:
                    r['score'] = r['score'] / max_score
            return results
        
        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Combine results
        combined = {}
        
        for result in semantic_results:
            doc_id = result['id']
            combined[doc_id] = result.copy()
            combined[doc_id]['score'] = self.semantic_weight * result['score']
            combined[doc_id]['strategy'] = 'hybrid'
        
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                # Add keyword score to existing semantic score
                combined[doc_id]['score'] += self.keyword_weight * result['score']
            else:
                # New document only from keyword search
                combined[doc_id] = result.copy()
                combined[doc_id]['score'] = self.keyword_weight * result['score']
                combined[doc_id]['strategy'] = 'hybrid'
        
        # Convert to list and sort
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        RULES:
        - Remove stop words (the, is, and, etc.)
        - Keep numbers (employee IDs, amounts)
        - Keep capitalized terms (H-1B, PPO, etc.)
        - Split on spaces and punctuation
        """
        # Simple stop words list
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'or', 'a', 'an',
            'what', 'how', 'when', 'where', 'who', 'why', 'show', 'me',
            'tell', 'about', 'for', 'with', 'in', 'to', 'of'
        }
        
        # Split and clean
        words = re.findall(r'\b\w+\b', query)
        
        # Keep important terms
        keywords = [
            word for word in words
            if word.lower() not in stop_words or word.isupper() or word.isdigit()
        ]
        
        return keywords
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document counts for all collections
        """
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = collection.count()
        return stats


def main():
    """
    Test retrieval engine
    """
    print("="*70)
    print("RETRIEVAL ENGINE - TESTING")
    print("="*70)
    
    # Initialize engine
    engine = RetrievalEngine()
    
    # Show collection stats
    print("\nCollection Stats:")
    stats = engine.get_collection_stats()
    for name, count in stats.items():
        print(f"  {name}: {count} documents")
    
    # Test queries
    test_queries = [
        ("What is the copay for primary care?", "semantic"),
        ("Employee 1503 salary", "keyword"),
        ("H-1B visa expiration", "hybrid"),
    ]
    
    for query, strategy in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"Strategy: {strategy}")
        print("-"*70)
        
        results = engine.retrieve(query, top_k=3, strategy=strategy)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['score']:.3f}] {result['collection']}")
            print(f"   {result['text'][:200]}...")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()