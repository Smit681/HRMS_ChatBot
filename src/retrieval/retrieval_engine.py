"""
Retrieval Engine - Hybrid Search System

Combines semantic search (meaning) with keyword search (exact match).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import re
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Hybrid retrieval: semantic search + keyword matching
    
    Why hybrid?
    - Semantic finds similar meaning ("salary" matches "compensation")
    - Keyword finds exact terms (employee ID "1503")
    - Combined = best results
    """
    
    def __init__(self):
        """Initialize retrieval engine"""
        logger.info("Initializing Retrieval Engine...")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        
        # Load embedding model
        logger.info(f"Loading {Config.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(
            Config.EMBEDDING_MODEL,
            device=Config.EMBEDDING_DEVICE
        )
        logger.info(f"Model loaded on {Config.EMBEDDING_DEVICE}")
        
        # Load collections
        self.collections = {}
        for name in Config.COLLECTIONS:
            try:
                self.collections[name] = self.client.get_collection(name)
                count = self.collections[name].count()
                logger.info(f"Loaded {name}: {count} documents")
            except Exception as e:
                logger.warning(f"Collection {name} not found: {e}")
        
        logger.info("✅ Retrieval Engine ready!")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        collections: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
            collections: Which collections to search (default: all)
        
        Returns:
            List of documents with scores
        """
        top_k = top_k or Config.TOP_K
        collections = collections or list(self.collections.keys())
        
        logger.info(f"Retrieving top-{top_k} for: {query}")
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k * 2, collections)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2, collections)
        
        # Combine with weighting
        combined = self._combine_results(semantic_results, keyword_results)
        
        # Return top-k
        final_results = combined[:top_k]
        logger.info(f"Retrieved {len(final_results)} documents")
        
        return final_results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        collections: List[str]
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                # Format results
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 / (1 + results['distances'][0][i]),  # Convert distance to score
                        'collection': col_name
                    })
            
            except Exception as e:
                logger.error(f"Error in {col_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        collections: List[str]
    ) -> List[Dict[str, Any]]:
        """Keyword search - exact term matching"""
        keywords = self._extract_keywords(query)
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                # Get all documents from collection
                all_docs = collection.get()
                
                # Score each document
                for i in range(len(all_docs['ids'])):
                    doc_text = all_docs['documents'][i].lower()
                    
                    # Count keyword matches
                    score = sum(
                        doc_text.count(kw.lower()) * (1 + 1/len(kw))
                        for kw in keywords
                    )
                    
                    if score > 0:
                        all_results.append({
                            'id': all_docs['ids'][i],
                            'text': all_docs['documents'][i],
                            'metadata': all_docs['metadatas'][i],
                            'score': score,
                            'collection': col_name
                        })
            
            except Exception as e:
                logger.error(f"Error in {col_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Stop words to ignore
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'or', 'a', 'an',
            'what', 'how', 'when', 'where', 'who', 'why', 'show', 'me',
            'tell', 'about', 'for', 'with', 'in', 'to', 'of'
        }
        
        # Split into words
        words = re.findall(r'\b\w+\b', query)
        
        # Keep important terms (not stop words, or numbers/uppercase)
        keywords = [
            word for word in words
            if word.lower() not in stop_words or word.isupper() or word.isdigit()
        ]
        
        return keywords
    
    def _combine_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """Combine semantic and keyword results with weighting"""
        # Normalize scores to 0-1
        def normalize(results):
            if not results:
                return results
            max_score = max(r['score'] for r in results)
            if max_score > 0:
                for r in results:
                    r['score'] = r['score'] / max_score
            return results
        
        semantic_results = normalize(semantic_results)
        keyword_results = normalize(keyword_results)
        
        # Combine by document ID
        combined = {}
        
        for result in semantic_results:
            doc_id = result['id']
            combined[doc_id] = result.copy()
            combined[doc_id]['score'] = Config.SEMANTIC_WEIGHT * result['score']
        
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['score'] += Config.KEYWORD_WEIGHT * result['score']
            else:
                combined[doc_id] = result.copy()
                combined[doc_id]['score'] = Config.KEYWORD_WEIGHT * result['score']
        
        # Convert to list and sort
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results
    
    def get_stats(self) -> Dict[str, int]:
        """Get document counts for all collections"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = collection.count()
        return stats


def main():
    """Test retrieval engine"""
    print("=" * 70)
    print("RETRIEVAL ENGINE - TESTING")
    print("=" * 70)
    
    engine = RetrievalEngine()
    
    # Show stats
    print("\nCollection Stats:")
    for name, count in engine.get_stats().items():
        print(f"  {name}: {count} documents")
    
    # Test queries
    test_queries = [
        "What is the copay for primary care?",
        "Employee 1503 salary",
        "H-1B visa benefits"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("-" * 70)
        
        results = engine.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['collection']}] Score: {result['score']:.3f}")
            print(f"   {result['text'][:150]}...")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()