"""
Retrieval Engine - Entity-Aware Hybrid Search System

Multi-stage retrieval:
1. Entity Detection (Employee IDs, Visa Types, etc.)
2. Exact Match Search (for detected entities)
3. Hybrid Search (semantic + keyword for remaining queries)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Entity-aware hybrid retrieval system
    
    Retrieval Strategy:
    1. EXACT MATCH (if employee ID/entity detected) → 100% guarantee
    2. HYBRID SEARCH (semantic + keyword) → flexible matching
    3. METADATA FILTERING → efficient pre-filtering
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
        Retrieve relevant documents with entity-aware search
        
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
        
        # STAGE 1: Detect entities (employee IDs, visa types, etc.)
        entities = self._detect_entities(query)
        logger.info(f"Detected entities: {entities}")
        
        # STAGE 2: If employee ID detected, do exact match first
        if entities['employee_ids']:
            exact_results = self._exact_employee_search(
                entities['employee_ids'],
                query,
                top_k
            )
            
            if exact_results:
                logger.info(f"✅ Found {len(exact_results)} exact employee matches")
                
                # If we have enough exact matches, return those
                if len(exact_results) >= top_k:
                    return exact_results[:top_k]
                
                # Otherwise, get additional context documents
                remaining_k = top_k - len(exact_results)
                context_results = self._hybrid_search(
                    query,
                    remaining_k,
                    collections,
                    exclude_ids=[r['id'] for r in exact_results]
                )
                
                return exact_results + context_results
        
        # STAGE 3: If visa type detected, filter by metadata
        if entities['visa_types']:
            filtered_results = self._filtered_search(
                query,
                top_k,
                collections=['employees'],
                metadata_filters={'visa_type': entities['visa_types']}
            )
            
            if filtered_results:
                logger.info(f"✅ Found {len(filtered_results)} visa-filtered results")
                return filtered_results[:top_k]
        
        # STAGE 4: Regular hybrid search
        hybrid_results = self._hybrid_search(query, top_k, collections)
        
        return hybrid_results[:top_k]
    
    def _detect_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Detect structured entities in query
        
        Returns:
            Dict with entity types and values
        """
        entities = {
            'employee_ids': [],
            'visa_types': [],
            'dates': [],
            'amounts': []
        }
        
        # Employee IDs - patterns like "1503", "employee 1503", "emp 1503", "id 1503"
        emp_patterns = [
            r'employee\s*(?:id\s*)?(\d{4})',  # "employee 1503" or "employee id 1503"
            r'emp\s*(\d{4})',                  # "emp 1503"
            r'id\s*(\d{4})',                   # "id 1503"
            r'\b(\d{4})\b'                     # standalone "1553"
        ]
        
        for pattern in emp_patterns:
            matches = re.findall(pattern, query.lower())
            entities['employee_ids'].extend(matches)
        
        # Remove duplicates, keep order
        entities['employee_ids'] = list(dict.fromkeys(entities['employee_ids']))
        
        # Visa types - H-1B, CPT, OPT, Green Card, Citizen, TN Visa
        visa_patterns = [
            r'h-?1b',
            r'\bcpt\b',
            r'\bopt\b',
            r'green\s*card',
            r'citizen',
            r'tn\s*visa',
            r'opt-extension'
        ]
        
        for pattern in visa_patterns:
            if re.search(pattern, query.lower()):
                # Normalize the visa type
                if 'h' in pattern and '1' in pattern:
                    entities['visa_types'].append('H-1B')
                elif 'cpt' in pattern:
                    entities['visa_types'].append('CPT')
                elif 'opt-ext' in pattern:
                    entities['visa_types'].append('OPT-Extension')
                elif 'opt' in pattern:
                    entities['visa_types'].append('OPT')
                elif 'green' in pattern:
                    entities['visa_types'].append('Green Card')
                elif 'citizen' in pattern:
                    entities['visa_types'].append('Citizen')
                elif 'tn' in pattern:
                    entities['visa_types'].append('TN Visa')
        
        return entities
    
    def _exact_employee_search(
        self,
        employee_ids: List[str],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Exact search for specific employee IDs
        GUARANTEED to return correct employee documents
        """
        if 'employees' not in self.collections:
            return []
        
        collection = self.collections['employees']
        exact_results = []
        
        try:
            # Get ALL employee documents
            all_docs = collection.get()
            
            # Find exact employee ID matches
            for emp_id in employee_ids:
                # Look for this employee in documents
                for i in range(len(all_docs['ids'])):
                    doc_id = all_docs['ids'][i]
                    doc_text = all_docs['documents'][i]
                    metadata = all_docs['metadatas'][i]
                    
                    # Check if this is the right employee
                    # Method 1: Check document ID (emp_1553)
                    if doc_id == f"emp_{emp_id}":
                        exact_results.append({
                            'id': doc_id,
                            'text': doc_text,
                            'metadata': metadata,
                            'score': 1.0,  # Perfect match!
                            'collection': 'employees',
                            'match_type': 'exact_employee_id'
                        })
                        logger.info(f"✅ EXACT MATCH: Found employee {emp_id}")
                        break
                    
                    # Method 2: Check metadata
                    if metadata.get('employee_id') == emp_id:
                        exact_results.append({
                            'id': doc_id,
                            'text': doc_text,
                            'metadata': metadata,
                            'score': 1.0,
                            'collection': 'employees',
                            'match_type': 'exact_employee_id'
                        })
                        logger.info(f"✅ EXACT MATCH: Found employee {emp_id}")
                        break
                    
                    # Method 3: Check document text (as backup)
                    if f"Employee ID: {emp_id}" in doc_text:
                        exact_results.append({
                            'id': doc_id,
                            'text': doc_text,
                            'metadata': metadata,
                            'score': 0.99,  # Slightly lower since it's text match
                            'collection': 'employees',
                            'match_type': 'text_employee_id'
                        })
                        logger.info(f"✅ TEXT MATCH: Found employee {emp_id}")
                        break
        
        except Exception as e:
            logger.error(f"Error in exact employee search: {e}")
        
        return exact_results
    
    def _filtered_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        metadata_filters: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Search with metadata pre-filtering"""
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                # Get documents matching metadata filter
                all_docs = collection.get()
                
                # Filter by metadata
                filtered_indices = []
                for i in range(len(all_docs['ids'])):
                    metadata = all_docs['metadatas'][i]
                    
                    # Check if matches any filter
                    for filter_key, filter_values in metadata_filters.items():
                        if metadata.get(filter_key) in filter_values:
                            filtered_indices.append(i)
                            break
                
                if not filtered_indices:
                    continue
                
                # Now do semantic search on filtered documents
                query_embedding = self.embedding_model.encode(query).tolist()
                
                # Get embeddings for filtered documents
                filtered_docs = [all_docs['documents'][i] for i in filtered_indices]
                filtered_ids = [all_docs['ids'][i] for i in filtered_indices]
                filtered_metadata = [all_docs['metadatas'][i] for i in filtered_indices]
                
                # Compute similarities
                doc_embeddings = self.embedding_model.encode(filtered_docs)
                similarities = self.embedding_model.similarity(
                    query_embedding,
                    doc_embeddings
                )[0]
                
                # Create results
                for i, sim in enumerate(similarities):
                    all_results.append({
                        'id': filtered_ids[i],
                        'text': filtered_docs[i],
                        'metadata': filtered_metadata[i],
                        'score': float(sim),
                        'collection': col_name,
                        'match_type': 'filtered_search'
                    })
            
            except Exception as e:
                logger.error(f"Error in filtered search: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic + keyword"""
        exclude_ids = exclude_ids or []
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k * 2, collections)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2, collections)
        
        # Combine with weighting
        combined = self._combine_results(semantic_results, keyword_results)
        
        # Remove excluded IDs
        filtered = [r for r in combined if r['id'] not in exclude_ids]
        
        return filtered[:top_k]
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        collections: List[str]
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count())
                )
                
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 / (1 + results['distances'][0][i]),
                        'collection': col_name,
                        'match_type': 'semantic'
                    })
            
            except Exception as e:
                logger.error(f"Error in semantic search {col_name}: {e}")
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        collections: List[str]
    ) -> List[Dict[str, Any]]:
        """Enhanced keyword search with entity prioritization"""
        keywords = self._extract_keywords(query)
        
        all_results = []
        
        for col_name in collections:
            if col_name not in self.collections:
                continue
            
            collection = self.collections[col_name]
            
            try:
                all_docs = collection.get()
                
                for i in range(len(all_docs['ids'])):
                    doc_text = all_docs['documents'][i].lower()
                    
                    # Score with entity boosting
                    score = 0
                    for kw in keywords:
                        kw_lower = kw.lower()
                        count = doc_text.count(kw_lower)
                        
                        # Boost employee IDs and numbers heavily
                        if kw.isdigit():
                            score += count * 5.0  # 5x boost for numbers
                        else:
                            score += count * (1 + 1/len(kw))
                    
                    if score > 0:
                        all_results.append({
                            'id': all_docs['ids'][i],
                            'text': all_docs['documents'][i],
                            'metadata': all_docs['metadatas'][i],
                            'score': score,
                            'collection': col_name,
                            'match_type': 'keyword'
                        })
            
            except Exception as e:
                logger.error(f"Error in keyword search {col_name}: {e}")
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords with entity priority"""
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'or', 'a', 'an',
            'what', 'how', 'when', 'where', 'who', 'why', 'show', 'me',
            'tell', 'about', 'for', 'with', 'in', 'to', 'of', 'did'
        }
        
        words = re.findall(r'\b\w+\b', query)
        
        # Prioritize: numbers > proper nouns > non-stopwords
        keywords = []
        
        for word in words:
            # Always keep numbers (employee IDs!)
            if word.isdigit():
                keywords.append(word)
            # Keep non-stopwords
            elif word.lower() not in stop_words:
                keywords.append(word)
            # Keep if uppercase (proper noun)
            elif word[0].isupper():
                keywords.append(word)
        
        return keywords
    
    def _combine_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """Combine semantic and keyword results with weighting"""
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
    """Test retrieval engine with entity detection"""
    print("=" * 70)
    print("IMPROVED RETRIEVAL ENGINE - TESTING")
    print("=" * 70)
    
    engine = RetrievalEngine()
    
    # Show stats
    print("\nCollection Stats:")
    for name, count in engine.get_stats().items():
        print(f"  {name}: {count} documents")
    
    # Test queries with employee IDs
    test_queries = [
        "when did employee with id 1553 joined the company?",
        "Employee 1503 salary information",
        "What is the visa status of emp 1504?",
        "Tell me about 1557",
        "How many H-1B employees do we have?",
        "What is the copay for primary care?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("-" * 70)
        
        results = engine.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            match_type = result.get('match_type', 'unknown')
            print(f"\n{i}. [{result['collection']}] Score: {result['score']:.3f} | Type: {match_type}")
            print(f"   ID: {result['id']}")
            print(f"   {result['text'][:200]}...")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()