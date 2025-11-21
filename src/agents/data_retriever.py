"""
DATA RETRIEVER AGENT - Specialized Retrieval Strategies

Executes different retrieval strategies based on query type.

STRATEGIES:
1. DIRECT - Get specific document by ID
2. STANDARD - Normal semantic search (top-k)
3. FILTERED - Search with metadata filters
4. COMPREHENSIVE - Search all collections
5. MULTI_SOURCE - Get from multiple collections for comparison

WHY DIFFERENT STRATEGIES?
- "Employee 1503 salary" → DIRECT (know exact doc)
- "H-1B benefits" → STANDARD (semantic search)
- "Employees with H-1B" → FILTERED (metadata filter)
- "How many X?" → COMPREHENSIVE (check all docs)
- "Compare plans" → MULTI_SOURCE (get specific plans)
"""

import os
import sys
from typing import Dict, Any, List, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)


class DataRetriever:
    """
    Specialized retrieval strategies for different query types
    
    Each strategy is optimized for specific use cases:
    - Direct: Fast lookup by ID
    - Standard: General semantic search
    - Filtered: Narrow by metadata
    - Comprehensive: Exhaustive search
    - Multi-source: Cross-collection queries
    """
    
    def __init__(self, chroma_path: str = "data/embeddings"):
        """
        Initialize data retriever
        
        Args:
            chroma_path: Path to ChromaDB storage
        """
        logger.info("Initializing Data Retriever...")
        
        self.retrieval_engine = RetrievalEngine(chroma_path=chroma_path)
        
        logger.info("✅ Data Retriever ready!")
    
    def retrieve(
        self,
        query: str,
        strategy: str,
        entities: Optional[Dict[str, List[Any]]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using specified strategy
        
        Args:
            query: Search query
            strategy: Retrieval strategy name
            entities: Extracted entities from query
            top_k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with scores
        """
        logger.info(f"Retrieving with strategy: {strategy}")
        
        # Map strategy to method
        strategy_methods = {
            'direct': self._direct_retrieval,
            'standard': self._standard_retrieval,
            'filtered': self._filtered_retrieval,
            'comprehensive': self._comprehensive_retrieval,
            'multi_source': self._multi_source_retrieval
        }
        
        if strategy not in strategy_methods:
            logger.warning(f"Unknown strategy '{strategy}', using standard")
            strategy = 'standard'
        
        # Execute strategy
        method = strategy_methods[strategy]
        results = method(query, entities or {}, top_k)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _direct_retrieval(
        self,
        query: str,
        entities: Dict[str, List[Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Direct retrieval by employee ID or specific identifier
        
        WHEN TO USE:
        - Query mentions specific employee ID
        - Looking for exact document
        
        EXAMPLE:
        Query: "What is employee 1503's salary?"
        → Look for documents with metadata: employee_id=1503
        """
        logger.info("Using DIRECT retrieval")
        
        # Check for employee IDs
        if entities.get('employee_ids'):
            emp_id = entities['employee_ids'][0]
            logger.info(f"Looking for employee: {emp_id}")
            
            # Search employee_visa collection with filter
            results = self.retrieval_engine.retrieve(
                query=query,
                top_k=top_k,
                collections=['employee_visa'],
                filters={'employeeid': emp_id},
                strategy='hybrid'
            )
            
            if results:
                return results
        
        # Fallback to standard if no direct match
        logger.warning("No direct match found, falling back to standard")
        return self._standard_retrieval(query, entities, top_k)
    
    def _standard_retrieval(
        self,
        query: str,
        entities: Dict[str, List[Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Standard semantic + keyword hybrid search
        
        WHEN TO USE:
        - General queries
        - No specific entities
        - Conceptual searches
        
        EXAMPLE:
        Query: "What are the dental benefits?"
        → Semantic search across relevant collections
        """
        logger.info("Using STANDARD retrieval")
        
        # Determine relevant collections based on query
        collections = self._infer_collections(query, entities)
        
        # Execute hybrid search
        results = self.retrieval_engine.retrieve(
            query=query,
            top_k=top_k,
            collections=collections,
            strategy='hybrid'
        )
        
        return results
    
    def _filtered_retrieval(
        self,
        query: str,
        entities: Dict[str, List[Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Filtered search with metadata constraints
        
        WHEN TO USE:
        - "Show employees with X"
        - "Only H-1B visa holders"
        - "Salaries above $100k"
        
        EXAMPLE:
        Query: "Employees with H-1B visas"
        → Filter by visa_type='H-1B'
        """
        logger.info("Using FILTERED retrieval")
        
        # Build filters from entities
        filters = {}
        
        if entities.get('visa_types'):
            visa_type = entities['visa_types'][0]
            filters['visa_type'] = visa_type
            logger.info(f"Filter: visa_type={visa_type}")
        
        # Get more results since filtering reduces count
        results = self.retrieval_engine.retrieve(
            query=query,
            top_k=top_k * 2,  # Get extra to account for filtering
            collections=['employee_visa'],
            filters=filters if filters else None,
            strategy='hybrid'
        )
        
        return results[:top_k]  # Trim to requested size
    
    def _comprehensive_retrieval(
        self,
        query: str,
        entities: Dict[str, List[Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive search across all collections
        
        WHEN TO USE:
        - Aggregation queries ("how many")
        - Need complete picture
        - Uncertain which collection has answer
        
        EXAMPLE:
        Query: "How many employees have H-1B visas?"
        → Search all collections, prioritize employee_visa
        """
        logger.info("Using COMPREHENSIVE retrieval")
        
        # Search all collections
        all_collections = list(self.retrieval_engine.collections.keys())
        
        # Get results from all collections
        results = self.retrieval_engine.retrieve(
            query=query,
            top_k=top_k * 2,  # Get more for comprehensive view
            collections=all_collections,
            strategy='hybrid'
        )
        
        # Sort by relevance
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def _multi_source_retrieval(
        self,
        query: str,
        entities: Dict[str, List[Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from multiple specific sources for comparison
        
        WHEN TO USE:
        - Comparison queries
        - "Compare X and Y"
        - Need specific documents from different sources
        
        EXAMPLE:
        Query: "Compare PPO 1000 and PPO 2500"
        → Get PPO 1000 docs + PPO 2500 docs
        """
        logger.info("Using MULTI-SOURCE retrieval")
        
        results = []
        
        # If comparing plans, get each plan separately
        if entities.get('plan_names'):
            for plan in entities['plan_names']:
                plan_results = self.retrieval_engine.retrieve(
                    query=plan,
                    top_k=3,  # Get a few docs for each plan
                    collections=['medical_plans', 'dental_plans', 'vision_plans'],
                    strategy='hybrid'
                )
                results.extend(plan_results)
        else:
            # General multi-source: search relevant collections
            collections = self._infer_collections(query, entities)
            results = self.retrieval_engine.retrieve(
                query=query,
                top_k=top_k,
                collections=collections,
                strategy='hybrid'
            )
        
        # Deduplicate and sort
        seen_ids = set()
        unique_results = []
        for r in results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:top_k]
    
    def _infer_collections(
        self,
        query: str,
        entities: Dict[str, List[Any]]
    ) -> List[str]:
        """
        Infer which collections are relevant for query
        
        RULES:
        - Employee/visa keywords → employee_visa
        - Medical/health/insurance → medical_plans
        - Dental → dental_plans
        - Vision/eye → vision_plans
        - Policy/agreement → employment_agreement
        - General questions → general_questions
        """
        query_lower = query.lower()
        relevant = []
        
        # Employee/visa related
        if any(word in query_lower for word in [
            'employee', 'visa', 'h-1b', 'opt', 'cpt', 'citizen',
            'salary', 'position', 'hire', 'termination'
        ]):
            relevant.append('employee_visa')
        
        # Medical plans
        if any(word in query_lower for word in [
            'medical', 'health', 'insurance', 'ppo', 'copay',
            'deductible', 'premium', 'doctor', 'hospital'
        ]):
            relevant.append('medical_plans')
        
        # Dental
        if any(word in query_lower for word in [
            'dental', 'teeth', 'dentist', 'cleaning'
        ]):
            relevant.append('dental_plans')
        
        # Vision
        if any(word in query_lower for word in [
            'vision', 'eye', 'glasses', 'contacts', 'exam'
        ]):
            relevant.append('vision_plans')
        
        # Employment agreement
        if any(word in query_lower for word in [
            'policy', 'agreement', 'contract', 'vacation',
            'sick', 'holiday', 'pto', 'termination'
        ]):
            relevant.append('employment_agreement')
        
        # General questions
        if any(word in query_lower for word in [
            'what is', 'explain', 'define', 'tell me about'
        ]):
            relevant.append('general_questions')
        
        # If no specific collections, search all
        if not relevant:
            relevant = list(self.retrieval_engine.collections.keys())
        
        logger.info(f"Inferred collections: {relevant}")
        return relevant


def main():
    """
    Test data retriever
    """
    print("="*70)
    print("DATA RETRIEVER - TESTING")
    print("="*70)
    
    # Initialize retriever
    retriever = DataRetriever()
    
    # Test different strategies
    test_cases = [
        {
            'query': "Employee 1503 salary",
            'strategy': 'direct',
            'entities': {'employee_ids': [1503]}
        },
        {
            'query': "What are the dental benefits?",
            'strategy': 'standard',
            'entities': {}
        },
        {
            'query': "Employees with H-1B visas",
            'strategy': 'filtered',
            'entities': {'visa_types': ['H-1B']}
        },
        {
            'query': "How many employees are there?",
            'strategy': 'comprehensive',
            'entities': {}
        },
        {
            'query': "Compare PPO 1000 and PPO 2500",
            'strategy': 'multi_source',
            'entities': {'plan_names': ['PPO 1000', 'PPO 2500']}
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Query: {test['query']}")
        print(f"Strategy: {test['strategy']}")
        print("-"*70)
        
        results = retriever.retrieve(
            query=test['query'],
            strategy=test['strategy'],
            entities=test['entities'],
            top_k=3
        )
        
        print(f"\nRetrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. [{doc['collection']}] Score: {doc['score']:.3f}")
            print(f"   {doc['text'][:150]}...")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()