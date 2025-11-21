"""
HR CHATBOT - Main Interface with Smart Routing

This is the primary interface for your HR chatbot. It intelligently routes queries
between a fast simple path and a thorough multi-agent path based on complexity.

USAGE:
    from hr_chatbot import HRChatbot
    
    chatbot = HRChatbot()
    result = chatbot.ask("How many H-1B employees need renewal?")
    print(result['answer'])

ARCHITECTURE:
    User Query → Complexity Detection → Route to Appropriate Path
                                      ├─→ Simple Path (fast, 70% of queries)
                                      └─→ Multi-Agent Path (thorough, 30% of queries)
"""

import os
import sys
from typing import Dict, Any, List
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_pipeline import SimpleRAGPipeline
from agents.orchestrator import MultiAgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HRChatbot:
    """
    Unified HR Chatbot Interface
    
    Automatically routes queries to the optimal processing path:
    - Simple Path: Direct retrieval + LLM (fast)
    - Multi-Agent Path: Specialized agents (thorough)
    
    WHY TWO PATHS?
    - 70% of queries are simple lookups → fast path is sufficient
    - 30% need complex reasoning → multi-agent path ensures accuracy
    
    COMPLEXITY DETECTION:
    - Simple: Direct facts, single entity, no calculations
    - Complex: Aggregations, comparisons, predictions, multi-step reasoning
    """
    
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        ollama_model: str = "qwen2.5:14b",
        auto_route: bool = True
    ):
        """
        Initialize HR Chatbot
        
        Args:
            chroma_path: Path to ChromaDB persistent storage
            ollama_model: Ollama model name for LLM
            auto_route: Enable automatic routing between simple/complex paths
        """
        logger.info("Initializing HR Chatbot...")
        
        self.auto_route = auto_route
        self.chroma_path = chroma_path
        self.ollama_model = ollama_model
        
        # Initialize both paths
        try:
            logger.info("Loading Simple RAG Pipeline...")
            self.simple_pipeline = SimpleRAGPipeline(
                chroma_path=chroma_path,
                ollama_model=ollama_model
            )
            
            logger.info("Loading Multi-Agent Orchestrator...")
            self.orchestrator = MultiAgentOrchestrator(
                chroma_path=chroma_path,
                ollama_model=ollama_model
            )
            
            logger.info("✅ HR Chatbot ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
    
    def ask(
        self,
        query: str,
        force_path: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Ask a question to the HR chatbot
        
        Args:
            query: User's question
            force_path: Force 'simple' or 'complex' path (overrides auto-routing)
            session_id: Session identifier for conversation tracking
        
        Returns:
            {
                'answer': str,           # Natural language response
                'sources': List[str],    # Source documents used
                'path_used': str,        # 'simple' or 'complex'
                'confidence': float,     # Answer confidence (0-1)
                'processing_time': float # Seconds taken
            }
        """
        start_time = datetime.now()
        logger.info(f"Processing query: {query}")
        
        # Determine which path to use
        if force_path:
            use_complex_path = (force_path == 'complex')
            logger.info(f"Forced to use {force_path} path")
        elif self.auto_route:
            use_complex_path = self._should_use_complex_path(query)
            logger.info(f"Auto-routed to {'complex' if use_complex_path else 'simple'} path")
        else:
            use_complex_path = False  # Default to simple
        
        # Route to appropriate path
        try:
            if use_complex_path:
                result = self.orchestrator.process_query(query, session_id)
                path_used = 'complex'
            else:
                result = self.simple_pipeline.query(query)
                path_used = 'simple'
            
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result['path_used'] = path_used
            result['processing_time'] = processing_time
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s via {path_used} path")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'answer': f"I encountered an error processing your query: {str(e)}",
                'sources': [],
                'path_used': path_used if 'path_used' in locals() else 'unknown',
                'confidence': 0.0,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e)
            }
    
    def _should_use_complex_path(self, query: str) -> bool:
        """
        Determine if query requires multi-agent (complex) path
        
        SIMPLE PATH INDICATORS (use simple):
        - Direct fact lookup: "What is X's salary?"
        - Single entity queries: "Show me employee 1503"
        - Simple definitions: "What is H-1B?"
        
        COMPLEX PATH INDICATORS (use multi-agent):
        - Aggregations: "How many", "Average", "Total"
        - Comparisons: "Compare X and Y", "Difference between"
        - Filters: "Employees with", "Only show", "Exclude"
        - Predictions: "Will", "Expected to", "Forecast"
        - Multi-step: "Find X and calculate Y"
        
        Returns:
            True if complex path needed, False for simple path
        """
        query_lower = query.lower()
        
        # Keywords indicating complex queries
        aggregation_keywords = ['how many', 'count', 'total', 'average', 'mean', 'sum']
        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'better', 'best']
        filter_keywords = ['with', 'where', 'who have', 'only', 'exclude', 'filter']
        prediction_keywords = ['will', 'forecast', 'predict', 'trend', 'expected']
        calculation_keywords = ['calculate', 'compute', 'percentage', 'ratio']
        
        # Check for complex indicators
        for keyword in (aggregation_keywords + comparison_keywords + 
                       filter_keywords + prediction_keywords + calculation_keywords):
            if keyword in query_lower:
                return True
        
        # Simple queries (direct lookup)
        if any(phrase in query_lower for phrase in ['what is', 'show me', 'tell me about']):
            # But check if it's a complex "what is" like "what is the average"
            if not any(word in query_lower for word in ['average', 'total', 'count', 'how many']):
                return False
        
        # Default to simple path (conservative approach)
        return False
    
    def batch_ask(
        self,
        queries: List[str],
        force_path: str = None,
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Useful for:
        - Testing multiple questions
        - Report generation
        - Batch analysis
        
        Args:
            queries: List of questions
            force_path: Force all to use same path
            session_id: Session ID for all queries
        
        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Batch {i}/{len(queries)}: {query}")
            result = self.ask(query, force_path, session_id)
            results.append(result)
        
        logger.info(f"✅ Batch processing complete")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chatbot statistics and health info
        
        Returns:
            {
                'status': str,
                'collections': dict,
                'models_loaded': bool,
                'cache_stats': dict
            }
        """
        stats = {
            'status': 'operational',
            'collections': {},
            'models_loaded': True,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get ChromaDB collection stats
            collections = self.simple_pipeline.retrieval_engine.client.list_collections()
            for col in collections:
                stats['collections'][col.name] = col.count()
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats['status'] = 'degraded'
            stats['error'] = str(e)
        
        return stats


def main():
    """
    Example usage and testing
    """
    print("="*70)
    print("HR CHATBOT - TESTING")
    print("="*70)
    
    # Initialize chatbot
    chatbot = HRChatbot()
    
    # Test queries (mix of simple and complex)
    test_queries = [
        # Simple queries (should use simple path)
        "What is the copay for primary care in PPO 1000?",
        "What is employee 1503's position?",
        "Tell me about dental benefits",
        
        # Complex queries (should use multi-agent path)
        "How many employees have H-1B visas?",
        "What is the average salary of technical project managers?",
        "Compare PPO 1000 and PPO 2500 medical plans",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {query}")
        print("-"*70)
        
        result = chatbot.ask(query)
        
        print(f"Path: {result['path_used']}")
        print(f"Time: {result['processing_time']:.2f}s")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"\nA: {result['answer']}")
        
        if result.get('sources'):
            print(f"\nSources: {len(result['sources'])} documents")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    
    # Show stats
    print("\n" + "="*70)
    print("CHATBOT STATS")
    print("="*70)
    stats = chatbot.get_stats()
    print(f"Status: {stats['status']}")
    print(f"Collections loaded: {len(stats['collections'])}")
    for name, count in stats['collections'].items():
        print(f"  - {name}: {count} documents")


if __name__ == "__main__":
    main()