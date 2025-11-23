"""
Simple Pipeline - Fast Lookups
===============================

Handles direct lookup queries:
- "What is X?"
- "Show me Y"
- "Tell me about Z"

Retrieval: top-k=3 (few, highly relevant docs)
Processing: Retrieve → Build context → LLM → Return
Speed: 2-3 seconds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from retrieval.retrieval_engine import RetrievalEngine
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePipeline:
    
    def __init__(self):
        """Initialize simple pipeline"""
        logger.info("Initializing Simple Pipeline...")
        
        self.retrieval_engine = RetrievalEngine()
        self.context_builder = ContextBuilder()
        self.llm = OllamaLLM()
        
        logger.info("✅ Simple Pipeline ready!")
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a simple lookup query
        
        Args:
            query: User's question
        
        Returns:
            {
                'answer': str,
                'sources': List[dict],
                'num_sources': int,
                'confidence': float,
                'pipeline': str
            }
        """
        logger.info(f"[Simple Pipeline] Processing: {query}")
        
        # Step 1: Retrieve top-3 documents
        logger.info("[1/3] Retrieving top-3 documents...")
        retrieved_docs = self.retrieval_engine.retrieve(query, top_k=3)
        
        if not retrieved_docs:
            logger.warning("No documents found")
            return self._no_results_response(query)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Build context
        logger.info("[2/3] Building context...")
        context = self.context_builder.build_context(retrieved_docs)
        
        # Step 3: Generate answer
        logger.info("[3/3] Generating answer...")
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['default']
        )
        
        llm_response = self.llm.generate(prompt)
        
        # Calculate confidence
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        confidence = min(avg_score, 1.0)
        
        result = {
            'answer': llm_response['text'],
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'confidence': confidence,
            'pipeline': 'simple'
        }
        
        logger.info(f"✅ Query complete (confidence: {confidence:.2f})")
        return result
    
    def _no_results_response(self, query: str) -> Dict[str, Any]:
        """Handle no results"""
        return {
            'answer': f"I couldn't find relevant information to answer: '{query}'",
            'sources': [],
            'num_sources': 0,
            'confidence': 0.0,
            'pipeline': 'simple'
        }

def main():
    """Test simple pipeline"""
    print("=" * 70)
    print("SIMPLE PIPELINE - TESTING")
    print("=" * 70)
    
    pipeline = SimplePipeline()
    
    test_queries = [
        "What is the copay for primary care?",
        "Tell me about dental benefits",
        "What is employee 1503's position?",
        "Explain the vision insurance plan"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print("-" * 70)
        
        result = pipeline.process(query)
        
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['num_sources']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()