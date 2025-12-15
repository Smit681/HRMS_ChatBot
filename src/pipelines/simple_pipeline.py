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

import asyncio
import sys
from pathlib import Path
# Add BOTH paths at the very top
sys.path.append(str(Path(__file__).parent.parent))


from config import Config
from retrieval.retrieval_engine import RetrievalEngine
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM
from typing import AsyncIterator, Dict, Any

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "backend" / "src" / "utils"))
    from cache_manager import get_cache_manager
    CACHE_AVAILABLE = True
    logger.info("✅ Cache manager found - caching enabled")
except ImportError:
    logger.warning("❌ Cache manager not found - caching disabled")
    CACHE_AVAILABLE = False

class SimplePipeline:
    
    def __init__(self):
        """Initialize simple pipeline"""
        logger.info("Initializing Simple Pipeline...")
        
        self.retrieval_engine = RetrievalEngine()
        self.context_builder = ContextBuilder()
        self.llm = OllamaLLM()

        # Initialize cache 
        logger.info("Initializing cache manager...")
        # Initialize cache manager
        if CACHE_AVAILABLE:
            logger.info("Initializing cache manager...")
            self.cache = get_cache_manager()
            logger.info("✅ Cache manager initialized")
        else:
            self.cache = None
            logger.warning("⚠️  Running without cache")

        
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
    
    async def process_stream(self, query: str) -> AsyncIterator[dict]:
        """
        Process query with streaming response
        
        Args:
            query: User's question
        
        Yields:
            dict: Progress updates and token chunks
        """
        logger.info(f"[Simple Pipeline] Processing with streaming: {query}")

        # Check cache first
        if self.cache:
            logger.info("🔍 Checking cache...")
            cached_result = self.cache.get(query, query_type='simple')
            
            if cached_result:
                # CACHE HIT - stream the cached answer
                cache_type = cached_result.get('cache_type', 'unknown')
                logger.info(f"✅ CACHE HIT ({cache_type})")
                
                yield {
                    'type': 'status',
                    'message': f'✅ Found cached result ({cache_type} match)'
                }
                
                # Stream the cached answer token by token (simulate streaming)
                cached_answer = cached_result.get('answer', '')
                for char in cached_answer:
                    yield {'type': 'token', 'content': char}
                    await asyncio.sleep(0.001)  # Small delay to simulate streaming
                
                # Send metadata
                yield {
                    'type': 'metadata',
                    'sources': cached_result.get('metadata', {}).get('sources', []),
                    'num_sources': cached_result.get('metadata', {}).get('num_sources', 0),
                    'confidence': cached_result.get('metadata', {}).get('confidence', 0.0),
                    'pipeline': 'simple',
                    'is_cache': True,
                    'cache_type': cache_type
                }
                
                logger.info("✅ Cached result streamed successfully")
                return
            
            logger.info("❌ Cache MISS - processing query...")
            yield {'type': 'status', 'message': 'Cache miss - processing query...'}
        
        # Step 1: Retrieve (not streamed)
        yield {'type': 'status', 'message': 'Retrieving documents...'}
        retrieved_docs = self.retrieval_engine.retrieve(query, top_k=3)
        
        if not retrieved_docs:
            yield {'type': 'error', 'message': 'No documents found'}
            return
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        yield {'type': 'status', 'message': f'Found {len(retrieved_docs)} documents'}
        
        # Step 2: Build 
        logger.info("[2/3] Building context...")
        yield {'type': 'status', 'message': 'Building context...'}
        context = self.context_builder.build_context(retrieved_docs)
        
        # Step 3: Generate answer with streaming
        logger.info("[3/3] Generating answer...")

        yield {'type': 'status', 'message': 'Generating answer...'}

        
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['default']
        )

        full_answer = ""        
        # Stream tokens
        async for token in self.llm.generate_stream(prompt):
            full_answer += token
            yield {'type': 'token', 'content': token}
            await asyncio.sleep(0)
            
        # Send metadata after completion
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        if self.cache:
            logger.info("💾 Caching result...")
            
            metadata = {
                'sources': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'confidence': min(avg_score, 1.0),
                'pipeline': 'simple'
            }
            
            cache_success = self.cache.set(
                query=query,
                answer=full_answer,
                metadata=metadata,
                query_type='simple'
            )
            
            if cache_success:
                logger.info("✅ Result cached successfully")
            else:
                logger.warning("⚠️  Failed to cache result")

        
        yield {
            'type': 'metadata',
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'confidence': min(avg_score, 1.0),
            'pipeline': 'simple'
        }
        logger.info(f"✅ Query complete (confidence: {min(avg_score, 1.0):.2f})")
    
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
    """Test simple pipeline with caching"""
    print("=" * 70)
    print("SIMPLE PIPELINE - CACHING TEST")
    print("=" * 70)
    
    pipeline = SimplePipeline()
    
    test_queries = [
        "What is the copay for primary care?",
        "What is the copay for primary care?",  # Exact match (should hit cache)
        "Tell me the primary care copay",  # Paraphrase (should hit semantic cache)
        "What is employee 1503's position?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {query}")
        print("-" * 70)
        
        result = pipeline.process_stream(query)
        
        print(f"\n📝 Answer: {result['answer'][:100]}...")
        print(f"📊 Sources: {result['num_sources']}")
        print(f"✅ Confidence: {result['confidence']:.2f}")        
        if i < len(test_queries):
            input("\nPress Enter for next test...")
    
    # Show cache stats
    if pipeline.cache:
        print("\n" + "=" * 70)
        print("CACHE STATISTICS")
        print("=" * 70)
        stats = pipeline.cache.get_stats()
        print(f"Total cached queries: {stats['total_cached_queries']}")
        print(f"Exact cache entries: {stats['total_exact']}")
        print(f"Semantic cache entries: {stats['total_semantic']}")
        print(f"Memory used: {stats['memory_used']}")
        print(f"Semantic caching: {'Enabled' if stats['semantic_enabled'] else 'Disabled'}")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()