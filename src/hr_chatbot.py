"""
HR Chatbot - Main Interface
============================

Intelligent HR assistant that routes queries to appropriate pipelines:
- Simple queries → Simple pipeline (ChromaDB + LLM)
- Aggregation queries → Aggregation pipeline (MongoDB + LLM)
- Ultra-complex queries → Ultra-complex pipeline (MongoDB + Batch LLM)

Uses BERT classifier for accurate query routing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from classification.query_classifier import QueryClassifier
from pipelines.simple_pipeline import SimplePipeline
from pipelines.aggregation_pipeline import AggregationPipeline
from pipelines.ultra_complex_pipeline import UltraComplexPipeline
from typing import AsyncIterator, Dict, Any, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRChatbot:
    def __init__(self):
        """Initialize HR Chatbot"""
        logger.info("=" * 70)
        logger.info("HR CHATBOT - INITIALIZING")
        logger.info("=" * 70)
        
        # Initialize BERT classifier
        logger.info("Loading BERT classifier...")
        self.classifier = QueryClassifier()
        
        # Initialize pipelines
        logger.info("Loading pipelines...")
        self.simple_pipeline = SimplePipeline()
        self.aggregation_pipeline = AggregationPipeline()
        self.ultra_complex_pipeline = UltraComplexPipeline()
        
        logger.info("=" * 70)
        logger.info("✅ HR CHATBOT READY!")
        logger.info("=" * 70)
    
    def ask(
        self,
        query: str,
        auto_confirm_ultra: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question to the HR chatbot
        
        Args:
            query: User's question
            auto_confirm_ultra: Auto-confirm ultra-complex (skip prompt)
        
        Returns:
            {
                'answer': str,
                'query_type': str,
                'confidence': float,
                'processing_time': float,
                'pipeline_used': str,
                'metadata': dict
            }
        """
        start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print(f"Q: {query}")
        print("=" * 70)
        
        # Step 1: Classify query using BERT
        print(f"\n🔍 Analyzing query...")
        classification = self.classifier.classify(query)
        
        query_type = classification['label']
        bert_confidence = classification['confidence']
        
        print(f"📊 Query Type: {query_type} (confidence: {bert_confidence:.2f})")
        
        # Step 2: Route to appropriate pipeline
        if query_type == 'ultra_complex':
            # Ultra-complex - needs user confirmation
            print(f"\n⚠️  ULTRA-COMPLEX QUERY DETECTED")
            print(f"⏱️  Estimated time: ~2 minutes")
            print(f"📋 This will perform deep analysis on all employees")
            
            # Ask for confirmation (unless auto-confirmed)
            if not auto_confirm_ultra:
                response = input("\n❓ Continue? (yes/no): ").strip().lower()
                
                if response not in ['yes', 'y']:
                    return {
                        'answer': "Query cancelled by user.",
                        'query_type': query_type,
                        'confidence': 0.0,
                        'processing_time': 0,
                        'pipeline_used': 'ultra_complex',
                        'metadata': {'cancelled': True},
                        'cancelled': True
                    }
            
            print(f"\n🚀 Processing with ultra-complex pipeline...")
            result = self.ultra_complex_pipeline.process(query)
            
            response = {
                'answer': result['answer'],
                'query_type': query_type,
                'confidence': bert_confidence,
                'processing_time': result['processing_time'],
                'pipeline_used': 'ultra_complex',
                'metadata': {
                    'total_analyzed': result['total_analyzed'],
                    'batch_results': len(result['batch_results'])
                }
            }
        
        elif query_type == 'aggregation':
            # Aggregation - use MongoDB
            print(f"\n🚀 Processing with aggregation pipeline...")
            result = self.aggregation_pipeline.process(query)
            
            response = {
                'answer': result['answer'],
                'query_type': query_type,
                'confidence': result['confidence'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'pipeline_used': 'aggregation',
                'metadata': {
                    'mongodb_operation': result['mongodb_operation'],
                    'result': result['result'],
                    'result_type': result['result_type']
                }
            }
        
        else:  # simple
            # Simple - use ChromaDB retrieval
            print(f"\n🚀 Processing with simple pipeline...")
            result = self.simple_pipeline.process(query)
            
            response = {
                'answer': result['answer'],
                'query_type': query_type,
                'confidence': result['confidence'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'pipeline_used': 'simple',
                'metadata': {
                    'num_sources': result['num_sources'],
                    'sources': result['sources'][:3]  # Top 3 sources
                }
            }
        
        # Display results
        self._display_results(response)
        
        return response
    
    async def ask_stream(
    self,
    query: str,
    auto_confirm_ultra: bool = False,
    conversation_history: List[Dict[str, str]] = None
) -> AsyncIterator[dict]:
        """
        Ask a question with streaming response
        
        Args:
            query: User's question
            auto_confirm_ultra: Auto-confirm ultra-complex (skip prompt)
        
        Yields:
            dict: Various message types during processing
        """
        from datetime import datetime
        start_time = datetime.now()
        
        # Classify query
        yield {'type': 'status', 'message': 'Analyzing query...'}
        classification = self.classifier.classify(query)
        
        query_type = classification['label']
        bert_confidence = classification['confidence']
        
        yield {
            'type': 'classification',
            'query_type': query_type,
            'confidence': bert_confidence
        }
        
        # Route to pipeline with streaming
        if query_type == 'ultra_complex':
            # Check if confirmed (in real app, handle this via frontend)
            # if not auto_confirm_ultra:
            #     yield {
            #         'type': 'confirmation_required',
            #         'message': 'Ultra-complex query detected. Estimated time: ~2 minutes. Continue?',
            #         'query_type': query_type
            #     }
            #     # In frontend, wait for user confirmation, then call again with auto_confirm_ultra=True
            #     return
            
            # Stream ultra-complex pipeline
            async for chunk in self.ultra_complex_pipeline.process_stream(query,                    conversation_history=conversation_history):
                yield chunk
        
        elif query_type == 'aggregation':
            # Stream aggregation pipeline
            async for chunk in self.aggregation_pipeline.process_stream(query, conversation_history=conversation_history):
                yield chunk
        
        else:  # simple
            # Stream simple pipeline
            async for chunk in self.simple_pipeline.process_stream(query, conversation_history=conversation_history):
                yield chunk
        
        # Final completion message
        processing_time = (datetime.now() - start_time).total_seconds()
        yield {
            'type': 'complete',
            'query_type': query_type,
            'processing_time': processing_time
        }
    
    def _display_results(self, response: Dict[str, Any]):
        """Display results to user"""
        print(f"\n{'=' * 70}")
        print(f"ANSWER")
        print("=" * 70)
        print(f"\n{response['answer']}")
        print(f"\n{'=' * 70}")
        print(f"📊 Query Type: {response['query_type']}")
        print(f"🔧 Pipeline: {response['pipeline_used']}")
        print(f"⏱️  Time: {response['processing_time']:.1f}s")
        print(f"✅ Confidence: {response['confidence']:.2f}")
        
        # Show additional metadata based on pipeline
        if response['pipeline_used'] == 'aggregation':
            print(f"🔢 Result: {response['metadata'].get('result')}")
        elif response['pipeline_used'] == 'ultra_complex':
            print(f"📈 Employees Analyzed: {response['metadata'].get('total_analyzed')}")
        elif response['pipeline_used'] == 'simple':
            print(f"📚 Sources: {response['metadata'].get('num_sources')}")
        
        print("=" * 70)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        from retrieval.retrieval_engine import RetrievalEngine
        from pymongo import MongoClient
        from config import Config
        
        # ChromaDB stats
        retrieval_engine = RetrievalEngine()
        chroma_stats = retrieval_engine.get_stats()
        
        # MongoDB stats
        client = MongoClient(Config.MONGODB_URI)
        db = client[Config.MONGODB_DB_NAME]
        mongo_stats = {
            'employees': db['employees_structured'].count_documents({})
        }
        client.close()
        
        return {
            'status': 'operational',
            'chromadb_collections': chroma_stats,
            'mongodb_collections': mongo_stats,
            'total_chromadb_docs': sum(chroma_stats.values()),
            'total_mongodb_docs': sum(mongo_stats.values()),
            'classifier_model': 'DistilBERT (fine-tuned)',
            'llm_model': Config.LLM_MODEL
        }
    
    def close(self):
        """Close all connections"""
        logger.info("Closing connections...")
        self.aggregation_pipeline.close()
        self.ultra_complex_pipeline.close()
        logger.info("✅ Connections closed")


def main():
    """Interactive chatbot mode"""
    print("=" * 70)
    print("HR CHATBOT - INTERACTIVE MODE")
    print("=" * 70)
    
    # Initialize chatbot
    chatbot = HRChatbot()
    
    # Show stats
    print("\n📊 SYSTEM STATS:")
    stats = chatbot.get_stats()
    print(f"  - ChromaDB Documents: {stats['total_chromadb_docs']}")
    print(f"  - MongoDB Employees: {stats['mongodb_collections']['employees']}")
    print(f"  - LLM: {stats['llm_model']}")
    print(f"  - Classifier: {stats['classifier_model']}")
    
    # Test queries
    print("\n" + "=" * 70)
    print("RUNNING TEST QUERIES")
    print("=" * 70)
    
    test_queries = [
        # Simple
        ("What is the copay for primary care in PPO 1000?", "simple"),
        
        # Aggregation
        ("How many employees have H-1B visas?", "aggregation"),
        ("Calculate the average salary", "aggregation"),
        
        #Ultra-complex
        ("Predict which employees are candidates for raises", "ultra_complex")
    ]
    
    for query, expected_type in test_queries:
        result = chatbot.ask(query, auto_confirm_ultra=True)
        
        # Verify classification
        if result['query_type'] == expected_type:
            print(f"\n✅ Classification correct: {expected_type}")
        else:
            print(f"\n⚠️  Classification mismatch: expected {expected_type}, got {result['query_type']}")
        
        input("\nPress Enter to continue to next query...")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 70)
    
    # Interactive mode
    print("\n💬 Entering interactive mode (type 'exit' to quit)...")
    print("Commands:")
    print("  - Type any question")
    print("  - 'stats' - Show system statistics")
    print("  - 'exit' - Quit")
    
    while True:
        print("\n" + "-" * 70)
        user_query = input("Your question: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_query.lower() == 'stats':
            stats = chatbot.get_stats()
            print(f"\n📊 System Stats:")
            print(f"  ChromaDB: {stats['total_chromadb_docs']} documents")
            print(f"  MongoDB: {stats['mongodb_collections']['employees']} employees")
            print(f"  Status: {stats['status']}")
            continue
        
        if not user_query:
            continue
        
        chatbot.ask(user_query)
    
    # Cleanup
    chatbot.close()


if __name__ == "__main__":
    main()
