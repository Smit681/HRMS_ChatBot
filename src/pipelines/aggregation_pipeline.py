"""
Aggregation Pipeline - MongoDB-Powered Queries
===============================================

Handles aggregation queries using MongoDB:

Flow:
1. User query → MongoDB Query Agent (generates MongoDB query)
2. Execute on MongoDB (fast, accurate)
3. Format result with LLM
4. Return answer
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from retrieval.llm_interface import OllamaLLM
from retrieval.context_builder import ContextBuilder
from agents.mongodb_query_agent import MongoDBQueryAgent
from typing import Dict, Any, Iterator
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregationPipeline:
    
    def __init__(self):
        """Initialize aggregation pipeline"""
        logger.info("Initializing Aggregation Pipeline...")
        
        self.mongodb_agent = MongoDBQueryAgent()
        self.llm = OllamaLLM()
        self.context_builder = ContextBuilder()
        
        logger.info("✅ Aggregation Pipeline ready!")
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process an aggregation query
        
        Args:
            query: User's question
        
        Returns:
            {
                'answer': str,
                'mongodb_operation': dict,
                'result': Any,
                'result_type': str,
                'num_sources': int,
                'confidence': float,
                'pipeline': str
            }
        """
        logger.info(f"[Aggregation Pipeline] Processing: {query}")
        
        # Step 1: Execute MongoDB query
        logger.info("[1/3] Executing MongoDB query...")
        mongo_result = self.mongodb_agent.execute(query)
        
        if not mongo_result['success']:
            logger.warning("MongoDB query failed")
            return self._error_response(query, mongo_result.get('error'))
        
        logger.info(f"MongoDB result: {mongo_result['result']}")
        
        # Step 2: Format result for LLM
        logger.info("[2/3] Formatting result...")
        context = self._format_mongo_result(query, mongo_result)
        
        # Step 3: Generate natural language answer
        logger.info("[3/3] Generating answer...")
        answer = self._generate_answer(query, context)
        
        result = {
            'answer': answer,
            'mongodb_operation': mongo_result['mongodb_operation'],
            'result': mongo_result['result'],
            'result_type': mongo_result['result_type'],
            'num_sources': self._count_sources(mongo_result),
            'confidence': 0.95,  # MongoDB results are highly accurate
            'pipeline': 'aggregation'
        }
        
        logger.info("✅ Query complete")
        return result
    
    def process_stream(self, query: str) -> Iterator[dict]:
        """
        Process aggregation query with streaming
        
        Args:
            query: User's question
        
        Yields:
            dict: Status updates and token chunks
        """
        logger.info(f"[Aggregation Pipeline] Processing with streaming: {query}")
        
        # Step 1: Execute MongoDB query (not streamed)
        yield {'type': 'status', 'message': 'Executing MongoDB query...'}
        mongo_result = self.mongodb_agent.execute(query)
        
        if not mongo_result['success']:
            yield {'type': 'error', 'message': mongo_result.get('error')}
            return
        
        yield {'type': 'status', 'message': f"Result: {mongo_result['result']}"}
        
        # Step 2: Format result
        yield {'type': 'status', 'message': 'Formatting result...'}
        context = self._format_mongo_result(query, mongo_result)
        
        # Step 3: Generate natural language answer with streaming
        yield {'type': 'status', 'message': 'Generating answer...'}
        
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['aggregation']
        )
        
        # Stream tokens
        for token in self.llm.generate_stream(prompt, temperature=0.2):
            yield {'type': 'token', 'content': token}
        
        # Send metadata
        yield {
            'type': 'metadata',
            'mongodb_operation': mongo_result['mongodb_operation'],
            'result': mongo_result['result'],
            'result_type': mongo_result['result_type'],
            'pipeline': 'aggregation'
        }
    
    def _format_mongo_result(self, query: str, mongo_result: dict) -> str:
        """
        Format MongoDB result for LLM consumption
        
        Args:
            query: Original query
            mongo_result: MongoDB execution result
        
        Returns:
            Formatted context string
        """
        result_type = mongo_result['result_type']
        result_value = mongo_result['result']
        
        context_parts = []
        context_parts.append("MongoDB Query Result:")
        context_parts.append("=" * 70)
        
        if result_type == 'count':
            # Simple count
            context_parts.append(f"Total Count: {result_value}")
        
        elif result_type == 'aggregate':
            # Aggregation results
            context_parts.append("Aggregation Results:")
            
            if isinstance(result_value, list):
                for item in result_value:
                    context_parts.append(json.dumps(item, indent=2, default=str))
            else:
                context_parts.append(str(result_value))
        
        elif result_type == 'documents':
            # Document list
            context_parts.append(f"Found {len(result_value)} documents:")
            
            # Show first 5 as examples
            for i, doc in enumerate(result_value[:5], 1):
                context_parts.append(f"\nDocument {i}:")
                context_parts.append(f"  Employee ID: {doc.get('employeeId')}")
                context_parts.append(f"  Position: {doc.get('position')}")
                context_parts.append(f"  Salary: ${doc.get('salary'):,.2f}" if doc.get('salary') else "  Salary: N/A")
                
                if doc.get('visas'):
                    visa_types = [v['visaType'] for v in doc['visas']]
                    context_parts.append(f"  Visas: {', '.join(visa_types)}")
            
            if len(result_value) > 5:
                context_parts.append(f"\n... and {len(result_value) - 5} more")
        
        context_parts.append("=" * 70)
        
        # Add MongoDB operation for transparency
        context_parts.append("\nMongoDB Operation Used:")
        context_parts.append(json.dumps(mongo_result['mongodb_operation'], indent=2))
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate natural language answer from MongoDB result
        
        Args:
            query: Original query
            context: Formatted MongoDB result
        
        Returns:
            Natural language answer
        """
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['aggregation']
        )
        
        response = self.llm.generate(prompt, temperature=0.2)
        return response['text']
    
    def _count_sources(self, mongo_result: dict) -> int:
        """Count how many sources were analyzed"""
        if mongo_result['result_type'] == 'documents':
            return len(mongo_result['result'])
        elif mongo_result['result_type'] == 'count':
            return mongo_result['result']
        else:
            return 1
    
    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Handle errors"""
        return {
            'answer': f"I encountered an error processing your query: {error}",
            'mongodb_operation': None,
            'result': None,
            'result_type': 'error',
            'num_sources': 0,
            'confidence': 0.0,
            'pipeline': 'aggregation'
        }
    
    def close(self):
        """Close MongoDB connection"""
        self.mongodb_agent.close()


def main():
    """Test aggregation pipeline"""
    print("=" * 70)
    print("AGGREGATION PIPELINE - TESTING")
    print("=" * 70)
    
    pipeline = AggregationPipeline()
    
    test_queries = [
        "How many employees have H-1B visas?",
        "Calculate the average salary",
        "How many Software Developers are there?",
        "Count employees with salary over $100,000",
        "How many employees have both health insurance and 401k?",
        "What is the total number of active employees?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print("-" * 70)
        
        result = pipeline.process(query)
        
        print(f"\nA: {result['answer']}")
        print(f"\nMongoDB Operation: {json.dumps(result['mongodb_operation'], indent=2)}")
        print(f"Result: {result['result']}")
        print(f"Sources analyzed: {result['num_sources']}")
        
        input("\nPress Enter to continue...")
    
    pipeline.close()
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()