"""
MongoDB Query Agent
===================

Converts natural language queries to MongoDB operations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from retrieval.llm_interface import OllamaLLM
from pymongo import MongoClient
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBQueryAgent:
    """
    Converts natural language to MongoDB queries and executes them
    
    Flow:
    1. User query → LLM generates MongoDB operation
    2. Execute MongoDB query
    3. Return structured result
    """
    
    def __init__(self):
        """Initialize MongoDB query agent"""
        logger.info("Initializing MongoDB Query Agent...")
        
        # Connect to MongoDB
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.MONGODB_DB_NAME]
        self.collection = self.db['employees_structured']
        
        # Initialize LLM
        self.llm = OllamaLLM()
        
        logger.info("✅ MongoDB Query Agent ready!")
    
    def execute(self, query: str) -> dict:
        """
        Execute natural language query on MongoDB
        
        Args:
            query: Natural language question
        
        Returns:
            {
                'query': str,
                'mongodb_operation': dict,
                'result': Any,
                'result_type': str,
                'success': bool
            }
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Generate MongoDB query
        mongodb_op = self._generate_mongodb_query(query)
        
        if not mongodb_op:
            return {
                'query': query,
                'mongodb_operation': None,
                'result': None,
                'result_type': 'error',
                'success': False,
                'error': 'Failed to generate MongoDB query'
            }
        
        logger.info(f"Generated MongoDB operation: {mongodb_op}")
        
        # Step 2: Execute MongoDB operation
        try:
            result = self._execute_mongodb_operation(mongodb_op)
            
            return {
                'query': query,
                'mongodb_operation': mongodb_op,
                'result': result['value'],
                'result_type': result['type'],
                'success': True
            }
        
        except Exception as e:
            logger.error(f"MongoDB execution failed: {e}")
            return {
                'query': query,
                'mongodb_operation': mongodb_op,
                'result': None,
                'result_type': 'error',
                'success': False,
                'error': str(e)
            }
    
    def _generate_mongodb_query(self, query: str) -> dict:
        """
        Use LLM to generate MongoDB query
        
        Args:
            query: Natural language question
        
        Returns:
            MongoDB operation dict or None
        """
        # Build prompt
        prompt = Config.SYSTEM_PROMPTS['MONGODB_QUERY_PROMPT'].format(query=query)
        
        # Generate with low temperature (precise)
        response = self.llm.generate(prompt, temperature=0.1)
        
        # Extract JSON from response
        try:
            response_text = response['text'].strip()
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            # Parse JSON
            mongodb_op = json.loads(response_text)
            
            return mongodb_op
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response['text']}")
            return None
    
    def _execute_mongodb_operation(self, mongodb_op: dict) -> dict:
        """
        Execute MongoDB operation
        
        Args:
            mongodb_op: MongoDB operation dict
        
        Returns:
            {
                'type': 'count' | 'aggregate' | 'documents',
                'value': result value
            }
        """
        operation = mongodb_op.get('operation')
        
        if operation == 'count':
            # Count documents
            filter_query = mongodb_op.get('filter', {})
            count = self.collection.count_documents(filter_query)
            
            return {
                'type': 'count',
                'value': count
            }
        
        elif operation == 'aggregate':
            # Aggregation pipeline
            pipeline = mongodb_op.get('pipeline', [])
            results = list(self.collection.aggregate(pipeline))

            # Convert ObjectId to string for ALL results
            for doc in results:
                if '_id' in doc:
                    # Handle both ObjectId and dict _id
                    if hasattr(doc['_id'], '__str__'):
                        doc['_id'] = str(doc['_id'])
            
            return {
                'type': 'aggregate',
                'value': results
            }
        
        elif operation == 'find':
            # Find documents
            filter_query = mongodb_op.get('filter', {})
            limit = mongodb_op.get('limit', 100)
            
            results = list(self.collection.find(filter_query).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return {
                'type': 'documents',
                'value': results
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


def main():
    """Test MongoDB query agent"""
    print("=" * 70)
    print("MONGODB QUERY AGENT - TESTING")
    print("=" * 70)
    
    agent = MongoDBQueryAgent()
    
    # Test queries
    test_queries = [
        "How many employees have H-1B visas?",
        "Calculate the average salary",
        "How many Software Developers are there?",
        "Count employees with salary over $100,000",
        "How many employees have both health insurance and 401k?",
        "What is the total number of active employees?",
        "Average salary by position",
        "How many employees are on client assignment?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print("-" * 70)
        
        result = agent.execute(query)
        
        if result['success']:
            print(f"\n✅ Success!")
            print(f"MongoDB Operation: {json.dumps(result['mongodb_operation'], indent=2)}")
            print(f"Result Type: {result['result_type']}")
            print(f"Result: {result['result']}")
        else:
            print(f"\n❌ Failed!")
            print(f"Error: {result.get('error')}")
        
        input("\nPress Enter to continue...")
    
    agent.close()
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()