"""
MULTI-AGENT ORCHESTRATOR - System Coordinator

The "brain" that coordinates all agents for complex queries.

FLOW:
1. Classify query (determine intent)
2. Retrieve data (using appropriate strategy)
3. Calculate (if numbers needed)
4. Generate response (LLM)
5. Validate (check quality)
6. Return (formatted response)

WHY ORCHESTRATOR?
- Complex queries need multiple steps
- Different agents specialize in different tasks
- Coordinator ensures everything works together
- Can retry/fallback if steps fail
"""

import os
import sys
from typing import Dict, Any, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_classifier import QueryClassifier
from agents.data_retriever import DataRetriever
from agents.calculator import Calculator
from agents.validator import Validator
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents for complex query processing
    
    AGENTS:
    1. QueryClassifier - Understand intent
    2. DataRetriever - Get relevant documents
    3. Calculator - Precise calculations
    4. LLM - Generate natural language
    5. Validator - Quality assurance
    
    FLOW:
    Query → Classify → Retrieve → Calculate → Generate → Validate → Response
    """
    
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        ollama_model: str = "qwen2.5:14b"
    ):
        """
        Initialize multi-agent orchestrator
        
        Args:
            chroma_path: Path to ChromaDB storage
            ollama_model: Ollama model name
        """
        logger.info("Initializing Multi-Agent Orchestrator...")
        
        # Initialize all agents
        self.classifier = QueryClassifier()
        self.retriever = DataRetriever(chroma_path=chroma_path)
        self.calculator = Calculator()
        self.validator = Validator()
        self.context_builder = ContextBuilder(max_context_tokens=2000)
        self.llm = OllamaLLM(model=ollama_model)
        
        logger.info("✅ Multi-Agent Orchestrator ready!")
    
    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a complex query through multi-agent pipeline
        
        Args:
            query: User's question
            session_id: Session identifier (for conversation tracking)
        
        Returns:
            {
                'answer': str,              # Final answer
                'sources': List[dict],      # Source documents
                'confidence': float,        # Confidence score
                'classification': dict,     # Query classification
                'calculations': dict,       # Calculation results
                'validation': dict          # Validation results
            }
        """
        logger.info("="*70)
        logger.info(f"Processing complex query: {query}")
        logger.info("="*70)
        
        # Step 1: Classify query
        logger.info("\n[Step 1] CLASSIFY QUERY")
        classification = self.classifier.classify(query)
        logger.info(f"  Type: {classification['query_type']}")
        logger.info(f"  Complexity: {classification['complexity']}")
        logger.info(f"  Strategy: {classification['suggested_strategy']}")
        
        # Step 2: Retrieve relevant documents
        logger.info("\n[Step 2] RETRIEVE DATA")
        retrieved_docs = self.retriever.retrieve(
            query=query,
            strategy=classification['suggested_strategy'],
            entities=classification['entities'],
            top_k=5
        )
        logger.info(f"  Retrieved: {len(retrieved_docs)} documents")
        
        if not retrieved_docs:
            logger.warning("  No documents retrieved!")
            return self._no_data_response(query, classification)
        
        # Step 3: Perform calculations (if needed)
        calculations = None
        if classification['query_type'] == 'aggregation':
            logger.info("\n[Step 3] CALCULATE")
            calculations = self._perform_calculations(
                query,
                retrieved_docs,
                classification
            )
            logger.info(f"  Result: {calculations.get('result')}")
        else:
            logger.info("\n[Step 3] CALCULATE (skipped - not needed)")
        
        # Step 4: Build context
        logger.info("\n[Step 4] BUILD CONTEXT")
        context = self.context_builder.build_context(
            query=query,
            retrieved_docs=retrieved_docs,
            include_metadata=True
        )
        
        # Add calculation results to context
        if calculations:
            calc_text = f"\n\nCalculation Result: {calculations['operation']} = {calculations['result']}"
            context += calc_text
        
        # Step 5: Generate response with LLM
        logger.info("\n[Step 5] GENERATE RESPONSE")
        response_text = self._generate_response(
            query=query,
            context=context,
            classification=classification,
            calculations=calculations
        )
        logger.info(f"  Generated {len(response_text)} characters")
        
        # Step 6: Validate response
        logger.info("\n[Step 6] VALIDATE")
        validation = self.validator.validate(
            response=response_text,
            query=query,
            sources=retrieved_docs,
            calculations=calculations
        )
        logger.info(f"  Valid: {validation['is_valid']}")
        logger.info(f"  Confidence: {validation['confidence']}")
        
        if validation['issues']:
            logger.warning(f"  Issues: {validation['issues']}")
        
        # Step 7: Format final response
        logger.info("\n[Step 7] FORMAT RESPONSE")
        final_response = {
            'answer': response_text,
            'sources': retrieved_docs,
            'confidence': validation['confidence'],
            'classification': classification,
            'calculations': calculations,
            'validation': validation,
            'num_sources': len(retrieved_docs)
        }
        
        logger.info("="*70)
        logger.info("✅ Query processing complete!")
        logger.info("="*70)
        
        return final_response
    
    def _perform_calculations(
        self,
        query: str,
        documents: list,
        classification: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Determine what calculation to perform and execute it
        
        DETECTION:
        - "How many" → COUNT
        - "Total/Sum" → SUM
        - "Average/Mean" → AVERAGE
        - "Highest/Lowest" → MAX/MIN
        """
        query_lower = query.lower()
        
        # Determine operation
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            operation = 'count'
            field = None
        elif 'average' in query_lower or 'mean' in query_lower:
            operation = 'average'
            field = self._infer_numeric_field(query_lower)
        elif 'total' in query_lower or 'sum' in query_lower:
            operation = 'sum'
            field = self._infer_numeric_field(query_lower)
        elif 'highest' in query_lower or 'maximum' in query_lower or 'max' in query_lower:
            operation = 'max'
            field = self._infer_numeric_field(query_lower)
        elif 'lowest' in query_lower or 'minimum' in query_lower or 'min' in query_lower:
            operation = 'min'
            field = self._infer_numeric_field(query_lower)
        else:
            # Default to count
            operation = 'count'
            field = None
        
        # Execute calculation
        result = self.calculator.calculate(
            operation=operation,
            data=documents,
            field=field
        )
        
        return result
    
    def _infer_numeric_field(self, query: str) -> Optional[str]:
        """
        Infer which numeric field to calculate on
        
        EXAMPLES:
        - "salary" → 'salary'
        - "premium" → 'premium'
        - "copay" → 'copay'
        """
        if 'salary' in query:
            return 'salary'
        elif 'premium' in query:
            return 'premium'
        elif 'copay' in query:
            return 'copay'
        elif 'deductible' in query:
            return 'deductible'
        else:
            return 'value'  # Generic field name
    
    def _generate_response(
        self,
        query: str,
        context: str,
        classification: dict,
        calculations: Optional[dict]
    ) -> str:
        """
        Generate natural language response using LLM
        
        CUSTOMIZED INSTRUCTIONS based on query type:
        - Aggregation → Include the calculated number
        - Comparison → Structured comparison format
        - Filter → List matching items
        """
        # Build system instructions based on query type
        query_type = classification['query_type']
        
        if query_type == 'aggregation':
            system_instructions = """You are an HR analytics assistant.
Answer the question using the provided context and calculation results.
Be precise with numbers - use the exact calculation result.
Format: Give the answer directly, then briefly explain the context.
Keep it concise and professional."""
        
        elif query_type == 'comparison':
            system_instructions = """You are an HR comparison assistant.
Compare the items objectively using the provided context.
Structure: Present key differences clearly.
Be balanced and factual."""
        
        else:
            system_instructions = """You are an HR assistant.
Answer accurately based on the provided context.
Be concise and professional.
Cite your sources."""
        
        # Build prompt
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_instructions=system_instructions
        )
        
        # Generate with appropriate temperature
        temperature = 0.1 if query_type == 'aggregation' else 0.2
        
        llm_result = self.llm.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return llm_result['text']
    
    def _no_data_response(
        self,
        query: str,
        classification: dict
    ) -> Dict[str, Any]:
        """
        Handle case when no relevant documents found
        """
        response_text = (
            f"I couldn't find relevant information to answer your question about "
            f"'{query}'. The system searched for {classification['query_type']} "
            f"information but didn't find matching documents."
        )
        
        return {
            'answer': response_text,
            'sources': [],
            'confidence': 0.0,
            'classification': classification,
            'calculations': None,
            'validation': {
                'is_valid': False,
                'confidence': 0.0,
                'issues': ['No relevant documents found'],
                'warnings': [],
                'checks': {}
            },
            'num_sources': 0
        }


def main():
    """
    Test multi-agent orchestrator
    """
    print("="*70)
    print("MULTI-AGENT ORCHESTRATOR - TESTING")
    print("="*70)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Test queries (different types)
    test_queries = [
        # Aggregation
        "How many employees have H-1B visas?",
        
        # Comparison
        "Compare PPO 1000 and PPO 2500 medical plans",
        
        # Simple lookup (should still work)
        "What is the copay for specialists in PPO 1000?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("="*70)
        
        result = orchestrator.process_query(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources: {result['num_sources']}")
        
        if result['calculations']:
            print(f"Calculation: {result['calculations']['operation']} = {result['calculations']['result']}")
        
        if result['validation']['issues']:
            print(f"Issues: {result['validation']['issues']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()