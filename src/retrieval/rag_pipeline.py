"""
SIMPLE RAG PIPELINE - Fast Path for Basic Queries

A streamlined RAG implementation that wraps the multi-agent core.
Used for 70% of queries that don't need complex reasoning.

ARCHITECTURE:
    Query → Retrieve Documents → Build Context → LLM Generate → Response

WHY SIMPLE PATH?
- Fast execution (~2-3 seconds)
- Sufficient for direct lookups
- Lower computational cost
- Good for "What is X?" queries

WHEN TO USE:
- Direct fact lookups
- Single entity queries
- Simple definitions
- Policy/document retrieval
"""

import os
import sys
from typing import Dict, Any, List
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retrieval_engine import RetrievalEngine
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAGPipeline:
    """
    Simple RAG Pipeline - Retrieve + Generate
    
    FLOW:
    1. Retrieve relevant documents (hybrid search)
    2. Build context from documents
    3. Generate answer with LLM
    4. Return formatted response
    
    This is the "fast path" - simple but effective for most queries.
    """
    
    def __init__(
        self,
        chroma_path: str = "data/embeddings",
        ollama_model: str = "qwen2.5:14b",
        top_k: int = 5
    ):
        """
        Initialize simple RAG pipeline
        
        Args:
            chroma_path: Path to ChromaDB storage
            ollama_model: Ollama model name
            top_k: Number of documents to retrieve
        """
        logger.info("Initializing Simple RAG Pipeline...")
        
        self.top_k = top_k
        
        # Initialize components
        self.retrieval_engine = RetrievalEngine(chroma_path=chroma_path)
        self.context_builder = ContextBuilder(max_context_tokens=2000)
        self.llm = OllamaLLM(model=ollama_model, temperature=0.2)
        
        logger.info("✅ Simple RAG Pipeline ready!")
    
    def query(
        self,
        question: str,
        top_k: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            top_k: Override default number of documents
            temperature: Override default LLM temperature
        
        Returns:
            {
                'answer': str,              # Generated answer
                'sources': List[dict],      # Source documents
                'num_sources': int,         # Number of sources used
                'confidence': float,        # Confidence score (0-1)
                'retrieval_strategy': str   # Strategy used
            }
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        k = top_k or self.top_k
        retrieved_docs = self.retrieval_engine.retrieve(
            query=question,
            top_k=k,
            strategy="hybrid"
        )
        
        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'num_sources': 0,
                'confidence': 0.0,
                'retrieval_strategy': 'hybrid'
            }
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Build context
        context = self.context_builder.build_context(
            query=question,
            retrieved_docs=retrieved_docs,
            include_metadata=True
        )
        
        # Step 3: Build prompt with instructions
        system_instructions = """You are an HR assistant for Itlize Global.
Answer questions accurately based on the provided context.
Be concise and professional.
If the answer is not in the context, say so clearly.
Always cite which source(s) you used."""
        
        prompt = self.context_builder.build_prompt(
            query=question,
            context=context,
            system_instructions=system_instructions
        )
        
        # Step 4: Generate answer
        temp = temperature or 0.2
        llm_response = self.llm.generate(
            prompt=prompt,
            temperature=temp
        )
        
        # Step 5: Calculate confidence based on retrieval scores
        avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        confidence = min(avg_score, 1.0)
        
        # Step 6: Format response
        result = {
            'answer': llm_response['text'],
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'confidence': confidence,
            'retrieval_strategy': 'hybrid',
            'llm_tokens': llm_response.get('tokens', 0),
            'llm_duration_ms': llm_response.get('duration_ms', 0)
        }
        
        logger.info(f"✅ Query processed (confidence: {confidence:.2f})")
        return result
    
    def batch_query(
        self,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions
        
        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} queries")
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Batch {i}/{len(questions)}: {question}")
            result = self.query(question)
            results.append(result)
        
        logger.info("✅ Batch processing complete")
        return results


def main():
    """
    Test simple RAG pipeline
    """
    print("="*70)
    print("SIMPLE RAG PIPELINE - TESTING")
    print("="*70)
    
    # Initialize pipeline
    pipeline = SimpleRAGPipeline()
    
    # Test queries
    test_queries = [
        "What is employee 1503's position?",
        "What is the copay for primary care in PPO 1000?",
        "What are the dental benefits?",
        "How many sick days do employees get?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("-"*70)
        
        result = pipeline.query(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {result['num_sources']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"LLM Time: {result['llm_duration_ms']}ms")
        
    print("\n" + "="*70)
    print("✅ Simple pipeline testing complete!")


if __name__ == "__main__":
    main()