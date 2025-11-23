"""
Context Builder - Formats Documents for LLM
============================================

Formats retrieved documents into clean context.
Now includes batch formatting for ultra-complex queries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Formats retrieved documents into LLM-ready context
    
    Methods:
    1. build_context - Normal queries
    2. build_batch_context - Ultra-complex batch analysis
    3. build_synthesis_context - Combine batch results
    4. build_prompt - Complete prompt with system instructions
    """
    
    def __init__(self):
        """Initialize context builder"""
        logger.info("Context Builder initialized")
    
    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Build formatted context from retrieved documents
        
        Args:
            query: Original user query
            retrieved_docs: Documents from retrieval engine
        
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        
        # Add header
        context_parts.append("Based on the following information:")
        context_parts.append("")
        
        # Add each document
        for i, doc in enumerate(retrieved_docs, 1):
            # Source header
            source = doc.get('collection', 'unknown').replace('_', ' ').title()
            score = doc.get('score', 0)
            context_parts.append(f"[Source {i}: {source} (Relevance: {score:.2f})]")
            
            # Document text
            text = doc.get('text', '')
            context_parts.append(text)
            context_parts.append("")  # Blank line
        
        # Join all parts
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        max_chars = Config.MAX_CONTEXT_TOKENS * 4
        if len(full_context) > max_chars:
            logger.warning(f"Context truncated: {len(full_context)} → {max_chars} chars")
            full_context = full_context[:max_chars] + "\n\n[Context truncated...]"
        
        return full_context
    
    def build_batch_context(
        self,
        query: str,
        batch_employees: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int
    ) -> str:
        """
        Build context for batch analysis (ultra-complex)
        
        Args:
            query: Original query
            batch_employees: Employee documents in this batch
            batch_num: Current batch number
            total_batches: Total number of batches
        
        Returns:
            Formatted batch context
        """
        context_parts = []
        
        # Batch header
        context_parts.append(f"Batch {batch_num}/{total_batches} - Analyzing {len(batch_employees)} employees")
        context_parts.append("")
        context_parts.append(f"Query: {query}")
        context_parts.append("")
        context_parts.append("Employee Data:")
        context_parts.append("-" * 70)
        
        # Add each employee
        for i, emp in enumerate(batch_employees, 1):
            context_parts.append(f"\nEmployee {i}:")
            context_parts.append(emp.get('text', ''))
        
        context_parts.append("")
        context_parts.append("-" * 70)
        context_parts.append("Analyze each employee and provide insights relevant to the query.")
        
        return "\n".join(context_parts)
    
    def build_synthesis_context(
        self,
        query: str,
        batch_results: List[str],
        total_employees: int
    ) -> str:
        """
        Build context for synthesizing batch results
        
        Args:
            query: Original query
            batch_results: Results from each batch
            total_employees: Total employees analyzed
        
        Returns:
            Synthesis context
        """
        context_parts = []
        
        # Header
        context_parts.append(f"Synthesizing results from {len(batch_results)} batches")
        context_parts.append(f"Total employees analyzed: {total_employees}")
        context_parts.append("")
        context_parts.append(f"Original Query: {query}")
        context_parts.append("")
        context_parts.append("Batch Results:")
        context_parts.append("=" * 70)
        
        # Add each batch result
        for i, result in enumerate(batch_results, 1):
            context_parts.append(f"\nBatch {i} Results:")
            context_parts.append("-" * 70)
            context_parts.append(result)
            context_parts.append("")
        
        context_parts.append("=" * 70)
        context_parts.append("Provide a comprehensive final answer combining all batch insights.")
        
        return "\n".join(context_parts)
    
    def build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: str = None
    ) -> str:
        """
        Build complete prompt for LLM
        
        Args:
            query: User's question
            context: Formatted context
            system_prompt: System instructions (optional)
        
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System instructions
        if system_prompt:
            prompt_parts.append(system_prompt)
        else:
            prompt_parts.append(Config.SYSTEM_PROMPTS['default'])
        
        prompt_parts.append("")
        
        # Context
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Query
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)


def main():
    """Test context builder"""
    print("=" * 70)
    print("CONTEXT BUILDER - TESTING")
    print("=" * 70)
    
    builder = ContextBuilder()
    
    # Test 1: Normal context
    print("\n--- Test 1: Normal Context ---")
    mock_docs = [
        {
            'text': 'Employee 1503 works as Technical Project Manager, salary $135,000.',
            'collection': 'employees',
            'score': 0.95
        }
    ]
    
    context = builder.build_context(mock_docs)
    print(context[:200] + "...")
    
    # Test 2: Batch context
    print("\n--- Test 2: Batch Context ---")
    batch_employees = [
        {'text': 'Employee 1503: PM, $135k, 3 years tenure'},
        {'text': 'Employee 1504: Developer, $95k, 2 years tenure'}
    ]
    
    batch_context = builder.build_batch_context(
        query="Predict raise candidates",
        batch_employees=batch_employees,
        batch_num=1,
        total_batches=5
    )
    print(batch_context[:200] + "...")
    
    # Test 3: Synthesis context
    print("\n--- Test 3: Synthesis Context ---")
    batch_results = [
        "Batch 1: Identified 2 candidates (1503, 1504)",
        "Batch 2: Identified 1 candidate (1520)"
    ]
    
    synthesis_context = builder.build_synthesis_context(
        query="Predict raise candidates",
        batch_results=batch_results,
        total_employees=20
    )
    print(synthesis_context)
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()