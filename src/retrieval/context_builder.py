"""
CONTEXT BUILDER - Formats Retrieved Documents for LLM

Takes raw retrieved documents and formats them into structured context
that the LLM can understand and use effectively.

WHY NEEDED?
- LLMs need well-formatted, structured input
- Token limits require smart truncation
- Relevance ordering improves answer quality
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds formatted context from retrieved documents
    
    RESPONSIBILITIES:
    1. Format documents into readable context
    2. Handle token limits (truncate if needed)
    3. Add metadata (sources, timestamps)
    4. Structure for optimal LLM comprehension
    """
    
    def __init__(self, max_context_tokens: int = 2000):
        """
        Initialize context builder
        
        Args:
            max_context_tokens: Maximum tokens for context (leave room for query + response)
        """
        self.max_context_tokens = max_context_tokens
        logger.info(f"Context Builder initialized (max tokens: {max_context_tokens})")
    
    def build_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """
        Build formatted context from retrieved documents
        
        Args:
            query: Original user query
            retrieved_docs: List of retrieved document dictionaries
            include_metadata: Include source/collection info
        
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
            # Document header
            if include_metadata:
                source = doc.get('collection', 'unknown').replace('_', ' ').title()
                score = doc.get('score', 0)
                context_parts.append(f"[Source {i}: {source} (Relevance: {score:.2f})]")
            else:
                context_parts.append(f"[Document {i}]")
            
            # Document content
            text = doc.get('text', '')
            context_parts.append(text)
            context_parts.append("")  # Blank line between documents
        
        # Join all parts
        full_context = "\n".join(context_parts)
        
        # Truncate if needed (simple character-based approximation)
        max_chars = self.max_context_tokens * 4  # ~4 chars per token
        if len(full_context) > max_chars:
            logger.warning(f"Context truncated from {len(full_context)} to {max_chars} characters")
            full_context = full_context[:max_chars] + "\n\n[Context truncated due to length...]"
        
        return full_context
    
    def build_prompt(
        self,
        query: str,
        context: str,
        system_instructions: str = None
    ) -> str:
        """
        Build complete prompt for LLM
        
        Args:
            query: User's question
            context: Formatted context from retrieved docs
            system_instructions: Optional system-level instructions
        
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System instructions
        if system_instructions:
            prompt_parts.append(system_instructions)
            prompt_parts.append("")
        else:
            # Default instructions
            prompt_parts.append("You are an HR assistant. Answer questions accurately based on the provided context.")
            prompt_parts.append("If the answer is not in the context, say so clearly.")
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
    """
    Test context builder
    """
    print("="*70)
    print("CONTEXT BUILDER - TESTING")
    print("="*70)
    
    # Mock retrieved documents
    mock_docs = [
        {
            'text': 'Employee 1503 works as a Technical Project Manager with a salary of $135,000.',
            'collection': 'employee_visa',
            'score': 0.95
        },
        {
            'text': 'Technical Project Managers are responsible for overseeing client projects.',
            'collection': 'general_questions',
            'score': 0.78
        }
    ]
    
    # Initialize builder
    builder = ContextBuilder(max_context_tokens=2000)
    
    # Build context
    query = "What is employee 1503's position and salary?"
    context = builder.build_context(query, mock_docs)
    
    print("\n--- CONTEXT ---")
    print(context)
    
    # Build full prompt
    prompt = builder.build_prompt(query, context)
    
    print("\n--- FULL PROMPT ---")
    print(prompt)
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()