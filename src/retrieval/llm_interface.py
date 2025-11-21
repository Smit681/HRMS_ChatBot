"""
LLM INTERFACE - Connects to Ollama for Text Generation

Handles all interactions with the local LLM (DeepSeek-R1) via Ollama.

WHY OLLAMA?
- Free, runs locally (no API costs)
- GPU acceleration (uses your T4)
- Multiple model support
- Simple API

KEY CONCEPTS:
- Prompt: Formatted input to LLM (context + question)
- Temperature: Randomness (0=deterministic, 1=creative)
- Streaming: Get response word-by-word (faster perceived response)
- Token limits: Max input/output length
"""

import requests
import json
from typing import Dict, Any, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Interface to Ollama-hosted LLMs
    
    RESPONSIBILITIES:
    1. Send prompts to Ollama API
    2. Handle streaming/non-streaming responses
    3. Manage errors and retries
    4. Track token usage
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1000
    ):
        """
        Initialize Ollama LLM interface
        
        Args:
            model: Model name (e.g., "qwen2.5:14b", "llama2:7b")
            base_url: Ollama API endpoint
            temperature: Sampling temperature (0=factual, 1=creative)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized Ollama interface: {model} @ {base_url}")
        
        # Verify Ollama is running
        try:
            self._check_ollama()
        except Exception as e:
            logger.error(f"Ollama not accessible: {e}")
            raise
    
    def _check_ollama(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
            else:
                logger.info(f"✅ Model {self.model} is available")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response from LLM (non-streaming)
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            system_prompt: Optional system-level instructions
        
        Returns:
            {
                'text': str,           # Generated response
                'tokens': int,         # Tokens used
                'duration_ms': int,    # Generation time
                'model': str           # Model used
            }
        """
        # Build request
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature,
                'num_predict': max_tokens or self.max_tokens
            }
        }
        
        if system_prompt:
            payload['system'] = system_prompt
        
        logger.info(f"Generating response (temp={payload['options']['temperature']})")
        
        try:
            # Send request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return {
                'text': result.get('response', ''),
                'tokens': result.get('eval_count', 0),
                'duration_ms': result.get('total_duration', 0) // 1_000_000,  # Convert ns to ms
                'model': result.get('model', self.model)
            }
            
        except requests.exceptions.Timeout:
            logger.error("LLM generation timed out")
            return {
                'text': "I apologize, but the request timed out. Please try again or rephrase your question.",
                'tokens': 0,
                'duration_ms': 120000,
                'model': self.model,
                'error': 'timeout'
            }
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                'text': f"I encountered an error: {str(e)}",
                'tokens': 0,
                'duration_ms': 0,
                'model': self.model,
                'error': str(e)
            }
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Generate response with streaming (word-by-word)
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            system_prompt: Optional system instructions
        
        Yields:
            str: Response chunks as they're generated
        """
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': True,
            'options': {
                'temperature': temperature or self.temperature
            }
        }
        
        if system_prompt:
            payload['system'] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                    if chunk.get('done', False):
                        break
                        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn chat (maintains conversation history)
        
        Args:
            messages: List of {'role': 'user'|'assistant', 'content': str}
            temperature: Override default temperature
        
        Returns:
            Dict with response and metadata
        """
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                'text': result.get('message', {}).get('content', ''),
                'role': result.get('message', {}).get('role', 'assistant'),
                'tokens': result.get('eval_count', 0),
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                'text': f"Error: {str(e)}",
                'role': 'assistant',
                'tokens': 0,
                'model': self.model,
                'error': str(e)
            }


def main():
    """
    Test LLM interface
    """
    print("="*70)
    print("LLM INTERFACE - TESTING")
    print("="*70)
    
    # Initialize
    llm = OllamaLLM(model="qwen2.5:14b")
    
    # Test 1: Simple generation
    print("\n--- Test 1: Simple Generation ---")
    result = llm.generate(
        "What is 2 + 2? Answer briefly.",
        temperature=0.1
    )
    print(f"Response: {result['text']}")
    print(f"Tokens: {result['tokens']}, Duration: {result['duration_ms']}ms")
    
    # Test 2: HR query
    print("\n--- Test 2: HR Context Query ---")
    context = "Employee 1503 is a Technical Project Manager earning $135,000 annually."
    prompt = f"Context: {context}\n\nQuestion: What is employee 1503's position?\n\nAnswer:"
    
    result = llm.generate(prompt, temperature=0.2)
    print(f"Response: {result['text']}")
    
    # Test 3: Streaming
    print("\n--- Test 3: Streaming Response ---")
    print("Response: ", end='', flush=True)
    for chunk in llm.generate_stream("List 3 benefits of using RAG systems. Be brief."):
        print(chunk, end='', flush=True)
    print("\n")
    
    print("="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()