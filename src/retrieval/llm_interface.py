"""
LLM Interface - Connects to Ollama

Sends prompts to Ollama and gets responses back.
"""

import json
import sys
from pathlib import Path
from typing import Iterator, AsyncIterator
import httpx
import asyncio
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Interface to Ollama LLM
    
    Simple: Send prompt → Get response
    """
    
    def __init__(self):
        """Initialize Ollama interface"""
        logger.info(f"Connecting to Ollama: {Config.LLM_MODEL}")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("✅ Ollama is running")
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}")
            logger.error("   Start Ollama: ollama serve")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = None
    ) -> dict:
        """
        Generate response from LLM
        
        Args:
            prompt: Input text
            temperature: Randomness (None = use config default)
        
        Returns:
            {
                'text': str,        # Generated response
                'tokens': int,      # Tokens used
                'duration_ms': int  # Time taken
            }
        """
        temperature = temperature or Config.LLM_TEMPERATURE
        
        # Build request
        payload = {
            'model': Config.LLM_MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': Config.LLM_MAX_TOKENS
            }
        }
        
        logger.info(f"Generating response (temp={temperature})...")
        
        try:
            # Send request
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return {
                'text': result.get('response', ''),
                'tokens': result.get('eval_count', 0),
                'duration_ms': result.get('total_duration', 0) // 1_000_000
            }
        
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return {
                'text': "Sorry, the request timed out. Please try again.",
                'tokens': 0,
                'duration_ms': 120000,
                'error': 'timeout'
            }
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'text': f"Error: {str(e)}",
                'tokens': 0,
                'duration_ms': 0,
                'error': str(e)
            }

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from LLM (ASYNC)
        """
        temperature = temperature or Config.LLM_TEMPERATURE
        
        payload = {
            'model': Config.LLM_MODEL,
            'prompt': prompt,
            'stream': True,
            'options': {
                'temperature': temperature,
                'num_predict': Config.LLM_MAX_TOKENS
            }
        }
        
        logger.info(f"Generating streaming response (temp={temperature})...")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{Config.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                                await asyncio.sleep(0)  # Yield control
                            
                            if chunk.get('done', False):
                                break
        
        except httpx.TimeoutException:
            logger.error("Request timed out")
            yield "Sorry, the request timed out. Please try again."
        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"


def main():
    """Test LLM interface"""
    print("=" * 70)
    print("LLM INTERFACE - TESTING")
    print("=" * 70)
    
    llm = OllamaLLM()
    
    # Test 1: Simple question
    print("\n--- Test 1: Simple Question ---")
    result = llm.generate("What is 2 + 2? Answer briefly.")
    print(f"Response: {result['text']}")
    print(f"Tokens: {result['tokens']}, Time: {result['duration_ms']}ms")
    
    # Test 2: HR question
    print("\n--- Test 2: HR Context ---")
    prompt = """Context: Employee 1503 is a Technical Project Manager earning $135,000.

Question: What is employee 1503's position?

Answer:"""
    
    result = llm.generate(prompt)
    print(f"Response: {result['text']}")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()