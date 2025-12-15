"""
Hybrid Cache Manager for HR Chatbot
====================================

Two-tier caching system:
1. Exact Match Cache: Hash-based (instant O(1) lookup)
2. Semantic Cache: Embedding similarity (slower but catches paraphrases)

Flow:
- Check exact match → HIT: return
- Check semantic similarity → HIT (>0.85): return
- MISS: process query, cache with both methods

Cache Strategy:
- Exact: SHA256 hash key
- Semantic: Store query embeddings in Redis with vector similarity search
- TTL: 1 hour (simple/aggregation), 24 hours (ultra-complex)
"""

import redis
import hashlib
import json
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
from config import Config

# Import embedding model
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HybridCacheManager:
    """
    Hybrid caching: Exact match + Semantic similarity
    """
    
    # Semantic cache settings
    SEMANTIC_THRESHOLD = 0.50  # Cosine similarity threshold (0-1)
    MAX_SEMANTIC_CANDIDATES = 100  # Max cached queries to check
    
    def __init__(self, enable_semantic: bool = True):
        """
        Initialize hybrid cache manager
        
        Args:
            enable_semantic: Enable semantic caching (default: True)
        """
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=False,  # Binary mode for embeddings
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis cache connected successfully")
            
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("Cache disabled - queries will not be cached")
            self.redis_client = None
            enable_semantic = False
        
        # Initialize semantic caching
        self.enable_semantic = enable_semantic
        self.embed_model = None
        
        if enable_semantic and self.redis_client:
            try:
                logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
                self.embed_model = SentenceTransformer(
                    Config.EMBEDDING_MODEL,
                    device=Config.EMBEDDING_DEVICE
                )
                logger.info(f"✅ Semantic caching enabled on {Config.EMBEDDING_DEVICE}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                logger.warning("Semantic caching disabled - using exact match only")
                self.enable_semantic = False
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching"""
        return ' '.join(query.lower().strip().split())
    
    def _generate_exact_key(self, query: str, query_type: str = None) -> str:
        """Generate exact match cache key (hash-based)"""
        normalized_query = self._normalize_query(query)
        cache_input = f"{query_type}:{normalized_query}" if query_type else normalized_query
        
        hash_object = hashlib.sha256(cache_input.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        return f"chatbot:exact:{hash_hex}"
    
    def _generate_semantic_key(self, query: str, query_type: str = None) -> str:
        """Generate semantic cache metadata key"""
        # Use same hash for linking exact <-> semantic
        exact_key = self._generate_exact_key(query, query_type)
        return f"chatbot:semantic:{exact_key.split(':')[-1]}"
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding for query
        
        Args:
            query: User query
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.embed_model:
            return None
        
        try:
            normalized = self._normalize_query(query)
            embedding = self.embed_model.encode(normalized, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get(self, query: str, query_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result (exact match first, then semantic)
        
        Args:
            query: User query
            query_type: Query classification type
            
        Returns:
            Cached data or None
        """
        if not self.redis_client:
            return None
        
        # Step 1: Try exact match (fast)
        exact_result = self._get_exact(query, query_type)
        if exact_result:
            logger.info(f"✅ EXACT CACHE HIT: {query[:50]}...")
            return exact_result
        
        # Step 2: Try semantic match (slower, but catches paraphrases)
        if self.enable_semantic:
            semantic_result = self._get_semantic(query, query_type)
            if semantic_result:
                logger.info(f"✅ SEMANTIC CACHE HIT: {query[:50]}...")
                logger.info(f"   Matched: {semantic_result.get('original_query', 'N/A')[:50]}...")
                return semantic_result
        
        logger.info(f"❌ Cache MISS: {query[:50]}...")
        return None
    
    def _get_exact(self, query: str, query_type: str = None) -> Optional[Dict[str, Any]]:
        """Try exact match cache"""
        try:
            exact_key = self._generate_exact_key(query, query_type)
            cached_data = self.redis_client.get(exact_key)
            
            if cached_data:
                result = json.loads(cached_data.decode('utf-8'))
                result['is_cache'] = True
                result['cache_type'] = 'exact'
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Exact cache retrieval error: {e}")
            return None
    
    def _get_semantic(self, query: str, query_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Try semantic similarity match
        
        Process:
        1. Generate query embedding
        2. Get all semantic cache keys for this query_type
        3. Calculate similarity with each
        4. Return best match if > threshold
        """
        try:
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                logger.debug("Failed to generate query embedding")
                return None
            
            # Get all semantic cache keys
            pattern = b"chatbot:semantic:*"  # Use bytes pattern
            semantic_keys = self.redis_client.keys(pattern)
            
            if not semantic_keys:
                logger.debug("No semantic cache entries found")
                return None
            
            logger.debug(f"Found {len(semantic_keys)} semantic cache entries to check")
            
            # Limit candidates to avoid slow searches
            if len(semantic_keys) > self.MAX_SEMANTIC_CANDIDATES:
                import random
                semantic_keys = random.sample(semantic_keys, self.MAX_SEMANTIC_CANDIDATES)
                logger.debug(f"Limited to {self.MAX_SEMANTIC_CANDIDATES} candidates")
            
            best_match = None
            best_similarity = 0.0
            
            # Check each cached query
            for sem_key in semantic_keys:
                try:
                    # Get semantic metadata
                    sem_data = self.redis_client.get(sem_key)
                    if not sem_data:
                        continue
                    
                    # Parse metadata
                    sem_meta = json.loads(sem_data.decode('utf-8'))
                    
                    # Check if query_type matches (if specified)
                    if query_type and sem_meta.get('query_type') != query_type:
                        continue
                    
                    # Get cached embedding - FIX: properly decode from base64
                    import base64
                    embedding_bytes = base64.b64decode(sem_meta['embedding'])
                    cached_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    logger.debug(f"Similarity with '{sem_meta.get('query', 'N/A')[:30]}...': {similarity:.3f}")
                    
                    # Track best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'semantic_key': sem_key,
                            'exact_key': sem_meta['exact_key'].encode('utf-8') if isinstance(sem_meta['exact_key'], str) else sem_meta['exact_key'],
                            'similarity': similarity,
                            'original_query': sem_meta.get('query', 'N/A')
                        }
                
                except Exception as e:
                    logger.debug(f"Error checking semantic key: {e}")
                    continue
            
            # Check if best match meets threshold
            if best_match and best_similarity >= self.SEMANTIC_THRESHOLD:
                logger.debug(f"Best match: {best_match['original_query'][:50]}... (similarity: {best_similarity:.3f})")
                
                # Retrieve the actual cached result
                cached_result = self.redis_client.get(best_match['exact_key'])
                if cached_result:
                    result = json.loads(cached_result.decode('utf-8'))
                    result['is_cache'] = True
                    result['cache_type'] = 'semantic'
                    result['similarity_score'] = best_similarity
                    result['original_query'] = best_match['original_query']
                    return result
            else:
                logger.debug(f"No match above threshold (best: {best_similarity:.3f}, threshold: {self.SEMANTIC_THRESHOLD})")
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic cache retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set(
        self,
        query: str,
        answer: str,
        metadata: Dict[str, Any],
        query_type: str = None
    ) -> bool:
        """
        Cache query result (both exact and semantic)
        
        Args:
            query: User query
            answer: Generated answer
            metadata: Result metadata
            query_type: Query classification
            
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False
        
        try:
            ttl = Config.REDIS_CACHE_TTL_SECONDS
            
            # Prepare cache data
            cache_data = {
                'answer': answer,
                'metadata': metadata,
                'query_type': query_type,
                'cached_at': datetime.utcnow().isoformat(),
                'query': query
            }
            
            cache_json = json.dumps(cache_data).encode('utf-8')
            
            # 1. Store exact match
            exact_key = self._generate_exact_key(query, query_type)
            self.redis_client.setex(
                name=exact_key,
                time=ttl,
                value=cache_json
            )
            
            # 2. Store semantic metadata (if enabled)
            # 2. Store semantic metadata (if enabled)
            if self.enable_semantic:
                query_embedding = self._get_query_embedding(query)
                if query_embedding is not None:
                    semantic_key = self._generate_semantic_key(query, query_type)
                    
                    # Store embedding + link to exact cache
                    # FIX: encode embedding as base64 string for JSON compatibility
                    import base64
                    embedding_bytes = query_embedding.astype(np.float32).tobytes()
                    embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                    
                    semantic_meta = {
                        'query': query,
                        'query_type': query_type,
                        'exact_key': exact_key,
                        'embedding': embedding_b64  # Store as base64 string
                    }
                    
                    semantic_json = json.dumps(semantic_meta).encode('utf-8')
                    
                    self.redis_client.setex(
                        name=semantic_key,
                        time=ttl,
                        value=semantic_json
                    )
                    
                    logger.debug(f"Stored semantic cache with embedding dim: {len(query_embedding)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
            return False
    
    
    def invalidate(self, query: str, query_type: str = None) -> bool:
        """Invalidate both exact and semantic cache"""
        if not self.redis_client:
            return False
        
        try:
            exact_key = self._generate_exact_key(query, query_type)
            semantic_key = self._generate_semantic_key(query, query_type)
            
            deleted_exact = self.redis_client.delete(exact_key)
            deleted_semantic = self.redis_client.delete(semantic_key)
            
            if deleted_exact or deleted_semantic:
                logger.info(f"🗑️ Invalidated cache: {query[:50]}...")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return False
    
    def clear_all(self) -> int:
        """Clear all chatbot caches"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys("chatbot:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.warning(f"🗑️ Cleared {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {
                'total_exact': 0,
                'total_semantic': 0,
                'memory_used': '0B',
                'connected': False,
                'semantic_enabled': False
            }
        
        try:
            exact_keys = self.redis_client.keys("chatbot:exact:*")
            semantic_keys = self.redis_client.keys("chatbot:semantic:*")
            
            info = self.redis_client.info('memory')
            memory_used = info.get('used_memory_human', 'Unknown')
            
            return {
                'total_exact': len(exact_keys),
                'total_semantic': len(semantic_keys),
                'total_cached_queries': len(exact_keys),  # Exact = actual queries
                'memory_used': memory_used,
                'connected': True,
                'semantic_enabled': self.enable_semantic
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_exact': 0,
                'total_semantic': 0,
                'memory_used': '0B',
                'connected': False,
                'semantic_enabled': False
            }
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")


# Singleton instance
_cache_manager_instance = None

def get_cache_manager() -> HybridCacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = HybridCacheManager(enable_semantic=True)
    
    return _cache_manager_instance


def main():
    """Test hybrid cache manager"""
    print("=" * 70)
    print("HYBRID CACHE MANAGER - TESTING")
    print("=" * 70)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cache = HybridCacheManager(enable_semantic=True)
    
    # Test 1: Stats
    print("\n--- Test 1: Initial Stats ---")
    stats = cache.get_stats()
    print(f"Connected: {stats['connected']}")
    print(f"Semantic enabled: {stats['semantic_enabled']}")
    print(f"Cached queries: {stats['total_cached_queries']}")
    print(f"Memory: {stats['memory_used']}")
    
    # Test 2: Cache MISS
    print("\n--- Test 2: Cache MISS ---")
    query1 = "How many employees have H-1B visas?"
    result = cache.get(query1, query_type="aggregation")
    print(f"Result: {result}")
    
    # Test 3: Cache SET
    print("\n--- Test 3: Cache SET ---")
    success = cache.set(
        query=query1,
        answer="There are 26 employees with H-1B visas.",
        metadata={'num_sources': 26, 'confidence': 0.95},
        query_type="aggregation"
    )
    print(f"Cached: {success}")
    
    # Test 4: Exact match HIT
    print("\n--- Test 4: Exact Match HIT ---")
    result = cache.get(query1, query_type="aggregation")
    print(f"Cache type: {result['cache_type']}")
    print(f"Answer: {result['answer'][:50]}...")
    
    # Test 5: Semantic match HIT (paraphrase)
    print("\n--- Test 5: Semantic Match HIT (paraphrase) ---")
    query2 = "Count of employees with H1B visa"  # Different wording
    result = cache.get(query2, query_type="aggregation")
    if result:
        print(f"Cache type: {result['cache_type']}")
        print(f"Similarity: {result.get('similarity_score', 0):.3f}")
        print(f"Original query: {result.get('original_query', 'N/A')[:50]}...")
        print(f"Answer: {result['answer'][:50]}...")
    else:
        print("No semantic match (threshold too high or embeddings too different)")
    
    # Test 6: Completely different query (MISS)
    print("\n--- Test 6: Different Query (MISS) ---")
    query3 = "What is the dental insurance copay?"
    result = cache.get(query3, query_type="simple")
    print(f"Result: {result}")
    
    # Test 7: Final stats
    print("\n--- Test 7: Final Stats ---")
    stats = cache.get_stats()
    print(f"Exact cache entries: {stats['total_exact']}")
    print(f"Semantic cache entries: {stats['total_semantic']}")
    print(f"Total cached queries: {stats['total_cached_queries']}")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid Cache Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()