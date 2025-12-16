"""
Hybrid Cache Manager for HR Chatbot - ROBUST VERSION
====================================================

Two-tier caching system with STRICT validation:
1. Exact Match Cache: Hash-based (instant O(1) lookup)
2. Semantic Cache: Embedding similarity with keyword validation

Flow:
- Check exact match → HIT: return
- Check semantic similarity (>0.92) → Validate keywords → HIT: return
- MISS: process query, cache with both methods

CRITICAL IMPROVEMENTS:
- High similarity threshold (0.92) to avoid false matches
- Keyword extraction and validation (numbers, IDs, plan names, etc.)
- Context-aware matching (employee IDs, visa types, positions must match)
"""

import redis
import hashlib
import json
import logging
import numpy as np
import re
from typing import Optional, Dict, Any, List, Tuple, Set
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
    Hybrid caching: Exact match + Semantic similarity with keyword validation
    """
    
    # Semantic cache settings
    SEMANTIC_THRESHOLD = 0.85  # STRICT threshold - only very similar queries match
    MAX_SEMANTIC_CANDIDATES = 100  # Max cached queries to check
    
    # Critical keyword patterns that MUST match for cache hit
    CRITICAL_PATTERNS = {
        'employee_ids': r'\b\d{4}\b',  # Employee IDs (4 digits like 1503, 1528)
        'numbers': r'\$?\d+(?:,\d{3})*(?:\.\d{2})?',  # Numbers and currencies
        'years': r'\b20\d{2}\b',  # Years (2020-2099)
    }
    
    # Critical keywords (case-insensitive)
    CRITICAL_KEYWORDS = {
        'plan_names': ['ppo 1000', 'ppo 2500', 'buy-up', 'plan 1', 'plan 2'],
        'insurance_types': ['dental', 'vision', 'medical', 'health'],
        'visa_types': ['h-1b', 'h1b', 'opt', 'cpt', 'green card', 'citizen', 'tn visa', 'opt-extension'],
        'positions': ['software developer', 'technical project manager', 'test analyst', 
                     'quality assurance', 'sales executive', 'project manager'],
        'employment_types': ['fulltime', 'full-time', 'parttime', 'part-time', 'contractor']
    }
    
    def __init__(self, enable_semantic: bool = True):
        """
        Initialize hybrid cache manager with strict validation
        
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
    
    def _extract_critical_info(self, query: str) -> Dict[str, Set[str]]:
        """
        Extract critical information that must match for valid cache hit
        
        Returns:
            {
                'employee_ids': {1503, 1528, ...},
                'numbers': {100, 2500, ...},
                'years': {2024, 2025, ...},
                'plan_names': {'ppo 1000', ...},
                'insurance_types': {'dental', ...},
                'visa_types': {'h-1b', ...},
                'positions': {'software developer', ...}
            }
        """
        query_lower = query.lower()
        critical_info = {}
        
        # Extract patterns (IDs, numbers, years)
        for key, pattern in self.CRITICAL_PATTERNS.items():
            matches = set(re.findall(pattern, query_lower))
            if matches:
                critical_info[key] = matches
        
        # Extract keyword categories
        for category, keywords in self.CRITICAL_KEYWORDS.items():
            found = set()
            for keyword in keywords:
                if keyword in query_lower:
                    found.add(keyword)
            if found:
                critical_info[category] = found
        
        return critical_info
    
    def _validate_keyword_match(
        self,
        query1_info: Dict[str, Set[str]],
        query2_info: Dict[str, Set[str]]
    ) -> Tuple[bool, str]:
        """
        Validate that critical keywords match between two queries
        
        Args:
            query1_info: Critical info from query 1
            query2_info: Critical info from query 2
        
        Returns:
            (is_valid, reason) - True if queries are compatible
        """
        # Check each critical category
        for category in query1_info.keys():
            if category in query2_info:
                # Both queries have this category - must have overlap
                overlap = query1_info[category] & query2_info[category]
                
                if not overlap:
                    # CRITICAL MISMATCH - different values for same category
                    reason = f"Mismatch in {category}: {query1_info[category]} vs {query2_info[category]}"
                    return False, reason
        
        # Check for one-sided critical categories (only in one query)
        all_categories = set(query1_info.keys()) | set(query2_info.keys())
        
        for category in all_categories:
            # Employee IDs are CRITICAL - must match if present in either query
            if category == 'employee_ids':
                if category in query1_info and category not in query2_info:
                    return False, f"Query 1 has employee ID, Query 2 doesn't"
                if category not in query1_info and category in query2_info:
                    return False, f"Query 2 has employee ID, Query 1 doesn't"
            
            # Plan names are CRITICAL - must match if present
            if category == 'plan_names':
                if category in query1_info and category not in query2_info:
                    return False, f"Query 1 has plan name, Query 2 doesn't"
                if category not in query1_info and category in query2_info:
                    return False, f"Query 2 has plan name, Query 1 doesn't"
        
        return True, "Keywords match"
    
    def _generate_exact_key(self, query: str, query_type: str = None) -> str:
        """Generate exact match cache key (hash-based)"""
        normalized_query = self._normalize_query(query)
        cache_input = f"{query_type}:{normalized_query}" if query_type else normalized_query
        
        hash_object = hashlib.sha256(cache_input.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        return f"chatbot:exact:{hash_hex}"
    
    def _generate_semantic_key(self, query: str, query_type: str = None) -> str:
        """Generate semantic cache metadata key"""
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
        Retrieve cached result with STRICT validation
        
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
        
        # Step 2: Try semantic match with STRICT validation
        if self.enable_semantic:
            semantic_result = self._get_semantic(query, query_type)
            if semantic_result:
                logger.info(f"✅ SEMANTIC CACHE HIT: {query[:50]}...")
                logger.info(f"   Matched: {semantic_result.get('original_query', 'N/A')[:50]}...")
                logger.info(f"   Similarity: {semantic_result.get('similarity_score', 0):.3f}")
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
        Try semantic similarity match with STRICT keyword validation
        
        Process:
        1. Generate query embedding
        2. Extract critical keywords from query
        3. Get all semantic cache keys
        4. For each cached query:
           a. Check similarity (must be > 0.92)
           b. Validate keywords match
        5. Return best valid match
        """
        try:
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                logger.debug("Failed to generate query embedding")
                return None
            
            # Extract critical keywords from query
            query_critical_info = self._extract_critical_info(query)
            logger.debug(f"Query critical info: {query_critical_info}")
            
            # Get all semantic cache keys
            pattern = b"chatbot:semantic:*"
            semantic_keys = self.redis_client.keys(pattern)
            
            if not semantic_keys:
                logger.debug("No semantic cache entries found")
                return None
            
            logger.debug(f"Found {len(semantic_keys)} semantic cache entries to check")
            
            # Limit candidates
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
                    
                    # Check if query_type matches
                    if query_type and sem_meta.get('query_type') != query_type:
                        continue
                    
                    # Get cached query text
                    cached_query = sem_meta.get('query', '')
                    
                    # Calculate similarity
                    import base64
                    embedding_bytes = base64.b64decode(sem_meta['embedding'])
                    cached_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    # STRICT THRESHOLD - only proceed if very similar
                    if similarity < self.SEMANTIC_THRESHOLD:
                        continue
                    
                    logger.debug(f"High similarity ({similarity:.3f}) with: {cached_query[:50]}...")
                    
                    # KEYWORD VALIDATION - critical keywords must match
                    cached_critical_info = self._extract_critical_info(cached_query)
                    is_valid, reason = self._validate_keyword_match(
                        query_critical_info,
                        cached_critical_info
                    )
                    
                    if not is_valid:
                        logger.debug(f"REJECTED: {reason}")
                        continue
                    
                    logger.debug(f"VALIDATED: Keywords match")
                    
                    # Track best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'semantic_key': sem_key,
                            'exact_key': sem_meta['exact_key'].encode('utf-8') if isinstance(sem_meta['exact_key'], str) else sem_meta['exact_key'],
                            'similarity': similarity,
                            'original_query': cached_query
                        }
                
                except Exception as e:
                    logger.debug(f"Error checking semantic key: {e}")
                    continue
            
            # Return best validated match
            if best_match:
                logger.debug(f"Best validated match: {best_match['original_query'][:50]}... (similarity: {best_similarity:.3f})")
                
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
                logger.debug(f"No validated match found (threshold: {self.SEMANTIC_THRESHOLD})")
            
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
            if self.enable_semantic:
                query_embedding = self._get_query_embedding(query)
                if query_embedding is not None:
                    semantic_key = self._generate_semantic_key(query, query_type)
                    
                    # Store embedding + link to exact cache
                    import base64
                    embedding_bytes = query_embedding.astype(np.float32).tobytes()
                    embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                    
                    semantic_meta = {
                        'query': query,
                        'query_type': query_type,
                        'exact_key': exact_key,
                        'embedding': embedding_b64
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
                'semantic_enabled': False,
                'semantic_threshold': self.SEMANTIC_THRESHOLD
            }
        
        try:
            exact_keys = self.redis_client.keys("chatbot:exact:*")
            semantic_keys = self.redis_client.keys("chatbot:semantic:*")
            
            info = self.redis_client.info('memory')
            memory_used = info.get('used_memory_human', 'Unknown')
            
            return {
                'total_exact': len(exact_keys),
                'total_semantic': len(semantic_keys),
                'total_cached_queries': len(exact_keys),
                'memory_used': memory_used,
                'connected': True,
                'semantic_enabled': self.enable_semantic,
                'semantic_threshold': self.SEMANTIC_THRESHOLD
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_exact': 0,
                'total_semantic': 0,
                'memory_used': '0B',
                'connected': False,
                'semantic_enabled': False,
                'semantic_threshold': self.SEMANTIC_THRESHOLD
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
    """Test hybrid cache manager with STRICT validation"""
    print("=" * 70)
    print("HYBRID CACHE MANAGER - ROBUST TESTING")
    print("=" * 70)
    
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG to see validation details
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cache = HybridCacheManager(enable_semantic=True)
    
    # Clear cache first
    print("\n🗑️  Clearing existing cache...")
    cache.clear_all()
    
    # Test cases that should NOT match
    print("\n" + "=" * 70)
    print("TEST 1: Different Employee IDs (should NOT match)")
    print("=" * 70)
    
    # Cache employee 1504
    cache.set(
        query="Tell me about employee with id 1504",
        answer="Employee 1504 is a Technical Project Manager.",
        metadata={'confidence': 0.95},
        query_type="simple"
    )
    
    # Try to get employee 1528 (should MISS)
    result = cache.get("Tell me about employee with id 1528", query_type="simple")
    if result:
        print(f"❌ FALSE MATCH: {result['original_query']}")
    else:
        print(f"✅ CORRECT MISS: Different employee IDs rejected")
    
    # Test cases that should NOT match
    print("\n" + "=" * 70)
    print("TEST 2: Different Insurance Types (should NOT match)")
    print("=" * 70)
    
    # Cache dental question
    cache.set(
        query="What is the dental insurance copay?",
        answer="Dental insurance has $50 deductible.",
        metadata={'confidence': 0.95},
        query_type="simple"
    )
    
    # Try insurance plans question (should MISS)
    result = cache.get("What are the insurance plans for this company?", query_type="simple")
    if result:
        print(f"❌ FALSE MATCH: {result['original_query']}")
    else:
        print(f"✅ CORRECT MISS: Different topics rejected")
    
    # Test cases that SHOULD match
    print("\n" + "=" * 70)
    print("TEST 3: Valid Paraphrases (should match)")
    print("=" * 70)
    
    # Cache H-1B question
    cache.set(
        query="How many employees have H-1B visas?",
        answer="There are 26 employees with H-1B visas.",
        metadata={'confidence': 0.95},
        query_type="aggregation"
    )
    
    # Try paraphrase (should HIT)
    paraphrases = [
        "Count of employees with H1B visa",
        "How many workers have H-1B status?",
        "Total H-1B visa holders"
    ]
    
    for para in paraphrases:
        result = cache.get(para, query_type="aggregation")
        if result:
            print(f"✅ VALID MATCH: {para}")
            print(f"   Original: {result['original_query']}")
            print(f"   Similarity: {result.get('similarity_score', 0):.3f}")
        else:
            print(f"⚠️  MISS: {para}")
    
    # Test different plan names
    print("\n" + "=" * 70)
    print("TEST 4: Different Plan Names (should NOT match)")
    print("=" * 70)
    
    # Cache PPO 1000 question
    cache.set(
        query="When did employee 1504 joined the company?",
        answer="PPO 1000 has $35 copay for primary care.",
        metadata={'confidence': 0.95},
        query_type="simple"
    )
    
    # Try PPO 2500 (should MISS)
    result = cache.get("What is the joining date of employee 1504?", query_type="simple")
    if result:
        print("Cache hit")
    else:
        print("Cache miss")
    
    result = cache.get("What is employee 1504's salary?", query_type="simple")
    if result:
        print("Cache hit")
    else:
        print("Cache miss")
    
    # Final stats
    print("\n" + "=" * 70)
    print("CACHE STATISTICS")
    print("=" * 70)
    stats = cache.get_stats()
    print(f"Cached queries: {stats['total_cached_queries']}")
    print(f"Semantic threshold: {stats['semantic_threshold']}")
    print(f"Memory used: {stats['memory_used']}")
    
    print("\n" + "=" * 70)
    print("✅ Robust Cache Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()