"""
Optimized caching system for predictions and models
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Optional, Dict
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = None

    def is_expired(self) -> bool:
        """Check if the entry has expired"""
        return time.time() - self.timestamp > self.ttl

    def access(self) -> Any:
        """Mark access and return the value"""
        self.access_count += 1
        self.last_access = time.time()
        return self.value


class OptimizedCache:
    """Optimized cache with LRU and TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None

            return entry.access()

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            if len(self._cache) >= self._max_size:
                await self._evict_lru()

            ttl = ttl or self._default_ttl
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )

    async def _evict_lru(self) -> None:
        """Remove least-recently-used entry"""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access or 0
        )
        del self._cache[lru_key]

    async def clear(self) -> None:
        """Clear the cache"""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": self._calculate_hit_rate(),
            "entries": [
                {
                    "key": key[:50] + "..." if len(key) > 50 else key,
                    "age_seconds": time.time() - entry.timestamp,
                    "access_count": entry.access_count
                }
                for key, entry in list(self._cache.items())[:10]
            ]
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        return total_accesses / max(len(self._cache), 1)


def cache_key_from_candidate(candidate) -> str:
    """Generate cache key from candidate"""
    # Use hash of flux data + parameters to generate a unique key
    flux_hash = hashlib.md5(
        json.dumps(candidate.light_curve.flux, sort_keys=True).encode()
    ).hexdigest()[:16]

    params_str = f"{candidate.target_name}_{candidate.light_curve.mission}_{flux_hash}"
    return hashlib.md5(params_str.encode()).hexdigest()


def cached_prediction(cache_instance, ttl: Optional[float] = None):
    """Decorator for caching predictions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Assume the first argument is the candidate
            candidate = args[1] if len(args) > 1 else kwargs.get('candidate')
            if not candidate:
                return await func(*args, **kwargs)

            cache_key = cache_key_from_candidate(candidate)

            # Try to fetch from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {candidate.target_name}")
                return cached_result

            # Execute function and store result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl)
            logger.info(f"Cache miss for {candidate.target_name} - result stored")

            return result
        return wrapper
    return decorator


# Global cache instances
prediction_cache = OptimizedCache(max_size=1000, default_ttl=3600)
model_cache = OptimizedCache(max_size=10, default_ttl=86400)  # 24h for models
