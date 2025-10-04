"""
Sistema de cache otimizado para predições e modelos
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
    """Entrada do cache com metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = None

    def is_expired(self) -> bool:
        """Verifica se a entrada expirou"""
        return time.time() - self.timestamp > self.ttl

    def access(self) -> Any:
        """Marca acesso e retorna valor"""
        self.access_count += 1
        self.last_access = time.time()
        return self.value


class OptimizedCache:
    """Cache otimizado com LRU e TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache"""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None

            return entry.access()

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Define valor no cache"""
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
        """Remove entrada menos recentemente usada"""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access or 0
        )
        del self._cache[lru_key]

    async def clear(self) -> None:
        """Limpa o cache"""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """Remove entradas expiradas"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Estatísticas do cache"""
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
        """Calcula taxa de acerto do cache"""
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        return total_accesses / max(len(self._cache), 1)


def cache_key_from_candidate(candidate) -> str:
    """Gera chave de cache a partir de candidato"""
    # Usa hash dos dados de fluxo + parâmetros para gerar chave única
    flux_hash = hashlib.md5(
        json.dumps(candidate.light_curve.flux, sort_keys=True).encode()
    ).hexdigest()[:16]

    params_str = f"{candidate.target_name}_{candidate.light_curve.mission}_{flux_hash}"
    return hashlib.md5(params_str.encode()).hexdigest()


def cached_prediction(cache_instance, ttl: Optional[float] = None):
    """Decorator para cache de predições"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Assume que o primeiro argumento é o candidato
            candidate = args[1] if len(args) > 1 else kwargs.get('candidate')
            if not candidate:
                return await func(*args, **kwargs)

            cache_key = cache_key_from_candidate(candidate)

            # Tenta buscar no cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit para {candidate.target_name}")
                return cached_result

            # Executa função e armazena resultado
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl)
            logger.info(f"Cache miss para {candidate.target_name} - resultado armazenado")

            return result
        return wrapper
    return decorator


# Instância global do cache
prediction_cache = OptimizedCache(max_size=1000, default_ttl=3600)
model_cache = OptimizedCache(max_size=10, default_ttl=86400)  # 24h para modelos
