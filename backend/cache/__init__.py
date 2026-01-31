# Intelligent caching layer with Redis backend
from .redis_cache import IntelligentCache, get_cache

__all__ = ["IntelligentCache", "get_cache"]
