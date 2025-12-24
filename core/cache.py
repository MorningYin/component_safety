"""
Activation cache with disk persistence.

Provides caching for activation data using pickle files.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import pickle
import hashlib
import json


class ActivationCache:
    """Cache for activation data with disk persistence."""
    
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, Any] = {}
    
    def _key_to_hash(self, cache_key: Dict) -> str:
        """Convert cache key dict to a hash string for filename."""
        key_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def load(self, namespace: str, cache_key: Dict) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            namespace: Category of data (e.g., 'activations')
            cache_key: Dict identifying the data
        
        Returns:
            Cached data if found, None otherwise.
        """
        key_hash = self._key_to_hash(cache_key)
        cache_path = self.cache_dir / namespace / f"{key_hash}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return None
        return None
    
    def save(self, namespace: str, cache_key: Dict, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            namespace: Category of data (e.g., 'activations')
            cache_key: Dict identifying the data
            data: Data to cache
        """
        key_hash = self._key_to_hash(cache_key)
        namespace_dir = self.cache_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = namespace_dir / f"{key_hash}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def has(self, namespace: str, cache_key: Dict) -> bool:
        """Check if data exists in cache."""
        key_hash = self._key_to_hash(cache_key)
        cache_path = self.cache_dir / namespace / f"{key_hash}.pkl"
        return cache_path.exists()
    
    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache. If namespace given, clear only that namespace."""
        if namespace:
            namespace_dir = self.cache_dir / namespace
            if namespace_dir.exists():
                for f in namespace_dir.glob("*.pkl"):
                    f.unlink()
        else:
            for f in self.cache_dir.rglob("*.pkl"):
                f.unlink()


# Global cache instance
GLOBAL_CACHE = ActivationCache()
