"""
Generic Cache Manager for ComfyUI Przewodo Utils

This module provides a universal caching system for any type of model or object,
with automatic CPU/GPU memory management and smart key generation.
"""

class CacheManager:
    """
    Generic cache manager for storing and retrieving any type of model or object.
    Supports automatic CPU/GPU memory management with user-provided cache keys.
    """
    def __init__(self):
        self._cache = {}
    
    def get_from_cache(self, cache_key, target_device="cuda"):
        """
        Retrieve an object from cache and move it to the target device.
        
        Args:
            cache_key (str): The cache key to look up
            target_device (str): Device to move the object to ("cuda", "cpu", etc.)
            
        Returns:
            The cached object moved to target device, or None if not found
        """
        if cache_key in self._cache:
            cached_obj = self._cache[cache_key]
            if hasattr(cached_obj, 'clone'):
                # Clone the object (for PyTorch models)
                obj = cached_obj.clone()
                if hasattr(obj, 'to') and target_device:
                    obj = obj.to(target_device)
                return obj
            else:
                # For non-PyTorch objects, return as-is
                return cached_obj
        return None
    
    def store_in_cache(self, cache_key, obj, storage_device="cpu"):
        """
        Store an object in cache, optionally moving it to a storage device.
        
        Args:
            cache_key (str): The cache key to store under
            obj: The object to cache
            storage_device (str): Device to store on ("cpu" to save VRAM, "cuda" to keep on GPU)
        """
        if hasattr(obj, 'clone'):
            # Clone and optionally move to storage device
            cached_obj = obj.clone()
            if hasattr(cached_obj, 'to') and storage_device:
                cached_obj = cached_obj.to(storage_device)
            self._cache[cache_key] = cached_obj
        else:
            # Store non-PyTorch objects directly
            self._cache[cache_key] = obj
    
    def clear_cache(self, cache_key=None):
        """
        Clear cache entries.
        
        Args:
            cache_key (str, optional): Specific key to clear. If None, clears all cache.
        """
        if cache_key:
            if cache_key in self._cache:
                del self._cache[cache_key]
        else:
            self._cache.clear()
    
    def get_cache_info(self):
        """Get information about cached objects."""
        return list(self._cache.keys())
    
    def cache_size(self):
        """Get the number of cached objects."""
        return len(self._cache)
