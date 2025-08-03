"""
Unit tests for CacheManager utility class.
"""
import pytest
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cache_manager import CacheManager
except ImportError:
    CacheManager = None

@pytest.mark.unit
class TestCacheManager:
    """Test cases for CacheManager class."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create a CacheManager instance for testing."""
        if CacheManager is None:
            pytest.skip("CacheManager not available")
        return CacheManager()
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test that CacheManager initializes correctly."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        assert hasattr(cache_manager, '_cache')
        assert isinstance(cache_manager._cache, dict)
    
    def test_store_and_retrieve_cache(self, cache_manager):
        """Test storing and retrieving items from cache."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
            
        test_key = "test_model_key"
        test_object = Mock()
        
        # Store in cache
        cache_manager.store_in_cache(test_key, test_object)
        
        # Retrieve from cache
        retrieved = cache_manager.get_from_cache(test_key)
        
        assert retrieved == test_object
    
    def test_cache_miss(self, cache_manager):
        """Test cache miss returns default value."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
            
        result = cache_manager.get_from_cache("nonexistent_key", default="default_value")
        assert result == "default_value"
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_cache_with_device_movement(self, mock_cuda_available, cache_manager):
        """Test caching with device movement (GPU to CPU)."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
            
        # Create a mock tensor that supports .cpu()
        mock_tensor = Mock()
        mock_cpu_tensor = Mock()
        mock_tensor.cpu.return_value = mock_cpu_tensor
        
        cache_manager.store_in_cache("tensor_key", mock_tensor, storage_device='cpu')
        
        # Should call .cpu() when storing with CPU device
        mock_tensor.cpu.assert_called_once()
    
    def test_cache_clear(self, cache_manager):
        """Test clearing the cache."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
            
        # Store some items
        cache_manager.store_in_cache("key1", "value1")
        cache_manager.store_in_cache("key2", "value2")
        
        # Verify items are stored
        assert cache_manager.get_from_cache("key1") == "value1"
        
        # Clear cache
        if hasattr(cache_manager, 'clear_cache'):
            cache_manager.clear_cache()
            
            # Verify cache is empty
            assert cache_manager.get_from_cache("key1") is None
    
    def test_cache_size_limit(self, cache_manager):
        """Test cache respects size limits if implemented."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
            
        # This test assumes cache might have size limits
        # Store many items to test behavior
        for i in range(100):
            cache_manager.store_in_cache(f"key_{i}", f"value_{i}")
        
        # At minimum, the last stored item should be retrievable
        assert cache_manager.get_from_cache("key_99") == "value_99"
