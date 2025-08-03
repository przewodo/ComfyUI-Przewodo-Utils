"""
Simple test to verify pytest is working correctly.
"""
import pytest
import torch
from unittest.mock import Mock

def test_basic_functionality():
    """Basic test to ensure pytest is working."""
    assert 1 + 1 == 2

def test_torch_available():
    """Test that PyTorch is available."""
    assert torch.cuda.is_available() or True  # Pass regardless of CUDA availability
    tensor = torch.randn(2, 3)
    assert tensor.shape == (2, 3)

def test_mock_functionality():
    """Test that mocking is working."""
    mock_obj = Mock()
    mock_obj.test_method.return_value = "mocked"
    assert mock_obj.test_method() == "mocked"
    mock_obj.test_method.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
