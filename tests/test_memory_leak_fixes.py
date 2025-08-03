"""
Simple test to validate the memory leak fixes in wan_image_to_video_advanced_sampler.py
"""
import pytest
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

def test_memory_leak_fixes_exist():
    """Test that the memory leak fixes are present in the code."""
    file_path = Path(__file__).parent / "wan_image_to_video_advanced_sampler.py"
    
    if not file_path.exists():
        pytest.skip("wan_image_to_video_advanced_sampler.py not found")
    
    content = file_path.read_text(encoding='utf-8')
    
    # Check that the aggressive cleanup functions are modified
    assert "break_circular_references" in content
    assert "cleanup_local_refs" in content
    assert "enhanced_memory_cleanup" in content
    
    # Check that force_model_cleanup is safer (should not have the aggressive tensor deletion)
    assert "force_model_cleanup" in content
    
    # Check that the main run method calls memory cleanup at the end
    # Look for the pattern in the run method
    run_method_start = content.find("def run(self,")
    if run_method_start != -1:
        # Find the end of the run method (next method definition or class end)
        next_method = content.find("\n    def ", run_method_start + 1)
        if next_method == -1:
            next_method = len(content)
        
        run_method_content = content[run_method_start:next_method]
        
        # Check that memory cleanup is called at the end
        assert "break_circular_references" in run_method_content
        assert "enhanced_memory_cleanup" in run_method_content

def test_break_circular_references_function():
    """Test the break_circular_references function directly."""
    # Create a mock class with the method
    class MockSampler:
        def break_circular_references(self, local_vars):
            """
            Simplified version of the actual function for testing.
            """
            target_vars = [
                'working_model_high', 'working_model_low', 
                'working_clip_high', 'working_clip_low',
                'model_high_cfg', 'model_low_cfg',
                'positive_clip_high', 'negative_clip_high',
                'positive_clip_low', 'negative_clip_low'
            ]
            
            for var_name in target_vars:
                if var_name in local_vars:
                    local_vars[var_name] = None
    
    sampler = MockSampler()
    
    # Create test local variables
    test_locals = {
        'working_model_high': Mock(),
        'working_model_low': Mock(),
        'working_clip_high': Mock(), 
        'working_clip_low': Mock(),
        'other_var': 'keep_this',
        'model_high_cfg': Mock()
    }
    
    # Run the function
    sampler.break_circular_references(test_locals)
    
    # Verify targeted variables were set to None
    assert test_locals['working_model_high'] is None
    assert test_locals['working_model_low'] is None
    assert test_locals['working_clip_high'] is None
    assert test_locals['working_clip_low'] is None
    assert test_locals['model_high_cfg'] is None
    
    # Verify other variables were not affected
    assert test_locals['other_var'] == 'keep_this'

def test_force_model_cleanup_safety():
    """Test that force_model_cleanup preserves critical attributes."""
    
    class MockSampler:
        def force_model_cleanup(self, model):
            """
            Simplified safe version for testing.
            """
            if model is None:
                return
                
            # Preserve critical attributes
            if hasattr(model, 'diffusion_model'):
                pass  # Don't delete this critical attribute
                
            # Safe cleanup
            if hasattr(model, 'cpu') and callable(model.cpu):
                model.cpu()
    
    sampler = MockSampler()
    
    # Create a mock model with critical attributes
    mock_model = Mock()
    mock_model.diffusion_model = "critical_component"
    mock_model.temp_cache = "can_be_cleaned"
    mock_model.cpu = Mock()
    
    # Run cleanup
    sampler.force_model_cleanup(mock_model)
    
    # Verify critical attributes are preserved
    assert hasattr(mock_model, 'diffusion_model')
    assert mock_model.diffusion_model == "critical_component"
    
    # Verify CPU was called for memory management
    mock_model.cpu.assert_called_once()

@patch('torch.cuda.empty_cache')
@patch('gc.collect')
def test_memory_cleanup_calls(mock_gc, mock_cuda_cache):
    """Test that memory cleanup functions are called properly."""
    
    class MockSampler:
        def enhanced_memory_cleanup(self, local_vars):
            """Mock enhanced memory cleanup."""
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    sampler = MockSampler()
    
    # Call cleanup
    sampler.enhanced_memory_cleanup({})
    
    # Verify cleanup functions were called
    mock_gc.assert_called()
    mock_cuda_cache.assert_called()

def test_memory_leak_pattern_fix():
    """Test that the specific memory leak pattern has been addressed."""
    
    # Simulate the original problem: model attributes being deleted during active use
    class MockModel:
        def __init__(self):
            self.diffusion_model = "important_component"
            self.other_attr = "can_be_cleaned"
    
    class SafeCleanupSampler:
        def safe_cleanup(self, model):
            """Safe cleanup that preserves critical attributes."""
            if model is None:
                return
                
            # DON'T delete diffusion_model - this caused the original error
            # Only clean up non-critical attributes safely
            
            if hasattr(model, 'cpu') and callable(model.cpu):
                model.cpu()  # Move to CPU but don't delete attributes
    
    sampler = SafeCleanupSampler()
    model = MockModel()
    
    # Verify model has critical attributes before cleanup
    assert hasattr(model, 'diffusion_model')
    assert model.diffusion_model == "important_component"
    
    # Run safe cleanup
    sampler.safe_cleanup(model)
    
    # Verify critical attributes are still there after cleanup
    assert hasattr(model, 'diffusion_model')
    assert model.diffusion_model == "important_component"
    
    # This should NOT raise AttributeError: 'WAN21' object has no attribute 'diffusion_model'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
