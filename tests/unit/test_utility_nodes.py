"""
Unit tests for utility nodes.
"""
import pytest
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@pytest.mark.unit
class TestImageSizerNode:
    """Test cases for ImageSizer node."""
    
    def test_image_sizer_import(self):
        """Test that ImageSizer can be imported."""
        try:
            from image_sizer_node import ImageSizer
            assert ImageSizer is not None
        except ImportError:
            pytest.skip("ImageSizer not available")
    
    def test_image_sizer_calculation(self):
        """Test image size calculations."""
        try:
            from image_sizer_node import ImageSizer
            sizer = ImageSizer()
            
            # Test with mock parameters
            width, height = sizer.run("720p", 16, 9)
            
            # Basic validation that we get numeric results
            assert isinstance(width, (int, float))
            assert isinstance(height, (int, float))
            assert width > 0
            assert height > 0
        except ImportError:
            pytest.skip("ImageSizer not available")

@pytest.mark.unit  
class TestUtilityNodes:
    """Test cases for simple utility nodes."""
    
    def test_float_if_else_node(self):
        """Test FloatIfElse node functionality."""
        try:
            from float_if_else import FloatIfElse
            node = FloatIfElse()
            
            # Test true condition
            result = node.run(True, 1.5, 2.5)
            assert result == (1.5,)
            
            # Test false condition
            result = node.run(False, 1.5, 2.5)
            assert result == (2.5,)
        except ImportError:
            pytest.skip("FloatIfElse not available")
    
    def test_compare_numbers_node(self):
        """Test CompareNumbers node functionality."""
        try:
            from compare_numbers_to_combo import CompareNumbersToCombo
            node = CompareNumbersToCombo()
            
            # Test equal comparison
            result = node.run(5.0, 5.0, "==")
            assert result == (True,)
            
            # Test greater than comparison
            result = node.run(10.0, 5.0, ">")
            assert result == (True,)
            
            # Test less than comparison
            result = node.run(3.0, 5.0, "<")
            assert result == (True,)
        except ImportError:
            pytest.skip("CompareNumbersToCombo not available")
    
    def test_has_input_value_node(self):
        """Test HasInputValue node functionality."""
        try:
            from has_input_value import HasInputValue
            node = HasInputValue()
            
            # Test with value
            result = node.run("test_value")
            assert result == (True,)
            
            # Test with None
            result = node.run(None)
            assert result == (False,)
            
            # Test with empty string
            result = node.run("")
            assert result == (False,)
        except ImportError:
            pytest.skip("HasInputValue not available")

@pytest.mark.unit
class TestWanUtilityNodes:
    """Test cases for Wan-specific utility nodes."""
    
    def test_wan_model_type_selector(self):
        """Test WanModelTypeSelector node."""
        try:
            from wan_model_type_selector import WanModelTypeSelector
            node = WanModelTypeSelector()
            
            # Test INPUT_TYPES structure
            input_types = node.INPUT_TYPES()
            assert "required" in input_types
            
            # Test basic functionality
            result = node.run("720p")
            assert isinstance(result, tuple)
        except ImportError:
            pytest.skip("WanModelTypeSelector not available")
    
    def test_wan_video_generation_mode_selector(self):
        """Test WanVideoGenerationModeSelector node."""
        try:
            from wan_video_generation_mode_selector import WanVideoGenerationModeSelector
            node = WanVideoGenerationModeSelector()
            
            # Test INPUT_TYPES structure
            input_types = node.INPUT_TYPES()
            assert "required" in input_types
            
            # Test basic functionality
            result = node.run("START_IMAGE")
            assert isinstance(result, tuple)
        except ImportError:
            pytest.skip("WanVideoGenerationModeSelector not available")
    
    def test_wan_prompt_chunk_stacker(self):
        """Test WanPromptChunkStacker node."""
        try:
            from wan_prompt_chunck_stacker import WanPromptChunckStacker
            node = WanPromptChunckStacker()
            
            # Test INPUT_TYPES structure  
            input_types = node.INPUT_TYPES()
            assert "required" in input_types
            
            # Test basic stacking functionality
            result = node.run(
                positive="test positive",
                negative="test negative", 
                chunk_index_start=0,
                lora_stack=None,
                prompt_stack=None
            )
            assert isinstance(result, tuple)
            assert len(result) == 1  # Should return a tuple with the prompt stack
        except ImportError:
            pytest.skip("WanPromptChunckStacker not available")
