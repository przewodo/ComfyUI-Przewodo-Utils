"""
Test configuration and shared fixtures for ComfyUI-Przewodo-Utils tests.
"""
import pytest
import torch
import sys
import os
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Add the parent directory to the path so we can import the custom nodes
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_device():
    """Fixture to provide the appropriate test device (CUDA if available, CPU otherwise)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session") 
def mock_comfy_model_management():
    """Mock ComfyUI's model management module."""
    mock_mm = Mock()
    mock_mm.throw_exception_if_processing_interrupted = Mock()
    mock_mm.unload_all_models = Mock()
    mock_mm.soft_empty_cache = Mock()
    return mock_mm

@pytest.fixture(scope="session")
def mock_comfy_nodes():
    """Mock ComfyUI's nodes module."""
    mock_nodes = Mock()
    mock_nodes.VAELoader = Mock()
    mock_nodes.CLIPLoader = Mock()
    mock_nodes.CLIPSetLastLayer = Mock()
    mock_nodes.UNETLoader = Mock()
    mock_nodes.LoraLoader = Mock()
    mock_nodes.KSamplerAdvanced = Mock()
    mock_nodes.CLIPTextEncode = Mock()
    mock_nodes.CLIPVisionLoader = Mock()
    mock_nodes.CLIPVisionEncode = Mock()
    return mock_nodes

@pytest.fixture(scope="function")
def mock_folder_paths():
    """Mock ComfyUI's folder_paths module."""
    mock_fp = Mock()
    mock_fp.get_filename_list = Mock(return_value=["test_model.safetensors"])
    mock_fp.models_dir = "/fake/models/dir"
    return mock_fp

@pytest.fixture(scope="function")
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    # Create a sample image tensor: [batch, height, width, channels]
    return torch.randn(1, 512, 512, 3)

@pytest.fixture(scope="function")
def sample_latent_tensor():
    """Create a sample latent tensor for testing."""
    # Create a sample latent tensor: [batch, channels, height, width]
    return torch.randn(1, 4, 64, 64)

@pytest.fixture(scope="function")
def mock_cache_manager():
    """Mock the CacheManager class."""
    mock_cache = Mock()
    mock_cache.get_from_cache = Mock(return_value=None)
    mock_cache.store_in_cache = Mock()
    return mock_cache

@pytest.fixture(scope="function")
def basic_node_inputs():
    """Provide basic input parameters for node testing."""
    return {
        "GGUF_High": "test_model.gguf",
        "GGUF_Low": "None",
        "Diffusor_High": "None", 
        "Diffusor_Low": "None",
        "Diffusor_weight_dtype": "default",
        "Use_Model_Type": "GGUF",
        "positive": "A beautiful landscape",
        "negative": "blurry, low quality",
        "clip": "test_clip.safetensors",
        "clip_type": "wan",
        "clip_device": "auto",
        "vae": "test_vae.safetensors",
        "use_tea_cache": True,
        "use_SLG": True,
        "SLG_blocks": "10",
        "use_sage_attention": True,
        "sage_attention_mode": "auto",
        "use_shift": True,
        "shift": 8.0,
        "use_block_swap": True,
        "block_swap": 20,
        "large_image_side": 832,
        "image_generation_mode": "START_IMAGE",
        "wan_model_size": "720p",
        "total_video_seconds": 1,
        "total_video_chunks": 1,
        "clip_vision_model": "None",
        "clip_vision_strength": 1.0,
        "use_dual_samplers": True,
        "high_cfg": 1.0,
        "low_cfg": 1.0,
        "total_steps": 15,
        "total_steps_high_cfg": 5,
        "noise_seed": 42,
        "lora_stack": None,
        "start_image": None,
        "start_image_clip_vision_enabled": True,
        "end_image": None,
        "end_image_clip_vision_enabled": True,
        "video_enhance_enabled": True,
        "use_cfg_zero_star": True,
        "apply_color_match": True,
        "causvid_lora": "None",
        "high_cfg_causvid_strength": 1.0,
        "low_cfg_causvid_strength": 1.0,
        "high_denoise": 1.0,
        "low_denoise": 1.0,
        "prompt_stack": None,
        "fill_noise_latent": 0.5,
        "frames_interpolation": False,
        "frames_engine": "None",
        "frames_multiplier": 2,
        "frames_clear_cache_after_n_frames": 100,
        "frames_use_cuda_graph": True,
        "frames_overlap_chunks": 8,
        "frames_overlap_chunks_blend": 0.3,
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, mock_comfy_model_management, mock_comfy_nodes, mock_folder_paths):
    """Automatically set up the test environment for all tests."""
    # Mock the ComfyUI imports that might not be available during testing
    import sys
    
    # Add mocks to sys.modules instead of using monkeypatch for module names
    sys.modules['comfy.model_management'] = mock_comfy_model_management
    sys.modules['nodes'] = mock_comfy_nodes
    sys.modules['folder_paths'] = mock_folder_paths
    
    # Mock the external dependencies that might not be available
    mock_external_modules = {
        'teacache': Mock(),
        'comfyui-kjnodes': Mock(), 
        'ComfyUI-GGUF': Mock(),
        'wanblockswap': Mock(),
        'ComfyUI-Rife-Tensorrt': Mock(),
    }
    
    for module_name, mock_module in mock_external_modules.items():
        sys.modules[module_name] = mock_module
        sys.modules[module_name] = mock_module

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set up any global test configuration here
    pass

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "workflow" in str(item.fspath):
            item.add_marker(pytest.mark.workflow)
            
        # Add GPU marker for tests that require CUDA
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
            
        # Add slow marker for tests that might take longer
        if "slow" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)
