"""
Integration tests for WanImageToVideoAdvancedSampler node.

These tests verify that the node works correctly when integrated
with other ComfyUI components and dependencies.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@pytest.mark.integration
class TestWanImageToVideoIntegration:
    """Integration tests for WanImageToVideoAdvancedSampler."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Set up mock dependencies for integration testing."""
        dependencies = {
            'model_high': Mock(),
            'model_low': Mock(), 
            'vae': Mock(),
            'clip_model': Mock(),
            'sage_attention': Mock(),
            'model_shift': Mock(),
            'wanBlockSwap': Mock(),
            'tea_cache': Mock(),
            'slg_wanvideo': Mock(),
        }
        
        # Configure mocks to return appropriate values
        for model in [dependencies['model_high'], dependencies['model_low']]:
            model.clone.return_value = Mock()
            
        dependencies['clip_model'].clone.return_value = Mock()
        
        return dependencies
    
    @pytest.fixture 
    def integration_inputs(self, sample_image_tensor):
        """Provide inputs for integration testing."""
        return {
            "GGUF_High": "test_model.gguf",
            "GGUF_Low": "None",
            "Diffusor_High": "None",
            "Diffusor_Low": "None", 
            "Diffusor_weight_dtype": "default",
            "Use_Model_Type": "GGUF",
            "positive": "A beautiful landscape video",
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
            "total_video_chunks": 2,  # Test multi-chunk generation
            "clip_vision_model": "test_clip_vision.safetensors",
            "clip_vision_strength": 1.0,
            "use_dual_samplers": True,
            "high_cfg": 1.0,
            "low_cfg": 1.0,
            "total_steps": 15,
            "total_steps_high_cfg": 5,
            "noise_seed": 42,
            "lora_stack": [["test_lora.safetensors", 1.0, 1.0]],
            "start_image": sample_image_tensor,
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
    
    @patch('wan_image_to_video_advanced_sampler.WanVideoBlockSwap')
    @patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful')
    @patch('wan_image_to_video_advanced_sampler.output_to_terminal')
    def test_full_pipeline_single_chunk(self, mock_terminal, mock_terminal_success, mock_blockswap, 
                                       mock_dependencies, integration_inputs):
        """Test the full pipeline with a single chunk."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        # Modify inputs for single chunk
        integration_inputs["total_video_chunks"] = 1
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Mock all the complex dependencies
        with patch.object(sampler, 'load_model', return_value=mock_dependencies['model_high']):
            with patch.object(sampler, 'initialize_tea_cache_and_slg', return_value=(mock_dependencies['tea_cache'], mock_dependencies['slg_wanvideo'])):
                with patch.object(sampler, 'initialize_sage_attention', return_value=mock_dependencies['sage_attention']):
                    with patch.object(sampler, 'initialize_model_shift', return_value=mock_dependencies['model_shift']):
                        with patch.object(sampler, 'postprocess', return_value=(torch.randn(16, 512, 512, 3), 16.0)):
                            with patch('nodes.VAELoader') as mock_vae_loader:
                                with patch('nodes.CLIPLoader') as mock_clip_loader:
                                    with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                                        # Configure mocks
                                        mock_vae_loader().load_vae.return_value = (mock_dependencies['vae'],)
                                        mock_clip_loader().load_clip.return_value = (mock_dependencies['clip_model'],)
                                        mock_clip_set_layer().set_last_layer.return_value = (mock_dependencies['clip_model'],)
                                        
                                        # Run the test
                                        result = sampler.run(**integration_inputs)
        
        # Verify results
        assert isinstance(result, tuple)
        assert len(result) == 2
        image_result, fps_result = result
        assert torch.is_tensor(image_result)
        assert isinstance(fps_result, float)
    
    @patch('wan_image_to_video_advanced_sampler.WanVideoBlockSwap')
    @patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful')
    @patch('wan_image_to_video_advanced_sampler.output_to_terminal')
    def test_multi_chunk_generation(self, mock_terminal, mock_terminal_success, mock_blockswap,
                                   mock_dependencies, integration_inputs):
        """Test multi-chunk video generation."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        # Configure for 3 chunks
        integration_inputs["total_video_chunks"] = 3
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Mock the postprocess method to simulate multi-chunk behavior
        chunk_outputs = [
            torch.randn(16, 512, 512, 3),  # Chunk 1
            torch.randn(16, 512, 512, 3),  # Chunk 2  
            torch.randn(16, 512, 512, 3),  # Chunk 3
        ]
        
        with patch.object(sampler, 'load_model', return_value=mock_dependencies['model_high']):
            with patch.object(sampler, 'initialize_tea_cache_and_slg', return_value=(mock_dependencies['tea_cache'], mock_dependencies['slg_wanvideo'])):
                with patch.object(sampler, 'initialize_sage_attention', return_value=mock_dependencies['sage_attention']):
                    with patch.object(sampler, 'initialize_model_shift', return_value=mock_dependencies['model_shift']):
                        with patch.object(sampler, 'postprocess') as mock_postprocess:
                            # Configure postprocess to return concatenated chunks
                            final_output = torch.cat(chunk_outputs, dim=0)
                            mock_postprocess.return_value = (final_output, 16.0)
                            
                            with patch('nodes.VAELoader') as mock_vae_loader:
                                with patch('nodes.CLIPLoader') as mock_clip_loader:
                                    with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                                        # Configure mocks
                                        mock_vae_loader().load_vae.return_value = (mock_dependencies['vae'],)
                                        mock_clip_loader().load_clip.return_value = (mock_dependencies['clip_model'],)
                                        mock_clip_set_layer().set_last_layer.return_value = (mock_dependencies['clip_model'],)
                                        
                                        # Run the test
                                        result = sampler.run(**integration_inputs)
        
        # Verify results
        assert isinstance(result, tuple)
        assert len(result) == 2
        image_result, fps_result = result
        assert torch.is_tensor(image_result)
        # Should have more frames due to multiple chunks
        assert image_result.shape[0] > 16  # More than single chunk
    
    @patch('wan_image_to_video_advanced_sampler.WanVideoBlockSwap')
    def test_prompt_stack_integration(self, mock_blockswap, mock_dependencies, integration_inputs):
        """Test integration with prompt stacking functionality."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        # Add prompt stack to inputs
        integration_inputs["total_video_chunks"] = 3
        integration_inputs["prompt_stack"] = [
            ("landscape in spring", "winter, snow", 0, None),
            ("landscape in summer", "winter, snow", 1, None), 
            ("landscape in autumn", "winter, snow", 2, None),
        ]
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Test that get_current_prompt works correctly with the stack
        for chunk_idx in range(3):
            positive, negative, loras = sampler.get_current_prompt(
                integration_inputs["prompt_stack"], 
                chunk_idx,
                "default_positive",
                "default_negative"
            )
            
            expected_prompts = [
                "landscape in spring",
                "landscape in summer", 
                "landscape in autumn"
            ]
            assert positive == expected_prompts[chunk_idx]
            assert negative == "winter, snow"
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_memory_management_integration(self, mock_dependencies, integration_inputs):
        """Test memory management during processing."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Monitor memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        with patch.object(sampler, 'load_model', return_value=mock_dependencies['model_high']):
            with patch.object(sampler, 'postprocess', return_value=(torch.randn(16, 512, 512, 3), 16.0)):
                # Test memory cleanup functions
                test_locals = {
                    'working_model_high': Mock(),
                    'working_model_low': Mock(),
                    'temp_data': torch.randn(1000, 1000).cuda(),  # Large GPU tensor
                }
                
                # Run cleanup
                sampler.enhanced_memory_cleanup(test_locals)
                sampler.break_circular_references(test_locals)
                
                # Verify cleanup occurred
                assert test_locals.get('working_model_high') is None
                assert test_locals.get('working_model_low') is None
        
        # Memory should not increase significantly
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but not excessive
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
