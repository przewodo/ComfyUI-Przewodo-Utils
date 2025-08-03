"""
Unit tests for WanImageToVideoAdvancedSampler node.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add ComfyUI root directory to path for nodes import
comfyui_root = str(Path(__file__).parent.parent.parent.parent.parent)
sys.path.insert(0, comfyui_root)

try:
    from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
except ImportError:
    # Handle case where ComfyUI dependencies are not available
    WanImageToVideoAdvancedSampler = None

try:
    from cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from nodes import LoadImage
except ImportError:
    LoadImage = None

try:
    from wan_prompt_chunk_stacker import WanPromptChunkStacker
except ImportError:
    WanPromptChunkStacker = None

try:
    from wan_video_lora_stack import WanVideoLoraStack
except ImportError:
    WanVideoLoraStack = None

@pytest.mark.unit
class TestWanImageToVideoAdvancedSampler:
    """Test cases for WanImageToVideoAdvancedSampler class."""
    
    @pytest.fixture
    def realistic_params(self):
        """Provide realistic parameters for testing the WAN sampler."""
        return {
            # Model configuration
            "GGUF_High": "Wan\\I2V\\wan2.1-i2v-14b-480p-Q8_0.gguf",
            "GGUF_Low": "None",
            "Diffusor_High": "None", 
            "Diffusor_Low": "None",
            "Diffusor_weight_dtype": "default",
            "Use_Model_Type": "GGUF",
            
            # Text & CLIP configuration
            "positive": "",
            "negative": "",
            "clip": "umt5_xxl_fp16.safetensors",
            "clip_type": "wan",
            "clip_device": "default",
            "vae": "Wan\\wan_2.1_vae.safetensors",
            
            # TeaCache optimization
            "use_tea_cache": False,
            "tea_cache_model_type": "wan2.1_i2v_480p_14B",
            "tea_cache_rel_l1_thresh": 0.05,
            "tea_cache_start_percent": 0.2,
            "tea_cache_end_percent": 0.8,
            "tea_cache_cache_device": "cuda",
            
            # Skip Layer Guidance
            "use_SLG": False,
            "SLG_blocks": "10",
            "SLG_start_percent": 0.2,
            "SLG_end_percent": 0.8,
            
            # Attention & Model optimizations
            "use_sage_attention": True,
            "sage_attention_mode": "auto",
            "use_shift": True,
            "shift": 8.0,
            "use_block_swap": True,
            "block_swap": 20,
            
            # Video generation settings
            "large_image_side": 832,
            "image_generation_mode": "start_image",
            "wan_model_size": "480p",
            "total_video_seconds": 5,
            "total_video_chunks": 3,
            
            # CLIP Vision settings
            "clip_vision_model": "CLIP-ViT-H-fp16.safetensors",
            "clip_vision_strength": 1.0,
            "start_image_clip_vision_enabled": True,
            "end_image_clip_vision_enabled": True,
            
            # Sampling configuration
            "use_dual_samplers": False,
            "high_cfg": 1.0,
            "low_cfg": 1.0,
            "high_denoise": 1.0,
            "low_denoise": 1.0,
            "total_steps": 6,
            "total_steps_high_cfg": 70,
            "fill_noise_latent": 0.5,
            "noise_seed": 349290219136119,
            
            # CausVid enhancement
            "causvid_lora": "Wan\\T2V\\Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            "high_cfg_causvid_strength": 0.3,
            "low_cfg_causvid_strength": 0.5,
            
            # Post-processing options
            "video_enhance_enabled": True,
            "use_cfg_zero_star": True,
            "apply_color_match": True,
            "frames_interpolation": False,
            "frames_engine": "rife49_ensemble_True_scale_1_sim.engine",
            "frames_multiplier": 2,
            "frames_clear_cache_after_n_frames": 100,
            "frames_use_cuda_graph": True,
            "frames_overlap_chunks": 8,
            "frames_overlap_chunks_blend": 0.3,
            
            # Optional parameters
            "lora_stack": None,
            "prompt_stack": None,
            "start_image": None,
            "end_image": None,
        }
    
    @pytest.fixture
    def loaded_start_image(self):
        """Load actual start image using ComfyUI's LoadImage node."""
        if LoadImage is None:
            pytest.skip("LoadImage node not available")
        
        # Create LoadImage instance
        load_image_node = LoadImage()
        
        # Load the specific image
        try:
            image_tensor, mask_tensor = load_image_node.load_image("Image_Upscaled_00002_.png")
            return image_tensor
        except Exception as e:
            pytest.skip(f"Could not load Image_Upscaled_00002_.png: {e}")
    
    @pytest.fixture
    def loaded_end_image(self):
        """Load actual end image using ComfyUI's LoadImage node."""
        if LoadImage is None:
            pytest.skip("LoadImage node not available")
        
        # Create LoadImage instance
        load_image_node = LoadImage()
        
        # Load the specific image
        try:
            image_tensor, mask_tensor = load_image_node.load_image("2197_anim_p3-Camera 1_29.jpg")
            return image_tensor
        except Exception as e:
            pytest.skip(f"Could not load 2197_anim_p3-Camera 1_29.jpg: {e}")
    
    @pytest.fixture
    def mock_start_image(self):
        """Create a mock start image tensor as fallback."""
        # ComfyUI format: [batch, height, width, channels]
        return torch.randn(1, 512, 512, 3)
    
    @pytest.fixture
    def realistic_params_with_image(self, realistic_params, loaded_start_image, loaded_end_image, realistic_lora_stack, realistic_prompt_stack):
        """Realistic parameters with actual loaded images and realistic stacks."""
        params = realistic_params.copy()
        params['start_image'] = loaded_start_image
        params['end_image'] = loaded_end_image
        params['lora_stack'] = realistic_lora_stack
        params['prompt_stack'] = realistic_prompt_stack
        params['total_video_chunks'] = 2  # Updated to match our 2-chunk prompt stack
        return params

    @pytest.fixture
    def mock_end_image(self):
        """Create a mock end image tensor."""
        # ComfyUI format: [batch, height, width, channels] 
        return torch.randn(1, 512, 512, 3)
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a real CacheManager instance for the sampler."""
        if CacheManager is None:
            pytest.skip("CacheManager not available")
        return CacheManager()
    
    @pytest.fixture
    def realistic_prompt_lora_stack(self):
        """Create realistic LoRA stack for prompt chunks using WanVideoLoraStack node."""
        if WanVideoLoraStack is None:
            pytest.skip("WanVideoLoraStack node not available")
        
        # Create WanVideoLoraStack instance
        lora_stack_node = WanVideoLoraStack()
        
        # Create the LoRA stack with specified parameters
        try:
            lora_stack_result = lora_stack_node.load_lora(
                lora_name="Wan\\I2V\\jfj-deepthroat-v1.safetensors",
                strength_model=0.9,
                strength_clip=1.0,
                previous_lora=None
            )
            return lora_stack_result[0]  # Return the LoRA stack
        except Exception as e:
            pytest.skip(f"Could not create LoRA stack: {e}")
    
    @pytest.fixture
    def realistic_lora_stack(self):
        """Create realistic LoRA stack with 3 stacked LoRAs using WanVideoLoraStack node."""
        if WanVideoLoraStack is None:
            pytest.skip("WanVideoLoraStack node not available")
        
        # Create WanVideoLoraStack instance
        lora_stack_node = WanVideoLoraStack()
        
        try:
            # LoRA Entry 1
            lora1_result = lora_stack_node.load_lora(
                lora_name="Wan\\I2V\\lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
                strength_model=1.0,
                strength_clip=1.0,
                previous_lora=None
            )
            
            # LoRA Entry 2 (uses LoRA Entry 1 as previous_lora)
            lora2_result = lora_stack_node.load_lora(
                lora_name="Wan\\I2V\\wan-nsfw-e14-fixed.safetensors",
                strength_model=0.9,
                strength_clip=1.0,
                previous_lora=lora1_result[0]
            )
            
            # LoRA Entry 3 (uses LoRA Entry 2 as previous_lora)
            lora3_result = lora_stack_node.load_lora(
                lora_name="Wan\\T2V\\detailz-wan.safetensors",
                strength_model=0.2,
                strength_clip=1.0,
                previous_lora=lora2_result[0]
            )
            
            return lora3_result[0]  # Return the final LoRA stack with all 3 LoRAs
        except Exception as e:
            pytest.skip(f"Could not create 3-LoRA stack: {e}")
    
    @pytest.fixture
    def realistic_prompt_stack(self, realistic_prompt_lora_stack):
        """Create realistic prompt stack using WanPromptChunkStacker node."""
        if WanPromptChunkStacker is None:
            pytest.skip("WanPromptChunkStacker node not available")
        
        # Create WanPromptChunkStacker instance
        prompt_stacker_node = WanPromptChunkStacker()
        
        try:
            # Chunk 1
            chunk1_result = prompt_stacker_node.run(
                previous_prompt=None,
                lora_stack=realistic_prompt_lora_stack,
                positive_prompt="Realistic, hardcore, close-up of a petite blonde girl with blue eyes and a playful expression, wearing a black bikini top with crisscross straps, fishnet gloves, and thigh-high stockings. She is kneeling on a bed with light-colored sheets, holding a large, veiny, glistening penis with precum. The girl gently grabs the penis, place the glans penis on her lips and then inserts the penis into her mouth, and begins a passionate, sexy blowjob. Her horny expressions and the intimate, sexually charged atmosphere are highlighted by soft lighting. The camera remains still, framing her actions in a cinematic, realistic 8K style,",
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, blurry, lowres, out of focus, deformed, distorted, poorly drawn, bad anatomy, bad proportions, watermark, logo, text overlay, artifact, glitch, flicker, noise, oversaturated, underexposed, disfigured, multiple faces, duplicate, double image, censorship, cropped, cut off, head cut off, missing limb, extra limb, bad hands, poorly drawn hands, bad face, bad lips, bad tongue, unnatural pose, bad perspective, incorrect lighting, color banding, unnatural colors, grain, pixelation, oversharpened, cartoonish, child, watermark, text, penis on girl, penis on woman",
                chunk_index_start=0
            )
            
            # Chunk 2
            chunk2_result = prompt_stacker_node.run(
                previous_prompt=chunk1_result[0],
                lora_stack=realistic_prompt_lora_stack,
                positive_prompt="Realistic, hardcore, close-up of a petite blonde girl with blue eyes and a playful expression, wearing a black bikini top with crisscross straps, fishnet gloves, and thigh-high stockings. She is kneeling on a bed with light-colored sheets, making a deepthroat. The girl inserts the huge veiny penis all inside on her mouth till she reaches his balls with her tongue. Her horny expressions and the intimate, sexually charged atmosphere are highlighted by soft lighting. The camera remains still, framing her actions in a cinematic, realistic 8K style,",
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, blurry, lowres, out of focus, deformed, distorted, poorly drawn, bad anatomy, bad proportions, watermark, logo, text overlay, artifact, glitch, flicker, noise, oversaturated, underexposed, disfigured, multiple faces, duplicate, double image, censorship, cropped, cut off, head cut off, missing limb, extra limb, bad hands, poorly drawn hands, bad face, bad lips, bad tongue, unnatural pose, bad perspective, incorrect lighting, color banding, unnatural colors, grain, pixelation, oversharpened, cartoonish, child, watermark, text, penis on girl, penis on woman",
                chunk_index_start=3
            )
            
            return chunk2_result[0]  # Return the final prompt stack
        except Exception as e:
            pytest.skip(f"Could not create prompt stack: {e}")
    
    @pytest.fixture
    def mock_lora_stack(self):
        """Create a mock LoRA stack."""
        return [
            ["test_lora1.safetensors", 0.8, 0.8],  # [name, model_strength, clip_strength]
            ["test_lora2.safetensors", 0.6, 0.6],
        ]
    
    @pytest.fixture
    def mock_prompt_stack(self):
        """Create a mock prompt stack."""
        return [
            ["Beautiful landscape", "blurry", 0, None],  # [positive, negative, chunk_start, lora_stack]
            ["Sunset scene", "dark", 1, None],
        ]
    
    @pytest.fixture
    def sampler_node(self, mock_cache_manager):
        """Create a WanImageToVideoAdvancedSampler instance for testing."""
        if WanImageToVideoAdvancedSampler is None:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        # Mock the class-level cache manager
        with patch.object(WanImageToVideoAdvancedSampler, '_cache_manager', mock_cache_manager):
            return WanImageToVideoAdvancedSampler()
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for the sampler."""
        mocks = {}
        
        # Mock ComfyUI nodes
        with patch('nodes.VAELoader') as mock_vae_loader:
            mock_vae_loader.return_value.load_vae.return_value = (Mock(),)
            mocks['vae_loader'] = mock_vae_loader
            
        with patch('nodes.CLIPLoader') as mock_clip_loader:
            mock_clip_loader.return_value.load_clip.return_value = (Mock(),)
            mocks['clip_loader'] = mock_clip_loader
            
        with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
            mock_clip_set_layer.return_value.set_last_layer.return_value = (Mock(),)
            mocks['clip_set_layer'] = mock_clip_set_layer
            
        with patch('nodes.UNETLoader') as mock_unet_loader:
            mock_unet_loader.return_value.load_unet.return_value = (Mock(),)
            mocks['unet_loader'] = mock_unet_loader
            
        with patch('nodes.KSamplerAdvanced') as mock_sampler:
            mock_sampler.return_value = Mock()
            mocks['k_sampler'] = mock_sampler
            
        with patch('nodes.CLIPTextEncode') as mock_text_encode:
            mock_text_encode.return_value.encode.return_value = (Mock(),)
            mocks['text_encode'] = mock_text_encode
            
        with patch('nodes.CLIPVisionLoader') as mock_clip_vision_loader:
            mock_clip_vision_loader.return_value.load_clip.return_value = (Mock(),)
            mocks['clip_vision_loader'] = mock_clip_vision_loader
            
        with patch('nodes.CLIPVisionEncode') as mock_clip_vision_encode:
            mock_clip_vision_encode.return_value = Mock()
            mocks['clip_vision_encode'] = mock_clip_vision_encode
            
        with patch('nodes.LoraLoader') as mock_lora_loader:
            mock_lora_loader.return_value.load_lora.return_value = (Mock(), Mock())
            mocks['lora_loader'] = mock_lora_loader
        
        # Mock external optimization libraries
        with patch('comfy.model_management') as mock_mm:
            mock_mm.throw_exception_if_processing_interrupted = Mock()
            mock_mm.unload_all_models = Mock()
            mock_mm.soft_empty_cache = Mock()
            mocks['model_management'] = mock_mm
            
        # Mock custom nodes
        with patch('wan_image_to_video_advanced_sampler.WanFirstLastFirstFrameToVideo') as mock_wan_i2v:
            mock_instance = Mock()
            mock_instance.encode.return_value = (Mock(), Mock(), Mock(), Mock(), Mock())
            mock_wan_i2v.return_value = mock_instance
            mocks['wan_i2v'] = mock_wan_i2v
            
        with patch('wan_image_to_video_advanced_sampler.WanVideoVaeDecode') as mock_wan_vae_decode:
            mock_instance = Mock()
            mock_instance.decode.return_value = (torch.randn(16, 512, 512, 3),)  # Mock video frames
            mock_wan_vae_decode.return_value = mock_instance
            mocks['wan_vae_decode'] = mock_wan_vae_decode
            
        return mocks
    
    def test_input_types_structure(self):
        """Test that INPUT_TYPES returns the expected structure."""
        if WanImageToVideoAdvancedSampler is None:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
            
        with patch('folder_paths.get_filename_list', return_value=['test_model.safetensors']):
            with patch('os.listdir', return_value=['test_engine.onnx']):
                with patch('folder_paths.models_dir', '/fake/models'):
                    input_types = WanImageToVideoAdvancedSampler.INPUT_TYPES()
        
        assert "required" in input_types
        assert "optional" in input_types
        assert isinstance(input_types["required"], dict)
        assert isinstance(input_types["optional"], dict)
        
        # Check that key parameters are present
        required_params = [
            "GGUF_High", "positive", "negative", "clip", "vae",
            "use_tea_cache", "use_SLG", "use_sage_attention",
            "total_steps", "noise_seed"
        ]
        for param in required_params:
            assert param in input_types["required"]
    
    def test_return_types(self):
        """Test that RETURN_TYPES are correctly defined.""" 
        if WanImageToVideoAdvancedSampler is None:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
            
        assert hasattr(WanImageToVideoAdvancedSampler, 'RETURN_TYPES')
        assert WanImageToVideoAdvancedSampler.RETURN_TYPES == ("IMAGE", "FLOAT")
        assert WanImageToVideoAdvancedSampler.RETURN_NAMES == ("IMAGE", "FPS")
        assert WanImageToVideoAdvancedSampler.FUNCTION == "run"
    
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_memory_cleanup_calls(self, mock_gc, mock_cuda_cache, sampler_node):
        """Test that memory cleanup functions are called appropriately.""" 
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        # Test the break_circular_references method
        test_locals = {
            'working_model_high': Mock(),
            'working_model_low': Mock(),
            'working_clip_high': Mock(),
            'working_clip_low': Mock(),
            'other_var': 'test'
        }
        
        sampler_node.break_circular_references(test_locals)
        
        # Verify that specific variables were set to None
        assert test_locals['working_model_high'] is None
        assert test_locals['working_model_low'] is None
        assert test_locals['other_var'] == 'test'  # Should be unchanged
        
        # Verify cleanup was called
        mock_gc.assert_called()
    
    def test_load_model_gguf_success(self, sampler_node):
        """Test successful GGUF model loading."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_gguf_loader = Mock()
        mock_model = Mock()
        mock_gguf_loader.load_unet.return_value = (mock_model,)
        
        with patch('wan_image_to_video_advanced_sampler.UnetLoaderGGUF', return_value=mock_gguf_loader):
            with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
                result = sampler_node.load_model("test.gguf", "None", "GGUF", "default")
        
        assert result == mock_model
        mock_gguf_loader.load_unet.assert_called_once_with("test.gguf", None, None, True)
    
    def test_load_model_diffusion_success(self, sampler_node):
        """Test successful diffusion model loading."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_unet_loader = Mock()
        mock_model = Mock()
        mock_unet_loader.load_unet.return_value = (mock_model,)
        
        with patch('nodes.UNETLoader', return_value=mock_unet_loader):
            with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
                result = sampler_node.load_model("None", "test_diffusion.safetensors", "Diffusion", "default")
        
        assert result == mock_model
        mock_unet_loader.load_unet.assert_called_once_with(
            unet_name="test_diffusion.safetensors", 
            weight_dtype="default"
        )
    
    def test_load_model_invalid_type(self, sampler_node):
        """Test loading model with invalid type raises error."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        with pytest.raises(ValueError, match="Invalid model type"):
            sampler_node.load_model("test.gguf", "None", "INVALID", "default")
    
    def test_initialize_tea_cache_and_slg_enabled(self, sampler_node):
        """Test TeaCache and SLG initialization when enabled."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_tea_cache = Mock()
        mock_slg = Mock()
        
        with patch('wan_image_to_video_advanced_sampler.TeaCache', return_value=mock_tea_cache):
            with patch('wan_image_to_video_advanced_sampler.SkipLayerGuidanceWanVideo', return_value=mock_slg):
                with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
                    tea_cache, slg = sampler_node.initialize_tea_cache_and_slg(True, True, "10")
        
        assert tea_cache == mock_tea_cache
        assert slg == mock_slg
    
    def test_initialize_tea_cache_and_slg_disabled(self, sampler_node):
        """Test TeaCache and SLG initialization when disabled."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        with patch('wan_image_to_video_advanced_sampler.output_to_terminal_error'):
            tea_cache, slg = sampler_node.initialize_tea_cache_and_slg(False, False, "10")
        
        assert tea_cache is None
        assert slg is None
    
    def test_process_lora_stack_with_loras(self, sampler_node):
        """Test LoRA stack processing with actual LoRAs."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_model = Mock()
        mock_clip = Mock()
        mock_model_clone = Mock()
        mock_clip_clone = Mock()
        mock_model.clone.return_value = mock_model_clone
        mock_clip.clone.return_value = mock_clip_clone
        
        mock_lora_loader = Mock()
        mock_lora_loader.load_lora.return_value = (mock_model_clone, mock_clip_clone)
        
        lora_stack = [
            ["test_lora.safetensors", 1.0, 1.0],
            ["test_lora2.safetensors", 0.8, 0.8]
        ]
        
        with patch('nodes.LoraLoader', return_value=mock_lora_loader):
            with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
                result_model, result_clip = sampler_node.process_lora_stack(lora_stack, mock_model, mock_clip)
        
        assert result_model == mock_model_clone
        assert result_clip == mock_clip_clone
        assert mock_lora_loader.load_lora.call_count == 2
    
    def test_process_lora_stack_empty(self, sampler_node):
        """Test LoRA stack processing with empty stack."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_model = Mock()
        mock_clip = Mock()
        mock_model_clone = Mock()
        mock_clip_clone = Mock()
        mock_model.clone.return_value = mock_model_clone
        mock_clip.clone.return_value = mock_clip_clone
        
        with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
            result_model, result_clip = sampler_node.process_lora_stack(None, mock_model, mock_clip)
        
        assert result_model == mock_model_clone
        assert result_clip == mock_clip_clone
    
    def test_force_model_cleanup_safety(self, sampler_node):
        """Test that force_model_cleanup doesn't corrupt models."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        mock_model = Mock()
        mock_model.cpu = Mock(return_value=mock_model)
        mock_model.device = Mock()
        mock_model.device.type = "cuda"
        mock_model.__dict__ = {"temp_cache": "test", "diffusion_model": "important"}
        
        with patch('gc.collect'):
            with patch('torch.cuda.empty_cache'):
                with patch('torch.cuda.is_available', return_value=True):
                    sampler_node.force_model_cleanup(mock_model)
        
        # Verify that critical attributes are preserved
        assert hasattr(mock_model, 'diffusion_model')
        # Verify CPU was called for GPU tensors
        mock_model.cpu.assert_called()
    
    @pytest.mark.parametrize("chunk_index,expected_positive", [
        (0, "positive_prompt_0"),
        (1, "positive_prompt_1"), 
        (5, "positive_prompt_1"),  # Should use last available prompt
    ])
    def test_get_current_prompt_stack(self, sampler_node, chunk_index, expected_positive):
        """Test prompt stack selection logic."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        prompt_stack = [
            ("positive_prompt_0", "negative_prompt_0", 0, None),
            ("positive_prompt_1", "negative_prompt_1", 1, None),
        ]
        
        with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
            positive, negative, loras = sampler_node.get_current_prompt(
                prompt_stack, chunk_index, "default_pos", "default_neg"
            )
        
        assert positive == expected_positive
    
    def test_get_current_prompt_no_stack(self, sampler_node):
        """Test prompt selection with no prompt stack."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
            
        positive, negative, loras = sampler_node.get_current_prompt(
            None, 0, "default_pos", "default_neg"
        )
        
        assert positive == "default_pos"
        assert negative == "default_neg"
        assert loras is None

    def test_load_image_node_functionality(self, loaded_start_image):
        """Test that LoadImage node can successfully load the specified image."""
        if LoadImage is None:
            pytest.skip("LoadImage node not available")
        
        # Verify the image was loaded successfully
        assert loaded_start_image is not None
        assert isinstance(loaded_start_image, torch.Tensor)
        
        # Verify ComfyUI image format: [batch, height, width, channels]
        assert len(loaded_start_image.shape) == 4
        assert loaded_start_image.shape[0] >= 1  # batch size
        assert loaded_start_image.shape[3] == 3  # RGB channels
        
        # Verify values are in expected range [0, 1]
        assert loaded_start_image.min() >= 0.0
        assert loaded_start_image.max() <= 1.0
        
        print(f"Loaded start image shape: {loaded_start_image.shape}")
        print(f"Start image value range: {loaded_start_image.min():.3f} to {loaded_start_image.max():.3f}")

    def test_load_end_image_node_functionality(self, loaded_end_image):
        """Test that LoadImage node can successfully load the specified end image."""
        if LoadImage is None:
            pytest.skip("LoadImage node not available")
        
        # Verify the image was loaded successfully
        assert loaded_end_image is not None
        assert isinstance(loaded_end_image, torch.Tensor)
        
        # Verify ComfyUI image format: [batch, height, width, channels]
        assert len(loaded_end_image.shape) == 4
        assert loaded_end_image.shape[0] >= 1  # batch size
        assert loaded_end_image.shape[3] == 3  # RGB channels
        
        # Verify values are in expected range [0, 1]
        assert loaded_end_image.min() >= 0.0
        assert loaded_end_image.max() <= 1.0
        
        print(f"Loaded end image shape: {loaded_end_image.shape}")
        print(f"End image value range: {loaded_end_image.min():.3f} to {loaded_end_image.max():.3f}")

    def test_both_images_loaded(self, loaded_start_image, loaded_end_image):
        """Test that both start and end images are loaded correctly."""
        if LoadImage is None:
            pytest.skip("LoadImage node not available")
        
        # Verify both images are loaded
        assert loaded_start_image is not None
        assert loaded_end_image is not None
        
        # They should be different tensors
        assert not torch.equal(loaded_start_image, loaded_end_image)
        
        print(f"Start image: {loaded_start_image.shape}, End image: {loaded_end_image.shape}")
        print(f"Images are different: {not torch.equal(loaded_start_image, loaded_end_image)}")

    def test_realistic_lora_stack_creation(self, realistic_lora_stack):
        """Test that WanVideoLoraStack creates proper 3-LoRA stack."""
        if WanVideoLoraStack is None:
            pytest.skip("WanVideoLoraStack node not available")
        
        # Verify LoRA stack was created
        assert realistic_lora_stack is not None
        assert isinstance(realistic_lora_stack, list)
        assert len(realistic_lora_stack) == 3  # Three LoRAs stacked
        
        # Verify LoRA 1 structure: [name, model_strength, clip_strength]
        lora1_entry = realistic_lora_stack[0]
        assert lora1_entry[0] == "Wan\\I2V\\lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors"
        assert lora1_entry[1] == 1.0  # model strength
        assert lora1_entry[2] == 1.0  # clip strength
        
        # Verify LoRA 2 structure
        lora2_entry = realistic_lora_stack[1]
        assert lora2_entry[0] == "Wan\\I2V\\wan-nsfw-e14-fixed.safetensors"
        assert lora2_entry[1] == 0.9  # model strength
        assert lora2_entry[2] == 1.0  # clip strength
        
        # Verify LoRA 3 structure
        lora3_entry = realistic_lora_stack[2]
        assert lora3_entry[0] == "Wan\\T2V\\detailz-wan.safetensors"
        assert lora3_entry[1] == 0.2  # model strength
        assert lora3_entry[2] == 1.0  # clip strength
        
        print(f"LoRA stack with {len(realistic_lora_stack)} entries: {realistic_lora_stack}")

    def test_realistic_prompt_stack_creation(self, realistic_prompt_stack):
        """Test that WanPromptChunkStacker creates proper prompt stack."""
        if WanPromptChunkStacker is None:
            pytest.skip("WanPromptChunkStacker node not available")
        
        # Verify prompt stack was created
        assert realistic_prompt_stack is not None
        assert isinstance(realistic_prompt_stack, list)
        assert len(realistic_prompt_stack) == 2  # Two chunks
        
        # Verify chunk 1 structure: [positive, negative, chunk_start, lora_stack]
        chunk1 = realistic_prompt_stack[0]
        assert "blonde girl" in chunk1[0]  # positive prompt
        assert "blurry" in chunk1[1]  # negative prompt
        assert chunk1[2] == 0  # chunk_index_start
        assert chunk1[3] is not None  # lora_stack
        
        # Verify chunk 2 structure
        chunk2 = realistic_prompt_stack[1]
        assert "deepthroat" in chunk2[0]  # positive prompt
        assert "blurry" in chunk2[1]  # negative prompt
        assert chunk2[2] == 3  # chunk_index_start
        assert chunk2[3] is not None  # lora_stack
        
        print(f"Prompt stack chunks: {len(realistic_prompt_stack)}")
        print(f"Chunk 1 start index: {chunk1[2]}")
        print(f"Chunk 2 start index: {chunk2[2]}")

    @patch('folder_paths.get_filename_list')
    @patch('os.listdir')
    @patch('folder_paths.models_dir', '/fake/models')
    def test_full_run_with_realistic_params(self, mock_listdir, mock_get_filename_list, sampler_node, realistic_params_with_image, mock_dependencies):
        """Test the full run method with realistic parameters and actual loaded image."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
        
        # Setup mock returns
        mock_get_filename_list.return_value = ['test_model.safetensors'] 
        mock_listdir.return_value = ['test_engine.onnx']
        
        # Mock all the external dependencies and classes
        with patch('wan_image_to_video_advanced_sampler.UnetLoaderGGUF') as mock_gguf_loader_class:
            mock_gguf_loader = Mock()
            mock_model = Mock()
            mock_model.clone.return_value = Mock()  # For model cloning
            mock_gguf_loader.load_unet.return_value = (mock_model,)
            mock_gguf_loader_class.return_value = mock_gguf_loader
            
            with patch('wan_image_to_video_advanced_sampler.nodes') as mock_nodes:
                # Setup all node mocks
                mock_nodes.VAELoader.return_value.load_vae.return_value = (Mock(),)
                mock_nodes.CLIPLoader.return_value.load_clip.return_value = (Mock(),)
                mock_nodes.CLIPSetLastLayer.return_value.set_last_layer.return_value = (Mock(),)
                mock_nodes.KSamplerAdvanced.return_value = Mock()
                mock_nodes.CLIPTextEncode.return_value.encode.return_value = (Mock(),)
                mock_nodes.CLIPVisionLoader.return_value.load_clip.return_value = (Mock(),)
                mock_nodes.CLIPVisionEncode.return_value = Mock()
                mock_nodes.LoraLoader.return_value.load_lora.return_value = (Mock(), Mock())
                
                with patch('wan_image_to_video_advanced_sampler.mm') as mock_mm:
                    mock_mm.throw_exception_if_processing_interrupted = Mock()
                    mock_mm.unload_all_models = Mock()
                    mock_mm.soft_empty_cache = Mock()
                    
                    with patch('wan_image_to_video_advanced_sampler.WanFirstLastFirstFrameToVideo') as mock_wan_i2v_class:
                        mock_wan_i2v = Mock()
                        mock_wan_i2v.encode.return_value = (Mock(), Mock(), Mock(), Mock(), Mock())
                        mock_wan_i2v_class.return_value = mock_wan_i2v
                        
                        with patch('wan_image_to_video_advanced_sampler.WanVideoVaeDecode') as mock_wan_vae_decode_class:
                            mock_wan_vae_decode = Mock()
                            # Return a proper video tensor: [frames, height, width, channels]
                            mock_video_output = torch.randn(16, 512, 512, 3)
                            mock_wan_vae_decode.decode.return_value = (mock_video_output,)
                            mock_wan_vae_decode_class.return_value = mock_wan_vae_decode
                            
                            with patch('wan_image_to_video_advanced_sampler.WanGetMaxImageResolutionByAspectRatio') as mock_resolution_class:
                                mock_resolution = Mock()
                                mock_resolution.run.return_value = (832, 832)
                                mock_resolution_class.return_value = mock_resolution
                                
                                with patch('wan_image_to_video_advanced_sampler.ImageResizeKJv2') as mock_resizer_class:
                                    mock_resizer = Mock()
                                    mock_resizer.resize.return_value = (realistic_params_with_image['start_image'],)
                                    mock_resizer_class.return_value = mock_resizer
                                    
                                    with patch('wan_image_to_video_advanced_sampler.WanVideoBlockSwap') as mock_block_swap_class:
                                        mock_block_swap = Mock()
                                        mock_block_swap_class.return_value = mock_block_swap
                                        
                                        with patch('wan_image_to_video_advanced_sampler.output_to_terminal_successful'):
                                            with patch('wan_image_to_video_advanced_sampler.output_to_terminal'):
                                                with patch('wan_image_to_video_advanced_sampler.output_to_terminal_error'):
                                                    with patch('torch.cuda.empty_cache'):
                                                        with patch('gc.collect'):
                                                            # Run the actual method
                                                            result_image, result_fps = sampler_node.run(**realistic_params_with_image)
        
        # Verify the results
        assert result_image is not None
        assert isinstance(result_fps, float)
        assert result_fps > 0
        
        # Verify that video output has correct shape (should be video frames minus 1)
        assert result_image.shape[0] == 15  # 16 frames - 1
        assert result_image.shape[1] == 512  # height
        assert result_image.shape[2] == 512  # width 
        assert result_image.shape[3] == 3    # channels
        
        # Verify that models were loaded
        mock_gguf_loader.load_unet.assert_called()
        mock_nodes.VAELoader.return_value.load_vae.assert_called()
        mock_nodes.CLIPLoader.return_value.load_clip.assert_called()
        
        # Verify that video generation pipeline was executed
        mock_wan_i2v.encode.assert_called()
        mock_wan_vae_decode.decode.assert_called()
        
        # Verify memory cleanup was called
        mock_mm.unload_all_models.assert_called()
        mock_mm.soft_empty_cache.assert_called()
    
    def test_run_with_lora_stack(self, sampler_node, realistic_params, realistic_lora_stack, loaded_start_image, loaded_end_image):
        """Test run method with realistic LoRA stack applied."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
        
        params_with_loras = realistic_params.copy()
        params_with_loras['lora_stack'] = realistic_lora_stack
        params_with_loras['start_image'] = loaded_start_image
        params_with_loras['end_image'] = loaded_end_image
        
        with patch('folder_paths.get_filename_list', return_value=['test_model.safetensors']):
            with patch('os.listdir', return_value=['test_engine.onnx']):
                with patch('folder_paths.models_dir', '/fake/models'):
                    with patch.object(sampler_node, 'postprocess') as mock_postprocess:
                        mock_postprocess.return_value = (torch.randn(15, 512, 512, 3), 16.0)
                        
                        with patch.object(sampler_node, 'load_model') as mock_load_model:
                            mock_load_model.return_value = Mock()
                            
                            with patch('wan_image_to_video_advanced_sampler.nodes') as mock_nodes:
                                mock_nodes.VAELoader.return_value.load_vae.return_value = (Mock(),)
                                mock_nodes.CLIPLoader.return_value.load_clip.return_value = (Mock(),)
                                mock_nodes.CLIPSetLastLayer.return_value.set_last_layer.return_value = (Mock(),)
                                
                                with patch('wan_image_to_video_advanced_sampler.mm') as mock_mm:
                                    mock_mm.throw_exception_if_processing_interrupted = Mock()
                                    mock_mm.unload_all_models = Mock()
                                    mock_mm.soft_empty_cache = Mock()
                                    
                                    with patch('torch.cuda.empty_cache'):
                                        with patch('gc.collect'):
                                            result_image, result_fps = sampler_node.run(**params_with_loras)
        
        # Verify LoRA stack was passed to postprocess
        mock_postprocess.assert_called_once()
        call_args = mock_postprocess.call_args[1]  # Get keyword arguments
        assert call_args['lora_stack'] == realistic_lora_stack
        
        # Verify results
        assert result_image is not None
        assert result_fps == 16.0
    
    def test_run_with_prompt_stack(self, sampler_node, realistic_params, realistic_prompt_stack, loaded_start_image, loaded_end_image):
        """Test run method with realistic prompt stack for multi-chunk generation."""
        if sampler_node is None:
            pytest.skip("Sampler node not available")
        
        params_with_prompts = realistic_params.copy()
        params_with_prompts['prompt_stack'] = realistic_prompt_stack
        params_with_prompts['total_video_chunks'] = 2  # Test multi-chunk
        params_with_prompts['start_image'] = loaded_start_image
        params_with_prompts['end_image'] = loaded_end_image
        
        with patch('folder_paths.get_filename_list', return_value=['test_model.safetensors']):
            with patch('os.listdir', return_value=['test_engine.onnx']):
                with patch('folder_paths.models_dir', '/fake/models'):
                    with patch.object(sampler_node, 'postprocess') as mock_postprocess:
                        mock_postprocess.return_value = (torch.randn(31, 512, 512, 3), 16.0)  # 2 chunks
                        
                        with patch.object(sampler_node, 'load_model') as mock_load_model:
                            mock_load_model.return_value = Mock()
                            
                            with patch('wan_image_to_video_advanced_sampler.nodes') as mock_nodes:
                                mock_nodes.VAELoader.return_value.load_vae.return_value = (Mock(),)
                                mock_nodes.CLIPLoader.return_value.load_clip.return_value = (Mock(),)
                                mock_nodes.CLIPSetLastLayer.return_value.set_last_layer.return_value = (Mock(),)
                                
                                with patch('wan_image_to_video_advanced_sampler.mm') as mock_mm:
                                    mock_mm.throw_exception_if_processing_interrupted = Mock()
                                    mock_mm.unload_all_models = Mock()
                                    mock_mm.soft_empty_cache = Mock()
                                    
                                    with patch('torch.cuda.empty_cache'):
                                        with patch('gc.collect'):
                                            result_image, result_fps = sampler_node.run(**params_with_prompts)
        
        # Verify prompt stack was passed to postprocess
        mock_postprocess.assert_called_once()
        call_args = mock_postprocess.call_args[1]  # Get keyword arguments
        assert call_args['prompt_stack'] == realistic_prompt_stack
        assert call_args['total_video_chunks'] == 2
        
        # Verify results
        assert result_image is not None
        assert result_fps == 16.0
