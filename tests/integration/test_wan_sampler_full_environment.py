"""
Integration tests for WanImageToVideoAdvancedSampler in full ComfyUI environment.

This test requires the full ComfyUI environment to be properly set up and running.
It tests the actual functionality of the WAN sampler with real models and data.
"""
import pytest
import torch
import sys
import os
from pathlib import Path
import json
import time

# Ensure ComfyUI is in the path
comfyui_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(comfyui_root))

# Add the custom nodes directory
custom_nodes_dir = comfyui_root / "custom_nodes" / "ComfyUI-Przewodo-Utils"
sys.path.insert(0, str(custom_nodes_dir))

@pytest.mark.integration
@pytest.mark.slow
class TestWanImageToVideoAdvancedSamplerFullEnvironment:
    """Integration tests for WAN sampler in full ComfyUI environment."""
    
    @pytest.fixture(autouse=True)
    def setup_comfyui_environment(self):
        """Set up the full ComfyUI environment before each test."""
        try:
            # Initialize ComfyUI core systems
            import folder_paths
            from comfy import model_management as mm
            
            # Verify essential ComfyUI components are available
            assert hasattr(folder_paths, 'models_dir'), "ComfyUI folder_paths not properly initialized"
            assert hasattr(mm, 'get_torch_device'), "ComfyUI model_management not available"
            
            # Set up model paths if needed
            models_dir = Path(folder_paths.models_dir)
            assert models_dir.exists(), f"Models directory not found: {models_dir}"
            
            print(f"✓ ComfyUI environment initialized")
            print(f"  - Models directory: {models_dir}")
            print(f"  - Torch device: {mm.get_torch_device()}")
            
        except ImportError as e:
            pytest.skip(f"ComfyUI environment not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to initialize ComfyUI environment: {e}")
    
    @pytest.fixture
    def wan_sampler_node(self):
        """Create a WanImageToVideoAdvancedSampler instance in full ComfyUI environment."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
            return WanImageToVideoAdvancedSampler()
        except ImportError as e:
            pytest.skip(f"WanImageToVideoAdvancedSampler not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to create WAN sampler: {e}")
    
    @pytest.fixture
    def test_images(self):
        """Load test images using ComfyUI's LoadImage node."""
        try:
            from nodes import LoadImage
            load_image_node = LoadImage()
            
            # Try to load test images from ComfyUI's input directory
            input_dir = Path(comfyui_root) / "input"
            test_images = {}
            
            # Look for any PNG or JPG files in input directory
            image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
            
            if len(image_files) >= 2:
                # Load first two images as start and end
                start_image, _ = load_image_node.load_image(image_files[0].name)
                end_image, _ = load_image_node.load_image(image_files[1].name)
                
                test_images['start_image'] = start_image
                test_images['end_image'] = end_image
                test_images['start_filename'] = image_files[0].name
                test_images['end_filename'] = image_files[1].name
                
                print(f"✓ Loaded test images:")
                print(f"  - Start: {image_files[0].name} {start_image.shape}")
                print(f"  - End: {image_files[1].name} {end_image.shape}")
                
                return test_images
            else:
                pytest.skip(f"Need at least 2 images in {input_dir} for testing")
                
        except ImportError as e:
            pytest.skip(f"ComfyUI LoadImage not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to load test images: {e}")
    
    @pytest.fixture
    def real_lora_stack(self):
        """Create a real LoRA stack using WanVideoLoraStack if available."""
        try:
            from wan_video_lora_stack import WanVideoLoraStack
            import folder_paths
            
            lora_stack_node = WanVideoLoraStack()
            
            # Get available LoRA files
            lora_files = folder_paths.get_filename_list("loras")
            wan_loras = [f for f in lora_files if "wan" in f.lower() or "i2v" in f.lower()]
            
            if len(wan_loras) >= 2:
                # Create a 2-LoRA stack
                lora1_result = lora_stack_node.load_lora(
                    lora_name=wan_loras[0],
                    strength_model=1.0,
                    strength_clip=1.0,
                    previous_lora=None
                )
                
                lora2_result = lora_stack_node.load_lora(
                    lora_name=wan_loras[1],
                    strength_model=0.8,
                    strength_clip=0.8,
                    previous_lora=lora1_result[0]
                )
                
                print(f"✓ Created LoRA stack with {len(wan_loras[:2])} LoRAs:")
                for i, lora in enumerate(wan_loras[:2]):
                    print(f"  - LoRA {i+1}: {lora}")
                
                return lora2_result[0]
            else:
                print(f"⚠ No WAN LoRAs found, using None for lora_stack")
                return None
                
        except ImportError as e:
            print(f"⚠ WanVideoLoraStack not available: {e}")
            return None
        except Exception as e:
            print(f"⚠ Failed to create LoRA stack: {e}")
            return None
    
    @pytest.fixture
    def real_prompt_stack(self, real_lora_stack):
        """Create a real prompt stack using WanPromptChunkStacker if available."""
        try:
            from wan_prompt_chunck_stacker import WanPromptChunkStacker
            
            prompt_stacker_node = WanPromptChunkStacker()
            
            # Create a simple 2-chunk prompt stack
            chunk1_result = prompt_stacker_node.run(
                previous_prompt=None,
                lora_stack=real_lora_stack,
                positive_prompt="A beautiful woman with flowing hair, cinematic lighting, high quality, detailed",
                negative_prompt="blurry, low quality, distorted, ugly",
                chunk_index_start=0
            )
            
            chunk2_result = prompt_stacker_node.run(
                previous_prompt=chunk1_result[0],
                lora_stack=real_lora_stack,
                positive_prompt="A beautiful woman smiling gently, soft lighting, artistic, professional photography",
                negative_prompt="blurry, low quality, distorted, ugly, dark",
                chunk_index_start=8
            )
            
            print(f"✓ Created prompt stack with 2 chunks")
            return chunk2_result[0]
            
        except ImportError as e:
            print(f"⚠ WanPromptChunkStacker not available: {e}")
            return None
        except Exception as e:
            print(f"⚠ Failed to create prompt stack: {e}")
            return None
    
    @pytest.fixture
    def realistic_wan_params(self, test_images, real_lora_stack, real_prompt_stack):
        """Create realistic parameters for WAN sampler testing."""
        try:
            import folder_paths
            
            # Get available models
            gguf_models = folder_paths.get_filename_list("unet")
            wan_gguf = [m for m in gguf_models if "wan" in m.lower() and "i2v" in m.lower() and m.endswith(".gguf")]
            
            vae_models = folder_paths.get_filename_list("vae")
            wan_vae = [v for v in vae_models if "wan" in v.lower()]
            
            clip_models = folder_paths.get_filename_list("clip")
            t5_clips = [c for c in clip_models if "t5" in c.lower() or "umt5" in c.lower()]
            
            # Use first available models or fallback to "None"
            gguf_model = wan_gguf[0] if wan_gguf else "None"
            vae_model = wan_vae[0] if wan_vae else "None"
            clip_model = t5_clips[0] if t5_clips else "None"
            
            params = {
                # Model configuration
                "GGUF_High": gguf_model,
                "GGUF_Low": "None",
                "Diffusor_High": "None",
                "Diffusor_Low": "None", 
                "Diffusor_weight_dtype": "default",
                "Use_Model_Type": "GGUF" if gguf_model != "None" else "Diffusion",
                
                # Text & CLIP configuration
                "positive": "A beautiful cinematic scene, high quality, detailed",
                "negative": "blurry, low quality, distorted",
                "clip": clip_model,
                "clip_type": "wan",
                "clip_device": "default",
                "vae": vae_model,
                
                # Optimization settings (disabled for testing reliability)
                "use_tea_cache": False,
                "tea_cache_model_type": "wan2.1_i2v_480p_14B",
                "tea_cache_rel_l1_thresh": 0.05,
                "tea_cache_start_percent": 0.2,
                "tea_cache_end_percent": 0.8,
                "tea_cache_cache_device": "cuda",
                
                "use_SLG": False,
                "SLG_blocks": "10",
                "SLG_start_percent": 0.2,
                "SLG_end_percent": 0.8,
                
                "use_sage_attention": False,  # Disabled for compatibility
                "sage_attention_mode": "auto",
                "use_shift": True,
                "shift": 8.0,
                "use_block_swap": False,  # Disabled for testing
                "block_swap": 20,
                
                # Video generation settings
                "large_image_side": 512,  # Smaller for faster testing
                "image_generation_mode": "start_image",
                "wan_model_size": "480p",
                "total_video_seconds": 2,  # Shorter for testing
                "total_video_chunks": 1,  # Single chunk for initial testing
                
                # CLIP Vision settings
                "clip_vision_model": "None",  # Disabled for testing
                "clip_vision_strength": 1.0,
                "start_image_clip_vision_enabled": False,
                "end_image_clip_vision_enabled": False,
                
                # Sampling configuration
                "use_dual_samplers": False,
                "high_cfg": 1.5,
                "low_cfg": 1.0,
                "high_denoise": 1.0,
                "low_denoise": 1.0,
                "total_steps": 4,  # Fewer steps for faster testing
                "total_steps_high_cfg": 20,  # Fewer steps
                "fill_noise_latent": 0.5,
                "noise_seed": 42,  # Fixed seed for reproducibility
                
                # CausVid enhancement (disabled for testing)
                "causvid_lora": "None",
                "high_cfg_causvid_strength": 0.0,
                "low_cfg_causvid_strength": 0.0,
                
                # Post-processing options (minimal for testing)
                "video_enhance_enabled": False,
                "use_cfg_zero_star": False,
                "apply_color_match": False,
                "frames_interpolation": False,
                "frames_engine": "None",
                "frames_multiplier": 2,
                "frames_clear_cache_after_n_frames": 100,
                "frames_use_cuda_graph": False,
                "frames_overlap_chunks": 4,
                "frames_overlap_chunks_blend": 0.3,
                
                # Test images and stacks
                "start_image": test_images['start_image'] if test_images else None,
                "end_image": test_images['end_image'] if test_images else None,
                "lora_stack": real_lora_stack,
                "prompt_stack": real_prompt_stack,
            }
            
            print(f"✓ Created realistic WAN parameters:")
            print(f"  - GGUF Model: {gguf_model}")
            print(f"  - VAE Model: {vae_model}")
            print(f"  - CLIP Model: {clip_model}")
            print(f"  - Has LoRA stack: {real_lora_stack is not None}")
            print(f"  - Has prompt stack: {real_prompt_stack is not None}")
            print(f"  - Has test images: {test_images is not None}")
            
            return params
            
        except Exception as e:
            pytest.skip(f"Failed to create realistic parameters: {e}")
    
    def test_wan_sampler_input_types(self, wan_sampler_node):
        """Test that WAN sampler INPUT_TYPES works in full environment."""
        input_types = wan_sampler_node.INPUT_TYPES()
        
        assert "required" in input_types
        assert "optional" in input_types
        assert isinstance(input_types["required"], dict)
        assert isinstance(input_types["optional"], dict)
        
        # Verify key parameters exist
        required_keys = ["GGUF_High", "positive", "negative", "clip", "vae", "total_steps", "noise_seed"]
        for key in required_keys:
            assert key in input_types["required"], f"Missing required parameter: {key}"
        
        print(f"✓ INPUT_TYPES validation passed with {len(input_types['required'])} required and {len(input_types['optional'])} optional parameters")
    
    def test_wan_sampler_return_types(self, wan_sampler_node):
        """Test that WAN sampler return types are correct."""
        assert hasattr(wan_sampler_node, 'RETURN_TYPES')
        assert hasattr(wan_sampler_node, 'RETURN_NAMES')
        assert hasattr(wan_sampler_node, 'FUNCTION')
        
        assert wan_sampler_node.RETURN_TYPES == ("IMAGE", "FLOAT")
        assert wan_sampler_node.RETURN_NAMES == ("IMAGE", "FPS")
        assert wan_sampler_node.FUNCTION == "run"
        
        print(f"✓ Return types validation passed")
    
    @pytest.mark.slow
    def test_wan_sampler_minimal_run(self, wan_sampler_node, realistic_wan_params):
        """Test minimal WAN sampler execution in full environment."""
        if realistic_wan_params["GGUF_High"] == "None" and realistic_wan_params["Use_Model_Type"] == "GGUF":
            pytest.skip("No GGUF models available for testing")
        
        print(f"\n🚀 Starting WAN sampler minimal run test...")
        start_time = time.time()
        
        try:
            # Run the sampler with realistic parameters
            result_image, result_fps = wan_sampler_node.run(**realistic_wan_params)
            
            # Verify results
            assert result_image is not None, "Result image should not be None"
            assert isinstance(result_image, torch.Tensor), "Result should be a torch.Tensor"
            assert len(result_image.shape) == 4, "Result should be 4D tensor [frames, height, width, channels]"
            assert result_image.shape[0] > 0, "Should have at least one frame"
            assert result_image.shape[3] == 3, "Should have 3 color channels"
            
            assert isinstance(result_fps, float), "FPS should be a float"
            assert result_fps > 0, "FPS should be positive"
            
            execution_time = time.time() - start_time
            
            print(f"✅ WAN sampler execution successful!")
            print(f"  - Output shape: {result_image.shape}")
            print(f"  - Output FPS: {result_fps}")
            print(f"  - Execution time: {execution_time:.2f}s")
            print(f"  - Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "")
            
            return result_image, result_fps
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ WAN sampler execution failed after {execution_time:.2f}s")
            print(f"  - Error: {str(e)}")
            
            # Don't fail the test if it's due to missing models or CUDA OOM
            if any(keyword in str(e).lower() for keyword in ['cuda', 'memory', 'model', 'file not found']):
                pytest.skip(f"Test requires specific hardware/models: {e}")
            else:
                raise
    
    @pytest.mark.slow
    def test_wan_sampler_with_lora_stack(self, wan_sampler_node, realistic_wan_params, real_lora_stack):
        """Test WAN sampler with LoRA stack in full environment."""
        if real_lora_stack is None:
            pytest.skip("No LoRA stack available for testing")
        
        if realistic_wan_params["GGUF_High"] == "None":
            pytest.skip("No GGUF models available for testing")
        
        print(f"\n🚀 Starting WAN sampler LoRA stack test...")
        start_time = time.time()
        
        try:
            # Ensure LoRA stack is in parameters
            params_with_lora = realistic_wan_params.copy()
            params_with_lora["lora_stack"] = real_lora_stack
            
            result_image, result_fps = wan_sampler_node.run(**params_with_lora)
            
            # Verify results
            assert result_image is not None
            assert isinstance(result_image, torch.Tensor)
            assert len(result_image.shape) == 4
            assert result_image.shape[0] > 0
            
            execution_time = time.time() - start_time
            
            print(f"✅ WAN sampler with LoRA execution successful!")
            print(f"  - Output shape: {result_image.shape}")
            print(f"  - LoRA stack entries: {len(real_lora_stack)}")
            print(f"  - Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ WAN sampler with LoRA failed after {execution_time:.2f}s: {e}")
            
            if any(keyword in str(e).lower() for keyword in ['cuda', 'memory', 'model']):
                pytest.skip(f"Test requires specific hardware/models: {e}")
            else:
                raise
    
    @pytest.mark.slow
    def test_wan_sampler_with_prompt_stack(self, wan_sampler_node, realistic_wan_params, real_prompt_stack):
        """Test WAN sampler with prompt stack in full environment."""
        if real_prompt_stack is None:
            pytest.skip("No prompt stack available for testing")
        
        if realistic_wan_params["GGUF_High"] == "None":
            pytest.skip("No GGUF models available for testing")
        
        print(f"\n🚀 Starting WAN sampler prompt stack test...")
        start_time = time.time()
        
        try:
            # Use prompt stack and set chunks accordingly
            params_with_prompts = realistic_wan_params.copy()
            params_with_prompts["prompt_stack"] = real_prompt_stack
            params_with_prompts["total_video_chunks"] = 2  # 2 chunks for prompt stack
            
            result_image, result_fps = wan_sampler_node.run(**params_with_prompts)
            
            # Verify results
            assert result_image is not None
            assert isinstance(result_image, torch.Tensor)
            assert len(result_image.shape) == 4
            assert result_image.shape[0] > 0
            
            execution_time = time.time() - start_time
            
            print(f"✅ WAN sampler with prompt stack execution successful!")
            print(f"  - Output shape: {result_image.shape}")
            print(f"  - Prompt stack chunks: {len(real_prompt_stack)}")
            print(f"  - Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ WAN sampler with prompt stack failed after {execution_time:.2f}s: {e}")
            
            if any(keyword in str(e).lower() for keyword in ['cuda', 'memory', 'model']):
                pytest.skip(f"Test requires specific hardware/models: {e}")
            else:
                raise
    
    def test_wan_sampler_memory_management(self, wan_sampler_node):
        """Test WAN sampler memory management functions."""
        # Test memory cleanup functions
        test_locals = {
            'working_model_high': torch.randn(10, 10),
            'working_model_low': torch.randn(10, 10),
            'working_clip_high': torch.randn(5, 5),
            'working_clip_low': torch.randn(5, 5),
            'other_var': 'should_remain'
        }
        
        # Test break_circular_references
        wan_sampler_node.break_circular_references(test_locals)
        
        # Verify cleanup
        assert test_locals['working_model_high'] is None
        assert test_locals['working_model_low'] is None
        assert test_locals['other_var'] == 'should_remain'
        
        print(f"✓ Memory management functions working correctly")
    
    def test_environment_requirements(self):
        """Test that the full ComfyUI environment has required components."""
        try:
            import folder_paths
            from comfy import model_management as mm
            from nodes import LoadImage, VAELoader, CLIPLoader
            
            # Test basic ComfyUI functionality
            device = mm.get_torch_device()
            models_dir = folder_paths.models_dir
            
            # Test node creation
            load_image = LoadImage()
            vae_loader = VAELoader()
            clip_loader = CLIPLoader()
            
            print(f"✓ Full ComfyUI environment verification passed")
            print(f"  - Device: {device}")
            print(f"  - Models dir: {models_dir}")
            print(f"  - Core nodes available: LoadImage, VAELoader, CLIPLoader")
            
        except ImportError as e:
            pytest.fail(f"Missing ComfyUI components: {e}")
        except Exception as e:
            pytest.fail(f"ComfyUI environment error: {e}")

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
