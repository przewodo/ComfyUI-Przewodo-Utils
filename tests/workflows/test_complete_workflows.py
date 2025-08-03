"""
Workflow tests for ComfyUI-Przewodo-Utils nodes.

These tests simulate complete ComfyUI workflows to ensure
nodes work correctly in real usage scenarios.
"""
import pytest
import torch
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@pytest.mark.workflow
class TestCompleteWorkflows:
    """Test complete workflows using Przewodo Utils nodes."""
    
    @pytest.fixture
    def basic_i2v_workflow(self, sample_image_tensor):
        """Basic image-to-video workflow configuration."""
        return {
            "nodes": {
                "1": {
                    "class_type": "WanImageToVideoAdvancedSampler",
                    "inputs": {
                        "GGUF_High": "wan2.1_i2v_720p_14B.gguf",
                        "positive": "A beautiful cinematic shot",
                        "negative": "blurry, low quality",
                        "clip": "wan_clip_vision.safetensors",
                        "vae": "wan_vae.safetensors",
                        "start_image": sample_image_tensor,
                        "total_video_seconds": 2,
                        "total_video_chunks": 1,
                        "use_dual_samplers": True,
                        "high_cfg": 3.0,
                        "low_cfg": 1.0,
                        "total_steps": 20,
                        "noise_seed": 42,
                    }
                }
            }
        }
    
    @pytest.fixture
    def multi_chunk_workflow(self, sample_image_tensor):
        """Multi-chunk video generation workflow."""
        return {
            "nodes": {
                "1": {
                    "class_type": "WanPromptChunckStacker", 
                    "inputs": {
                        "positive": "landscape in spring",
                        "negative": "blurry",
                        "chunk_index_start": 0,
                    }
                },
                "2": {
                    "class_type": "WanPromptChunckStacker",
                    "inputs": {
                        "positive": "landscape in summer", 
                        "negative": "blurry",
                        "chunk_index_start": 1,
                        "prompt_stack": ["1", 0],  # Reference to node 1
                    }
                },
                "3": {
                    "class_type": "WanImageToVideoAdvancedSampler",
                    "inputs": {
                        "GGUF_High": "wan2.1_i2v_720p_14B.gguf",
                        "positive": "default prompt",
                        "negative": "default negative",
                        "clip": "wan_clip_vision.safetensors",
                        "vae": "wan_vae.safetensors", 
                        "start_image": sample_image_tensor,
                        "total_video_seconds": 2,
                        "total_video_chunks": 2,
                        "prompt_stack": ["2", 0],  # Reference to stacked prompts
                        "use_dual_samplers": True,
                        "total_steps": 15,
                        "noise_seed": 42,
                    }
                }
            }
        }
    
    @pytest.fixture
    def lora_workflow(self, sample_image_tensor):
        """Workflow with LoRA stack integration."""
        return {
            "nodes": {
                "1": {
                    "class_type": "WanVideoLoraStack",
                    "inputs": {
                        "lora_name": "cinematic_style.safetensors",
                        "model_strength": 0.8,
                        "clip_strength": 0.8,
                    }
                },
                "2": {
                    "class_type": "WanVideoLoraStack",
                    "inputs": {
                        "lora_name": "enhance_details.safetensors", 
                        "model_strength": 0.6,
                        "clip_strength": 0.6,
                        "lora_stack": ["1", 0],  # Reference to previous LoRA
                    }
                },
                "3": {
                    "class_type": "WanImageToVideoAdvancedSampler",
                    "inputs": {
                        "GGUF_High": "wan2.1_i2v_720p_14B.gguf",
                        "positive": "cinematic masterpiece",
                        "negative": "low quality",
                        "clip": "wan_clip_vision.safetensors",
                        "vae": "wan_vae.safetensors",
                        "start_image": sample_image_tensor,
                        "lora_stack": ["2", 0],  # Reference to LoRA stack
                        "total_video_seconds": 1,
                        "total_video_chunks": 1,
                        "total_steps": 20,
                        "noise_seed": 123,
                    }
                }
            }
        }
    
    def test_basic_i2v_workflow_execution(self, basic_i2v_workflow):
        """Test basic image-to-video workflow execution."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        # Extract the node configuration
        node_config = basic_i2v_workflow["nodes"]["1"]
        inputs = node_config["inputs"]
        
        # Mock the required dependencies
        sampler = WanImageToVideoAdvancedSampler()
        
        with patch.object(sampler, 'load_model') as mock_load_model:
            with patch('nodes.VAELoader') as mock_vae_loader:
                with patch('nodes.CLIPLoader') as mock_clip_loader:
                    with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                        with patch.object(sampler, 'postprocess') as mock_postprocess:
                            # Configure mocks
                            mock_load_model.return_value = Mock()
                            mock_vae_loader().load_vae.return_value = (Mock(),)
                            mock_clip_loader().load_clip.return_value = (Mock(),)
                            mock_clip_set_layer().set_last_layer.return_value = (Mock(),)
                            
                            # Mock successful video generation
                            output_video = torch.randn(32, 720, 1280, 3)  # 2 seconds at 16fps
                            mock_postprocess.return_value = (output_video, 16.0)
                            
                            # Execute the workflow
                            result = sampler.run(**inputs)
        
        # Verify workflow execution
        assert isinstance(result, tuple)
        assert len(result) == 2
        video_output, fps = result
        assert torch.is_tensor(video_output)
        assert fps == 16.0
        
        # Verify the output has expected dimensions
        assert len(video_output.shape) == 4  # [frames, height, width, channels]
        assert video_output.shape[0] > 0  # Has frames
        assert video_output.shape[3] == 3  # RGB channels
    
    def test_prompt_stack_workflow(self, multi_chunk_workflow):
        """Test workflow with prompt stacking across chunks."""
        try:
            from wan_prompt_chunck_stacker import WanPromptChunckStacker
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("Required nodes not available")
        
        # Simulate the workflow execution order
        
        # Step 1: Create first prompt
        stacker1 = WanPromptChunckStacker()
        node1_inputs = multi_chunk_workflow["nodes"]["1"]["inputs"]
        prompt_stack1 = stacker1.run(**node1_inputs)
        
        # Step 2: Add second prompt to stack
        stacker2 = WanPromptChunckStacker()
        node2_inputs = multi_chunk_workflow["nodes"]["2"]["inputs"]
        node2_inputs["prompt_stack"] = prompt_stack1[0]  # Use output from step 1
        prompt_stack2 = stacker2.run(**node2_inputs)
        
        # Step 3: Generate video with prompt stack
        sampler = WanImageToVideoAdvancedSampler()
        node3_inputs = multi_chunk_workflow["nodes"]["3"]["inputs"]
        node3_inputs["prompt_stack"] = prompt_stack2[0]  # Use final prompt stack
        
        with patch.object(sampler, 'load_model') as mock_load_model:
            with patch('nodes.VAELoader') as mock_vae_loader:
                with patch('nodes.CLIPLoader') as mock_clip_loader:
                    with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                        with patch.object(sampler, 'postprocess') as mock_postprocess:
                            # Configure mocks
                            mock_load_model.return_value = Mock()
                            mock_vae_loader().load_vae.return_value = (Mock(),)
                            mock_clip_loader().load_clip.return_value = (Mock(),)
                            mock_clip_set_layer().set_last_layer.return_value = (Mock(),)
                            
                            # Mock multi-chunk video generation
                            output_video = torch.randn(64, 720, 1280, 3)  # 2 chunks
                            mock_postprocess.return_value = (output_video, 16.0)
                            
                            # Execute the final node
                            result = sampler.run(**node3_inputs)
        
        # Verify workflow results
        assert isinstance(result, tuple)
        video_output, fps = result
        assert torch.is_tensor(video_output)
        assert video_output.shape[0] > 32  # More frames due to multiple chunks
    
    def test_lora_stack_workflow(self, lora_workflow):
        """Test workflow with LoRA stacking."""
        try:
            from wan_video_lora_stack import WanVideoLoraStack
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("Required nodes not available")
        
        # Step 1: Create first LoRA
        lora_stack1 = WanVideoLoraStack()
        node1_inputs = lora_workflow["nodes"]["1"]["inputs"]
        lora_result1 = lora_stack1.run(**node1_inputs)
        
        # Step 2: Add second LoRA to stack
        lora_stack2 = WanVideoLoraStack()
        node2_inputs = lora_workflow["nodes"]["2"]["inputs"] 
        node2_inputs["lora_stack"] = lora_result1[0]  # Use output from step 1
        lora_result2 = lora_stack2.run(**node2_inputs)
        
        # Step 3: Generate video with LoRA stack
        sampler = WanImageToVideoAdvancedSampler()
        node3_inputs = lora_workflow["nodes"]["3"]["inputs"]
        node3_inputs["lora_stack"] = lora_result2[0]  # Use final LoRA stack
        
        with patch.object(sampler, 'load_model') as mock_load_model:
            with patch('nodes.VAELoader') as mock_vae_loader:
                with patch('nodes.CLIPLoader') as mock_clip_loader:
                    with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                        with patch.object(sampler, 'postprocess') as mock_postprocess:
                            # Configure mocks
                            mock_load_model.return_value = Mock()
                            mock_vae_loader().load_vae.return_value = (Mock(),)
                            mock_clip_loader().load_clip.return_value = (Mock(),)
                            mock_clip_set_layer().set_last_layer.return_value = (Mock(),)
                            
                            # Mock video generation with LoRAs
                            output_video = torch.randn(16, 720, 1280, 3)
                            mock_postprocess.return_value = (output_video, 16.0)
                            
                            # Execute the final node
                            result = sampler.run(**node3_inputs)
        
        # Verify LoRA workflow results
        assert isinstance(result, tuple)
        video_output, fps = result
        assert torch.is_tensor(video_output)
        assert video_output.shape[0] == 16  # Expected frame count
    
    @pytest.mark.slow
    def test_workflow_memory_efficiency(self, basic_i2v_workflow):
        """Test that workflows manage memory efficiently."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory efficiency test")
        
        # Monitor memory usage throughout workflow
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = initial_memory
        
        def memory_monitor():
            nonlocal peak_memory
            current = torch.cuda.memory_allocated()
            peak_memory = max(peak_memory, current)
        
        # Execute workflow with memory monitoring
        sampler = WanImageToVideoAdvancedSampler()
        node_config = basic_i2v_workflow["nodes"]["1"]
        inputs = node_config["inputs"]
        
        with patch.object(sampler, 'load_model') as mock_load_model:
            with patch.object(sampler, 'postprocess') as mock_postprocess:
                # Add memory monitoring to key points
                mock_load_model.side_effect = lambda *args, **kwargs: (memory_monitor(), Mock())[1]
                mock_postprocess.side_effect = lambda *args, **kwargs: (memory_monitor(), (torch.randn(16, 720, 1280, 3), 16.0))[1]
                
                with patch('nodes.VAELoader') as mock_vae_loader:
                    with patch('nodes.CLIPLoader') as mock_clip_loader:
                        with patch('nodes.CLIPSetLastLayer') as mock_clip_set_layer:
                            mock_vae_loader().load_vae.return_value = (Mock(),)
                            mock_clip_loader().load_clip.return_value = (Mock(),)
                            mock_clip_set_layer().set_last_layer.return_value = (Mock(),)
                            
                            result = sampler.run(**inputs)
        
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        peak_increase = peak_memory - initial_memory
        
        # Verify memory efficiency
        assert isinstance(result, tuple)
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500 * 1024 * 1024
        # Peak memory shouldn't be excessive (less than 1GB over initial)
        assert peak_increase < 1024 * 1024 * 1024
    
    def test_workflow_error_handling(self, basic_i2v_workflow):
        """Test workflow error handling and recovery."""
        try:
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            pytest.skip("WanImageToVideoAdvancedSampler not available")
        
        sampler = WanImageToVideoAdvancedSampler()
        node_config = basic_i2v_workflow["nodes"]["1"]
        inputs = node_config["inputs"]
        
        # Test with missing model
        inputs["GGUF_High"] = "nonexistent_model.gguf"
        
        with patch.object(sampler, 'load_model', side_effect=ValueError("Model not found")):
            with pytest.raises(ValueError, match="Model not found"):
                sampler.run(**inputs)
        
        # Test with invalid parameters
        inputs["total_steps"] = -1  # Invalid step count
        
        # The node should handle invalid parameters gracefully
        # (specific behavior depends on implementation)
