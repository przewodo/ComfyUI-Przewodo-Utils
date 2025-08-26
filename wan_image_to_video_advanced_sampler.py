import os
import torch
import torch.nn.functional as F
import numpy as np
import nodes
import folder_paths
import node_helpers
import gc
import weakref
import sys
import time
import comfy.model_management as mm
import comfy
from comfy.model_management import InterruptProcessingException, interrupt_processing_mutex, interrupt_processing
from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_cfg import CFGZeroStar
from .core import *
from .cache_manager import CacheManager
from .wan_get_max_image_resolution_by_aspect_ratio import WanGetMaxImageResolutionByAspectRatio
from .wan_video_enhance_a_video import WanVideoEnhanceAVideo
from .temporal_frame_buffer import TemporalFrameBuffer

# Import external custom nodes using the centralized import function
imported_nodes = {}
teacache_imports = import_nodes(["teacache"], ["TeaCache"])

# Import nodes from different custom node packages
kjnodes_imports = import_nodes(["comfyui-kjnodes"], ["SkipLayerGuidanceWanVideo", "PathchSageAttentionKJ", "ImageResizeKJv2", "ModelPatchTorchSettings", "ColorMatch"])
gguf_imports = import_nodes(["ComfyUI-GGUF"], ["UnetLoaderGGUF"])
wanblockswap = import_nodes(["wanblockswap"], ["WanVideoBlockSwap"])
rife_tensorrt = import_nodes(["ComfyUI-Rife-Tensorrt"], ["RifeTensorrt"])

imported_nodes.update(teacache_imports)
imported_nodes.update(kjnodes_imports)
imported_nodes.update(gguf_imports)
imported_nodes.update(wanblockswap)
imported_nodes.update(rife_tensorrt)

TeaCache = imported_nodes.get("TeaCache")
SkipLayerGuidanceWanVideo = imported_nodes.get("SkipLayerGuidanceWanVideo")
UnetLoaderGGUF = imported_nodes.get("UnetLoaderGGUF")
SageAttention = imported_nodes.get("PathchSageAttentionKJ")
WanVideoBlockSwap = imported_nodes.get("WanVideoBlockSwap")
ImageResizeKJv2 = imported_nodes.get("ImageResizeKJv2")
ModelPatchTorchSettings = imported_nodes.get("ModelPatchTorchSettings")
ColorMatch = imported_nodes.get("ColorMatch")
RifeTensorrt = imported_nodes.get("RifeTensorrt")
class WanImageToVideoAdvancedSampler:
	# Class-level generic cache manager
	_cache_manager = CacheManager()
	
	@classmethod
	def INPUT_TYPES(s):
		
		clip_names = [NONE] + folder_paths.get_filename_list("text_encoders")
		gguf_model_names = [NONE] + folder_paths.get_filename_list("unet_gguf")
		diffusion_models_names = [NONE] + folder_paths.get_filename_list("diffusion_models")
		vae_names = [NONE] + folder_paths.get_filename_list("vae")
		clip_vision_models = [NONE] + folder_paths.get_filename_list("clip_vision")		
		lora_names = [NONE] + folder_paths.get_filename_list("loras")
		rife_engines = [NONE] + os.listdir(os.path.join(folder_paths.models_dir, "tensorrt", "rife"))

		return {
			"required": OrderedDict([
				# ═════════════════════════════════════════════════════════════════
				# 🔧 MODEL CONFIGURATION
				# ═════════════════════════════════════════════════════════════════
				("GGUF_High", (gguf_model_names, {"default": NONE, "advanced": True, "tooltip": "GGUF model for high CFG/noise sampling phase. Used for initial generation with strong prompt adherence and detail capture."})),
				("GGUF_Low", (gguf_model_names, {"default": NONE, "advanced": True, "tooltip": "GGUF model for low CFG/noise sampling phase. Used for refinement and smoothing with reduced prompt influence."})),
				("Diffusor_High", (diffusion_models_names, {"default": NONE, "advanced": True, "tooltip": "Diffusion model for high CFG/noise sampling phase. Used for initial generation with strong prompt adherence and detail capture."})),
				("Diffusor_Low", (diffusion_models_names, {"default": NONE, "advanced": True, "tooltip": "Diffusion model for low CFG/noise sampling phase. Used for refinement and smoothing with reduced prompt influence."})),
				("Diffusor_weight_dtype", (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True, "tooltip": "Weight data type for the diffusion model. FP8 options provide memory optimization with potential speed improvements."})),
				("Use_Model_Type", (MODEL_TYPE_LIST, {"default": MODEL_GGUF, "advanced": True, "tooltip": "Choose between GGUF or Diffusion model types. GGUF models are optimized for efficiency."})),
				
				# ═════════════════════════════════════════════════════════════════
				# 📝 TEXT & CLIP CONFIGURATION  
				# ═════════════════════════════════════════════════════════════════
				("clip", (clip_names, {"default": None, "advanced": True, "tooltip": "CLIP text encoder model to use for processing text prompts."})),
				("clip_type", (CLIP_TYPE_LIST, {"default": CLIP_WAN, "advanced": True, "tooltip": "Type of CLIP encoder. WAN is optimized for Wan2.1 models."})),
				("clip_device", (CLIP_DEVICE_LIST, {"default": CLIP_DEVICE_DEFAULT, "advanced": True, "tooltip": "Device to run CLIP text encoding on (CPU/GPU)."})),
				("vae", (vae_names, {"default": NONE, "advanced": True, "tooltip": "Variational Auto-Encoder for encoding/decoding between pixel and latent space."})),
				
				# ═════════════════════════════════════════════════════════════════
				# ⚡ TEACACHE OPTIMIZATION
				# ═════════════════════════════════════════════════════════════════
				("use_tea_cache", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable TeaCache for faster inference by caching diffusion model outputs."})),
				("tea_cache_model_type", (["flux", "ltxv", "lumina_2", "hunyuan_video", "hidream_i1_dev", "hidream_i1_full", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode", "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode"], {"default": "wan2.1_i2v_720p_14B", "tooltip": "Supported diffusion model."})),
				("tea_cache_rel_l1_thresh", ("FLOAT", {"default": 0.05, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})),
				("tea_cache_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."})),
				("tea_cache_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."})),
				("tea_cache_cache_device", (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside"})),
				
				# ═════════════════════════════════════════════════════════════════
				# 🎯 SKIP LAYER GUIDANCE
				# ═════════════════════════════════════════════════════════════════
				("use_SLG", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable Skip Layer Guidance for improved video generation quality."})),
				("SLG_blocks", ("STRING", {"default": "10", "multiline": False, "tooltip": "Number of blocks to process in each step. You can comma separate the blocks like 8, 9, 10"})),
				("SLG_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "The start percentage of sampling steps where Skip Layer Guidance will be applied."})),
				("SLG_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "The end percentage of sampling steps where Skip Layer Guidance will be applied."})),
				
				# ═════════════════════════════════════════════════════════════════
				# 🧠 ATTENTION & MODEL OPTIMIZATIONS
				# ═════════════════════════════════════════════════════════════════
				("use_sage_attention", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable SageAttention for optimized attention computation and memory efficiency."})),
				("sage_attention_mode", (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": "auto", "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."})),
				("use_shift", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable Model Shift for improved sampling stability and quality."})),
				("shift", ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Shift value for ModelSamplingSD3. Higher values can improve sampling stability."})),
				("use_block_swap", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable Block Swap optimization for memory efficiency during video generation."})),
				("block_swap", ("INT", {"default": 20, "min": 1, "max": 40, "step":1, "tooltip": "Block swap threshold value. Controls when to swap model blocks for memory optimization."})),
				
				# ═════════════════════════════════════════════════════════════════
				# 🎬 VIDEO GENERATION SETTINGS
				# ═════════════════════════════════════════════════════════════════
				("large_image_side", ("INT", {"default": 832, "min": 2.0, "max": 1280, "step":2, "advanced": True, "tooltip": "The larger side of the image to resize to. The smaller side will be resized proportionally."})),
				("image_generation_mode", (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE, "tooltip": "Mode for video generation."})),
				("wan_model_size", (WAN_MODELS, {"default": WAN_720P, "tooltip": "The model type to use for the diffusion process."})),
				("total_video_seconds", ("INT", {"default": 1, "min": 1, "max": 5, "step":1, "advanced": True, "tooltip": "The total duration of the video in seconds."})),
				("divide_video_in_chunks", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable dividing the video into chunks of 5 seconds for processing."})),
				
				# ═════════════════════════════════════════════════════════════════
				# ️ CLIP VISION SETTINGS
				# ═════════════════════════════════════════════════════════════════
				("clip_vision_model", (clip_vision_models, {"default": NONE, "advanced": True, "tooltip": "CLIP Vision model for processing input images. Required for image-to-video generation."})),
				("clip_vision_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Strength of CLIP vision influence on the generation. Higher values make the output more similar to input images."})),
				("start_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the start image. If disabled, the start image will be used as a static frame."})),
				("end_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the end image. If disabled, the end image will be used as a static frame."})),
				
				# ═════════════════════════════════════════════════════════════════
				# ⚙️ SAMPLING CONFIGURATION
				# ═════════════════════════════════════════════════════════════════
				("use_dual_samplers", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Use dual samplers for better quality. First sampler with high CFG, then low CFG for refinement. If disabled, single sampler uses the High CFG parameters."})),
				("high_cfg", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Classifier-Free Guidance scale for the first (high CFG) sampling pass. Higher values follow prompts more closely."})),
				("low_cfg", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Classifier-Free Guidance scale for the second (low CFG) sampling pass. Used for refinement."})),
				("high_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.001, "tooltip": "Denoising strength for the first sampling pass. 0.0 = full denoising, lower values preserve more of the input."})),
				("low_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.001, "tooltip": "Denoising strength for the second sampling pass. Used for refinement when dual samplers are enabled. 0.0 = full denoising, lower values preserve more of the input."})),
				("total_steps", ("INT", {"default": 15, "min": 1, "max": 90, "step":1, "advanced": True, "tooltip": "Total number of sampling steps. More steps generally improve quality but increase generation time."})),
				("total_steps_high_cfg", ("INT", {"default": 5, "min": 1, "max": 90, "step":1, "advanced": True, "tooltip": "Percentage of total_steps dedicated to the high CFG pass when using dual samplers. Remaining steps use low CFG for refinement."})),
				("fill_noise_latent", ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.001, "tooltip": f"How strong the denoise mask will be on the latent over the frames to be generated. 0.0: 100% denoise, 1.0: 0% denoise."})),
				("noise_seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducible generation. Same seed with same settings produces identical results."})),
				
				# ═════════════════════════════════════════════════════════════════
				# 🎨 CAUS VID ENHANCEMENT
				# ═════════════════════════════════════════════════════════════════
				("causvid_lora", (lora_names, {"default": NONE, "tooltip": "CausVid LoRA model for enhanced video generation capabilities."})),
				("high_cfg_causvid_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "tooltip": "LoRA strength for CausVid during the high CFG sampling pass."})),
				("low_cfg_causvid_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "tooltip": "LoRA strength for CausVid during the low CFG sampling pass."})),
				
				# ═════════════════════════════════════════════════════════════════
				# ✨ POST-PROCESSING OPTIONS
				# ═════════════════════════════════════════════════════════════════
				("video_enhance_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable video enhancement processing for improved output quality and temporal consistency."})),
				("use_cfg_zero_star", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CFG Zero Star optimization for improved sampling efficiency and quality."})),
				("apply_color_match", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Apply color matching between start image and generated output for consistent color grading."})),
				("apply_color_match_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Strength of color matching between start image and generated output. 0.0 = no color matching, 1.0 = full color matching."})),
				("clip_vision_astetics_strength", ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01, "advanced": True, "tooltip": "How strong to preserve the astetics of the original image"})),
				("frames_interpolation", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Make frame interpolation with Rife TensorRT. This will make the video smoother by generating additional frames between existing ones."})),
				("frames_engine", (rife_engines, {"default": NONE, "tooltip": "Rife TensorRT engine to use for frame interpolation."})),
				("frames_multiplier", ("INT", {"default": 2, "min": 2, "max": 100, "step":1, "advanced": True, "tooltip": "Multiplier for the number of frames generated during interpolation."})),
				("frames_clear_cache_after_n_frames", ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear the cache after processing this many frames. Helps manage memory usage during long video generation."})),
				("frames_use_cuda_graph", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Use CUDA Graphs for frame interpolation. Improves performance by reducing overhead during inference."})),
				("frames_overlap_chunks", ("INT", {"default": 16, "min": 8, "max": 32, "step": 4, "advanced": True, "tooltip": "Number of overlapping frames between video chunks."})),
				("frames_overlap_chunks_blend", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step":0.01, "advanced": True, "tooltip": "Controls how strongly overlapping frames between chunks are blended together. Higher values (0.7-1.0) create smoother transitions using research-based StreamingT2V blending techniques, while lower values (0.3-0.6) preserve more chunk independence. Affects spectral frequency domain consistency and temporal coherence."})),
				("frames_overlap_chunks_motion_weight", ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step":0.01, "advanced": True, "tooltip": "Weight for motion-predicted guidance using optical flow analysis between chunks. Higher values (0.4-0.6) apply stronger motion consistency from Go-with-the-Flow research, lower values (0.1-0.3) allow more motion variation. Controls how much previous chunk motion influences next chunk generation."})),
				("frames_overlap_chunks_motion_smooth", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Smooth transition blending motion"})),
				("frames_overlap_chunks_motion_decay", ("FLOAT", {"default": 0.92, "min": 0.1, "max": 1.0, "step":0.01, "advanced": True, "tooltip": "How fast the last chunk motion decays over the next chunk"})),
			]),
			"optional": OrderedDict([
				("lora_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of LoRAs to apply to the diffusion model. Each LoRA modifies the model's behavior."})),
				("prompt_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of prompts to apply to the diffusion model on each chunck generated. If there is less prompts than chunks, the last prompt will be used for the remaining chunks."})),
				("start_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "Start image for the video generation process."})),
				("end_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "End image for the video generation process."})),
			]),
		}

	RETURN_TYPES = ("IMAGE", "FLOAT",)
	RETURN_NAMES = ("IMAGE", "FPS",)

	FUNCTION = "run"

	CATEGORY = "PrzewodoUtils/Wan"

	def run(self, GGUF_High, GGUF_Low, Diffusor_High, Diffusor_Low, Diffusor_weight_dtype, Use_Model_Type, clip, clip_type, clip_device, vae, use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.05, tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda", use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=8.0, use_block_swap=True, block_swap=20, large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size=WAN_720P, total_video_seconds=1, divide_video_in_chunks=True, clip_vision_model=NONE, clip_vision_strength=1.0, use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0, lora_stack=None, start_image=None, start_image_clip_vision_enabled=True, end_image=None, end_image_clip_vision_enabled=True, video_enhance_enabled=True, use_cfg_zero_star=True, apply_color_match=True, apply_color_match_strength=1.0, causvid_lora=NONE, high_cfg_causvid_strength=1.0, low_cfg_causvid_strength=1.0, high_denoise=1.0, low_denoise=1.0, prompt_stack=None, fill_noise_latent=0.5, frames_interpolation=False, frames_engine=NONE, frames_multiplier=2, frames_clear_cache_after_n_frames=100, frames_use_cuda_graph=True, frames_overlap_chunks=8, frames_overlap_chunks_blend=0.3, frames_overlap_chunks_motion_weight=0.3, frames_overlap_chunks_motion_smooth=True, frames_overlap_chunks_motion_decay=0.92, clip_vision_astetics_strength = 2.0):
		self.default_fps = 16.0

		gc.collect()
		torch.cuda.empty_cache()

		# ================================================================
		# 📊 PARAMETER LOGGING - Lines 105-207
		# ================================================================
		output_to_terminal("\nPARAMETER VALUES:")
		
		# LoRA Stack (show names list)
		if lora_stack is not None:
			lora_names = [lora[0] for lora in lora_stack] if isinstance(lora_stack, list) else ['Unknown']
			output_to_terminal(f"lora_stack: {lora_names}")
		else:
			output_to_terminal("lora_stack: None")
		
		# Prompt Stack (show count)
		if prompt_stack is not None:
			prompt_count = len(prompt_stack) if isinstance(prompt_stack, list) else 1
			output_to_terminal(f"prompt_stack: {prompt_count} prompts")
		else:
			output_to_terminal("prompt_stack: None")
		
		# Start Image (show name/filename)
		if start_image is not None:
			output_to_terminal(f"start_image: Present ({start_image.shape if hasattr(start_image, 'shape') else 'tensor'})")
		else:
			output_to_terminal("start_image: None")
		
		# End Image (show name/filename)
		if end_image is not None:
			output_to_terminal(f"end_image: Present ({end_image.shape if hasattr(end_image, 'shape') else 'tensor'})")
		else:
			output_to_terminal("end_image: None")
		
		# Model Parameters
		output_to_terminal(f"GGUF_High: {GGUF_High}")
		output_to_terminal(f"GGUF_Low: {GGUF_Low}")
		output_to_terminal(f"Diffusor_High: {Diffusor_High}")
		output_to_terminal(f"Diffusor_Low: {Diffusor_Low}")
		output_to_terminal(f"Diffusor_weight_dtype: {Diffusor_weight_dtype}")
		output_to_terminal(f"Use_Model_Type: {Use_Model_Type}")
		
		# CLIP Configuration
		output_to_terminal(f"clip: {clip}")
		output_to_terminal(f"clip_type: {clip_type}")
		output_to_terminal(f"clip_device: {clip_device}")
		output_to_terminal(f"vae: {vae}")
		
		# TeaCache Optimization
		output_to_terminal(f"use_tea_cache: {use_tea_cache}")
		output_to_terminal(f"tea_cache_model_type: {tea_cache_model_type}")
		output_to_terminal(f"tea_cache_rel_l1_thresh: {tea_cache_rel_l1_thresh}")
		output_to_terminal(f"tea_cache_start_percent: {tea_cache_start_percent}")
		output_to_terminal(f"tea_cache_end_percent: {tea_cache_end_percent}")
		output_to_terminal(f"tea_cache_cache_device: {tea_cache_cache_device}")
		
		# Skip Layer Guidance
		output_to_terminal(f"use_SLG: {use_SLG}")
		output_to_terminal(f"SLG_blocks: {SLG_blocks}")
		output_to_terminal(f"SLG_start_percent: {SLG_start_percent}")
		output_to_terminal(f"SLG_end_percent: {SLG_end_percent}")
		
		# Attention & Model Optimizations
		output_to_terminal(f"use_sage_attention: {use_sage_attention}")
		output_to_terminal(f"sage_attention_mode: {sage_attention_mode}")
		output_to_terminal(f"use_shift: {use_shift}")
		output_to_terminal(f"shift: {shift}")
		output_to_terminal(f"use_block_swap: {use_block_swap}")
		output_to_terminal(f"block_swap: {block_swap}")
		
		# Video Generation Settings
		output_to_terminal(f"large_image_side: {large_image_side}")
		output_to_terminal(f"image_generation_mode: {image_generation_mode}")
		output_to_terminal(f"wan_model_size: {wan_model_size}")
		output_to_terminal(f"total_video_seconds: {total_video_seconds}")
		output_to_terminal(f"divide_video_in_chunks: {divide_video_in_chunks}")
		
		# CLIP Vision Settings
		output_to_terminal(f"clip_vision_model: {clip_vision_model}")
		output_to_terminal(f"clip_vision_strength: {clip_vision_strength}")
		output_to_terminal(f"start_image_clip_vision_enabled: {start_image_clip_vision_enabled}")
		output_to_terminal(f"end_image_clip_vision_enabled: {end_image_clip_vision_enabled}")
		
		# Sampling Configuration
		output_to_terminal(f"use_dual_samplers: {use_dual_samplers}")
		output_to_terminal(f"high_cfg: {high_cfg}")
		output_to_terminal(f"low_cfg: {low_cfg}")
		output_to_terminal(f"high_denoise: {high_denoise}")
		output_to_terminal(f"low_denoise: {low_denoise}")
		output_to_terminal(f"total_steps: {total_steps}")
		output_to_terminal(f"total_steps_high_cfg: {total_steps_high_cfg}")
		output_to_terminal(f"fill_noise_latent: {fill_noise_latent}")
		output_to_terminal(f"noise_seed: {noise_seed}")
		
		# CausVid Enhancement
		output_to_terminal(f"causvid_lora: {causvid_lora}")
		output_to_terminal(f"high_cfg_causvid_strength: {high_cfg_causvid_strength}")
		output_to_terminal(f"low_cfg_causvid_strength: {low_cfg_causvid_strength}")
		
		# Post-processing Options
		output_to_terminal(f"video_enhance_enabled: {video_enhance_enabled}")
		output_to_terminal(f"use_cfg_zero_star: {use_cfg_zero_star}")
		output_to_terminal(f"apply_color_match: {apply_color_match}")
		output_to_terminal(f"apply_color_match_strength: {apply_color_match_strength}")
		output_to_terminal(f"clip_vision_astetics_strength: {clip_vision_astetics_strength}")		
		output_to_terminal(f"frames_interpolation: {frames_interpolation}")
		output_to_terminal(f"frames_engine: {frames_engine}")
		output_to_terminal(f"frames_multiplier: {frames_multiplier}")
		output_to_terminal(f"frames_clear_cache_after_n_frames: {frames_clear_cache_after_n_frames}")
		output_to_terminal(f"frames_use_cuda_graph: {frames_use_cuda_graph}")
		output_to_terminal(f"frames_overlap_chunks: {frames_overlap_chunks}")
		output_to_terminal(f"frames_overlap_chunks_blend: {frames_overlap_chunks_blend}")
		output_to_terminal(f"frames_overlap_chunks_motion_weight: {frames_overlap_chunks_motion_weight}")
		output_to_terminal(f"frames_overlap_chunks_motion_smooth: {frames_overlap_chunks_motion_smooth}")
		output_to_terminal(f"frames_overlap_chunks_motion_decay: {frames_overlap_chunks_motion_decay}")
		output_to_terminal("================================================================\n")		

		model_high = self.load_model(GGUF_High, Diffusor_High, Use_Model_Type, Diffusor_weight_dtype)
		self.interrupt_execution(locals())

		model_low = None
		if (GGUF_Low != NONE or Diffusor_Low != NONE):
			model_low = self.load_model(GGUF_Low, Diffusor_Low, Use_Model_Type, Diffusor_weight_dtype)
			self.interrupt_execution(locals())
		else:
			model_low = model_high

		output_to_terminal_successful("Loading VAE...")
		vae, = nodes.VAELoader().load_vae(vae)
		self.interrupt_execution(locals())

		output_to_terminal_successful("Loading CLIP...")

		clip_cache_key = f"{clip}_{clip_type}_{clip_device}"

		clip_model = self._cache_manager.get_from_cache(clip_cache_key, 'cpu')
		
		if clip_model is not None:
			output_to_terminal_successful(f"Loaded CLIP from cache...")
		else:
			clip_model, = nodes.CLIPLoader().load_clip(clip, clip_type, clip_device)
			clip_set_last_layer = nodes.CLIPSetLastLayer()
			clip_model, = clip_set_last_layer.set_last_layer(clip_model, -1)  # Use all layers but truncate tokens
			
			# Store new model in cache (move to CPU to save VRAM)
			self._cache_manager.store_in_cache(clip_cache_key, clip_model, storage_device='cpu')
			output_to_terminal_successful(f"Loaded CLIP from disk...")
		
		self.interrupt_execution(locals())

		gc.collect()
		torch.cuda.empty_cache()

		tea_cache = None
		sage_attention = None
		slg_wanvideo = None
		model_shift = None
		wanBlockSwap = WanVideoBlockSwap()

		# Initialize TeaCache and SkipLayerGuidanceWanVideo
		tea_cache, slg_wanvideo = self.initialize_tea_cache_and_slg(use_tea_cache, use_SLG, SLG_blocks)
		self.interrupt_execution(locals())

		# Initialize SageAttention
		sage_attention = self.initialize_sage_attention(use_sage_attention, sage_attention_mode)
		self.interrupt_execution(locals())

		# Initialize Model Shift
		model_shift = self.initialize_model_shift(use_shift, shift)
		self.interrupt_execution(locals())

		output_image, fps, = self.postprocess(model_high, model_low, vae, clip_model, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, apply_color_match_strength, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, divide_video_in_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, frames_overlap_chunks, frames_overlap_chunks_blend, frames_overlap_chunks_motion_weight, frames_overlap_chunks_motion_smooth, frames_overlap_chunks_motion_decay, clip_vision_astetics_strength)
		self.interrupt_execution(locals())

		self.full_memory_cleanup(locals())

		return (output_image, fps,)

	def postprocess(self, model_high, model_low, vae, clip_model, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, apply_color_match_strength, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, divide_video_in_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, frames_overlap_chunks, frames_overlap_chunks_blend, frames_overlap_chunks_motion_weight, frames_overlap_chunks_motion_smooth, frames_overlap_chunks_motion_decay, clip_vision_astetics_strength):
		gc.collect()
		torch.cuda.empty_cache()

		frame_buffer = TemporalFrameBuffer(frames_overlap_chunks * 3, frames_overlap_chunks)
		remainder_video_seconds = total_video_seconds
		total_frames = (total_video_seconds * 16) + 1
		total_video_chunks = 1 if (divide_video_in_chunks == False and total_video_seconds > 5) else int(math.ceil(total_video_seconds / 5.0))
		k_sampler = nodes.KSamplerAdvanced()
		text_encode = nodes.CLIPTextEncode()
		wan_max_resolution = WanGetMaxImageResolutionByAspectRatio()
		CLIPVisionLoader = nodes.CLIPVisionLoader()
		CLIPVisionEncoder = nodes.CLIPVisionEncode()
		resizer = ImageResizeKJv2()
		image_width = large_image_side
		image_height = large_image_side
		lora_loader = nodes.LoraLoader()
		wanVideoEnhanceAVideo = WanVideoEnhanceAVideo()
		cfgZeroStar = CFGZeroStar()
		colorMatch = ColorMatch()
		clip_vision_start_image = None
		clip_vision_end_image = None
		positive = None
		negative = None
		positive_clip_high = None
		positive_clip_low = None
		negative_clip_high = None
		negative_clip_low = None
		clip_vision = None
		model_high_cfg = None
		model_low_cfg = None

		# Load CLIP Vision Model
		if (start_image_clip_vision_enabled == True or end_image_clip_vision_enabled == True):
			clip_vision = self.load_clip_vision_model(clip_vision_model, CLIPVisionLoader)
		else:
			output_to_terminal_error("CLIP vision not enabled...")
		self.interrupt_execution(locals())

		# Generate video chunks sequentially
		original_image_start = start_image.clone()
		original_image_end = end_image
		output_images = None
		input_latent = None
		input_mask = None
		input_clip_latent = None
		all_output_frames = []
		
		output_to_terminal_successful("Generation started...")

		for chunk_index in range(total_video_chunks):
			chunck_seconds = 5 if (remainder_video_seconds > 5) else remainder_video_seconds
			chunk_frames = (chunck_seconds * 16) + 1 + (frames_overlap_chunks if (remainder_video_seconds > 5) else 0)
			remainder_video_seconds = remainder_video_seconds - 5

			working_model_high = model_high.clone()
			# Only clone low model if dual samplers are used
			working_model_low = model_low.clone() if use_dual_samplers else None

			working_clip_high = clip_model.clone()
			# Only clone low clip if dual samplers are used
			working_clip_low = clip_model.clone() if use_dual_samplers else None

			# Immediately clear CUDA cache after cloning to prevent accumulation
			torch.cuda.empty_cache()

			# Apply Model Patch Torch Settings
			if (wan_model_size != WAN_2_2):
				working_model_high = self.apply_model_patch_torch_settings(working_model_high)
				if use_dual_samplers:
					working_model_low = self.apply_model_patch_torch_settings(working_model_low)
				self.interrupt_execution(locals())

			# Apply Sage Attention
			working_model_high = self.apply_sage_attention(sage_attention, working_model_high, sage_attention_mode)
			if use_dual_samplers:
				working_model_low = self.apply_sage_attention(sage_attention, working_model_low, sage_attention_mode)
			self.interrupt_execution(locals())

			# Apply TeaCache and SLG
			working_model_high = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model_high, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent)
			if use_dual_samplers:
				working_model_low = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model_low, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent)
			self.interrupt_execution(locals())

			# Apply Model Shift
			working_model_high = self.apply_model_shift(model_shift, use_shift, working_model_high, shift)
			if use_dual_samplers:
				working_model_low = self.apply_model_shift(model_shift, use_shift, working_model_low, shift)
			self.interrupt_execution(locals())

			# Apply Video Enhance
			working_model_high = self.apply_video_enhance(video_enhance_enabled, working_model_high, wanVideoEnhanceAVideo, chunk_frames)
			if use_dual_samplers:
				working_model_low = self.apply_video_enhance(video_enhance_enabled, working_model_low, wanVideoEnhanceAVideo, chunk_frames)
			self.interrupt_execution(locals())

			# Apply CFG Zero Star
			working_model_high = self.apply_cfg_zero_star(use_cfg_zero_star, working_model_high, cfgZeroStar)
			if use_dual_samplers:
				working_model_low = self.apply_cfg_zero_star(use_cfg_zero_star, working_model_low, cfgZeroStar)
			self.interrupt_execution(locals())

			# Apply Block Swap
			working_model_high = self.apply_block_swap(use_block_swap, working_model_high, wanBlockSwap, block_swap)
			if use_dual_samplers:
				working_model_low = self.apply_block_swap(use_block_swap, working_model_low, wanBlockSwap, block_swap)
			self.interrupt_execution(locals())

			# Process LoRA stack
			working_model_high, working_clip_high = self.process_lora_stack(lora_stack, working_model_high, working_clip_high)
			if use_dual_samplers:
				working_model_low, working_clip_low = self.process_lora_stack(lora_stack, working_model_low, working_clip_low)
			self.interrupt_execution(locals())

			# Clean up memory after model configuration
			gc.collect()
			torch.cuda.empty_cache()
			self.interrupt_execution(locals())

			# Light memory cleanup before each chunk - don't interfere with active models
			gc.collect()
			torch.cuda.empty_cache()
			if hasattr(torch.cuda, 'synchronize'):
				torch.cuda.synchronize()  # Ensure all operations complete before cleanup

			output_to_terminal(f"Generating video chunk {chunk_index + 1}/{total_video_chunks}...")
			self.interrupt_execution(locals())
			
			if (prompt_stack is not None):
				positive, negative, prompt_loras = self.get_current_prompt(prompt_stack, chunk_index)
				working_model_high, working_clip_high = self.process_lora_stack(prompt_loras, working_model_high, working_clip_high)
				if use_dual_samplers:
					working_model_low, working_clip_low = self.process_lora_stack(prompt_loras, working_model_low, working_clip_low)
				self.interrupt_execution(locals())

			if start_image is not None and (image_generation_mode == START_IMAGE or image_generation_mode == START_END_IMAGE or image_generation_mode == START_TO_END_TO_START_IMAGE):
				output_to_terminal_successful(f"Original start_image dimensions: {start_image.shape[2]}x{start_image.shape[1]}")

				start_image, image_width, image_height, clip_vision_start_image = self.process_image(original_image_start,
					start_image, start_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, 
					CLIPVisionEncoder, large_image_side, wan_model_size, start_image.shape[2], start_image.shape[1], "Start Image",
					chunk_index
				)
				self.interrupt_execution(locals())

			if end_image is not None and (image_generation_mode == END_TO_START_IMAGE):
				output_to_terminal_successful(f"Original end_image dimensions: {end_image.shape[2]}x{end_image.shape[1]}")

				end_image, image_width, image_height, clip_vision_end_image = self.process_image(original_image_end,
					end_image, end_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution,
					CLIPVisionEncoder, large_image_side, wan_model_size, end_image.shape[2], end_image.shape[1], "End Image",
					chunk_index
				)
				self.interrupt_execution(locals())

			model_high_cfg, model_low_cfg, working_clip_high, working_clip_low = self.apply_causvid_lora_processing(working_model_high, working_model_low, working_clip_high, working_clip_low, lora_loader, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers)
			self.interrupt_execution(locals())

			output_to_terminal_successful("Encoding Positive CLIP text...")
			positive_clip_high, = text_encode.encode(working_clip_high, positive)
			positive_clip_low, = text_encode.encode(working_clip_low, positive) if use_dual_samplers else (None,)
			self.interrupt_execution(locals())

			output_to_terminal_successful("Encoding Negative CLIP text...")
			negative_clip_high, = text_encode.encode(working_clip_high, negative)
			negative_clip_low, = text_encode.encode(working_clip_low, negative) if use_dual_samplers else (None,)
			self.interrupt_execution(locals())

			positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low = self.apply_clipvision_astetics(chunk_index, original_image_start, positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, start_image_clip_vision_enabled, clip_vision, CLIPVisionEncoder, clip_vision_astetics_strength)
			self.interrupt_execution(locals())

			output_to_terminal_successful("Wan Image to Video started...")

			'''
			Start creating the latent, mask, and image tensors
			'''
			#image = None
			keep_frames = 0

			if chunk_index == 0:
				keep_frames = chunk_frames - frames_overlap_chunks - 1
			elif chunk_index == total_video_chunks - 1:
				remaining_frames = total_frames - sum(len(chunk) for chunk in all_output_frames)
				keep_frames = remaining_frames - 1
			else:
				keep_frames = chunk_frames - frames_overlap_chunks - 1

			input_latent = {}
			input_latent["samples"] = torch.zeros([1, 16, ((chunk_frames - 1) // 4) + 1, image_height // 8, image_width // 8])

			# Create conditioning with buffer management
			input_clip_latent, input_mask = self.create_buffer_managed_conditioning(
				vae,
				image_width,
				image_height,
				chunk_frames,
				frame_buffer,
				frames_overlap_chunks + 1,
				use_motion_prediction=True,
				start_image=start_image if chunk_index == 0 else None
			)
			'''
			End creating the latent, mask, and image tensors
			'''
			self.interrupt_execution(locals())

			positive_clip_high = node_helpers.conditioning_set_values(positive_clip_high, {"concat_latent_image": input_clip_latent, "concat_mask": input_mask})
			negative_clip_high = node_helpers.conditioning_set_values(negative_clip_high, {"concat_latent_image": input_clip_latent, "concat_mask": input_mask})
			positive_clip_low = node_helpers.conditioning_set_values(positive_clip_low, {"concat_latent_image": input_clip_latent, "concat_mask": input_mask}) if use_dual_samplers == True else None
			negative_clip_low = node_helpers.conditioning_set_values(negative_clip_low, {"concat_latent_image": input_clip_latent, "concat_mask": input_mask}) if use_dual_samplers == True else None

			if (chunk_index > 0 and image_generation_mode == START_IMAGE and clip_vision_start_image is not None):
				output_to_terminal_successful("Running clipvision for start sequence")

				start_hidden = clip_vision_start_image.penultimate_hidden_states * clip_vision_strength

				clip_vision_output = comfy.clip_vision.Output()
				clip_vision_output.penultimate_hidden_states = start_hidden

				positive_clip_high = node_helpers.conditioning_set_values(positive_clip_high, {"clip_vision_output": clip_vision_output})
				negative_clip_high = node_helpers.conditioning_set_values(negative_clip_high, {"clip_vision_output": clip_vision_output})
				positive_clip_low = node_helpers.conditioning_set_values(positive_clip_low, {"clip_vision_output": clip_vision_output}) if use_dual_samplers == True else None
				negative_clip_low = node_helpers.conditioning_set_values(negative_clip_low, {"clip_vision_output": clip_vision_output}) if use_dual_samplers == True else None
			self.interrupt_execution(locals())
					
			output_to_terminal(f"Chunk {chunk_index + 1}: Frame Count: {chunk_frames}")
			output_to_terminal(f"Chunk {chunk_index + 1}: Latent Shape: {input_latent["samples"].shape}")
			output_to_terminal(f"Chunk {chunk_index + 1}: Mask Shape: {input_mask.shape}")
			output_to_terminal(f"Chunk {chunk_index + 1}: CLIP Latent Image Shape: {input_clip_latent.shape}")

			self.interrupt_execution(locals())

			# Light cleanup without interfering with active models
			gc.collect()
			torch.cuda.empty_cache()
			self.interrupt_execution(locals())
			
			if (use_dual_samplers):
				input_latent = self.apply_dual_sampler_processing(model_high_cfg, model_low_cfg, k_sampler, noise_seed, total_steps, high_cfg, low_cfg, positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, input_latent, total_steps_high_cfg, high_denoise, low_denoise, wan_model_size)
			else:
				input_latent = self.apply_single_sampler_processing(model_high_cfg, k_sampler, noise_seed, total_steps, high_cfg, positive_clip_high, negative_clip_high, input_latent, high_denoise)
			self.interrupt_execution(locals())

			chunk_images = vae.decode(input_latent["samples"])
			if len(chunk_images.shape) == 5:
				chunk_images = chunk_images.reshape(-1, chunk_images.shape[-3], chunk_images.shape[-2], chunk_images.shape[-1])

			chunk_images, _, _, _ = self.process_image(original_image_start,
				chunk_images, False, None, resizer, wan_max_resolution, 
				None, large_image_side, wan_model_size, image_width, image_height, "Chunk Images",
				chunk_index
			)

			chunk_images = self.apply_color_match_to_image(original_image_start, chunk_images, apply_color_match, colorMatch, apply_color_match_strength)

			# Process chunk based on position
			if chunk_index == 0 and chunk_index < (total_video_chunks - 1):
				# First chunk: keep all frames except overlap buffer
				output_frames = chunk_images[:keep_frames]
				buffer_frames = chunk_images  # Initialize buffer with full chunk
				
				# Initialize buffer
				frame_buffer.initialize_buffer(buffer_frames, input_latent["samples"])
			else:
				# Subsequent chunks: skip overlap, keep new content
				new_frames = chunk_images[frames_overlap_chunks+1:]
				output_frames = new_frames[:keep_frames] if (chunk_index < (total_video_chunks - 1)) else chunk_images[:-1]
				
				# Update buffer with new frames
				frame_buffer.add_frames(new_frames, input_latent["samples"][:, :, frames_overlap_chunks+1:])

			# Merge the sampled chunk results back into the full input_latent
			all_output_frames.append(output_frames)
			output_to_terminal(f"Chunk {chunk_index + 1}: Merged sampled results shape {output_frames.shape}")

			# Clean up after reference frame operations
			gc.collect()
			torch.cuda.empty_cache()

			output_to_terminal_successful(f"Video chunk {chunk_index + 1} generated successfully")

			# Comprehensive memory cleanup for windowed processing
			self._cleanup_chunk_memory(
				chunk_index=chunk_index,
				total_chunks=total_video_chunks,
				variables_to_clean={
					'working_model_high': working_model_high,
					'working_model_low': working_model_low,
					'working_clip_high': working_clip_high,
					'working_clip_low': working_clip_low,
					'positive_clip_high': positive_clip_high,
					'positive_clip_low': positive_clip_low,
					'negative_clip_high': negative_clip_high,
					'negative_clip_low': negative_clip_low
				}
			)

			# Nullify local variables to help garbage collection
			working_model_high = None
			working_model_low = None
			working_clip_high = None
			working_clip_low = None
			positive_clip_high = None
			positive_clip_low = None
			negative_clip_high = None
			negative_clip_low = None
			new_frames = None
			output_frames = None
			chunk_images = None
			gc.collect()
			torch.cuda.empty_cache()
			self.interrupt_execution(locals())

		output_images = torch.cat(all_output_frames, dim=0)

		# Clear the variable to free memory
		del all_output_frames

		# Final comprehensive memory cleanup for all chunks
		self._final_memory_cleanup([input_mask, input_latent, input_clip_latent])
		
		# Force cleanup of main models before final processing - be less aggressive
		# Only break circular references, don't force cleanup active models
		self.break_circular_references(locals())
		self.break_circular_references(locals())
		self.break_circular_references(locals())

		self.interrupt_execution(locals())

		frame_buffer.clear_buffer()
		del frame_buffer

		# Light cleanup only
		gc.collect()
		torch.cuda.empty_cache()

		if (output_images is None):
			output_to_terminal_error("Failed to generate output image")
			return (None, self.default_fps,)

		if (frames_interpolation and frames_engine != NONE):
			gc.collect()
			torch.cuda.empty_cache()
			
			output_to_terminal_successful(f"Starting interpolation with engine: {frames_engine}, multiplier: {frames_multiplier}, clear cache after {frames_clear_cache_after_n_frames} frames, use CUDA graph: {frames_use_cuda_graph}")
			interpolationEngine = RifeTensorrt()
			output_images, = interpolationEngine.vfi(output_images, frames_engine, frames_clear_cache_after_n_frames, frames_multiplier, frames_use_cuda_graph, False)
			self.default_fps = self.default_fps * float(frames_multiplier)
			output_images = output_images[0:total_frames - (frames_multiplier+1), :, :, :]
			output_to_terminal_successful(f"Output Image Shape: {output_images.shape}")
			output_to_terminal_successful("All video chunks generated successfully")
		else:
			output_images = output_images[0:total_frames - 1, :, :, :]
			output_to_terminal_successful("All video chunks generated successfully")
			output_to_terminal_successful(f"Output Image Shape: {output_images.shape}")

		output_to_terminal(f"Final Output Image Shape: {output_images.shape}")

		return (output_images, self.default_fps,)

	def apply_clipvision_astetics(self, chunk_index, reference_image, clip_positive_high, clip_negative_high, clip_positive_low, clip_negative_low, image_clip_vision_enabled, clip_vision, CLIPVisionEncoder, clip_vision_astetics_strength):
		if (chunk_index > 0) and (image_clip_vision_enabled) and (clip_vision is not None):
			output_to_terminal_successful(f"Encoding CLIP Vision for Astetics...")

			clip_vision_image, = CLIPVisionEncoder.encode(clip_vision, reference_image, "center")

			hidden = clip_vision_image.penultimate_hidden_states * clip_vision_astetics_strength

			clip_vision_output = comfy.clip_vision.Output()
			clip_vision_output.penultimate_hidden_states = hidden

			clip_positive_high = node_helpers.conditioning_set_values(clip_positive_high, {"clip_vision_output": clip_vision_output})
			clip_negative_high = node_helpers.conditioning_set_values(clip_negative_high, {"clip_vision_output": clip_vision_output})
			clip_positive_low = node_helpers.conditioning_set_values(clip_positive_low, {"clip_vision_output": clip_vision_output}) if clip_positive_low is not None else None
			clip_negative_low = node_helpers.conditioning_set_values(clip_negative_low, {"clip_vision_output": clip_vision_output}) if clip_negative_low is not None else None

		return clip_positive_high, clip_negative_high, clip_positive_low, clip_negative_low

	def get_current_prompt(self, prompt_stack, chunk_index):
		"""
		Get the current prompt based on the chunk index and prompt stack.
		"""
		if prompt_stack is not None:
			# Sort prompt_stack by chunk_index_start (third element in each array)
			prompt_stack = sorted(prompt_stack, key=lambda x: x[2])

			# Find the appropriate prompt where chunk_index >= chunk_index_start
			selected_prompt = None
			for prompt_entry in prompt_stack:
				if chunk_index >= prompt_entry[2]:  # chunk_index >= chunk_index_start
					selected_prompt = prompt_entry

			# If no prompt found, use the one with the biggest chunk_index_start
			if selected_prompt is None and len(prompt_stack) > 0:
				selected_prompt = max(prompt_stack, key=lambda x: x[2])

			# Use prompt stack if found, otherwise use defaults
			if selected_prompt is not None:
				positive = selected_prompt[0]  # First element is positive
				negative = selected_prompt[1]  # Second element is negative
				lora_stack = selected_prompt[3] # Fourth element is lora_stack
				output_to_terminal_successful(f"Using prompt from stack: index {selected_prompt[2]}")
				return positive, negative, lora_stack
			else:
				return "", "", None
		else:
			return "", "", None

	def load_model(self, GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype):
		"""
		Load the model based on the selected type.
		"""
		if Use_Model_Type == MODEL_GGUF:
			if UnetLoaderGGUF is not None and GGUF != NONE:
				# Use UnetLoaderGGUF to load the GGUF model
				output_to_terminal_successful(f"Loading GGUF model: {GGUF}")
				
				gguf_loader = UnetLoaderGGUF()
				model, = gguf_loader.load_unet(GGUF, None, None, True)
				output_to_terminal_successful(f"GGUF model {GGUF} loaded successfully using UnetLoaderGGUF")
				return model
			else:
				if UnetLoaderGGUF is None:
					output_to_terminal_error("UnetLoaderGGUF not available - cannot load GGUF model")
					raise ValueError("UnetLoaderGGUF not available - cannot load GGUF model")
				else:
					output_to_terminal_error("No GGUF model specified")
					raise ValueError("No GGUF model specified")
		elif Use_Model_Type == MODEL_DIFFUSION:
			if Diffusor != NONE:
				# Use ComfyUI's core UNETLoader to load the diffusion model
				output_to_terminal_successful(f"Loading Diffusion model: {Diffusor}")
				unet_loader = nodes.UNETLoader()
				model, = unet_loader.load_unet(unet_name=Diffusor, weight_dtype=Diffusor_weight_dtype)
				output_to_terminal_successful(f"Diffusion model {Diffusor} loaded successfully using UNETLoader")
				return model
			else:
				output_to_terminal_error("No Diffusion model specified")
				raise ValueError("No Diffusion model specified")
		else:
			output_to_terminal_error("Invalid model type selected. Please choose either GGUF or Diffusion model.")
			raise ValueError("Invalid model type selected. Please choose either GGUF or Diffusion model.")

	def initialize_tea_cache_and_slg(self, use_tea_cache, use_SLG, SLG_blocks):
		"""
		Initialize TeaCache and SkipLayerGuidanceWanVideo components.
		
		Args:
			use_tea_cache (bool): Whether to enable TeaCache
			use_SLG (bool): Whether to enable SkipLayerGuidanceWanVideo
			SLG_blocks (str): Block configuration for SkipLayerGuidanceWanVideo
			
		Returns:
			tuple: (tea_cache, slg_wanvideo) - Initialized components or None if disabled
		"""
		tea_cache = None
		slg_wanvideo = None
		
		# Create TeaCache node if available
		if TeaCache is not None and use_tea_cache:
			tea_cache = TeaCache()
			output_to_terminal_successful(f"TeaCache enabled")

			if (SkipLayerGuidanceWanVideo is not None) and use_SLG:
				slg_wanvideo = SkipLayerGuidanceWanVideo()
				output_to_terminal_successful(f"SkipLayerGuidanceWanVideo enabled with blocks: {SLG_blocks}")
			else:
				slg_wanvideo = None
				output_to_terminal_successful("SkipLayerGuidanceWanVideo disabled")
		else:
			tea_cache = None
			output_to_terminal_error("TeaCache disabled")
			
		return tea_cache, slg_wanvideo

	def initialize_sage_attention(self, use_sage_attention, sage_attention_mode):
		"""
		Initialize SageAttention component.
		
		Args:
			use_sage_attention (bool): Whether to enable SageAttention
			sage_attention_mode (str): Mode configuration for SageAttention
			
		Returns:
			SageAttention instance or None if disabled
		"""
		if (SageAttention is not None) and use_sage_attention:
			if sage_attention_mode == "disabled":
				sage_attention = None
				output_to_terminal_successful("SageAttention disabled")
			else:
				sage_attention = SageAttention()
				output_to_terminal_successful(f"SageAttention enabled with mode: {sage_attention_mode}")
		else:
			sage_attention = None
			output_to_terminal_error("SageAttention disabled")
			
		return sage_attention

	def initialize_model_shift(self, use_shift, shift):
		"""
		Initialize ModelSamplingSD3 (Model Shift) component.
		
		Args:
			use_shift (bool): Whether to enable Model Shift
			shift (float): Shift value configuration for ModelSamplingSD3
			
		Returns:
			ModelSamplingSD3 instance or None if disabled
		"""
		if (ModelSamplingSD3 is not None) and use_shift:
			model_shift = ModelSamplingSD3()
			model_shift.shift = shift
			output_to_terminal_successful(f"Model Shift enabled with shift: {shift}")
		else:
			model_shift = None
			output_to_terminal_error("Model Shift disabled")
			
		return model_shift

	def process_lora_stack(self, lora_stack, model, clip):
		"""
		Process and apply LoRA stack to the model and CLIP.
		
		Args:
			lora_stack: List of LoRA entries to apply
			model: The model to apply LoRAs to
			clip: The CLIP model to apply LoRAs to
			
		Returns:
			tuple: (updated_model, updated_clip) with LoRAs applied
		"""
		model_clone = model.clone()
		clip_clone = clip.clone()

		# Don't force cleanup the original models - just let them be garbage collected naturally
		# The original models may still be needed by ComfyUI's internal systems

		if lora_stack is not None and len(lora_stack) > 0:
			output_to_terminal_successful("Loading Lora Stack...")
			
			lora_loader = nodes.LoraLoader()
			lora_count = 0
			
			for lora_entry in lora_stack:
				lora_count += 1

				lora_name = lora_entry[0] if lora_entry[0] != NONE else None
				model_strength = lora_entry[1]
				clip_strength = lora_entry[2]

				if lora_name and lora_name != NONE:
					output_to_terminal_successful(f"Applying LoRA {lora_count}/{len(lora_stack)}: {lora_name} (model: {model_strength}, clip: {clip_strength})")
					model_clone, clip_clone = lora_loader.load_lora(model_clone, clip_clone, lora_name, model_strength, clip_strength)
				else:
					output_to_terminal_error(f"Skipping LoRA {lora_count}/{len(lora_stack)}: No valid LoRA name")
			
			output_to_terminal_successful(f"Successfully applied {len(lora_stack)} LoRAs to the model")
		
		return model_clone, clip_clone

	def apply_model_patch_torch_settings(self, working_model):
		"""
		Apply ModelPatchTorchSettings to the working model.
		
		Args:
			working_model: The model to apply torch settings patch to
			
		Returns:
			The updated working model with torch settings applied
		"""
		if (ModelPatchTorchSettings is not None):
			output_to_terminal_successful("Applying Model Patch Torch Settings...")
			working_model, = ModelPatchTorchSettings().patch(working_model, True)
		else:
			output_to_terminal_error("Model Patch Torch Settings not available, skipping...")
			
		return working_model

	def load_clip_vision_model(self, clip_vision_model, CLIPVisionLoader):
		"""
		Load CLIP Vision model if specified.
		
		Args:
			clip_vision_model: The CLIP vision model name to load
			CLIPVisionLoader: The CLIP vision loader instance
			
		Returns:
			The loaded CLIP vision model or None if not specified
		"""
		if (clip_vision_model != NONE):
			output_to_terminal_successful("Loading clip vision model...")
			clip_vision, = CLIPVisionLoader.load_clip(clip_vision_model)
			return clip_vision
		else:
			output_to_terminal_error("No clip vision model selected, skipping...")
			return None

	def process_image(self, reference_image, image, image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size, image_width, image_height, image_type, chunk_index):
		"""
		Process and resize an image, and encode CLIP vision if enabled.
		
		Args:
			image: The input image to process
			image_clip_vision_enabled (bool): Whether CLIP vision is enabled for this image
			clip_vision: The CLIP vision model
			resizer: The image resizer instance
			wan_max_resolution: The max resolution calculator
			CLIPVisionEncoder: The CLIP vision encoder
			large_image_side (int): The target size for the larger side
			wan_model_size (str): The model size configuration
			image_width (int): Current image width (used as default/starting value)
			image_height (int): Current image height (used as default/starting value)
			image_type (str): Type of image being processed ("Start Image" or "End Image")
			
		Returns:
			tuple: (processed_image, image_width, image_height, clip_vision_image)
		"""
		clip_vision_image = None
		
		if (image is not None):
			original_width = image.shape[2]
			original_height = image.shape[1]

			# Calculate new dimensions while maintaining aspect ratio
			new_width = large_image_side
			new_height = large_image_side
			
			# Calculate aspect ratio
			aspect_ratio = original_width / original_height
			
			# Determine new dimensions based on which side is larger
			if original_width >= original_height:
				# Width is larger or equal - scale based on width
				new_width = large_image_side
				new_height = int(large_image_side / aspect_ratio)
			else:
				# Height is larger - scale based on height
				new_height = large_image_side
				new_width = int(large_image_side * aspect_ratio)
			
			tmp_width, tmp_height, = wan_max_resolution.run(wan_model_size, image)

			wan_large_side = max(tmp_width, tmp_height)
			img_large_side = max(new_width, new_height)

			if (wan_large_side < img_large_side):
				image, image_width, image_height, _ = resizer.resize(image, tmp_width, tmp_height, "resize", "lanczos", 2, "0, 0, 0", "center", None, "cpu", None)
			else:
				image, image_width, image_height, _ = resizer.resize(image, new_width, new_height, "resize", "lanczos", 2, "0, 0, 0", "center", None, "cpu", None)

			output_to_terminal_successful(f"{image_type} final size: {image_width}x{image_height} | Image Shape: {image.shape}")

			if (chunk_index > 0) and (image_clip_vision_enabled) and (clip_vision is not None):
				output_to_terminal_successful(f"Encoding CLIP Vision for {image_type}...")
				clip_vision_image, = CLIPVisionEncoder.encode(clip_vision, reference_image, "center")
		else:
			output_to_terminal_error(f"{image_type} is not provided, skipping...")
			
		return image, image_width, image_height, clip_vision_image

	def apply_sage_attention(self, sage_attention, working_model, sage_attention_mode):
		"""
		Apply Sage Attention to the working model.
		
		Args:
			sage_attention: The SageAttention instance or None
			working_model: The model to apply SageAttention to
			sage_attention_mode (str): The mode configuration for SageAttention
			
		Returns:
			The updated working model with SageAttention applied (or unchanged if disabled)
		"""
		if (sage_attention is not None):
			output_to_terminal_successful("Applying Sage Attention...")
			working_model, = sage_attention.patch(working_model, sage_attention_mode)
		else:
			output_to_terminal_error("Sage Attention disabled, skipping...")
			
		return working_model

	def apply_tea_cache_and_slg(self, tea_cache, use_tea_cache, working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent):
		"""
		Apply TeaCache and Skip Layer Guidance to the working model.
		
		Args:
			tea_cache: The TeaCache instance or None
			use_tea_cache (bool): Whether TeaCache is enabled
			working_model: The model to apply TeaCache and SLG to
			tea_cache_model_type (str): Model type configuration for TeaCache
			tea_cache_rel_l1_thresh (float): Relative L1 threshold for TeaCache
			tea_cache_start_percent (float): Start percentage for TeaCache
			tea_cache_end_percent (float): End percentage for TeaCache
			tea_cache_cache_device (str): Device for TeaCache
			slg_wanvideo: The SkipLayerGuidanceWanVideo instance or None
			use_SLG (bool): Whether Skip Layer Guidance is enabled
			slg_wanvideo_blocks_string (str): Block configuration for SLG
			slg_wanvideo_start_percent (float): Start percentage for SLG
			slg_wanvideo_end_percent (float): End percentage for SLG
			
		Returns:
			The updated working model with TeaCache and SLG applied
		"""
		if (tea_cache is not None and use_tea_cache):
			output_to_terminal_successful("Applying TeaCache...")
			working_model, = tea_cache.apply_teacache(working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device)

			if (slg_wanvideo is not None and use_SLG) and (SLG_blocks is not None) and (SLG_blocks.strip() != ""):
				output_to_terminal_successful(f"Applying Skip Layer Guidance with blocks: {SLG_blocks}...")
				working_model, = slg_wanvideo.slg(working_model, SLG_start_percent, SLG_end_percent, SLG_blocks)
			else:
				output_to_terminal_error("SLG WanVideo not enabled or blocks not specified, skipping...")
		else:
			output_to_terminal_error("TeaCache not enabled, skipping...")
			
		return working_model

	def apply_model_shift(self, model_shift, use_shift, working_model, shift):
		"""
		Apply Model Shift to the working model.
		
		Args:
			model_shift: The ModelSamplingSD3 instance or None
			use_shift (bool): Whether Model Shift is enabled
			working_model: The model to apply Model Shift to
			shift (float): The shift value for ModelSamplingSD3
			
		Returns:
			The updated working model with Model Shift applied (or unchanged if disabled)
		"""
		if (model_shift is not None and use_shift):
			output_to_terminal_successful("Applying Model Shift...")
			working_model, = model_shift.patch(working_model, shift)
		else:
			output_to_terminal_error("Model Shift disabled, skipping...")
			
		return working_model

	def apply_block_swap(self, use_block_swap, working_model, wanBlockSwap, block_swap):
		"""
		Apply Block Swap to the working model.
		
		Args:
			use_block_swap (bool): Whether Block Swap is enabled
			working_model: The model to apply Block Swap to
			wanBlockSwap: The WanVideoBlockSwap instance
			block_swap (int): The block swap value
			
		Returns:
			The updated working model with Block Swap applied (or unchanged if disabled)
		"""
		if (use_block_swap):
			output_to_terminal_successful("Setting block swap...")
			working_model, = wanBlockSwap.set_callback(working_model, block_swap, True, True, True)
		else:
			output_to_terminal_error("Block swap disabled, skipping...")
			
		return working_model

	def apply_video_enhance(self, video_enhance_enabled, working_model, wanVideoEnhanceAVideo, total_frames):
		"""
		Apply Wan Video Enhance to the working model.
		
		Args:
			video_enhance_enabled (bool): Whether Video Enhance is enabled
			working_model: The model to apply Video Enhance to
			wanVideoEnhanceAVideo: The WanVideoEnhanceAVideo instance
			total_frames (int): The total number of frames
			
		Returns:
			The updated working model with Video Enhance applied (or unchanged if disabled)
		"""
		if (video_enhance_enabled):
			output_to_terminal_successful("Applying Wan Video Enhance...")
			working_model, = wanVideoEnhanceAVideo.enhance(working_model, 2, total_frames, 0)
		else:
			output_to_terminal_error("Wan Video Enhance is disabled, skipping...")
			
		return working_model

	def apply_cfg_zero_star(self, use_cfg_zero_star, working_model, cfgZeroStar):
		"""
		Apply CFG Zero Star patch to the model if enabled.
		
		Args:
			use_cfg_zero_star (bool): Whether to apply CFG Zero Star patch
			working_model: The model to apply the patch to
			cfgZeroStar: The CFG Zero Star instance
			
		Returns:
			tuple: Modified model or original model if disabled
		"""
		if (use_cfg_zero_star):
			output_to_terminal_successful("Applying CFG Zero Star Patch...")
			working_model, = cfgZeroStar.patch(working_model)
		else:
			output_to_terminal_error("CFG Zero Star disables. skipping...")
			
		return working_model

	def apply_dual_sampler_processing(self, model_high_cfg, model_low_cfg, k_sampler, noise_seed, total_steps, high_cfg, low_cfg, positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, in_latent, total_steps_high_cfg, high_denoise, low_denoise, wan_model_size):
		stop_steps = int(total_steps_high_cfg / 100 * total_steps)
		high_start = 0
		high_stop = stop_steps
		low_start = stop_steps
		low_stop = total_steps

		if (wan_model_size == WAN_2_2):
			stop_steps = total_steps // 2
			total_steps = stop_steps
			high_start = 0
			high_stop = stop_steps
			low_start = 0
			low_stop = stop_steps

		gc.collect()
		torch.cuda.empty_cache()
		
		output_to_terminal_successful("High CFG KSampler started...")
		out_latent, = k_sampler.sample(model_high_cfg, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", positive_clip_high, negative_clip_high, in_latent, high_start, high_stop, "enabled", high_denoise)
		self.interrupt_execution(locals())
		
		# Light cleanup between samplers
		gc.collect()
		torch.cuda.empty_cache()

		output_to_terminal_successful("Low CFG KSampler started...")
		out_latent, = k_sampler.sample(model_low_cfg, "disable", noise_seed, total_steps, low_cfg, "lcm", "simple", positive_clip_low, negative_clip_low, out_latent, low_start, low_stop, "enabled", low_denoise)
		self.interrupt_execution(locals())
		
		# Light cleanup after sampling
		gc.collect()
		torch.cuda.empty_cache()

		return out_latent

	def apply_single_sampler_processing(self, working_model, k_sampler, noise_seed, total_steps, high_cfg, positive_clip_high, negative_clip_high, in_latent, high_denoise):

		output_to_terminal_successful("KSampler started...")
		out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", positive_clip_high, negative_clip_high, in_latent, 0, total_steps, "enabled", high_denoise)

		return out_latent

	def apply_causvid_lora_processing(self, model_high, model_low, clip_high, clip_low, lora_loader, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers):
		"""
		Apply CausVid LoRA processing and return prepared models for sampling.
		
		Args:
			working_model: The base model to apply LoRA to
			clip: The CLIP model
			lora_loader: The LoRA loader instance  
			causvid_lora: The CausVid LoRA to apply
			high_cfg_causvid_strength (float): LoRA strength for high CFG model
			low_cfg_causvid_strength (float): LoRA strength for low CFG model  
			use_dual_samplers (bool): Whether dual samplers are being used
			
		Returns:
			tuple: (model_high_cfg, model_low_cfg, updated_clip) - Prepared models with LoRAs applied
		"""
		model_high_cfg = model_high.clone()
		# Only clone low model and clip if dual samplers are used
		model_low_cfg = model_low.clone() if use_dual_samplers and model_low is not None else None
		updated_clip_high = clip_high.clone()
		updated_clip_low = clip_low.clone() if use_dual_samplers and clip_low is not None else None

		# Don't force cleanup original models - they may still be needed
		# Just let Python's garbage collector handle them naturally

		if use_dual_samplers:
			# Apply CausVid LoRA for High CFG model
			if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
				output_to_terminal_successful(f"Applying CausVid LoRA for High CFG with strength: {high_cfg_causvid_strength}")
				model_high_cfg, updated_clip_high, = lora_loader.load_lora(model_high_cfg, updated_clip_high, causvid_lora, high_cfg_causvid_strength, 1.0)

			# Apply CausVid LoRA for Low CFG model
			if (causvid_lora != NONE and low_cfg_causvid_strength > 0.0):
				output_to_terminal_successful(f"Applying CausVid LoRA for Low CFG with strength: {low_cfg_causvid_strength}")
				model_low_cfg, updated_clip_low, = lora_loader.load_lora(model_low_cfg, updated_clip_low, causvid_lora, low_cfg_causvid_strength, 1.0)
		else:
			# Single sampler - only apply to high CFG model
			if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
				output_to_terminal_successful(f"Applying CausVid LoRA with strength: {high_cfg_causvid_strength}")
				model_high_cfg, updated_clip_high, = lora_loader.load_lora(model_high_cfg, updated_clip_high, causvid_lora, high_cfg_causvid_strength, 1.0)

		return model_high_cfg, model_low_cfg, updated_clip_high, updated_clip_low

	def apply_color_match_to_image(self, original_image, image, apply_color_match, colorMatch, strength=1.0):
		"""
		Apply color matching between original_image and images if enabled.
		
		Args:
			original_image: Reference image for color matching (or None)
			output_image: Target image to apply color correction to
			apply_color_match: Boolean flag to enable/disable color matching
			colorMatch: Color matching utility object
			
		Returns:
			output_image: Processed image with or without color matching applied
		"""
		if (image is not None and apply_color_match):
			output_to_terminal_successful("Applying color match to images...")
			image, = colorMatch.colormatch(original_image, image, "hm-mvgd-hm", strength)

		return image
	
	def enhanced_memory_cleanup(self, local_scope):
		"""
		Enhanced memory cleanup using weakref and sys for comprehensive memory management.
		
		This method implements advanced memory management techniques using Python's
		weakref and sys modules to break circular references and force garbage collection.
		
		Args:
			local_scope (dict): Local variables scope to clean up
			chunk_index (int): Current chunk index for logging
		"""
		try:
			# Create weak references to track large objects before deletion
			weak_refs = []
			memory_intensive_objects = []
			
			# Identify memory-intensive objects (models, tensors, images, latents)
			for name, obj in list(local_scope.items()):
				if obj is not None and not name.startswith('_'):
					# Check for PyTorch tensors, ComfyUI models, or large objects
					is_memory_intensive = (
						hasattr(obj, 'size') and callable(getattr(obj, 'size')) or  # PyTorch tensors
						hasattr(obj, 'model') or  # ComfyUI model wrappers
						hasattr(obj, 'patcher') or  # Model patchers
						'model' in name.lower() or
						'latent' in name.lower() or
						'image' in name.lower() or
						'clip' in name.lower() or
						'tensor' in str(type(obj)).lower()
					)
					
					if is_memory_intensive:
						memory_intensive_objects.append(name)
						try:
							# Create weak reference to track cleanup
							weak_refs.append((name, weakref.ref(obj)))
						except TypeError:
							# Some objects can't have weak references
							pass
			
			# Force deletion of memory-intensive objects - but be more conservative
			deleted_count = 0
			for obj_name in memory_intensive_objects:
				if obj_name in local_scope:
					try:
						# Only delete objects that are clearly temporary/cached
						if any(temp_word in obj_name.lower() for temp_word in ['temp', 'cache', '_cache', 'working_', 'updated_']):
							del local_scope[obj_name]
							deleted_count += 1
					except Exception as e:
						# Continue cleanup even if some deletions fail
						pass
			
			# Force Python reference counting cleanup
			# This is where sys module becomes useful
			if hasattr(sys, 'getrefcount'):
				# Check reference counts before cleanup
				high_ref_objects = []
				for name, weak_ref in weak_refs:
					if weak_ref() is not None:
						ref_count = sys.getrefcount(weak_ref())
						if ref_count > 3:  # More than expected references
							high_ref_objects.append((name, ref_count))
				
				if high_ref_objects:
					output_to_terminal_successful(f"High reference count objects detected: {len(high_ref_objects)} objects")
			
			# Multiple garbage collection passes to break circular references
			# This is crucial for complex object graphs in ML models
			total_collected = 0
			for gc_pass in range(5):  # Multiple passes for thorough cleanup
				collected = gc.collect()
				total_collected += collected
				if collected == 0:
					break  # No more objects to collect
				
				# Small delay to allow Python's memory manager to work
				if gc_pass < 4:  # Don't delay on last pass
					time.sleep(0.01)
			
			# Check cleanup effectiveness using weak references
			alive_objects = []
			for name, weak_ref in weak_refs:
				if weak_ref() is not None:
					alive_objects.append(name)
			
			# Report cleanup results
			if alive_objects:
				output_to_terminal_successful(f"{len(alive_objects)} objects still referenced after cleanup")
			
			output_to_terminal_successful(f"Cleaned {deleted_count} objects, collected {total_collected} garbage objects")
			
			# Final CUDA memory management
			if hasattr(torch.cuda, 'empty_cache'):
				torch.cuda.empty_cache()
			if hasattr(torch.cuda, 'synchronize'):
				torch.cuda.synchronize()
			
			# Report GPU memory status if available
			if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
				allocated_gb = torch.cuda.memory_allocated() / (1024**3)
				reserved_gb = torch.cuda.memory_reserved() / (1024**3)
				output_to_terminal_successful(f"GPU memory - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB")

		except Exception as e:
			output_to_terminal_error(f"Error in enhanced memory cleanup: {str(e)}")
			# Fallback to basic cleanup
			try:
				gc.collect()
				torch.cuda.empty_cache()
			except:
				pass	

	def cleanup_local_refs(self, local_vars_dict):
		"""
		Generic function to clean up local variable references for memory optimization.
		
		Args:
			local_vars_dict: Dictionary of local variables (typically from locals())
		
		Returns:
			int: Number of objects that are still referenced after cleanup (potential memory leaks)
		"""
		cleanup_refs = []
		
		# Identify all variables containing models, tensors, or large objects
		local_vars = list(local_vars_dict.items())
		for name, obj in local_vars:
			if (hasattr(obj, '__dict__') and any(keyword in name.lower() for keyword in ['model', 'clip', 'latent', 'image']) 
				and not name.startswith('_') and obj is not None):
				try:
					# Create weak reference to track the object
					cleanup_refs.append(weakref.ref(obj))
				except TypeError:
					# Some objects can't have weak references, skip them
					pass
		
		# Force deletion of specific variables to break circular references
		vars_to_delete = [name for name, obj in local_vars 
						if any(keyword in name.lower() for keyword in ['model', 'clip', 'latent', 'cache'])
						and not name.startswith('_')]
		
		for var_name in vars_to_delete:
			if var_name in local_vars_dict:
				try:
					del local_vars_dict[var_name]
				except:
					pass  # Continue if deletion fails
		
		# Clear local variables list to free memory
		del local_vars, vars_to_delete
		
		# Force Python's garbage collector to run multiple times
		# This helps break any remaining circular references
		for _ in range(3):
			collected = gc.collect()
			if collected == 0:
				break  # No more objects to collect
		
		# Check if weak references are still alive (indicates memory leak)
		alive_refs = sum(1 for ref in cleanup_refs if ref() is not None)
		if alive_refs > 0:
			output_to_terminal_successful(f"Warning: {alive_refs} objects still referenced after cleanup")
		
		# Clear the cleanup references
		cleanup_refs.clear()
		del cleanup_refs
		
		# Get current memory usage for monitoring
		if hasattr(torch.cuda, 'memory_allocated'):
			current_mem = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
			output_to_terminal_successful(f"GPU memory after cleanup: {current_mem:.2f} GB")
		
		return alive_refs

	def force_model_cleanup(self, *models):
		"""
		Safe cleanup of model references to prevent memory leaks without corrupting models.
		
		This method safely cleans up model objects without deleting critical
		attributes that are needed for model inference.
		
		Args:
			*models: Variable number of model objects to clean up
		"""
		for model in models:
			if model is not None:
				try:
					# DON'T delete critical model attributes - just move to CPU if possible
					if hasattr(model, 'cpu') and hasattr(model, 'device'):
						try:
							if hasattr(model.device, 'type') and model.device.type == 'cuda':
								# Move model to CPU to free GPU memory
								model.cpu()
						except:
							pass
					
					# Only clean up non-essential cached data
					if hasattr(model, '__dict__'):
						# Only remove safe-to-remove attributes
						safe_to_remove = ['cache', 'temp_cache', '_cache', 'temp_data']
						for attr_name in list(model.__dict__.keys()):
							if any(safe_attr in attr_name.lower() for safe_attr in safe_to_remove):
								try:
									delattr(model, attr_name)
								except:
									pass
					
					# Clean up model patcher patches but keep the patcher structure
					if hasattr(model, 'patcher'):
						patcher = model.patcher
						if hasattr(patcher, 'patches'):
							# Clear patches but don't delete the patches dict
							patcher.patches.clear()
						if hasattr(patcher, 'backup') and hasattr(patcher.backup, 'clear'):
							# Clear backup but don't delete the backup dict
							patcher.backup.clear()
					
				except Exception as e:
					# Continue cleanup even if individual model cleanup fails
					output_to_terminal_successful(f"Warning: Failed to clean up model: {str(e)}")
					continue
		
		# Standard garbage collection
		gc.collect()
		
		# Clear CUDA cache
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def break_circular_references(self, local_scope):
		"""
		Safely break circular references without corrupting active models.
		
		Args:
			local_scope (dict): Local variables scope to analyze for circular references
		"""
		# Identify potential circular reference chains - but be more conservative
		circular_ref_patterns = [
			'working_model_high', 'working_model_low', 
			'working_clip_high', 'working_clip_low',
		]
		
		# Only set completed/temporary models to None, don't cleanup active models
		for var_name in circular_ref_patterns:
			if var_name in local_scope and local_scope[var_name] is not None:
				# Just set to None to break reference - don't force cleanup
				local_scope[var_name] = None
		
		# Light garbage collection
		gc.collect()

	def _cleanup_chunk_memory(self, chunk_index, total_chunks, variables_to_clean):
		"""
		Comprehensive memory cleanup for each chunk with windowed processing optimization.
		
		Args:
			chunk_index: Current chunk index
			total_chunks: Total number of chunks
			variables_to_clean: Dictionary of variable names and their objects to clean
		"""
		try:
			# Log memory before cleanup
			if torch.cuda.is_available():
				mem_before = torch.cuda.memory_allocated() / (1024**3)
				
			# Clean up tensors and models efficiently
			tensor_objects = []
			model_objects = []
			
			for var_name, obj in variables_to_clean.items():
				if obj is not None:
					# Categorize objects for specialized cleanup
					if 'model' in var_name.lower() or 'clip' in var_name.lower():
						model_objects.append((var_name, obj))
					elif (hasattr(obj, 'shape') or isinstance(obj, dict) and 'samples' in obj):
						tensor_objects.append((var_name, obj))
			
			# Clean up tensor objects first (usually larger memory footprint)
			for var_name, tensor_obj in tensor_objects:
				try:
					if isinstance(tensor_obj, dict) and 'samples' in tensor_obj:
						# Handle latent dictionaries
						if hasattr(tensor_obj['samples'], 'cpu'):
							tensor_obj['samples'] = tensor_obj['samples'].cpu()
						del tensor_obj['samples']
						tensor_obj.clear()
					elif hasattr(tensor_obj, 'cpu'):
						# Move tensor to CPU before deletion
						tensor_obj.cpu()
					del tensor_obj
				except Exception as e:
					output_to_terminal_error(f"Error cleaning tensor {var_name}: {str(e)}")
					continue
			
			# Clean up model objects with special handling
			for var_name, model_obj in model_objects:
				try:
					# Use existing safe model cleanup
					self.force_model_cleanup(model_obj)
				except Exception as e:
					output_to_terminal_error(f"Error cleaning model {var_name}: {str(e)}")
					continue
			
			# Progressive garbage collection based on chunk position
			gc_iterations = 3 if chunk_index < total_chunks - 1 else 5  # More thorough cleanup on last chunk
			total_collected = 0
			
			for i in range(gc_iterations):
				collected = gc.collect()
				total_collected += collected
				if collected == 0:
					break
					
				# Small delay for memory manager
				if i < gc_iterations - 1:
					time.sleep(0.005)
			
			# CUDA memory cleanup
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if hasattr(torch.cuda, 'synchronize'):
					torch.cuda.synchronize()
				
				# Log cleanup effectiveness
				mem_after = torch.cuda.memory_allocated() / (1024**3)
				mem_freed = mem_before - mem_after
				output_to_terminal_successful(f"Chunk {chunk_index + 1} cleanup: freed {mem_freed:.2f}GB, collected {total_collected} objects")
				
		except Exception as e:
			output_to_terminal_error(f"Error in chunk memory cleanup: {str(e)}")
			# Fallback cleanup
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def _final_memory_cleanup(self, large_tensors):
		"""
		Comprehensive final memory cleanup after all chunks are processed.
		
		Args:
			large_tensors: List of large tensor objects to clean up
		"""
		try:
			output_to_terminal_successful("Starting final comprehensive memory cleanup...")
			
			# Log initial memory state
			if torch.cuda.is_available():
				initial_mem = torch.cuda.memory_allocated() / (1024**3)
				output_to_terminal_successful(f"Memory before final cleanup: {initial_mem:.2f}GB")
			
			# Clean up large tensors first
			for i, tensor in enumerate(large_tensors):
				if tensor is not None:
					try:
						if hasattr(tensor, 'cpu'):
							tensor.cpu()
						if isinstance(tensor, dict):
							for key in list(tensor.keys()):
								if hasattr(tensor[key], 'cpu'):
									tensor[key].cpu()
								del tensor[key]
							tensor.clear()
						del tensor
					except Exception as e:
						output_to_terminal_error(f"Error cleaning large tensor {i}: {str(e)}")
						continue
			
			# Clear the tensor list
			large_tensors.clear()
			
			# Aggressive garbage collection for final cleanup
			total_collected = 0
			for gc_round in range(7):  # More rounds for final cleanup
				collected = gc.collect()
				total_collected += collected
				if collected == 0:
					break
				time.sleep(0.01)  # Allow memory manager to work
			
			# Final CUDA cleanup
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if hasattr(torch.cuda, 'synchronize'):
					torch.cuda.synchronize()
				
				# Reset memory stats if available
				if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
					torch.cuda.reset_accumulated_memory_stats()
				
				final_mem = torch.cuda.memory_allocated() / (1024**3)
				freed_mem = initial_mem - final_mem
				output_to_terminal_successful(f"Final cleanup complete: freed {freed_mem:.2f}GB, collected {total_collected} objects")
				output_to_terminal_successful(f"Final memory usage: {final_mem:.2f}GB")
		
		except Exception as e:
			output_to_terminal_error(f"Error in final memory cleanup: {str(e)}")
			# Fallback cleanup
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			gc.collect()
	
	def comprehensive_circular_reference_cleanup(self, local_vars=None):
		"""
		Comprehensive function to detect and break all potential circular references.
		
		This function systematically identifies variables that could create circular references
		in ML/AI contexts, particularly models, tensors, caches, and complex objects.
		
		Args:
			local_vars: Dictionary of local variables to analyze and clean
		"""
		try:
			if local_vars is None:
				local_vars = {}
			
			cleanup_stats = {
				'models_cleaned': 0,
				'tensors_cleaned': 0,
				'caches_cleaned': 0,
				'objects_cleaned': 0,
				'circular_refs_broken': 0,
				'memory_freed_mb': 0
			}
			
			# Get initial memory state
			initial_memory = 0
			if torch.cuda.is_available():
				initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
			
			output_to_terminal_successful("Starting comprehensive circular reference cleanup...")
			
			# 1. DETECT AND CLEAN MODEL OBJECTS (highest priority)
			model_patterns = [
				'model', 'clip', 'vae', 'diffuser', 'unet', 'encoder', 'decoder',
				'sampler', 'scheduler', 'pipeline', 'processor', 'loader'
			]
			
			model_vars = []
			for var_name, var_obj in list(local_vars.items()):
				if var_obj is not None and any(pattern in var_name.lower() for pattern in model_patterns):
					model_vars.append(var_name)
			
			# Clean model objects with circular reference detection
			for var_name in model_vars:
				try:
					var_obj = local_vars.get(var_name)
					if var_obj is not None:
						# Check for common circular reference patterns in ML models
						if hasattr(var_obj, '__dict__'):
							obj_dict = var_obj.__dict__
							circular_refs = []
							
							# Detect self-references
							for attr_name, attr_value in obj_dict.items():
								if attr_value is var_obj:
									circular_refs.append(f"{var_name}.{attr_name}")
								elif hasattr(attr_value, '__dict__') and var_obj in attr_value.__dict__.values():
									circular_refs.append(f"{var_name}.{attr_name}")
							
							# Break detected circular references
							for ref_path in circular_refs:
								try:
									attr_path = ref_path.split('.')[1:]
									obj_ref = var_obj
									for attr in attr_path[:-1]:
										obj_ref = getattr(obj_ref, attr)
									setattr(obj_ref, attr_path[-1], None)
									cleanup_stats['circular_refs_broken'] += 1
								except:
									pass
						
						# Apply safe model cleanup
						self.force_model_cleanup(var_obj)
						local_vars[var_name] = None
						cleanup_stats['models_cleaned'] += 1
						
				except Exception as e:
					output_to_terminal_error(f"Error cleaning model {var_name}: {str(e)}")
					continue
			
			# 2. DETECT AND CLEAN TENSOR OBJECTS
			tensor_patterns = [
				'tensor', 'latent', 'image', 'frame', 'features', 'embedding',
				'hidden', 'output', 'input', 'batch', 'sample'
			]
			
			tensor_vars = []
			for var_name, var_obj in list(local_vars.items()):
				if var_obj is not None:
					# Check if it's a tensor or contains tensors
					is_tensor = (
						hasattr(var_obj, 'shape') and hasattr(var_obj, 'dtype') or
						isinstance(var_obj, dict) and any(hasattr(v, 'shape') for v in var_obj.values() if v is not None) or
						any(pattern in var_name.lower() for pattern in tensor_patterns)
					)
					if is_tensor:
						tensor_vars.append(var_name)
			
			# Clean tensor objects
			for var_name in tensor_vars:
				try:
					var_obj = local_vars.get(var_name)
					if var_obj is not None:
						# Handle dictionary of tensors
						if isinstance(var_obj, dict):
							for key, value in list(var_obj.items()):
								if hasattr(value, 'cpu'):
									value.cpu()
								var_obj[key] = None
							var_obj.clear()
						# Handle direct tensors
						elif hasattr(var_obj, 'cpu'):
							var_obj.cpu()
						
						local_vars[var_name] = None
						cleanup_stats['tensors_cleaned'] += 1
						
				except Exception as e:
					output_to_terminal_error(f"Error cleaning tensor {var_name}: {str(e)}")
					continue
			
			# 3. DETECT AND CLEAN CACHE OBJECTS
			cache_patterns = [
				'cache', 'buffer', 'memory', 'bank', 'store', 'registry',
				'pool', 'queue', 'stack', 'history', 'temporal'
			]
			
			cache_vars = []
			for var_name, var_obj in list(local_vars.items()):
				if var_obj is not None and any(pattern in var_name.lower() for pattern in cache_patterns):
					cache_vars.append(var_name)
			
			# Clean cache objects
			for var_name in cache_vars:
				try:
					var_obj = local_vars.get(var_name)
					if var_obj is not None:
						# Handle different cache types
						if isinstance(var_obj, dict):
							var_obj.clear()
						elif isinstance(var_obj, list):
							var_obj.clear()
						elif hasattr(var_obj, 'clear'):
							var_obj.clear()
						elif hasattr(var_obj, 'cpu'):
							var_obj.cpu()
						
						local_vars[var_name] = None
						cleanup_stats['caches_cleaned'] += 1
						
				except Exception as e:
					output_to_terminal_error(f"Error cleaning cache {var_name}: {str(e)}")
					continue
			
			# 4. DETECT AND CLEAN COMPLEX OBJECTS WITH POTENTIAL CIRCULAR REFS
			complex_object_vars = []
			for var_name, var_obj in list(local_vars.items()):
				if var_obj is not None and not var_name.startswith('_'):
					# Check for complex objects that might have circular references
					has_dict = hasattr(var_obj, '__dict__')
					has_complex_attrs = has_dict and len(getattr(var_obj, '__dict__', {})) > 5
					is_callable_class = hasattr(var_obj, '__class__') and hasattr(var_obj, '__call__')
					
					if has_complex_attrs or is_callable_class:
						complex_object_vars.append(var_name)
			
			# Analyze and clean complex objects
			for var_name in complex_object_vars:
				try:
					var_obj = local_vars.get(var_name)
					if var_obj is not None and hasattr(var_obj, '__dict__'):
						obj_dict = var_obj.__dict__
						
						# Detect potential circular references in object attributes
						circular_attrs = []
						for attr_name, attr_value in list(obj_dict.items()):
							# Check if attribute references parent object
							if attr_value is var_obj:
								circular_attrs.append(attr_name)
							# Check if attribute contains parent object in its dict
							elif hasattr(attr_value, '__dict__') and var_obj in getattr(attr_value, '__dict__', {}).values():
								circular_attrs.append(attr_name)
							# Check for cross-references between attributes
							elif hasattr(attr_value, '__dict__'):
								attr_dict = getattr(attr_value, '__dict__', {})
								if any(other_attr is attr_value for other_attr in obj_dict.values() if other_attr != attr_value):
									circular_attrs.append(attr_name)
						
						# Break circular references
						for attr_name in circular_attrs:
							try:
								setattr(var_obj, attr_name, None)
								cleanup_stats['circular_refs_broken'] += 1
							except:
								pass
						
						local_vars[var_name] = None
						cleanup_stats['objects_cleaned'] += 1
						
				except Exception as e:
					output_to_terminal_error(f"Error cleaning complex object {var_name}: {str(e)}")
					continue
			
			# 5. CLEAN CLASS INSTANCE VARIABLES THAT MIGHT HAVE CIRCULAR REFS
			class_cache_attrs = [
				'_memory_bank', '_spectral_blend_cache',
				'_optical_flow_features', '_noise_schedule_cache', '_previous_chunk_latents',
				'_temporal_features', '_original_clip_vision', '_original_image_reference',
				'_original_color_stats', '_drift_detection_reference'
			]
			
			for attr_name in class_cache_attrs:
				if hasattr(self, attr_name):
					try:
						attr_obj = getattr(self, attr_name)
						if attr_obj is not None:
							# Clean based on object type
							if isinstance(attr_obj, dict):
								# Check for circular references in dict values
								for key, value in list(attr_obj.items()):
									if value is attr_obj or (hasattr(value, '__dict__') and attr_obj in getattr(value, '__dict__', {}).values()):
										attr_obj[key] = None
										cleanup_stats['circular_refs_broken'] += 1
								attr_obj.clear()
							elif isinstance(attr_obj, list):
								# Check for circular references in list items
								for i, item in enumerate(attr_obj):
									if item is attr_obj or (hasattr(item, '__dict__') and attr_obj in getattr(item, '__dict__', {}).values()):
										attr_obj[i] = None
										cleanup_stats['circular_refs_broken'] += 1
								attr_obj.clear()
							elif hasattr(attr_obj, 'cpu'):
								attr_obj.cpu()
							
							setattr(self, attr_name, None)
							cleanup_stats['caches_cleaned'] += 1
					except Exception as e:
						output_to_terminal_error(f"Error cleaning class attribute {attr_name}: {str(e)}")
						try:
							setattr(self, attr_name, None)
						except:
							pass
			
			# 6. FORCE GARBAGE COLLECTION WITH CIRCULAR REFERENCE FOCUS
			# Multiple passes to ensure circular references are fully broken
			total_collected = 0
			for gc_pass in range(6):  # More passes for circular reference cleanup
				collected = gc.collect()
				total_collected += collected
				if collected == 0:
					break
				# Small delay to allow circular reference detection
				time.sleep(0.005)
			
			# 7. FINAL MEMORY CALCULATION AND REPORTING
			final_memory = 0
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
				final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
				cleanup_stats['memory_freed_mb'] = initial_memory - final_memory
			
			# Report comprehensive cleanup results
			output_to_terminal_successful(f"Circular reference cleanup complete:")
			output_to_terminal_successful(f"  Models: {cleanup_stats['models_cleaned']}, Tensors: {cleanup_stats['tensors_cleaned']}")
			output_to_terminal_successful(f"  Caches: {cleanup_stats['caches_cleaned']}, Objects: {cleanup_stats['objects_cleaned']}")
			output_to_terminal_successful(f"  Circular refs broken: {cleanup_stats['circular_refs_broken']}")
			output_to_terminal_successful(f"  Memory freed: {cleanup_stats['memory_freed_mb']:.1f}MB")
			output_to_terminal_successful(f"  GC collected: {total_collected} objects")
			
			return cleanup_stats
			
		except Exception as e:
			output_to_terminal_error(f"Error in comprehensive circular reference cleanup: {str(e)}")
			# Emergency fallback cleanup
			try:
				# Force clear all detected variables
				for var_name in list(local_vars.keys()):
					if not var_name.startswith('_') and local_vars.get(var_name) is not None:
						local_vars[var_name] = None
				
				# Force multiple GC passes
				for _ in range(3):
					gc.collect()
				
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					
				output_to_terminal_successful("Emergency circular reference cleanup completed")
			except:
				output_to_terminal_error("Emergency cleanup also failed")
			
			return {'error': True, 'message': str(e)}

	def interrupt_execution(self, local_vars=None):
		global interrupt_processing
		global interrupt_processing_mutex
		with interrupt_processing_mutex:
			if interrupt_processing:
				self.full_memory_cleanup(local_vars)
				interrupt_processing = False
				raise InterruptProcessingException()

	def full_memory_cleanup(self, local_vars=None):
		gc.collect()
		torch.cuda.empty_cache()

		# Use provided local variables or empty dict as fallback
		if local_vars is None:
			local_vars = {}

		# Comprehensive circular reference detection and cleanup
		self.comprehensive_circular_reference_cleanup(local_vars)
		
		# Additional standard cleanup (backup)
		self.break_circular_references(local_vars)
		self.cleanup_local_refs(local_vars)
		self.enhanced_memory_cleanup(local_vars)
		mm.unload_all_models()
		mm.soft_empty_cache()
						
	def create_buffer_managed_conditioning(self, vae, width, height, chunk_length, frame_buffer, overlap_frames=16, use_motion_prediction=True, start_image=None):
		"""
		Create conditioning using advanced buffer management with strong visual continuity
		"""
		# Get overlap data from buffer
		overlap_frames_data = frame_buffer.get_overlap_frames()
		
		# Create full image tensor for the entire chunk
		image_tensor = torch.ones((chunk_length, height, width, 3)) * 0.5
		
		# Encode first to get proper latent dimensions
		temp_latent = vae.encode(image_tensor[:1, :, :, :3])  # Encode just one frame to get dimensions
		
		# Scale the latent spatial dimensions to target size
		# Handle 5D tensor (batch, channels, frames, height, width) or 4D tensor (batch, channels, height, width)
		if len(temp_latent.shape) == 5:
			# 5D tensor: (batch, channels, frames, height, width)
			temp_latent = torch.nn.functional.interpolate(
				temp_latent, 
				size=(temp_latent.shape[2], height // 8, width // 8),  # (frames, height, width)
				mode='trilinear', 
				align_corners=False
			)
		else:
			# 4D tensor: (batch, channels, height, width)
			temp_latent = torch.nn.functional.interpolate(
				temp_latent, 
				size=(height // 8, width // 8), 
				mode='bilinear', 
				align_corners=False
			)

		latent_h, latent_w = temp_latent.shape[3], temp_latent.shape[4]
		device = temp_latent.device  # Get device from VAE output
		
		# Create mask with proper latent spatial dimensions and device
		mask = torch.ones((1, 1, chunk_length, latent_h, latent_w), device=device)
		
		if overlap_frames_data is not None:
			# Use MORE overlap frames for stronger continuity
			extended_overlap = min(overlap_frames * 2, overlap_frames_data.shape[0], chunk_length)
			
			# Place overlap frames at the beginning with extended coverage
			image_tensor[:extended_overlap] = overlap_frames_data[-extended_overlap:]
			
			# Create STRONGER conditioning mask
			mask[:, :, :extended_overlap] = 0.0  # Force use of overlap frames
			
			# Create gradual transition zone for smoother blending
			transition_frames = min(8, chunk_length - extended_overlap)
			for i in range(transition_frames):
				frame_idx = extended_overlap + i
				if frame_idx < chunk_length:
					# Gradual transition from strict to free
					alpha = i / transition_frames
					mask[:, :, frame_idx] = alpha * 0.5  # Partial conditioning
					
					# Use motion prediction for transition frames
					if use_motion_prediction:
						motion_pred = frame_buffer.get_motion_prediction()
						if motion_pred is not None:
							decay_factor = 0.9 ** i
							base_frame = overlap_frames_data[-1]
							predicted_frame = torch.clamp(
								base_frame + motion_pred * decay_factor, 0.0, 1.0
							)
							# Blend with existing content
							image_tensor[frame_idx] = (
								image_tensor[frame_idx] * alpha + 
								predicted_frame * (1 - alpha)
							)
		
		# Handle start image override (should be last to take priority)
		if start_image is not None:
			image_tensor[0:1] = start_image
			mask[:, :, 0:1] = 0.0
		
		# Apply feature matching for better visual continuity
		if overlap_frames_data is not None and overlap_frames_data.shape[0] > 4:
			# Extract visual features from last overlap frame
			reference_frame = overlap_frames_data[-1]
			
			# Get style features from buffer if available
			style_features = frame_buffer.get_style_features()
			color_stats = frame_buffer.get_color_stats()
			
			# Apply color and luminance matching to free generation area
			image_tensor = self.apply_color_consistency(
				image_tensor, reference_frame, extended_overlap, style_features, color_stats
			)
		
		# Encode entire image sequence to latent
		input_clip_latent = vae.encode(image_tensor[:, :, :, :3])
		
		# Scale the latent spatial dimensions to target size
		# Handle 5D tensor (batch, channels, frames, height, width) or 4D tensor (batch, channels, height, width)
		if len(input_clip_latent.shape) == 5:
			# 5D tensor: (batch, channels, frames, height, width)
			input_clip_latent = torch.nn.functional.interpolate(
				input_clip_latent, 
				size=(input_clip_latent.shape[2], height // 8, width // 8),  # (frames, height, width)
				mode='trilinear', 
				align_corners=False
			)
		else:
			# 4D tensor: (batch, channels, height, width)
			input_clip_latent = torch.nn.functional.interpolate(
				input_clip_latent, 
				size=(height // 8, width // 8), 
				mode='bilinear', 
				align_corners=False
			)
		
		# Ensure mask matches latent temporal dimension (VAE compresses by 4x)
		latent_frames = input_clip_latent.shape[2]
		
		if mask.shape[2] != latent_frames:
			# Use simple indexing/repetition instead of interpolate for temporal adjustment
			if mask.shape[2] > latent_frames:
				# Downsample: take every 4th frame approximately
				indices = torch.linspace(0, mask.shape[2] - 1, latent_frames).long()
				mask = mask[:, :, indices, :, :]
			else:
				# Upsample: repeat frames
				repeat_factor = latent_frames // mask.shape[2]
				remainder = latent_frames % mask.shape[2]
				mask_repeated = mask.repeat(1, 1, repeat_factor, 1, 1)
				if remainder > 0:
					mask_extra = mask[:, :, :remainder, :, :]
					mask = torch.cat([mask_repeated, mask_extra], dim=2)
		
		return input_clip_latent, mask
	
	def apply_color_consistency(self, image_tensor, reference_frame, transition_start, style_features=None, color_stats=None):
		"""
		Apply enhanced color and style consistency to maintain visual continuity
		"""
		# Use style features if available, otherwise extract from reference
		if style_features is not None and color_stats is not None:
			ref_mean = color_stats['mean']
			ref_std = color_stats['std']
			luma_target = style_features.get('luma_mean', 0.5)
		else:
			# Fallback to reference frame
			ref_mean = reference_frame.mean(dim=[0, 1], keepdim=True)
			ref_std = reference_frame.std(dim=[0, 1], keepdim=True)
			ref_luma = (reference_frame[:, :, 0] * 0.299 + 
					   reference_frame[:, :, 1] * 0.587 + 
					   reference_frame[:, :, 2] * 0.114)
			luma_target = ref_luma.mean()
		
		# Apply enhanced color matching to transition and free frames
		for i in range(transition_start, image_tensor.shape[0]):
			frame = image_tensor[i]
			frame_mean = frame.mean(dim=[0, 1], keepdim=True)
			frame_std = frame.std(dim=[0, 1], keepdim=True)
			
			# Stronger color transfer with slower decay for better consistency
			decay = 0.85 ** (i - transition_start)  # Slower decay
			target_mean = ref_mean * decay + frame_mean * (1 - decay)
			target_std = ref_std * decay + frame_std * (1 - decay)
			
			# Apply color transfer
			normalized = (frame - frame_mean) / (frame_std + 1e-8)
			image_tensor[i] = normalized * target_std + target_mean
			
			# Apply luminance consistency
			if style_features is not None:
				frame_luma = (image_tensor[i][:, :, 0] * 0.299 + 
							 image_tensor[i][:, :, 1] * 0.587 + 
							 image_tensor[i][:, :, 2] * 0.114)
				current_luma = frame_luma.mean()
				luma_diff = luma_target - current_luma
				luma_adjustment = luma_diff * decay * 0.3  # Gentle luminance adjustment
				
				# Apply luminance correction
				image_tensor[i] = image_tensor[i] + luma_adjustment
			
			image_tensor[i] = torch.clamp(image_tensor[i], 0.0, 1.0)
		
		return image_tensor