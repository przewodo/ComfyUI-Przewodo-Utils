import os
import torch
import nodes
import folder_paths
import gc
import weakref
import sys
import time
import comfy.model_management as mm
from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_cfg import CFGZeroStar
from .core import *
from .cache_manager import CacheManager
from .wan_first_last_first_frame_to_video import WanFirstLastFirstFrameToVideo
from .wan_video_vae_decode import WanVideoVaeDecode
from .wan_get_max_image_resolution_by_aspect_ratio import WanGetMaxImageResolutionByAspectRatio
from .wan_video_enhance_a_video import WanVideoEnhanceAVideo
from .image_sizer_node import ImageSizer

# Optional imports with try/catch blocks

try:
    from torchvision.transforms.functional import gaussian_blur
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

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
                ("positive", ("STRING", {"default": None, "advanced": True, "tooltip": "Positive text prompt describing what you want to generate in the video."})),
                ("negative", ("STRING", {"default": None, "advanced": True, "tooltip": "Negative text prompt describing what you want to avoid in the video generation."})),
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
                ("total_video_chunks", ("INT", {"default": 1, "min": 1, "max": 1000, "step":1, "advanced": True, "tooltip": "Number of sequential video chunks to generate. Each chunk extends the total video duration. Higher values create longer videos by generating chunks in sequence."})),
                
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
                ("frames_interpolation", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Make frame interpolation with Rife TensorRT. This will make the video smoother by generating additional frames between existing ones."})),
                ("frames_engine", (rife_engines, {"default": NONE, "tooltip": "Rife TensorRT engine to use for frame interpolation."})),
                ("frames_multiplier", ("INT", {"default": 2, "min": 2, "max": 100, "step":1, "advanced": True, "tooltip": "Multiplier for the number of frames generated during interpolation."})),
                ("frames_clear_cache_after_n_frames", ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear the cache after processing this many frames. Helps manage memory usage during long video generation."})),
                ("frames_use_cuda_graph", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Use CUDA Graphs for frame interpolation. Improves performance by reducing overhead during inference."})),
                ("frames_overlap_chunks", ("INT", {"default": 8, "min": 1, "max": 81, "step":1, "advanced": True, "tooltip": "Number of overlapping frames between video chunks to ensure seamless motion continuity. Higher values (8-16) create smoother transitions, while lower values may cause visible seams between chunks."})),
                ("frames_overlap_chunks_blend", ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1, "step":0.01, "advanced": True, "tooltip": "How much to influence from previous frames (0.0 = no influence, 1.0 = full replacement)"})),
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

    def run(self, GGUF_High, GGUF_Low, Diffusor_High, Diffusor_Low, Diffusor_weight_dtype, Use_Model_Type, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.05, tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda", use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=8.0, use_block_swap=True, block_swap=20, large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size=WAN_720P, total_video_seconds=1, total_video_chunks=1, clip_vision_model=NONE, clip_vision_strength=1.0, use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0, lora_stack=None, start_image=None, start_image_clip_vision_enabled=True, end_image=None, end_image_clip_vision_enabled=True, video_enhance_enabled=True, use_cfg_zero_star=True, apply_color_match=True, causvid_lora=NONE, high_cfg_causvid_strength=1.0, low_cfg_causvid_strength=1.0, high_denoise=1.0, low_denoise=1.0, prompt_stack=None, fill_noise_latent=0.5, frames_interpolation=False, frames_engine=NONE, frames_multiplier=2, frames_clear_cache_after_n_frames=100, frames_use_cuda_graph=True, frames_overlap_chunks=8, frames_overlap_chunks_blend=0.3):
        self.default_fps = 16.0

        gc.collect()
        torch.cuda.empty_cache()
        #mm.soft_empty_cache()

        model_high = self.load_model(GGUF_High, Diffusor_High, Use_Model_Type, Diffusor_weight_dtype)
        mm.throw_exception_if_processing_interrupted()

        model_low = None
        if (GGUF_Low != NONE or Diffusor_Low != NONE):
            model_low = self.load_model(GGUF_Low, Diffusor_Low, Use_Model_Type, Diffusor_weight_dtype)
            mm.throw_exception_if_processing_interrupted()
        else:
            model_low = model_high

        output_to_terminal_successful("Loading VAE...")
        vae, = nodes.VAELoader().load_vae(vae)
        mm.throw_exception_if_processing_interrupted()

        output_to_terminal_successful("Loading CLIP...")
        # Create cache key for CLIP model
        clip_cache_key = f"{clip}_{clip_type}_{clip_device}"
        # Check if CLIP model is already cached
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
        
        mm.throw_exception_if_processing_interrupted()

        # Clean up memory after model loading
        gc.collect()
        torch.cuda.empty_cache()

        tea_cache = None
        sage_attention = None
        slg_wanvideo = None
        model_shift = None
        wanBlockSwap = WanVideoBlockSwap()

        # Initialize TeaCache and SkipLayerGuidanceWanVideo
        tea_cache, slg_wanvideo = self.initialize_tea_cache_and_slg(use_tea_cache, use_SLG, SLG_blocks)
        mm.throw_exception_if_processing_interrupted()

        # Initialize SageAttention
        sage_attention = self.initialize_sage_attention(use_sage_attention, sage_attention_mode)
        mm.throw_exception_if_processing_interrupted()

        # Initialize Model Shift
        model_shift = self.initialize_model_shift(use_shift, shift)
        mm.throw_exception_if_processing_interrupted()

        output_image, fps, = self.postprocess(model_high, model_low, vae, clip_model, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, frames_overlap_chunks, frames_overlap_chunks_blend)

        # Aggressive cleanup of main models to prevent WanTEModel memory leaks
        # NOTE: Only do this AFTER processing is completely done
        # Don't cleanup models that might still be needed
        
        # Break any circular references
        self.break_circular_references(locals())
        
        # Standard cleanup
        self.cleanup_local_refs(locals())
        self.enhanced_memory_cleanup(locals())
        mm.unload_all_models()
        mm.soft_empty_cache()

        return (output_image, fps,)

    def postprocess(self, model_high, model_low, vae, clip_model, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, frames_overlap_chunks, frames_overlap_chunks_blend):
        gc.collect()
        torch.cuda.empty_cache()

        k_sampler = nodes.KSamplerAdvanced()
        text_encode = nodes.CLIPTextEncode()
        wan_image_to_video = WanFirstLastFirstFrameToVideo()
        wan_video_vae_decode = WanVideoVaeDecode()
        wan_max_resolution = WanGetMaxImageResolutionByAspectRatio()
        CLIPVisionLoader = nodes.CLIPVisionLoader()
        CLIPVisionEncoder = nodes.CLIPVisionEncode()
        resizer = ImageResizeKJv2()
        image_width = large_image_side
        image_height = large_image_side
        in_latent = None
        out_latent = None
        total_frames = (total_video_seconds * 16) + 1
        lora_loader = nodes.LoraLoader()
        wanVideoEnhanceAVideo = WanVideoEnhanceAVideo()
        cfgZeroStar = CFGZeroStar()
        colorMatch = ColorMatch()
        clip_vision_start_image = None
        clip_vision_end_image = None
        positive_clip_high = None
        positive_clip_low = None
        negative_clip_high = None
        negative_clip_low = None
        clip_vision = None
        model_high_cfg = None
        model_low_cfg = None

#        if (image_generation_mode == TEXT_TO_VIDEO):
#            start_image = None
#            end_image = None
#
#        if (image_generation_mode == TEXT_TO_VIDEO):
#            imageSizer = ImageSizer()
#            image_width, image_height, = imageSizer.run(wan_model_size, 9, 16)
#            wan_max_resolution

        # Load CLIP Vision Model
        clip_vision = self.load_clip_vision_model(clip_vision_model, CLIPVisionLoader)
        mm.throw_exception_if_processing_interrupted()

       # Generate video chunks sequentially
        images_chunk = []
        last_latent = None
        original_image_start = start_image
        original_image_end = end_image
        output_image = None
        reference_frames = []
        
        output_to_terminal_successful("Generation started...")

        for chunk_index in range(total_video_chunks):
            working_model_high = model_high.clone()
            working_model_low = model_low.clone()
            working_clip_high = clip_model.clone()
            working_clip_low = clip_model.clone()

            # Immediately clear CUDA cache after cloning to prevent accumulation
            torch.cuda.empty_cache()

            # Apply Model Patch Torch Settings
            working_model_high = self.apply_model_patch_torch_settings(working_model_high)
            working_model_low = self.apply_model_patch_torch_settings(working_model_low)
            mm.throw_exception_if_processing_interrupted()

            # Apply Sage Attention
            working_model_high = self.apply_sage_attention(sage_attention, working_model_high, sage_attention_mode)
            working_model_low = self.apply_sage_attention(sage_attention, working_model_low, sage_attention_mode)
            mm.throw_exception_if_processing_interrupted()

            # Apply TeaCache and SLG
            working_model_high = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model_high, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent)
            working_model_low = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model_low, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent)
            mm.throw_exception_if_processing_interrupted()

            # Apply Model Shift
            working_model_high = self.apply_model_shift(model_shift, use_shift, working_model_high, shift)
            working_model_low = self.apply_model_shift(model_shift, use_shift, working_model_low, shift)
            mm.throw_exception_if_processing_interrupted()

            # Apply Video Enhance
            working_model_high = self.apply_video_enhance(video_enhance_enabled, working_model_high, wanVideoEnhanceAVideo, total_frames)
            working_model_low = self.apply_video_enhance(video_enhance_enabled, working_model_low, wanVideoEnhanceAVideo, total_frames)
            mm.throw_exception_if_processing_interrupted()

            # Apply CFG Zero Star
            working_model_high = self.apply_cfg_zero_star(use_cfg_zero_star, working_model_high, cfgZeroStar)
            working_model_low = self.apply_cfg_zero_star(use_cfg_zero_star, working_model_low, cfgZeroStar)
            mm.throw_exception_if_processing_interrupted()

            # Apply Block Swap
            working_model_high = self.apply_block_swap(use_block_swap, working_model_high, wanBlockSwap, block_swap)
            working_model_low = self.apply_block_swap(use_block_swap, working_model_low, wanBlockSwap, block_swap)
            mm.throw_exception_if_processing_interrupted()

            # Process LoRA stack
            working_model_high, working_clip_high = self.process_lora_stack(lora_stack, working_model_high, working_clip_high)
            working_model_low, working_clip_low = self.process_lora_stack(lora_stack, working_model_low, working_clip_low)
            mm.throw_exception_if_processing_interrupted()

            # Clean up memory after model configuration
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()

            # Light memory cleanup before each chunk - don't interfere with active models
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()  # Ensure all operations complete before cleanup

#            if (image_generation_mode == TEXT_TO_VIDEO and chunk_index == 1):
#                start_image = images_chunk[len(images_chunk) - 1][len(images_chunk[len(images_chunk) - 1]) - 1]  # Use last frame of previous chunk as start image
#                original_image = images_chunk[len(images_chunk) - 1][0] # Use first frame of previous chunk as original image
#                image_generation_mode = START_IMAGE  # Switch to START_IMAGE mode after first chunk

            output_to_terminal(f"Generating video chunk {chunk_index + 1}/{total_video_chunks}...")
            mm.throw_exception_if_processing_interrupted()
            
            if (prompt_stack is not None):
                positive, negative, prompt_loras = self.get_current_prompt(prompt_stack, chunk_index, positive, negative)
                working_model_high, working_clip_high = self.process_lora_stack(prompt_loras, working_model_high, working_clip_high)
                working_model_low, working_clip_low = self.process_lora_stack(prompt_loras, working_model_low, working_clip_low)
                mm.throw_exception_if_processing_interrupted()

            # Get original start_image dimensions if available
            if start_image is not None and (image_generation_mode == START_IMAGE or image_generation_mode == START_END_IMAGE or image_generation_mode == START_TO_END_TO_START_IMAGE):
                # ComfyUI images are tensors with shape [batch, height, width, channels]
                output_to_terminal_successful(f"Original start_image dimensions: {start_image.shape[2]}x{start_image.shape[1]}")

                # Process Start Image
                start_image, image_width, image_height, clip_vision_start_image = self.process_image(original_image_start,
                    start_image, start_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, 
                    CLIPVisionEncoder, large_image_side, wan_model_size, start_image.shape[2], start_image.shape[1], "Start Image",
                    chunk_index
                )

            # Get original end_image dimensions if available
            if end_image is not None and (image_generation_mode == END_TO_START_IMAGE):
                # ComfyUI images are tensors with shape [batch, height, width, channels]
                output_to_terminal_successful(f"Original end_image dimensions: {end_image.shape[2]}x{end_image.shape[1]}")

                end_image, image_width, image_height, clip_vision_end_image = self.process_image(original_image_end,
                    end_image, end_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution,
                    CLIPVisionEncoder, large_image_side, wan_model_size, end_image.shape[2], end_image.shape[1], "End Image",
                    chunk_index
                )
            mm.throw_exception_if_processing_interrupted()

            # Apply CausVid LoRA processing for current chunk
            model_high_cfg, model_low_cfg, working_clip_high, working_clip_low = self.apply_causvid_lora_processing(working_model_high, working_model_low, working_clip_high, working_clip_low, lora_loader, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Positive CLIP text...")
            positive_clip_high, = text_encode.encode(working_clip_high, positive)
            positive_clip_low, = text_encode.encode(working_clip_low, positive)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Negative CLIP text...")
            negative_clip_high, = text_encode.encode(working_clip_high, negative)
            negative_clip_low, = text_encode.encode(working_clip_low, negative)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Wan Image to Video started...")
            positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, in_latent, = wan_image_to_video.encode(positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, vae, image_width, image_height, total_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, fill_noise_latent, image_generation_mode)
            mm.throw_exception_if_processing_interrupted()

            if (last_latent is not None):
                # Guide current chunk generation using previous chunk's motion
                in_latent = self.guide_next_chunk_generation(last_latent, in_latent, frames_overlap_chunks, frames_overlap_chunks_blend)
                
                # Clean up previous latent to free memory
                del last_latent
                last_latent = None
                
            # Light cleanup without interfering with active models
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()

            # high_denoise, low_denoise = self.apply_progressive_denoise_ramp(high_denoise, low_denoise, chunk_index, total_video_chunks)

            if (use_dual_samplers):
                # Apply dual sampler processing
                out_latent = self.apply_dual_sampler_processing(model_high_cfg, model_low_cfg, k_sampler, noise_seed, total_steps, high_cfg, low_cfg, positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, in_latent, total_steps_high_cfg, high_denoise, low_denoise)
            else:
                # Apply single sampler processing
                out_latent = self.apply_single_sampler_processing(model_high_cfg, k_sampler, noise_seed, total_steps, high_cfg, positive_clip_high, negative_clip_high, in_latent, high_denoise)
            mm.throw_exception_if_processing_interrupted()

            last_latent = out_latent

            output_image, = wan_video_vae_decode.decode(out_latent, vae, 0, image_generation_mode)
            mm.throw_exception_if_processing_interrupted()

            # Subsequent chunks: use original_image as reference for consistency
            output_image = self.apply_color_match_to_image(original_image_start, output_image, apply_color_match, colorMatch)
            mm.throw_exception_if_processing_interrupted()

            reference_frames.append(output_image[0:1].clone())
            if len(reference_frames) > 5:
                reference_frames.pop(0)
                
            # Clean up after reference frame operations
            gc.collect()
            torch.cuda.empty_cache()

            if (total_video_chunks > 1):
                start_image = output_image[output_image.shape[0] - 1:output_image.shape[0]].clone()
                if (chunk_index < total_video_chunks - 1):
                    images_chunk.append(output_image[:-1])
                else:
                    images_chunk.append(output_image)
            else:
                images_chunk.append(output_image)

            output_to_terminal_successful(f"Video chunk {chunk_index + 1} generated successfully")

            del in_latent
            if (chunk_index == total_video_chunks - 1):
                del last_latent
            
            # Clean up working models for this chunk - Use safe cleanup
            # Don't use aggressive cleanup during processing
            del working_model_high, working_model_low, working_clip_high, working_clip_low
            
            # Light cleanup only
            gc.collect()
            torch.cuda.empty_cache()
        
        del reference_frames

        output_to_terminal_successful("All video chunks generated successfully")

        # Force cleanup of main models before final processing - be less aggressive
        # Only break circular references, don't force cleanup active models
        self.break_circular_references(locals())

        # Merge all video chunks in sequence with overlap handling
        if len(images_chunk) > 1:
            output_to_terminal_successful(f"Merging {len(images_chunk)} video chunks with simple concatenation...")
            output_image = torch.cat(images_chunk, dim=0)
        elif len(images_chunk) == 1:
            output_image = images_chunk[0]

        mm.throw_exception_if_processing_interrupted()

        # Light cleanup only
        gc.collect()
        torch.cuda.empty_cache()

        if (output_image is None):
            return (None, self.default_fps,)

        if (frames_interpolation and frames_engine != NONE):
            gc.collect()
            torch.cuda.empty_cache()
            #mm.soft_empty_cache()
            output_to_terminal_successful(f"Starting interpolation with engine: {frames_engine}, multiplier: {frames_multiplier}, clear cache after {frames_clear_cache_after_n_frames} frames, use CUDA graph: {frames_use_cuda_graph}")
            interpolationEngine = RifeTensorrt()
            output_image, = interpolationEngine.vfi(output_image, frames_engine, frames_clear_cache_after_n_frames, frames_multiplier, frames_use_cuda_graph, False)
            self.default_fps = self.default_fps * float(frames_multiplier)
            return (output_image[:-frames_multiplier+1], self.default_fps,)
        else:
            return (output_image[:-1], self.default_fps,)

    def get_current_prompt(self, prompt_stack, chunk_index, default_positive, default_negative):
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
                return default_positive, default_negative, None
        else:
            return default_positive, default_negative, None
   
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
        else:
            output_to_terminal_successful("No LoRA stack provided, skipping LoRA application")
        
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

    def process_image(self, reference_image, image, image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size, image_width, image_height, image_type, chunck_index):
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

            if (chunck_index == 0):
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

                output_to_terminal_successful(f"{image_type} final size: {image_width}x{image_height}")
            else:
                # For subsequent chunks, maintain previous dimensions
                image_width = original_width
                image_height = original_height
                output_to_terminal_successful(f"{image_type} final size: {original_width}x{original_height}")

            if (image_clip_vision_enabled) and (clip_vision is not None):
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

    def apply_tea_cache_and_slg(self, tea_cache, use_tea_cache, working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent):
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

            if (slg_wanvideo is not None and use_SLG) and (slg_wanvideo_blocks_string is not None) and (slg_wanvideo_blocks_string.strip() != ""):
                output_to_terminal_successful(f"Applying Skip Layer Guidance with blocks: {slg_wanvideo_blocks_string}...")
                working_model, = slg_wanvideo.slg(working_model, slg_wanvideo_start_percent, slg_wanvideo_end_percent, slg_wanvideo_blocks_string)
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

    def apply_dual_sampler_processing(self, model_high_cfg, model_low_cfg, k_sampler, noise_seed, total_steps, high_cfg, low_cfg, positive_clip_high, negative_clip_high, positive_clip_low, negative_clip_low, in_latent, total_steps_high_cfg, high_denoise, low_denoise):
        stop_steps = int(total_steps_high_cfg / 100 * total_steps)

        gc.collect()
        torch.cuda.empty_cache()
        #mm.soft_empty_cache()
        output_to_terminal_successful("High CFG KSampler started...")
        out_latent, = k_sampler.sample(model_high_cfg, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", positive_clip_high, negative_clip_high, in_latent, 0, stop_steps, "enabled", high_denoise)
        mm.throw_exception_if_processing_interrupted()
        
        # Light cleanup between samplers
        gc.collect()
        torch.cuda.empty_cache()

        output_to_terminal_successful("Low CFG KSampler started...")
        out_latent, = k_sampler.sample(model_low_cfg, "disable", noise_seed, total_steps, low_cfg, "lcm", "simple", positive_clip_low, negative_clip_low, out_latent, stop_steps, 1000, "enabled", low_denoise)
        mm.throw_exception_if_processing_interrupted()
        
        # Light cleanup after sampling
        gc.collect()
        torch.cuda.empty_cache()

        return out_latent

    def apply_single_sampler_processing(self, working_model, k_sampler, noise_seed, total_steps, high_cfg, positive_clip_high, negative_clip_high, in_latent, high_denoise):

        output_to_terminal_successful("KSampler started...")
        out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", positive_clip_high, negative_clip_high, in_latent, 0, 1000, "enabled", high_denoise)

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
        model_low_cfg = model_low.clone()
        updated_clip_high = clip_high.clone()
        updated_clip_low = clip_low.clone()

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

    def apply_color_match_to_image(self, original_image, image, apply_color_match, colorMatch):
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
            image, = colorMatch.colormatch(original_image, image, "hm-mvgd-hm", strength=1.0)
#            current_mean = torch.mean(image)
#            current_std = torch.std(image)
#            
#            if original_image is not None:
#                ref_mean = torch.mean(original_image)
#                ref_std = torch.std(original_image)
#                
#                # Detect quality degradation
#                brightness_drift = abs(current_mean - ref_mean)
#                contrast_loss = abs(current_std - ref_std)
#                
#                if brightness_drift > 0.1 or contrast_loss > 0.05:
#                    output_to_terminal_error(f"Quality degradation detected: brightness_drift={brightness_drift:.3f}, contrast_loss={contrast_loss:.3f}")

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

    def guide_next_chunk_generation(self, last_latent, in_latent, frames_overlap_chunks, blend_strength=0.7):
        """
        Use last 16 frames from previous chunk to guide first 16 frames of current chunk for motion continuity.
        
        Args:
            last_latent: The latent output from the previous chunk containing motion information
            in_latent: The input latent for the current chunk to be guided
            blend_strength (float): How much to influence from previous frames (0.0 = no influence, 1.0 = full replacement)
            
        Returns:
            in_latent: The modified input latent with motion guidance applied
        """
        if last_latent is None:
            return in_latent
            
        # Use last frames_overlap_chunks frames from previous chunk to guide first 16 frames of current chunk
        output_to_terminal_successful(f"Blending last {frames_overlap_chunks} frames from previous chunk for motion continuity...")

        # Get the last frames_overlap_chunks frames from the previous latent
        last_frames_count = min(frames_overlap_chunks, last_latent['samples'].shape[0])
        last_frames = last_latent['samples'][-last_frames_count:]
        
        # Get the first frames from current latent that we want to replace/blend
        first_frames_count = min(frames_overlap_chunks, in_latent['samples'].shape[0])
        
        # Apply blending to the overlapping frames
        overlap_frames = min(last_frames_count, first_frames_count)
        for i in range(overlap_frames):
            # Create a fade from previous to current (stronger influence at start, weaker at end)
            frame_blend_weight = blend_strength * (1.0 - (i / overlap_frames))
            
            # Blend the latents: current_frame = (1-weight) * current + weight * previous
            in_latent['samples'][i] = (1.0 - frame_blend_weight) * in_latent['samples'][i] + frame_blend_weight * last_frames[-(overlap_frames-i)]
        
        output_to_terminal_successful(f"Blended {overlap_frames} frames for motion continuity with strength {blend_strength}")
        
        return in_latent
    
    def apply_progressive_denoise_ramp(self, high_denoise, low_denoise, chunk_index, total_chunks):
        """
        Apply progressive denoise ramping to prevent quality drops at boundaries.
        
        Args:
            high_denoise: Current high denoise value
            low_denoise: Current low denoise value
            chunk_index: Current chunk index
            total_chunks: Total number of chunks
            
        Returns:
            tuple: (adjusted_high_denoise, adjusted_low_denoise)
        """
        if chunk_index == 0:
            return high_denoise, low_denoise
        
        # Use more conservative reduction to avoid static frames
        # Linear reduction is safer than exponential for video generation
        max_reduction = 0.15  # Maximum 15% reduction from original values
        reduction_per_chunk = max_reduction / max(total_chunks - 1, 1)  # Spread reduction across all chunks
        total_reduction = min(reduction_per_chunk * chunk_index, max_reduction)
        
        # Apply reduction but maintain minimum thresholds suitable for video
        adjusted_high_denoise = max(high_denoise * (1.0 - total_reduction), 0.85)  # Higher minimum for video
        adjusted_low_denoise = max(low_denoise * (1.0 - total_reduction), 0.85)   # Higher minimum for video
        
        output_to_terminal_successful(f"Progressive denoise ramp: high={adjusted_high_denoise:.3f}, low={adjusted_low_denoise:.3f} (reduction: {total_reduction:.3f})")
        
        return adjusted_high_denoise, adjusted_low_denoise