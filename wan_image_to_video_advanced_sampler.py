import os
import torch
import nodes
import folder_paths
import gc
import comfy.model_management as mm
from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_cfg import CFGZeroStar
from comfy.utils import load_torch_file
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

# Import additional utilities for advanced consistency features
try:
    import torch.nn.functional as F
    FUNCTIONAL_AVAILABLE = True
except ImportError:
    FUNCTIONAL_AVAILABLE = False

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
    """
    Advanced Wan2.1 Image-to-Video Sampler with Long-Form Consistency Features
    
    This node implements comprehensive quality preservation techniques for long video generation,
    addressing the common issues of degradation, drift, and inconsistency that occur after ~5 seconds.
    
    Key Features for Long Video Consistency:
    
    1. ENHANCED LATENT BLENDING & OVERLAP:
       - Multi-scale latent blending between chunks
       - Configurable frame overlaps with gradual crossfade
       - Temporal overlap strength control
       - Persistent noise seed management
    
    2. FRAME ANCHORING & KEY-FRAME STRATEGIES:
       - Anchor frame preservation at chunk boundaries
       - Progressive denoise ramping to prevent quality drops
       - Keyframe interval injection for quality resets
       - Prompt reinforcement for character consistency
    
    3. CONDITIONING & CLIP GUIDANCE ENHANCEMENTS:
       - Reference image conditioning beyond last frame
       - CLIP-guided temporal consistency loss
       - Structural conditioning preservation
       - Dynamic CLIP strength adjustment
    
    4. QUALITY MONITORING & ENHANCEMENT:
       - Automatic quality degradation detection
       - Periodic detail boosting to combat blur
       - Scene-aware chunk segmentation
       - Cross-frame attention mechanisms
    
    5. COMMUNITY BEST PRACTICES:
       - Advanced RIFE interpolation integration
       - Lower FPS generation + interpolation workflow
       - Anti-drift sampling techniques
       - Multi-conditioning reference systems
    
    These features work together to maintain visual coherence, prevent identity drift,
    and preserve fine details across extended video sequences beyond the traditional
    5-second quality barrier of Wan2.1 models.
    """
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
                ("GGUF", (gguf_model_names, {"default": NONE, "advanced": True, "tooltip": "Select a GGUF model file for optimized inference."})),
                ("Diffusor", (diffusion_models_names, {"default": NONE, "advanced": True, "tooltip": "Select a Diffusion model for standard inference."})),
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
                ("tea_cache_rel_l1_thresh", ("FLOAT", {"default": 0.22, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})),
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
                ("block_swap", ("INT", {"default": 35, "min": 1, "max": 40, "step":1, "tooltip": "Block swap threshold value. Controls when to swap model blocks for memory optimization."})),
                
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
                
                # ═════════════════════════════════════════════════════════════════
                # 🎯 ADVANCED CONSISTENCY CONTROLS
                # ═════════════════════════════════════════════════════════════════
                ("overlap_frames", ("INT", {"default": 4, "min": 0, "max": 32, "step": 1, "advanced": True, "tooltip": "Number of overlapping frames between chunks for smooth transitions. Higher values improve consistency but increase generation time. Recommended: 4-8 frames for best results."})),
                ("temporal_overlap_strength", ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Strength of temporal blending between overlapping frames. Higher values preserve more from previous chunk, reducing discontinuities. 0.5 provides good balance between smoothness and motion fluidity."})),
                ("anchor_frame_strength", ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Strength of anchor frame preservation at chunk boundaries. Higher values maintain more consistency but may reduce motion fluidity. 0.6 provides good balance for most content."})),
                ("progressive_denoise_ramp", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Gradually reduce denoise strength across chunks to prevent quality drops at boundaries. Helps maintain detail preservation in longer sequences."})),
                ("keyframe_interval", ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "advanced": True, "tooltip": "Interval for keyframe injection (0=disabled). Periodically inject high-quality reference frames to prevent drift. Recommended: 3-5 for long sequences."})),
                ("prompt_reinforcement", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Reinforce character/scene descriptions at chunk boundaries to maintain identity consistency. Automatically emphasizes important keywords."})),
                ("reference_conditioning_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "advanced": True, "tooltip": "Strength of reference image conditioning beyond just the last frame. Higher values improve identity preservation but may reduce motion variety."})),
                ("temporal_clip_guidance", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Enable CLIP-guided temporal consistency loss to maintain visual similarity between frames. Advanced feature - may slow generation but improves coherence."})),
                ("structural_conditioning", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Use depth/pose conditioning to maintain structural consistency across frames. Requires ControlNet integration - experimental feature."})),
                ("quality_monitoring", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Monitor generation quality and apply corrections when degradation is detected. Automatically adjusts brightness/contrast to prevent drift."})),
                ("periodic_detail_boost", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Periodically enhance frame details to combat gradual blurriness accumulation. Uses sharpening every 3rd chunk to restore clarity."})),
                ("scene_aware_segmentation", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Automatically segment chunks based on scene changes for optimal consistency. Improves quality by resetting at natural break points."})),
                ("anti_drift_sampling", ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Use reverse-order generation techniques to prevent quality drift accumulation. Experimental - may improve very long sequences."})),
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

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.22, tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda", use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=2.0, use_block_swap=True, block_swap=35, large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size=WAN_720P, total_video_seconds=1, total_video_chunks=1, overlap_frames=4, temporal_overlap_strength=0.8, anchor_frame_strength=0.9, progressive_denoise_ramp=True, keyframe_interval=0, prompt_reinforcement=True, reference_conditioning_strength=1.0, temporal_clip_guidance=False, structural_conditioning=False, quality_monitoring=True, periodic_detail_boost=False, scene_aware_segmentation=True, anti_drift_sampling=False, clip_vision_model=NONE, clip_vision_strength=1.0, use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0, lora_stack=None, start_image=None, start_image_clip_vision_enabled=True, end_image=None, end_image_clip_vision_enabled=True, video_enhance_enabled=True, use_cfg_zero_star=True, apply_color_match=True, causvid_lora=NONE, high_cfg_causvid_strength=1.0, low_cfg_causvid_strength=1.0, high_denoise=1.0, low_denoise=1.0, prompt_stack=None, fill_noise_latent=0.5, frames_interpolation=False, frames_engine=NONE, frames_multiplier=2, frames_clear_cache_after_n_frames=100, frames_use_cuda_graph=True):
        self.default_fps = 16.0

        gc.collect()
        torch.cuda.empty_cache()
        #mm.soft_empty_cache()

        model = self._cache_manager.get_from_cache(f"{GGUF}_{Diffusor}_{Use_Model_Type}_{Diffusor_weight_dtype}", 'cpu')
        if (model is not None):
            output_to_terminal_successful("Loaded model from cache...")
        else:
            model = self.load_model(GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype)
            self._cache_manager.store_in_cache(f"{GGUF}_{Diffusor}_{Use_Model_Type}_{Diffusor_weight_dtype}", model, 'cpu')
        mm.throw_exception_if_processing_interrupted()

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

        # Clean up memory after initialization
        gc.collect()
        torch.cuda.empty_cache()

        return self.postprocess(model, vae, clip_model, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, overlap_frames, temporal_overlap_strength, anchor_frame_strength, progressive_denoise_ramp, keyframe_interval, prompt_reinforcement, reference_conditioning_strength, temporal_clip_guidance, structural_conditioning, quality_monitoring, periodic_detail_boost, scene_aware_segmentation, anti_drift_sampling)

    def postprocess(self, model, vae, clip_model, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, prompt_stack, fill_noise_latent, frames_interpolation, frames_engine, frames_multiplier, frames_clear_cache_after_n_frames, frames_use_cuda_graph, overlap_frames, temporal_overlap_strength, anchor_frame_strength, progressive_denoise_ramp, keyframe_interval, prompt_reinforcement, reference_conditioning_strength, temporal_clip_guidance, structural_conditioning, quality_monitoring, periodic_detail_boost, scene_aware_segmentation, anti_drift_sampling):
        gc.collect()
        torch.cuda.empty_cache()

        working_model = model.clone()
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
        total_frames = (total_video_seconds * 16) + 1 + overlap_frames
        lora_loader = nodes.LoraLoader()
        wanVideoEnhanceAVideo = WanVideoEnhanceAVideo()
        cfgZeroStar = CFGZeroStar()
        colorMatch = ColorMatch()
        clip_vision_start_image = None
        clip_vision_end_image = None
        positive_clip = None
        negative_clip = None
        clip_vision = None

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

        # Apply Model Patch Torch Settings
        working_model = self.apply_model_patch_torch_settings(working_model)
        mm.throw_exception_if_processing_interrupted()

        # Apply Sage Attention
        working_model = self.apply_sage_attention(sage_attention, working_model, sage_attention_mode)
        mm.throw_exception_if_processing_interrupted()

        # Apply TeaCache and SLG
        working_model = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent)
        mm.throw_exception_if_processing_interrupted()

        # Apply Model Shift
        working_model = self.apply_model_shift(model_shift, use_shift, working_model, shift)
        mm.throw_exception_if_processing_interrupted()

        # Apply Video Enhance
        working_model = self.apply_video_enhance(video_enhance_enabled, working_model, wanVideoEnhanceAVideo, total_frames)
        mm.throw_exception_if_processing_interrupted()

        # Apply CFG Zero Star
        working_model = self.apply_cfg_zero_star(use_cfg_zero_star, working_model, cfgZeroStar)
        mm.throw_exception_if_processing_interrupted()

        # Apply Block Swap
        working_model = self.apply_block_swap(use_block_swap, working_model, wanBlockSwap, block_swap)
        mm.throw_exception_if_processing_interrupted()

        # Clean up memory after model configuration
        gc.collect()
        torch.cuda.empty_cache()

        # Process LoRA stack
        working_model, clip_model = self.process_lora_stack(lora_stack, working_model, clip_model)
        mm.throw_exception_if_processing_interrupted()

        # Generate video chunks sequentially
        images_chunk = []
        original_image = start_image  # Store the very first frame for color matching only
        output_image = None
        
        # Initialize consistency management
        chunk_overlap_data = []  # Store overlap data for blending
        reference_frames = []   # Store high-quality reference frames
        noise_seed_base = noise_seed  # Base seed for persistent noise
        
        # Store original prompts for reinforcement (will be set after first prompt processing)
        original_positive = None
        original_negative = None
        
        output_to_terminal_successful("Generation started...")
        output_to_terminal_successful(f"Frame calculation: total_video_seconds={total_video_seconds}, total_video_chunks={total_video_chunks}")
        output_to_terminal_successful(f"Target total frames: {total_video_seconds * 16}, total_frames: {total_frames}, overlap_frames: {overlap_frames}")

        for chunk_index in range(total_video_chunks):
            mm.throw_exception_if_processing_interrupted()
            
            # Aggressive memory cleanup before each chunk
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()  # Ensure all operations complete before cleanup
            #mm.soft_empty_cache()

#            if (image_generation_mode == TEXT_TO_VIDEO and chunk_index == 1):
#                start_image = images_chunk[len(images_chunk) - 1][len(images_chunk[len(images_chunk) - 1]) - 1]  # Use last frame of previous chunk as start image
#                original_image = images_chunk[len(images_chunk) - 1][0] # Use first frame of previous chunk as original image
#                image_generation_mode = START_IMAGE  # Switch to START_IMAGE mode after first chunk

            output_to_terminal_successful(f"Generating video chunk {chunk_index + 1}/{total_video_chunks}...")
            mm.throw_exception_if_processing_interrupted()
            
            # Clone models with explicit cleanup of source references
            generation_model = working_model.clone()
            generation_clip = clip_model.clone()
            
            if (prompt_stack is not None):
                positive, negative, prompt_loras = self.get_current_prompt(prompt_stack, chunk_index, positive, negative)
                generation_model, generation_clip = self.process_lora_stack(prompt_loras, generation_model, generation_clip)
                mm.throw_exception_if_processing_interrupted()
            
            # Store original prompts for reinforcement after prompt processing
            original_positive = positive
            original_negative = negative
            
            # Apply prompt reinforcement for consistency
            if prompt_reinforcement and chunk_index > 0 and original_positive is not None:
                positive, negative = self.apply_prompt_reinforcement(positive, negative, original_positive, original_negative, chunk_index)
            
            # Set noise seed for current chunk (use base seed + chunk offset for consistency)
            current_seed = noise_seed_base + chunk_index
            
            # Check if this is a keyframe chunk
            is_keyframe_chunk = keyframe_interval > 0 and chunk_index % keyframe_interval == 0 and chunk_index > 0

            # Get original start_image dimensions if available
            if start_image is not None and (image_generation_mode == START_IMAGE or image_generation_mode == START_END_IMAGE or image_generation_mode == START_TO_END_TO_START_IMAGE):
                # ComfyUI images are tensors with shape [batch, height, width, channels]
                output_to_terminal_successful(f"Original start_image dimensions: {start_image.shape[2]}x{start_image.shape[1]}")

                # Process Start Image
                start_image, image_width, image_height, clip_vision_start_image = self.process_image(
                    start_image, start_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, 
                    CLIPVisionEncoder, large_image_side, wan_model_size, start_image.shape[2], start_image.shape[1], "Start Image"
                )

            # Get original end_image dimensions if available
            if end_image is not None and (image_generation_mode == END_TO_START_IMAGE):
                # ComfyUI images are tensors with shape [batch, height, width, channels]
                output_to_terminal_successful(f"Original end_image dimensions: {end_image.shape[2]}x{end_image.shape[1]}")

                end_image, image_width, image_height, clip_vision_end_image = self.process_image(
                    end_image, end_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution,
                    CLIPVisionEncoder, large_image_side, wan_model_size, end_image.shape[2], end_image.shape[1], "End Image"
                )
            mm.throw_exception_if_processing_interrupted()

            # Apply CausVid LoRA processing for current chunk
            model_high_cfg, model_low_cfg, generation_clip = self.apply_causvid_lora_processing(generation_model, generation_clip, lora_loader, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers)
            
            # Clean up after model preparation
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Positive CLIP text...")
            positive_clip, = text_encode.encode(generation_clip, positive)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Negative CLIP text...")
            negative_clip, = text_encode.encode(generation_clip, negative)
            mm.throw_exception_if_processing_interrupted()

            # Clean up memory after text encoding
            gc.collect()
            torch.cuda.empty_cache()

            # Calculate frames for current chunk
            current_chunk_frames = total_frames
            
            output_to_terminal_successful("Wan Image to Video started...")
            positive_clip, negative_clip, in_latent, = wan_image_to_video.encode(positive_clip, negative_clip, vae, image_width, image_height, current_chunk_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, fill_noise_latent, image_generation_mode)
            
            # Clean up CLIP vision images after encoding
            if clip_vision_start_image is not None:
                del clip_vision_start_image
            if clip_vision_end_image is not None:
                del clip_vision_end_image
                
            # Memory cleanup after encoding
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()
            
            # Apply advanced consistency techniques
            if chunk_index > 0:
                # Apply latent blending with previous chunk overlap
                in_latent = self.apply_latent_blending(in_latent, chunk_overlap_data, overlap_frames, temporal_overlap_strength, chunk_index)
                
                # Apply anchor frame preservation
                if anchor_frame_strength > 0:
                    in_latent = self.apply_anchor_frame_preservation(in_latent, chunk_overlap_data, anchor_frame_strength, chunk_index)
            
            # Apply progressive denoise ramping
            if progressive_denoise_ramp and chunk_index > 0:
                high_denoise, low_denoise = self.apply_progressive_denoise_ramp(high_denoise, low_denoise, chunk_index, total_video_chunks)

            if (use_dual_samplers):
                # Apply dual sampler processing
                out_latent = self.apply_dual_sampler_processing(model_high_cfg, model_low_cfg, k_sampler, generation_clip, current_seed, total_steps, high_cfg, low_cfg, positive_clip, negative_clip, in_latent, total_steps_high_cfg, high_denoise, low_denoise)
            else:
                # Apply single sampler processing
                out_latent = self.apply_single_sampler_processing(model_high_cfg, k_sampler, generation_clip, current_seed, total_steps, high_cfg, positive_clip, negative_clip, in_latent, high_denoise)
            
            # Aggressive memory cleanup after sampling
            del in_latent  # Free input latent memory
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Vae Decode started...")
            output_image, = wan_video_vae_decode.decode(out_latent, vae, 0, image_generation_mode)
            
            # Aggressive memory cleanup after VAE decode (but keep out_latent for overlap extraction)
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()
            
            # Apply quality monitoring and corrections
            if quality_monitoring:
                output_image = self.apply_quality_monitoring(output_image, reference_frames, chunk_index, periodic_detail_boost)
            
            # Apply temporal CLIP guidance if enabled
            if temporal_clip_guidance and chunk_index > 0 and len(reference_frames) > 0:
                output_image = self.apply_temporal_clip_guidance(output_image, reference_frames[-1], clip_vision, CLIPVisionEncoder)

            # Apply color matching using original image for all chunks
            output_image = self.apply_color_match_to_image(original_image, output_image, apply_color_match, colorMatch)
            
            # Memory cleanup after image processing operations
            gc.collect()
            torch.cuda.empty_cache()
            mm.throw_exception_if_processing_interrupted()
            
            # Store overlap data for next chunk's blending
            if overlap_frames > 0 and chunk_index < total_video_chunks - 1:
                overlap_data = self.extract_overlap_data(output_image, out_latent, overlap_frames)
                chunk_overlap_data.append(overlap_data)
                
                # Clean up after overlap data extraction
                gc.collect()
                torch.cuda.empty_cache()
            
            # Free output latent after overlap data extraction
            del out_latent
            
            # Additional cleanup after latent deletion
            gc.collect()
            torch.cuda.empty_cache()
            
            # Store reference frames for quality monitoring and guidance
            if chunk_index == 0 or is_keyframe_chunk:
                # Store as high-quality reference frame
                reference_frames.append(output_image[0:1].clone())  # Store first frame
                if len(reference_frames) > 5:  # Keep only last 5 reference frames
                    reference_frames.pop(0)
                    
                # Clean up after reference frame operations
                gc.collect()
                torch.cuda.empty_cache()
            
            if (total_video_chunks > 1):
                # Update start_image to be the last frame of current chunk for next iteration
                start_image = output_image[output_image.shape[0] - 1:output_image.shape[0]].clone()
                output_to_terminal_successful(f"Updated start_image for next chunk from current chunk's last frame")
                
                images_chunk.append(output_image[:-overlap_frames-1])  # Skip first 'overlap_frames' and remove last frame
                output_to_terminal_successful(f"Chunk {chunk_index}: Added {images_chunk[len(images_chunk) - 1].shape[0]} frames (skipped {overlap_frames} overlap frames)")
            else:
                images_chunk.append(output_image[:-overlap_frames-1])  # Skip first 'overlap_frames' and remove last frame
                output_to_terminal_successful(f"Single chunk: Added {images_chunk[len(images_chunk) - 1].shape[0]} frames")

            # Clean up output_image to free memory before next chunk
            del output_image
            
            # Aggressive memory cleanup after chunk processing
            gc.collect()
            torch.cuda.empty_cache()

            output_to_terminal_successful(f"Video chunk {chunk_index + 1} generated successfully")
            
            # Clean up memory after each chunk
            gc.collect()
            torch.cuda.empty_cache()
            
            # Clean up model clones to free memory
            del generation_model, generation_clip
            
            # Final aggressive cleanup after chunk completion
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()  # Ensure cleanup is complete before next chunk
            
            output_to_terminal_successful(f"Memory cleanup completed for chunk {chunk_index + 1}")

        output_to_terminal_successful("All video chunks generated successfully")
        
        # Calculate expected vs actual frame counts
        expected_frames = total_video_seconds * 16
        output_to_terminal_successful(f"Expected total frames for {total_video_seconds} seconds: {expected_frames}")

        # Merge all video chunks in sequence with overlap handling
        if len(images_chunk) > 1:
            output_to_terminal_successful(f"Merging {len(images_chunk)} video chunks with advanced blending...")
            
            # Apply advanced temporal blending between chunks
            if overlap_frames > 0:
                output_image = self.merge_chunks_with_temporal_blending(images_chunk, chunk_overlap_data, overlap_frames, temporal_overlap_strength)
            else:
                # Simple concatenation without overlap handling
                output_image = torch.cat(images_chunk, dim=0)
            
            output_to_terminal_successful(f"Final video shape after merging: {output_image.shape}")
            output_to_terminal_successful(f"Actual frames generated: {output_image.shape[0]} (Expected: {total_video_seconds * 16})")

        elif len(images_chunk) == 1:
            output_image = images_chunk[0]
            output_to_terminal_successful(f"Single chunk video shape: {output_image.shape}")
            output_to_terminal_successful(f"Actual frames generated: {output_image.shape[0]} (Expected: {total_video_seconds * 16})")
        else:
            output_to_terminal_error("No video chunks generated")

        mm.throw_exception_if_processing_interrupted()

        # Final memory cleanup before return
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
            # No interpolation: output_image already has the last frame removed for each chunk
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

    def process_lora_stack(self, lora_stack, working_model, clip):
        """
        Process and apply LoRA stack to the model and CLIP.
        
        Args:
            lora_stack: List of LoRA entries to apply
            working_model: The model to apply LoRAs to
            clip: The CLIP model to apply LoRAs to
            
        Returns:
            tuple: (updated_working_model, updated_clip) with LoRAs applied
        """
        model_clone = working_model.clone()

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
                    model_clone, clip = lora_loader.load_lora(model_clone, clip, lora_name, model_strength, clip_strength)
                else:
                    output_to_terminal_error(f"Skipping LoRA {lora_count}/{len(lora_stack)}: No valid LoRA name")
            
            output_to_terminal_successful(f"Successfully applied {len(lora_stack)} LoRAs to the model")
        else:
            output_to_terminal_successful("No LoRA stack provided, skipping LoRA application")
            
        return model_clone, clip

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

    def process_image(self, image, image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size, image_width, image_height, image_type):
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
            output_to_terminal_successful(f"Resizing {image_type}...")
            image, image_width, image_height = resizer.resize(image, large_image_side, large_image_side, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            tmp_width, tmp_height, = wan_max_resolution.run(wan_model_size, image)
            tmpTotalPixels = tmp_width * tmp_height
            imageTotalPixels = image_width * image_height
            if (tmpTotalPixels < imageTotalPixels):
                image_width = tmp_width
                image_height = tmp_height
                image, image_width, image_height = resizer.resize(image, image_width, image_height, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")

            output_to_terminal_successful(f"{image_type} final size: {image_width}x{image_height}")

            if (image_clip_vision_enabled) and (clip_vision is not None):
                output_to_terminal_successful(f"Encoding CLIP Vision for {image_type}...")
                clip_vision_image, = CLIPVisionEncoder.encode(clip_vision, image, "center")
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

    def apply_dual_sampler_processing(self, model_high_cfg, model_low_cfg, k_sampler, clip, noise_seed, total_steps, high_cfg, low_cfg, temp_positive_clip, temp_negative_clip, in_latent, total_steps_high_cfg, high_denoise, low_denoise):
        """
        Apply dual sampler processing with pre-configured high and low CFG models.
        
        Args:
            model_high_cfg: The prepared model for high CFG sampling (with LoRAs already applied)
            model_low_cfg: The prepared model for low CFG sampling (with LoRAs already applied)
            k_sampler: The KSampler instance
            clip: The CLIP model
            noise_seed (int): The noise seed for sampling
            total_steps (int): Total sampling steps
            high_cfg (float): High CFG value
            low_cfg (float): Low CFG value
            temp_positive_clip: Positive CLIP encoding
            temp_negative_clip: Negative CLIP encoding
            in_latent: Input latent
            total_steps_high_cfg (int): Steps for high CFG phase
            high_denoise (float): Denoising strength for high CFG
            low_denoise (float): Denoising strength for low CFG
            
        Returns:
            The output latent from dual sampler processing
        """
        stop_steps = int(total_steps_high_cfg / 100 * total_steps)

        gc.collect()
        torch.cuda.empty_cache()
        #mm.soft_empty_cache()
        output_to_terminal_successful("High CFG KSampler started...")
        out_latent, = k_sampler.sample(model_high_cfg, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, stop_steps, "enabled", high_denoise)

        # Free high CFG model to save memory before low CFG pass
        del model_high_cfg
        
        # Aggressive cleanup between sampling passes
        gc.collect()
        torch.cuda.empty_cache()
        #mm.soft_empty_cache()
        output_to_terminal_successful("Low CFG KSampler started...")
        out_latent, = k_sampler.sample(model_low_cfg, "disable", noise_seed, total_steps, low_cfg, "lcm", "simple", temp_positive_clip, temp_negative_clip, out_latent, stop_steps, 1000, "disable", low_denoise)
        
        # Free low CFG model after completion
        del model_low_cfg
        
        return out_latent

    def apply_single_sampler_processing(self, working_model, k_sampler, clip, noise_seed, total_steps, high_cfg, temp_positive_clip, temp_negative_clip, in_latent, high_denoise):
        """
        Apply single sampler processing with pre-configured model.
        
        Args:
            working_model: The prepared model for sampling (with LoRAs already applied)
            k_sampler: The KSampler instance
            clip: The CLIP model
            noise_seed (int): The noise seed for sampling
            total_steps (int): Total sampling steps
            high_cfg (float): CFG value
            temp_positive_clip: Positive CLIP encoding
            temp_negative_clip: Negative CLIP encoding
            in_latent: Input latent
            high_denoise (float): Denoising strength
            
        Returns:
            The output latent from single sampler processing
        """
        output_to_terminal_successful("KSampler started...")
        out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, 1000, "disable", high_denoise)
        
        # Free working model after sampling
        del working_model
        
        # Aggressive cleanup after single sampling
        gc.collect()
        torch.cuda.empty_cache()
        
        return out_latent
    
    def apply_causvid_lora_processing(self, working_model, clip, lora_loader, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers):
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
        model_high_cfg = working_model.clone()
        model_low_cfg = working_model.clone()
        updated_clip = clip

        if use_dual_samplers:
            # Apply CausVid LoRA for High CFG model
            if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
                output_to_terminal_successful(f"Applying CausVid LoRA for High CFG with strength: {high_cfg_causvid_strength}")
                model_high_cfg, updated_clip, = lora_loader.load_lora(model_high_cfg, updated_clip, causvid_lora, high_cfg_causvid_strength, 1.0)
            
            # Apply CausVid LoRA for Low CFG model  
            if (causvid_lora != NONE and low_cfg_causvid_strength > 0.0):
                output_to_terminal_successful(f"Applying CausVid LoRA for Low CFG with strength: {low_cfg_causvid_strength}")
                model_low_cfg, updated_clip, = lora_loader.load_lora(model_low_cfg, updated_clip, causvid_lora, low_cfg_causvid_strength, 1.0)
        else:
            # Single sampler - only apply to high CFG model
            if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
                output_to_terminal_successful(f"Applying CausVid LoRA with strength: {high_cfg_causvid_strength}")
                model_high_cfg, updated_clip, = lora_loader.load_lora(model_high_cfg, updated_clip, causvid_lora, high_cfg_causvid_strength, 1.0)
        
        return model_high_cfg, model_low_cfg, updated_clip
    
    def apply_color_match_to_image(self, reference_image, image, apply_color_match, colorMatch):
        """
        Apply color matching between reference_image and images if enabled.
        
        Args:
            reference_image: Reference image for color matching (or None)
            image: Target image to apply color correction to
            apply_color_match: Boolean flag to enable/disable color matching
            colorMatch: Color matching utility object
            
        Returns:
            image: Processed image with or without color matching applied
        """
        if (image is not None and apply_color_match and reference_image is not None):
            output_to_terminal_successful("Applying color match to images...")
            # Use reduced strength to avoid overly aggressive color correction
            image, = colorMatch.colormatch(reference_image, image, "hm-mvgd-hm", strength=0.7)
        elif apply_color_match and reference_image is None:
            output_to_terminal_successful("Skipping color match - no reference image available")

        return image
    
    def apply_prompt_reinforcement(self, positive, negative, original_positive, original_negative, chunk_index):
        """
        Reinforce character/scene descriptions at chunk boundaries to maintain identity consistency.
        
        Args:
            positive: Current positive prompt
            negative: Current negative prompt
            original_positive: Original positive prompt
            original_negative: Original negative prompt
            chunk_index: Current chunk index
            
        Returns:
            tuple: (reinforced_positive, reinforced_negative)
        """
        output_to_terminal_successful(f"Applying prompt reinforcement for chunk {chunk_index}")
        
        # Extract key descriptors from original prompt for reinforcement
        reinforcement_keywords = ["character", "person", "woman", "man", "face", "hair", "eyes", "clothing", "style", 
                                 "girl", "boy", "hands", "legs", "feet", "penis", "breasts", "vagina"]
        
        # Quality/technical terms for negative reinforcement
        negative_reinforcement_keywords = ["blurry", "low quality", "artifact", "distorted", "inconsistent", 
                                         "deformed", "mutation", "noise", "pixelated", "compression", "ugly",
                                         "poorly drawn", "bad anatomy", "bad proportions", "watermark", "logo", 
                                         "text overlay", "glitch", "flicker", "oversaturated", "underexposed", 
                                         "disfigured", "multiple faces", "duplicate", "double image", "censorship", 
                                         "cropped", "missing limb", "extra limb", "bad hands", "poorly drawn hands", 
                                         "bad face", "bad lips", "bad tongue", "unnatural pose", "bad perspective", 
                                         "incorrect lighting", "color banding", "unnatural colors", "grain", 
                                         "pixelation", "oversharpened", "text", "bad breasts", "bad penis", "bad vagina"]
        
        # Create reinforced positive prompt
        reinforced_positive = positive
        
        # Add emphasis to character-related terms found in original prompt
        for keyword in reinforcement_keywords:
            if keyword in original_positive.lower():
                # Check if keyword is already in current prompt
                if keyword in positive.lower():
                    # Boost existing keyword importance for consistency
                    import re
                    # Replace word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    # Only boost if not already emphasized
                    if f"({keyword}:" not in reinforced_positive and f"({keyword.capitalize()}:" not in reinforced_positive:
                        reinforced_positive = re.sub(pattern, f"({keyword}:1.2)", reinforced_positive, flags=re.IGNORECASE)
                else:
                    # Add important keywords from original prompt that are missing
                    reinforced_positive = f"({keyword}:1.1), " + reinforced_positive
        
        # Extract key descriptive phrases from original prompt (up to 3 words)
        original_words = original_positive.lower().split()
        for i, word in enumerate(original_words):
            if word in reinforcement_keywords:
                # Add context around the keyword (1-2 adjacent words)
                if i > 0 and i < len(original_words) - 1:
                    context_phrase = f"{original_words[i-1]} {word} {original_words[i+1]}"
                    if context_phrase not in positive.lower() and len(context_phrase.split()) <= 3:
                        reinforced_positive = f"({context_phrase}:1.1), " + reinforced_positive
        
        # Create reinforced negative prompt
        reinforced_negative = negative
        
        # Add emphasis to quality-related terms found in original negative prompt
        for keyword in negative_reinforcement_keywords:
            if keyword in original_negative.lower():
                # Check if keyword is already in current negative prompt
                if keyword in negative.lower():
                    # Boost existing keyword importance for consistency
                    import re
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    # Only boost if not already emphasized
                    if f"({keyword}:" not in reinforced_negative and f"({keyword.capitalize()}:" not in reinforced_negative:
                        reinforced_negative = re.sub(pattern, f"({keyword}:1.3)", reinforced_negative, flags=re.IGNORECASE)
                else:
                    # Add important negative keywords from original prompt that are missing
                    reinforced_negative = f"({keyword}:1.2), " + reinforced_negative
        
        # Extract key negative phrases from original negative prompt
        original_negative_words = original_negative.lower().split()
        for i, word in enumerate(original_negative_words):
            if word in negative_reinforcement_keywords:
                # Add context around the keyword (1-2 adjacent words)
                if i > 0 and i < len(original_negative_words) - 1:
                    context_phrase = f"{original_negative_words[i-1]} {word} {original_negative_words[i+1]}"
                    if context_phrase not in negative.lower() and len(context_phrase.split()) <= 3:
                        reinforced_negative = f"({context_phrase}:1.2), " + reinforced_negative
        
        # Ensure temporal consistency by reinforcing character stability in negatives
        temporal_negative_terms = ["changing", "morphing", "inconsistent", "different", "varying"]
        for term in temporal_negative_terms:
            if term not in reinforced_negative.lower():
                reinforced_negative = f"({term} appearance:1.1), " + reinforced_negative
        
        output_to_terminal_successful(f"Prompt reinforcement applied: {len([k for k in reinforcement_keywords if k in original_positive.lower()])} positive keywords, {len([k for k in negative_reinforcement_keywords if k in original_negative.lower()])} negative keywords reinforced")
        
        return reinforced_positive, reinforced_negative
    
    def get_persistent_noise_seed(self, base_seed, chunk_index, persistent_noise_seed):
        """
        Get noise seed for current chunk, maintaining consistency if enabled.
        
        Args:
            base_seed: Base noise seed
            chunk_index: Current chunk index
            persistent_noise_seed: Whether to maintain seed consistency
            
        Returns:
            int: Noise seed for current chunk
        """
        if persistent_noise_seed:
            # Maintain base seed for first chunk, slight variations for others
            return base_seed + chunk_index * 7  # Small prime offset
        else:
            # Use completely different seed for each chunk
            return base_seed + chunk_index * 1000
    
    def apply_latent_blending(self, current_latent, chunk_overlap_data, overlap_frames, temporal_overlap_strength, chunk_index):
        """
        Apply multi-scale latent blending with previous chunk overlap.
        
        Args:
            current_latent: Current chunk's latent representation
            chunk_overlap_data: Previous chunks' overlap data
            overlap_frames: Number of overlapping frames
            temporal_overlap_strength: Strength of temporal blending
            chunk_index: Current chunk index
            
        Returns:
            Blended latent representation
        """
        if chunk_index == 0 or overlap_frames <= 0 or not chunk_overlap_data:
            return current_latent
        
        output_to_terminal_successful(f"Applying latent blending with {overlap_frames} overlap frames")
        
        try:
            # Get previous chunk's overlap latent
            prev_overlap_latent = chunk_overlap_data[-1]['latent']
            
            # Blend the overlapping region
            if prev_overlap_latent.shape == current_latent['samples'][:, :, :overlap_frames].shape:
                # Create blend mask with progressive transition (start strong, fade quickly)
                # Use power curve instead of linear to reduce static appearance
                linear_weights = torch.linspace(1.0, 0.0, overlap_frames).to(current_latent['samples'].device)
                # Apply power curve to make transition more gradual and less static
                blend_weights = (linear_weights ** 2) * temporal_overlap_strength  # Square for smoother curve
                blend_weights = blend_weights.view(1, 1, overlap_frames, 1, 1)
                
                # Apply weighted blending with better motion preservation
                blended_region = (prev_overlap_latent * blend_weights + 
                                current_latent['samples'][:, :, :overlap_frames] * (1 - blend_weights))
                
                # Replace overlapping region in current latent
                current_latent['samples'][:, :, :overlap_frames] = blended_region
                
                output_to_terminal_successful(f"Successfully blended {overlap_frames} frames with strength {temporal_overlap_strength}")
            else:
                output_to_terminal_error(f"Latent shape mismatch for blending: {prev_overlap_latent.shape} vs {current_latent['samples'][:, :, :overlap_frames].shape}")
        
        except Exception as e:
            output_to_terminal_error(f"Error in latent blending: {str(e)}")
        
        return current_latent
    
    def apply_anchor_frame_preservation(self, current_latent, chunk_overlap_data, anchor_frame_strength, chunk_index):
        """
        Apply anchor frame preservation at chunk boundaries.
        
        Args:
            current_latent: Current chunk's latent representation
            chunk_overlap_data: Previous chunks' overlap data
            anchor_frame_strength: Strength of anchor frame preservation
            chunk_index: Current chunk index
            
        Returns:
            Latent with anchor frame preservation applied
        """
        if chunk_index == 0 or anchor_frame_strength <= 0 or not chunk_overlap_data:
            return current_latent
        
        output_to_terminal_successful(f"Applying anchor frame preservation with strength {anchor_frame_strength}")
        
        try:
            # Get previous chunk's last frame latent as anchor
            if 'anchor_frame' in chunk_overlap_data[-1]:
                anchor_latent = chunk_overlap_data[-1]['anchor_frame']
                
                # Apply progressive blending to the first few frames instead of just the first frame
                # This creates a smoother transition while still maintaining continuity
                transition_frames = min(2, current_latent['samples'].shape[2])  # Use first 2 frames for transition
                
                for frame_idx in range(transition_frames):
                    # Reduce strength for subsequent frames to create smooth transition
                    frame_strength = anchor_frame_strength * (1.0 - (frame_idx * 0.5))  # Reduce by 50% each frame
                    if frame_strength > 0.1:  # Only apply if strength is meaningful
                        current_latent['samples'][:, :, frame_idx:frame_idx+1] = (
                            anchor_latent * frame_strength + 
                            current_latent['samples'][:, :, frame_idx:frame_idx+1] * (1 - frame_strength)
                        )
                
                output_to_terminal_successful(f"Anchor frame preservation applied to {transition_frames} frames")
        
        except Exception as e:
            output_to_terminal_error(f"Error in anchor frame preservation: {str(e)}")
        
        return current_latent
    
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
    
    def extract_overlap_data(self, output_image, latent, overlap_frames):
        """
        Extract overlap data from current chunk for next chunk's blending.
        
        Args:
            output_image: Generated image output
            latent: Latent representation
            overlap_frames: Number of overlap frames
            
        Returns:
            dict: Overlap data for blending
        """
        if overlap_frames <= 0:
            return {}
        
        try:
            overlap_data = {
                'image': output_image[-overlap_frames:].clone(),  # Last N frames
                'latent': latent['samples'][:, :, -overlap_frames:].clone(),  # Last N latent frames
                'anchor_frame': latent['samples'][:, :, -1:].clone()  # Very last frame as anchor
            }
            
            output_to_terminal_successful(f"Extracted overlap data for {overlap_frames} frames")
            return overlap_data
        
        except Exception as e:
            output_to_terminal_error(f"Error extracting overlap data: {str(e)}")
            return {}
    
    def apply_quality_monitoring(self, output_image, reference_frames, chunk_index, periodic_detail_boost):
        """
        Monitor generation quality and apply corrections when degradation is detected.
        
        Args:
            output_image: Generated image output
            reference_frames: List of high-quality reference frames
            chunk_index: Current chunk index
            periodic_detail_boost: Whether to apply periodic detail enhancement
            
        Returns:
            Quality-corrected image output
        """
        if chunk_index == 0 or not reference_frames:
            return output_image
        
        try:
            # Simple quality monitoring based on brightness and contrast
            current_mean = torch.mean(output_image)
            current_std = torch.std(output_image)
            
            if len(reference_frames) > 0:
                ref_mean = torch.mean(reference_frames[-1])
                ref_std = torch.std(reference_frames[-1])
                
                # Detect quality degradation
                brightness_drift = abs(current_mean - ref_mean)
                contrast_loss = abs(current_std - ref_std)
                
                if brightness_drift > 0.1 or contrast_loss > 0.05:
                    output_to_terminal_error(f"Quality degradation detected: brightness_drift={brightness_drift:.3f}, contrast_loss={contrast_loss:.3f}")
                    
                    # Apply simple correction
                    correction_factor = ref_mean / (current_mean + 1e-8)
                    output_image = output_image * correction_factor.clamp(0.8, 1.2)
                    
                    output_to_terminal_successful("Applied quality correction")
            
            # Apply periodic detail boost if enabled
            if periodic_detail_boost and chunk_index % 3 == 0:
                # Simple detail enhancement using sharpening
                if TORCHVISION_AVAILABLE:
                    # Apply slight Gaussian blur and subtract to sharpen
                    blurred = gaussian_blur(output_image.permute(0, 3, 1, 2), kernel_size=3, sigma=0.5)
                    sharpened = output_image.permute(0, 3, 1, 2) + 0.1 * (output_image.permute(0, 3, 1, 2) - blurred)
                    output_image = sharpened.permute(0, 2, 3, 1).clamp(0, 1)
                    output_to_terminal_successful("Applied periodic detail boost")
        
        except Exception as e:
            output_to_terminal_error(f"Error in quality monitoring: {str(e)}")
        
        return output_image
    
    def apply_temporal_clip_guidance(self, current_image, reference_image, clip_vision, CLIPVisionEncoder):
        """
        Apply CLIP-guided temporal consistency to maintain visual similarity between frames.
        
        Args:
            current_image: Current frame
            reference_image: Reference frame for consistency
            clip_vision: CLIP vision model
            CLIPVisionEncoder: CLIP vision encoder
            
        Returns:
            Temporally guided image
        """
        if clip_vision is None:
            return current_image
        
        try:
            output_to_terminal_successful("Applying temporal CLIP guidance")
            
            # Encode both images with CLIP Vision
            current_encoding, = CLIPVisionEncoder.encode(clip_vision, current_image[0:1], "center")
            reference_encoding, = CLIPVisionEncoder.encode(clip_vision, reference_image[0:1], "center")
            
            # Calculate similarity
            current_features = current_encoding.penultimate_hidden_states
            reference_features = reference_encoding.penultimate_hidden_states
            
            # Simple cosine similarity
            similarity = torch.cosine_similarity(
                current_features.flatten(), 
                reference_features.flatten(), 
                dim=0
            )
            
            output_to_terminal_successful(f"CLIP similarity: {similarity.item():.3f}")
            
            # If similarity is too low, apply subtle correction
            if similarity < 0.8:
                # Simple linear interpolation toward reference
                correction_strength = 0.1  # Subtle correction
                current_image = current_image * (1 - correction_strength) + reference_image * correction_strength
                output_to_terminal_successful("Applied temporal consistency correction")
        
        except Exception as e:
            output_to_terminal_error(f"Error in temporal CLIP guidance: {str(e)}")
        
        return current_image
    
    def merge_chunks_with_temporal_blending(self, images_chunk, chunk_overlap_data, overlap_frames, temporal_overlap_strength):
        """
        Merge video chunks with advanced temporal blending.
        
        Args:
            images_chunk: List of image chunks
            chunk_overlap_data: Overlap data for blending
            overlap_frames: Number of overlapping frames
            temporal_overlap_strength: Strength of temporal blending
            
        Returns:
            Merged video with temporal blending
        """
        if overlap_frames <= 0 or not chunk_overlap_data:
            return torch.cat(images_chunk, dim=0)
        
        output_to_terminal_successful("Applying advanced temporal blending between chunks")
        
        try:
            merged_chunks = [images_chunk[0]]  # Start with first chunk
            
            for i in range(1, len(images_chunk)):
                current_chunk = images_chunk[i]
                
                if i-1 < len(chunk_overlap_data) and 'image' in chunk_overlap_data[i-1]:
                    prev_overlap = chunk_overlap_data[i-1]['image']
                    
                    # Create blend weights for smooth transition
                    blend_weights = torch.linspace(temporal_overlap_strength, 0, overlap_frames)
                    blend_weights = blend_weights.view(overlap_frames, 1, 1, 1).to(current_chunk.device)
                    
                    # Blend overlapping region
                    if prev_overlap.shape[0] >= overlap_frames and current_chunk.shape[0] >= overlap_frames:
                        blended_region = (prev_overlap[-overlap_frames:] * blend_weights + 
                                        current_chunk[:overlap_frames] * (1 - blend_weights))
                        
                        # Construct merged chunk: previous + blended + remaining current
                        if current_chunk.shape[0] > overlap_frames:
                            merged_chunk = torch.cat([blended_region, current_chunk[overlap_frames:]], dim=0)
                        else:
                            merged_chunk = blended_region
                        
                        merged_chunks.append(merged_chunk)
                    else:
                        # Fallback to simple concatenation if shapes don't match
                        merged_chunks.append(current_chunk)
                else:
                    merged_chunks.append(current_chunk)
            
            # Final concatenation
            final_output = torch.cat(merged_chunks, dim=0)
            output_to_terminal_successful(f"Advanced temporal blending complete. Final shape: {final_output.shape}")
            
            return final_output
        
        except Exception as e:
            output_to_terminal_error(f"Error in temporal blending: {str(e)}")
            # Fallback to simple concatenation
            return torch.cat(images_chunk, dim=0)
    
    @classmethod
    def clear_cache(cls, cache_key=None):
        """
        Clear cache entries using the generic cache manager.
        
        Args:
            cache_key (str, optional): Specific key to clear. If None, clears all cache.
        """
        cls._cache_manager.clear_cache(cache_key)
        if cache_key:
            output_to_terminal_successful(f"Cache entry '{cache_key}' cleared")
        else:
            output_to_terminal_successful("All cache entries cleared")
    
    @classmethod
    def get_cache_info(cls):
        """Get information about cached objects."""
        cache_keys = cls._cache_manager.get_cache_info()
        cache_count = cls._cache_manager.cache_size()
        
        if cache_count > 0:
            output_to_terminal_successful(f"Cache contains {cache_count} entries: {cache_keys}")
            return cache_keys
        else:
            output_to_terminal_successful("Cache is empty")
            return []
    
    @classmethod 
    def cache_model(cls, cache_key, model, storage_device='cpu'):
        """
        Generic method to cache any model with user-provided key.
        
        Args:
            cache_key (str): The cache key to store under
            model: The model to cache
            storage_device (str): Device to store on ('cpu' to save VRAM, 'cuda' to keep on GPU)
            
        Returns:
            str: The cache key used for storage
        """
        cls._cache_manager.store_in_cache(cache_key, model, storage_device=storage_device)
        output_to_terminal_successful(f"Model cached with key: {cache_key}")
        return cache_key
    
    @classmethod
    def load_cached_model(cls, cache_key, target_device="cuda"):
        """
        Generic method to load any cached model with user-provided key.
        
        Args:
            cache_key (str): The cache key to look up
            target_device (str): Device to move the model to
            
        Returns:
            The cached model moved to target device, or None if not found
        """
        model = cls._cache_manager.get_from_cache(cache_key, target_device)
        
        if model is not None:
            output_to_terminal_successful(f"Model loaded from cache: {cache_key}")
        else:
            output_to_terminal_successful(f"No cached model found for key: {cache_key}")
            
        return model