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
from .wan_first_last_first_frame_to_video import WanFirstLastFirstFrameToVideo
from .wan_video_vae_decode import WanVideoVaeDecode
from .wan_get_max_image_resolution_by_aspect_ratio import WanGetMaxImageResolutionByAspectRatio
from .wan_video_enhance_a_video import WanVideoEnhanceAVideo

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

imported_nodes.update(teacache_imports)
imported_nodes.update(kjnodes_imports)
imported_nodes.update(gguf_imports)
imported_nodes.update(wanblockswap)

TeaCache = imported_nodes.get("TeaCache")
SkipLayerGuidanceWanVideo = imported_nodes.get("SkipLayerGuidanceWanVideo")
UnetLoaderGGUF = imported_nodes.get("UnetLoaderGGUF")
SageAttention = imported_nodes.get("PathchSageAttentionKJ")
WanVideoBlockSwap = imported_nodes.get("WanVideoBlockSwap")
ImageResizeKJv2 = imported_nodes.get("ImageResizeKJv2")
ModelPatchTorchSettings = imported_nodes.get("ModelPatchTorchSettings")
ColorMatch = imported_nodes.get("ColorMatch")


class WanImageToVideoAdvancedSampler:
    @classmethod
    def INPUT_TYPES(s):
        
        clip_names = [NONE] + folder_paths.get_filename_list("text_encoders")
        gguf_model_names = [NONE] + folder_paths.get_filename_list("unet_gguf")
        diffusion_models_names = [NONE] + folder_paths.get_filename_list("diffusion_models")
        vae_names = [NONE] + folder_paths.get_filename_list("vae")
        clip_vision_models = [NONE] + folder_paths.get_filename_list("clip_vision")        
        lora_names = [NONE] + folder_paths.get_filename_list("loras")

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
                # 🔄 QUALITY PRESERVATION FOR MULTI-CHUNK GENERATION
                # ═════════════════════════════════════════════════════════════════
                ("enable_quality_preservation", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable advanced quality preservation techniques for multi-chunk video generation. Reduces degradation over time."})),
                ("temporal_overlap_frames", ("INT", {"default": 3, "min": 1, "max": 8, "step": 1, "advanced": True, "tooltip": "Number of frames to overlap between chunks for temporal coherence. More frames = smoother transitions but slower generation."})),
                ("latent_blend_strength", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Strength of latent space blending between chunks. Higher values preserve more continuity but may reduce variation."})),
                ("artifact_reduction", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Apply gaussian blur artifact reduction to transition frames. Helps reduce compression artifacts between chunks."})),
                
                # ═════════════════════════════════════════════════════════════════
                # 👁️ CLIP VISION SETTINGS
                # ═════════════════════════════════════════════════════════════════
                ("clip_vision_model", (clip_vision_models, {"default": NONE, "advanced": True, "tooltip": "CLIP Vision model for processing input images. Required for image-to-video generation."})),
                ("clip_vision_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Strength of CLIP vision influence on the generation. Higher values make the output more similar to input images."})),
                ("start_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the start image. If disabled, the start image will be used as a static frame."})),
                ("end_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the end image. If disabled, the end image will be used as a static frame."})),
                
                # ═════════════════════════════════════════════════════════════════
                # 🎯 FEATURE CONSISTENCY SETTINGS
                # ═════════════════════════════════════════════════════════════════
                ("feature_consistency_strength", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Strength of feature consistency enforcement. Higher values better preserve all body features including face, body shape, hands, feet, genitals, tattoos, jewelry, clothing patterns across frames."})),
                ("reference_frame_interval", ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "advanced": True, "tooltip": "How often to re-inject reference image features during generation. Lower values = more consistency but may reduce motion."})),
                ("detail_preservation_mode", (["disabled", "light", "medium", "strong"], {"default": "medium", "advanced": True, "tooltip": "Detail preservation mode. Strong mode best preserves features but may reduce motion fluidity."})),
                
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
                
                # ═════════════════════════════════════════════════════════════════
                # 🚀 MEMORY OPTIMIZATION
                # ═════════════════════════════════════════════════════════════════
                ("enable_aggressive_memory_optimization", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable aggressive VRAM optimization. Reduces memory usage at the cost of some performance."})),
            ]),
            "optional": OrderedDict([
                ("lora_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of LoRAs to apply to the diffusion model. Each LoRA modifies the model's behavior."})),
                ("prompt_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of prompts to apply to the diffusion model on each chunck generated. If there is less prompts than chunks, the last prompt will be used for the remaining chunks."})),
                ("start_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "Start image for the video generation process."})),
                ("end_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "End image for the video generation process."})),
            ]),
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN",)
    RETURN_NAMES = ("IMAGE", "IS_CHUNCK",)
    OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.22, tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda", use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=2.0, use_block_swap=True, block_swap=35, large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size=WAN_720P, total_video_seconds=1, total_video_chunks=1, enable_quality_preservation=True, temporal_overlap_frames=3, latent_blend_strength=0.2, artifact_reduction=True, clip_vision_model=NONE, clip_vision_strength=1.0, use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0, lora_stack=None, start_image=None, start_image_clip_vision_enabled=True, end_image=None, end_image_clip_vision_enabled=True, video_enhance_enabled=True, use_cfg_zero_star=True, apply_color_match=True, causvid_lora=NONE, high_cfg_causvid_strength=1.0, low_cfg_causvid_strength=1.0, high_denoise=1.0, low_denoise=1.0, prompt_stack=None, feature_consistency_strength=0.8, reference_frame_interval=3, detail_preservation_mode="medium", enable_aggressive_memory_optimization=True, fill_noise_latent=0.5):
        # Aggressive memory optimization setup
        if enable_aggressive_memory_optimization:
            output_to_terminal_successful("Enabling aggressive memory optimization...")
            
            # Force aggressive garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
            
            # Set memory fraction if available
            if hasattr(torch.cuda, 'set_memory_fraction'):
                try:
                    torch.cuda.set_memory_fraction(0.9)  # Use 90% of available VRAM
                    output_to_terminal_successful("Set CUDA memory fraction to 0.9")
                except Exception as e:
                    output_to_terminal_error(f"Failed to set memory fraction: {e}")
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                output_to_terminal_successful("Enabled efficient attention backends")
        
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()
        mm.throw_exception_if_processing_interrupted()
        model = self.load_model(GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype)
        output_to_terminal_successful("Loading VAE...")
        vae, = nodes.VAELoader().load_vae(vae)

        output_to_terminal_successful("Loading CLIP...")
        clip, = nodes.CLIPLoader().load_clip(clip, clip_type, clip_device)
        clip_set_last_layer = nodes.CLIPSetLastLayer()
        clip, = clip_set_last_layer.set_last_layer(clip, -1)  # Use all layers but truncate tokens

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

        for result in self.postprocess(model, vae, clip, clip_type, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, enable_quality_preservation, temporal_overlap_frames, latent_blend_strength, artifact_reduction, prompt_stack, feature_consistency_strength, reference_frame_interval, detail_preservation_mode, enable_aggressive_memory_optimization, fill_noise_latent):
            return result
        else:
            return self.postprocess(model, vae, clip, clip_type, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, enable_quality_preservation, temporal_overlap_frames, latent_blend_strength, artifact_reduction, prompt_stack, feature_consistency_strength, reference_frame_interval, detail_preservation_mode, enable_aggressive_memory_optimization, fill_noise_latent)

    def postprocess(self, model, vae, clip, clip_type, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks, enable_quality_preservation, temporal_overlap_frames, latent_blend_strength, artifact_reduction, prompt_stack, feature_consistency_strength, reference_frame_interval, detail_preservation_mode, enable_aggressive_memory_optimization, fill_noise_latent):

        output_to_terminal_successful("Generation started...")

        working_model = model.clone()
        k_sampler = nodes.KSamplerAdvanced()
        text_encode = nodes.CLIPTextEncode()
        wan_image_to_video = WanFirstLastFirstFrameToVideo()
        wan_video_vae_decode = WanVideoVaeDecode()
        wan_max_resolution = WanGetMaxImageResolutionByAspectRatio()
        CLIPVisionLoader = nodes.CLIPVisionLoader()
        CLIPVisionEncoder = nodes.CLIPVisionEncode()
        resizer = ImageResizeKJv2()
        image_width = 512
        image_height = 512
        in_latent = None
        out_latent = None
        total_frames = (total_video_seconds * 16) + 1
        lora_loader = nodes.LoraLoader()
        wanVideoEnhanceAVideo = WanVideoEnhanceAVideo()
        cfgZeroStar = CFGZeroStar()
        colorMatch = ColorMatch()

        mm.throw_exception_if_processing_interrupted()

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

        # Process LoRA stack
        working_model, clip = self.process_lora_stack(lora_stack, working_model, clip)
        mm.throw_exception_if_processing_interrupted()

        # Generate video chunks sequentially with quality preservation
        images_chunck = []
        original_image = start_image if start_image is not None else None
        last_latent = None  # Store latent for continuity
        reference_clip_vision = None  # Store reference CLIP vision features for consistency
        
        # Prepare reference features for consistency
        if feature_consistency_strength > 0 and original_image is not None and clip_vision is not None:
            try:
                reference_clip_vision, = CLIPVisionEncoder.encode(clip_vision, original_image, "center")
                output_to_terminal_successful(f"Reference CLIP vision features extracted for consistency (strength: {feature_consistency_strength})")
            except Exception as e:
                output_to_terminal_error(f"Failed to extract reference features: {e}")
                reference_clip_vision = None
        mm.throw_exception_if_processing_interrupted()
        
        for chunk_index in range(total_video_chunks):
            output_to_terminal_successful(f"Generating video chunk {chunk_index + 1}/{total_video_chunks}...")
            mm.throw_exception_if_processing_interrupted()
            
            # Aggressive memory cleanup before each chunk
            if enable_aggressive_memory_optimization:
                # Clean up previous chunk data
                if chunk_index > 0:
                    # Explicitly delete variables that might hold GPU memory
                    for var_name in ['output_image', 'out_latent', 'in_latent', 'temp_positive_clip', 'temp_negative_clip']:
                        if var_name in locals():
                            del locals()[var_name]
                    
                    # Force garbage collection and cache clearing
                    gc.collect()
                    torch.cuda.empty_cache()
                    mm.soft_empty_cache()
                    
                    # Report memory status
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        max_memory = torch.cuda.max_memory_allocated() / 1024**3
                        output_to_terminal_successful(f"VRAM: Current {current_memory:.2f}GB, Peak {max_memory:.2f}GB")
            
            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
            output_image = None
            clip_vision = None
            clip_vision_start_image = None
            clip_vision_end_image = None
            generation_model = working_model.clone()
            generation_clip = clip.clone()

            positive, negative, loras = self.get_current_prompt(prompt_stack, chunk_index, positive, negative)
            mm.throw_exception_if_processing_interrupted()

            if (loras is not None):
                generation_model, generation_clip = self.process_lora_stack(loras, generation_model, generation_clip)
            mm.throw_exception_if_processing_interrupted()

            # Quality preservation: Use multiple frames for smoother transition
            if enable_quality_preservation and chunk_index > 0 and images_chunck:
                # Extract last few frames for better temporal coherence
                prev_chunk = images_chunck[-1]
                overlap_frames = min(temporal_overlap_frames, prev_chunk.shape[0])
                
                # Use weighted average of last frames to reduce single-frame artifacts
                if overlap_frames > 1:
                    # Take last N frames and compute weighted average (more weight to recent frames)
                    weights = torch.linspace(0.3, 1.0, overlap_frames).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    weights = weights.to(prev_chunk.device)
                    
                    last_frames = prev_chunk[-overlap_frames:]
                    weighted_frames = last_frames * weights
                    start_image = weighted_frames.mean(dim=0, keepdim=True)
                    
                    output_to_terminal_successful(f"Using weighted average of last {overlap_frames} frames for temporal coherence")
                else:
                    # Fallback to single frame with noise reduction
                    start_image = prev_chunk[-1:]
                    
                # Apply artifact reduction if enabled
                if artifact_reduction:
                    try:
                        if TORCHVISION_AVAILABLE:
                            start_image = gaussian_blur(start_image, kernel_size=3, sigma=0.5)
                            output_to_terminal_successful("Applied artifact reduction to transition frame")
                        else:
                            output_to_terminal_error("torchvision not available for artifact reduction, skipping...")
                    except Exception as e:
                        output_to_terminal_error(f"Artifact reduction failed: {e}")
            elif chunk_index > 0 and images_chunck:
                # Basic fallback: just use last frame
                start_image = images_chunck[-1][-1:]
                output_to_terminal_successful("Using simple last frame transition (quality preservation disabled)")
            mm.throw_exception_if_processing_interrupted()

            # Process start and end images with enhanced feature consistency
            start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image = self.process_start_and_end_images_with_consistency(
                start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled,
                clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
                image_generation_mode, reference_clip_vision, feature_consistency_strength, 
                chunk_index, reference_frame_interval, detail_preservation_mode
            )
            mm.throw_exception_if_processing_interrupted()

            # Apply CausVid LoRA processing for current chunk
            model_high_cfg, model_low_cfg, generation_clip = self.apply_causvid_lora_processing(
                generation_model, generation_clip, lora_loader, causvid_lora, 
                high_cfg_causvid_strength, low_cfg_causvid_strength, use_dual_samplers
            )
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Positive CLIP text...")
            
            # Enhance positive prompt with consistency keywords based on detail preservation mode
            enhanced_positive = self.enhance_prompt_for_consistency(positive, detail_preservation_mode, chunk_index)
            temp_positive_clip, = text_encode.encode(generation_clip, enhanced_positive)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Encoding Negative CLIP text...")
            
            # Enhance negative prompt to avoid inconsistencies
            enhanced_negative = self.enhance_negative_prompt_for_consistency(negative, detail_preservation_mode)
            temp_negative_clip, = text_encode.encode(generation_clip, enhanced_negative)
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Wan Image to Video started...")
            temp_positive_clip, temp_negative_clip, in_latent, = wan_image_to_video.encode(temp_positive_clip, temp_negative_clip, vae, image_width, image_height, total_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, fill_noise_latent, image_generation_mode)
            mm.throw_exception_if_processing_interrupted()

            # Enhanced latent space continuity with feature consistency
            if enable_quality_preservation and chunk_index > 0 and last_latent is not None and latent_blend_strength > 0:
                try:
                    # Extract the actual tensor from last_latent (handle both dict and tensor formats)
                    if isinstance(last_latent, dict) and 'samples' in last_latent:
                        last_latent_tensor = last_latent['samples']
                    else:
                        last_latent_tensor = last_latent
                    
                    # Extract the actual tensor from in_latent (handle both dict and tensor formats)
                    if isinstance(in_latent, dict) and 'samples' in in_latent:
                        in_latent_tensor = in_latent['samples']
                        is_in_latent_dict = True
                    else:
                        in_latent_tensor = in_latent
                        is_in_latent_dict = False
                    
                    # Blend the new latent with the last latent for smoother transition
                    if in_latent_tensor.shape == last_latent_tensor.shape:
                        # Enhanced blending with feature consistency consideration
                        effective_blend_strength = latent_blend_strength
                        
                        # Increase blending strength for better feature consistency
                        if detail_preservation_mode == "strong":
                            effective_blend_strength = min(0.6, latent_blend_strength * 1.5)
                        elif detail_preservation_mode == "medium":
                            effective_blend_strength = min(0.5, latent_blend_strength * 1.2)
                        
                        # Take more frames for stronger consistency
                        blend_frames = min(3 if detail_preservation_mode == "strong" else 2, 
                                         last_latent_tensor.shape[2])
                        
                        blended_frames = (
                            in_latent_tensor[:, :, :blend_frames] * (1 - effective_blend_strength) + 
                            last_latent_tensor[:, :, -blend_frames:] * effective_blend_strength
                        )
                        
                        # Update the latent with blended frames
                        if is_in_latent_dict:
                            in_latent['samples'][:, :, :blend_frames] = blended_frames
                        else:
                            in_latent[:, :, :blend_frames] = blended_frames
                            
                        output_to_terminal_successful(f"Applied enhanced latent space blending (strength: {effective_blend_strength:.2f}, frames: {blend_frames}, mode: {detail_preservation_mode})")
                    else:
                        output_to_terminal_error(f"Latent shape mismatch: {in_latent_tensor.shape} vs {last_latent_tensor.shape}, skipping blend")
                except Exception as e:
                    output_to_terminal_error(f"Enhanced latent blending failed, continuing without: {e}")

            mm.throw_exception_if_processing_interrupted()
            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
            if (use_dual_samplers):
                # Apply dual sampler processing
                out_latent = self.apply_dual_sampler_processing(model_high_cfg, model_low_cfg, k_sampler, generation_clip, noise_seed, total_steps, high_cfg, low_cfg, temp_positive_clip, temp_negative_clip, in_latent, total_steps_high_cfg, high_denoise, low_denoise)
            else:
                # Apply single sampler processing
                out_latent = self.apply_single_sampler_processing(model_high_cfg, k_sampler, generation_clip, noise_seed, total_steps, high_cfg, temp_positive_clip, temp_negative_clip, in_latent, high_denoise)

            # Memory cleanup after sampling
            if enable_aggressive_memory_optimization:
                # Clear intermediate variables
                del temp_positive_clip, temp_negative_clip, in_latent
                if 'model_high_cfg' in locals():
                    del model_high_cfg
                if 'model_low_cfg' in locals():
                    del model_low_cfg
                gc.collect()
                torch.cuda.empty_cache()

            # Store latent for next chunk continuity
            if enable_quality_preservation:
                # Handle both tensor and dict formats from KSampler
                if isinstance(out_latent, dict) and 'samples' in out_latent:
                    last_latent = out_latent['samples'] if out_latent['samples'] is not None else None
                elif hasattr(out_latent, 'clone'):
                    last_latent = out_latent if out_latent is not None else None
                else:
                    last_latent = out_latent  # Fallback for other types
            mm.throw_exception_if_processing_interrupted()

            output_to_terminal_successful("Vae Decode started...")
            output_image, = wan_video_vae_decode.decode(out_latent, vae, 0, image_generation_mode)

            # Enhanced color matching: Match to original OR previous chunk
            if chunk_index == 0:
                # First chunk: match to original image
                output_image = self.apply_color_match(original_image, output_image, apply_color_match, colorMatch)
            else:
                # Subsequent chunks: maintain color consistency with previous chunk
                if apply_color_match and images_chunck and enable_quality_preservation:
                    prev_reference = images_chunck[-1][-1:]  # Last frame of previous chunk
                    output_image = self.apply_color_match(prev_reference, output_image, True, colorMatch)
                    output_to_terminal_successful("Applied color matching to previous chunk for consistency")
                elif apply_color_match and original_image is not None:
                    # Fallback to original image color matching
                    output_image = self.apply_color_match(original_image, output_image, apply_color_match, colorMatch)
            mm.throw_exception_if_processing_interrupted()
            
            images_chunck.append(output_image)
            
            quality_status = "with quality preservation" if enable_quality_preservation else "standard"
            output_to_terminal_successful(f"Video chunk {chunk_index + 1} generated successfully {quality_status}")
            if chunk_index < total_video_chunks:
                yield (output_image, True,)

        output_to_terminal_successful("All video chunks generated successfully")
        # Merge all video chunks in sequence with overlap handling
        if len(images_chunck) > 1:
            output_to_terminal_successful(f"Merging {len(images_chunck)} video chunks with smart concatenation...")

            if enable_quality_preservation:
                # Smart concatenation: remove overlapping frames to avoid duplicates
                merged_chunks = [images_chunck[0]]  # Start with first chunk
                
                for i in range(1, len(images_chunck)):
                    current_chunk = images_chunck[i]
                    
                    # Skip the first frame of subsequent chunks to avoid overlap
                    # (since we used the last frame of previous chunk as start)
                    if current_chunk.shape[0] > 1:
                        merged_chunks.append(current_chunk[1:])  # Skip first frame
                        output_to_terminal_successful(f"Merged chunk {i + 1} with overlap removal")
                    else:
                        # If chunk has only 1 frame, keep it but blend with previous
                        if i > 0 and merged_chunks:
                            # Blend the single frame with the last frame for smoother transition
                            prev_last = merged_chunks[-1][-1:]
                            blended_frame = (prev_last * 0.3 + current_chunk * 0.7)
                            merged_chunks.append(blended_frame)
                            output_to_terminal_successful(f"Blended single-frame chunk {i + 1}")
                        else:
                            merged_chunks.append(current_chunk)

                # Concatenate all processed chunks after the loop
                output_image = torch.cat(merged_chunks, dim=0)
                output_to_terminal_successful(f"Final video shape after smart concatenation: {output_image.shape}")
                
                # Clean up intermediate chunks to save memory
                del merged_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Verify temporal consistency
                total_expected_frames = sum(chunk.shape[0] for chunk in images_chunck) - (len(images_chunck) - 1)
                actual_frames = output_image.shape[0]
                output_to_terminal_successful(f"Frame count: Expected ~{total_expected_frames}, Got {actual_frames}")
            else:
                # Simple concatenation without overlap handling
                output_image = torch.cat(images_chunck, dim=0)
                output_to_terminal_successful(f"Final video shape after simple concatenation: {output_image.shape}")
            
        elif len(images_chunck) == 1:
            output_image = images_chunck[0]
            output_to_terminal_successful(f"Single chunk video shape: {output_image.shape}")
        else:
            output_to_terminal_error("No video chunks generated")

        # Final memory cleanup and reporting
        if enable_aggressive_memory_optimization:
            # Clean up any remaining variables
            del images_chunck
            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
            
            # Final memory report
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                output_to_terminal_successful(f"Final VRAM usage: Current {current_memory:.2f}GB, Peak {max_memory:.2f}GB")
                
                # Reset peak memory tracker for next run
                torch.cuda.reset_peak_memory_stats()

        yield (output_image, False,)

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
                model, = gguf_loader.load_unet(unet_name=GGUF)
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

    def process_start_and_end_images_with_consistency(self, start_image, start_image_clip_vision_enabled, 
                                                    end_image, end_image_clip_vision_enabled, clip_vision, 
                                                    resizer, wan_max_resolution, CLIPVisionEncoder, 
                                                    large_image_side, wan_model_size, image_generation_mode,
                                                    reference_clip_vision, feature_consistency_strength,
                                                    chunk_index, reference_frame_interval, detail_preservation_mode):
        """
        Enhanced version of process_start_and_end_images with feature consistency support.
        
        This method maintains consistency of features like tattoos, jewelry, clothing patterns
        across video chunks by blending CLIP vision features with the original reference.
        """
        
        # First, process images normally
        start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image = self.process_start_and_end_images(
            start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled,
            clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
            image_generation_mode
        )
        
        # Apply feature consistency if enabled and we have reference features
        if (feature_consistency_strength > 0 and reference_clip_vision is not None and 
            clip_vision_start_image is not None and detail_preservation_mode != "disabled"):
            
            try:
                # Determine consistency strength based on mode and frame interval
                consistency_factor = feature_consistency_strength
                
                # Adjust based on detail preservation mode
                mode_multipliers = {
                    "light": 0.6,
                    "medium": 1.0,
                    "strong": 1.4
                }
                consistency_factor *= mode_multipliers.get(detail_preservation_mode, 1.0)
                
                # Apply periodic stronger consistency injections
                if chunk_index % reference_frame_interval == 0:
                    consistency_factor *= 1.3  # Boost consistency at intervals
                    output_to_terminal_successful(f"Applying enhanced feature consistency at frame interval (factor: {consistency_factor:.2f})")
                
                # Blend CLIP vision features for consistency
                # This preserves the original features while allowing for some motion/pose changes
                original_clip_features = clip_vision_start_image
                
                # Create a weighted blend: more original features = better consistency
                blended_features = self.blend_clip_vision_features(
                    original_clip_features, reference_clip_vision, consistency_factor
                )
                
                clip_vision_start_image = blended_features
                
                # Apply same logic to end image if it exists and is enabled
                if clip_vision_end_image is not None and end_image_clip_vision_enabled:
                    blended_end_features = self.blend_clip_vision_features(
                        clip_vision_end_image, reference_clip_vision, consistency_factor * 0.8  # Slightly less for end frame
                    )
                    clip_vision_end_image = blended_end_features
                
                output_to_terminal_successful(f"Applied feature consistency blending (strength: {consistency_factor:.2f}, mode: {detail_preservation_mode})")
                
            except Exception as e:
                output_to_terminal_error(f"Feature consistency blending failed: {e}")
                # Continue with original features if blending fails
                pass
        
        return start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image

    def blend_clip_vision_features(self, current_features, reference_features, blend_strength):
        """
        Blend CLIP vision features to maintain consistency while allowing motion.
        
        Args:
            current_features: Current frame's CLIP vision features
            reference_features: Original reference image's CLIP vision features  
            blend_strength: How much to blend (0.0 = all current, 1.0 = all reference)
            
        Returns:
            Blended CLIP vision features
        """
        try:
            # Ensure blend_strength is in valid range
            blend_strength = max(0.0, min(1.0, blend_strength))
            
            # Simple linear interpolation between features
            # This maintains feature consistency while allowing for pose/motion changes
            blended = current_features * (1.0 - blend_strength) + reference_features * blend_strength
            
            return blended
            
        except Exception as e:
            output_to_terminal_error(f"CLIP vision feature blending failed: {e}")
            return current_features  # Return original if blending fails

    def process_start_and_end_images(self, start_image, start_image_clip_vision_enabled, 
                                   end_image, end_image_clip_vision_enabled, clip_vision, 
                                   resizer, wan_max_resolution, CLIPVisionEncoder, 
                                   large_image_side, wan_model_size, image_generation_mode):
        """
        Original method for processing start and end images.
        This is kept as a separate method for backward compatibility and cleaner code.
        """
        
        image_width = 512
        image_height = 512
        clip_vision_start_image = None
        clip_vision_end_image = None
        
        # Process start image
        if start_image is not None:
            start_image, image_width, image_height, clip_vision_start_image = self.process_image(
                start_image, start_image_clip_vision_enabled, clip_vision, resizer,
                wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
                image_width, image_height, "start image"
            )
        
        # Process end image
        if end_image is not None:
            end_image, _, _, clip_vision_end_image = self.process_image(
                end_image, end_image_clip_vision_enabled, clip_vision, resizer,
                wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
                image_width, image_height, "end image"
            )
        
        return start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image

    def enhance_prompt_for_consistency(self, positive_prompt, detail_preservation_mode, chunk_index):
        """
        Enhance the positive prompt to encourage feature consistency.
        
        Args:
            positive_prompt: Original positive prompt
            detail_preservation_mode: Level of detail preservation 
            chunk_index: Current chunk being generated
            
        Returns:
            Enhanced prompt with consistency keywords
        """
        if detail_preservation_mode == "disabled":
            return positive_prompt
        
        # Base consistency keywords
        consistency_keywords = []
        
        if detail_preservation_mode == "light":
            consistency_keywords = ["consistent appearance", "same person", "identical face", "same body shape"]
        elif detail_preservation_mode == "medium":
            consistency_keywords = ["consistent appearance", "same person", "identical features", "maintaining details", 
                                  "same face", "identical body shape", "consistent hands", "same feet", "preserved body parts"]
        elif detail_preservation_mode == "strong":
            consistency_keywords = ["consistent appearance", "same person", "identical features", "maintaining details", 
                                  "same face", "identical facial features", "consistent body shape", "same body proportions",
                                  "preserved hands", "consistent feet", "identical body parts", "same genitals", 
                                  "preserved tattoos", "consistent jewelry", "same clothing patterns", "unchanged accessories",
                                  "anatomical consistency", "body part preservation", "facial consistency"]
        
        # Add temporal consistency for multi-chunk generation
        if chunk_index > 0:
            consistency_keywords.extend(["temporal consistency", "smooth continuation", "anatomical continuity"])
        
        # Combine with original prompt
        if consistency_keywords:
            enhanced_prompt = f"{positive_prompt}, {', '.join(consistency_keywords)}"
            return enhanced_prompt
        
        return positive_prompt

    def enhance_negative_prompt_for_consistency(self, negative_prompt, detail_preservation_mode):
        """
        Enhance the negative prompt to avoid inconsistencies.
        
        Args:
            negative_prompt: Original negative prompt
            detail_preservation_mode: Level of detail preservation
            
        Returns:
            Enhanced negative prompt that discourages inconsistencies
        """
        if detail_preservation_mode == "disabled":
            return negative_prompt
        
        # Inconsistency keywords to avoid
        inconsistency_keywords = []
        
        if detail_preservation_mode == "light":
            inconsistency_keywords = ["changing appearance", "different person", "morphing face", "changing body shape"]
        elif detail_preservation_mode == "medium":
            inconsistency_keywords = ["changing appearance", "different person", "inconsistent features", "morphing details",
                                    "face morphing", "changing facial features", "body shape changes", "hand morphing", "feet changes"]
        elif detail_preservation_mode == "strong":
            inconsistency_keywords = ["changing appearance", "different person", "inconsistent features", "morphing details",
                                    "face morphing", "changing facial features", "facial inconsistency", "different face",
                                    "body shape changes", "changing body proportions", "morphing body parts",
                                    "hand morphing", "changing hands", "different hands", "feet changes", "morphing feet",
                                    "genital changes", "morphing genitals", "body part inconsistency",
                                    "disappearing tattoos", "changing jewelry", "different clothing patterns", "missing accessories",
                                    "feature inconsistency", "detail morphing", "identity changes", "anatomical inconsistency"]
        
        # Combine with original negative prompt
        if inconsistency_keywords:
            if negative_prompt:
                enhanced_negative = f"{negative_prompt}, {', '.join(inconsistency_keywords)}"
            else:
                enhanced_negative = ', '.join(inconsistency_keywords)
            return enhanced_negative
        
        return negative_prompt if negative_prompt else ""

    def process_image(self, image, image_clip_vision_enabled, clip_vision, resizer, 
                     wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
                     image_width, image_height, image_type):
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

    def process_start_and_end_images(self, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled,
                                    clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
                                    image_generation_mode):
        """
        Process start and end images, getting their dimensions and processing them.
        
        Args:
            start_image: The start image tensor
            start_image_clip_vision_enabled (bool): Whether CLIP vision is enabled for start image
            end_image: The end image tensor  
            end_image_clip_vision_enabled (bool): Whether CLIP vision is enabled for end image
            clip_vision: The CLIP vision model
            resizer: The image resizer instance
            wan_max_resolution: The max resolution calculator
            CLIPVisionEncoder: The CLIP vision encoder
            large_image_side (int): The target size for the larger side
            wan_model_size (str): The model size configuration
            
        Returns:
            tuple: (start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image)
        """
        clip_vision_end_image = None
        clip_vision_start_image = None

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
        if end_image is not None and (image_generation_mode == END_IMAGE or image_generation_mode == END_TO_START_IMAGE or image_generation_mode == START_TO_END_TO_START_IMAGE):
            # ComfyUI images are tensors with shape [batch, height, width, channels]
            output_to_terminal_successful(f"Original end_image dimensions: {end_image.shape[2]}x{end_image.shape[1]}")

            # Process End Image
            if (image_generation_mode == END_IMAGE or image_generation_mode == END_TO_START_IMAGE):
                end_image, image_width, image_height, clip_vision_end_image = self.process_image(
                    end_image, end_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution,
                    CLIPVisionEncoder, large_image_side, wan_model_size, end_image.shape[2], end_image.shape[1], "End Image"
                )
            else:
                end_image, _, _, clip_vision_end_image = self.process_image(
                    end_image, end_image_clip_vision_enabled, clip_vision, resizer, wan_max_resolution,
                    CLIPVisionEncoder, large_image_side, wan_model_size, image_width, image_height, "End Image"
                )
        
        return start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image

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
        mm.soft_empty_cache()
        output_to_terminal_successful("High CFG KSampler started...")
        out_latent, = k_sampler.sample(model_high_cfg, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, stop_steps, "enabled", high_denoise)

        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()
        output_to_terminal_successful("Low CFG KSampler started...")
        out_latent, = k_sampler.sample(model_low_cfg, "disable", noise_seed, total_steps, low_cfg, "lcm", "simple", temp_positive_clip, temp_negative_clip, out_latent, stop_steps, 1000, "disable", low_denoise)
        
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
                model_high_cfg, updated_clip, = lora_loader.load_lora(model_high_cfg, updated_clip, causvid_lora, high_cfg_causvid_strength, high_cfg_causvid_strength)
            
            # Apply CausVid LoRA for Low CFG model  
            if (causvid_lora != NONE and low_cfg_causvid_strength > 0.0):
                output_to_terminal_successful(f"Applying CausVid LoRA for Low CFG with strength: {low_cfg_causvid_strength}")
                model_low_cfg, updated_clip, = lora_loader.load_lora(model_low_cfg, updated_clip, causvid_lora, low_cfg_causvid_strength, low_cfg_causvid_strength)
        else:
            # Single sampler - only apply to high CFG model
            if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
                output_to_terminal_successful(f"Applying CausVid LoRA with strength: {high_cfg_causvid_strength}")
                model_high_cfg, updated_clip, = lora_loader.load_lora(model_high_cfg, updated_clip, causvid_lora, high_cfg_causvid_strength, high_cfg_causvid_strength)
        
        return model_high_cfg, model_low_cfg, updated_clip
    
    def apply_color_match(self, start_image, output_image, apply_color_match, colorMatch):
        """
        Apply color matching between start and output images if enabled.
        
        Args:
            start_image: Reference image for color matching (or None)
            output_image: Target image to apply color correction to
            apply_color_match: Boolean flag to enable/disable color matching
            colorMatch: Color matching utility object
            
        Returns:
            output_image: Processed image with or without color matching applied
        """
        if (start_image is not None and apply_color_match):
            output_to_terminal_successful("Applying color match to images...")
            output_image, = colorMatch.colormatch(start_image, output_image, "hm-mvgd-hm", strength=1.0)
            
        return output_image
