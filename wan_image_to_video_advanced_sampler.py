import os
from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_cfg import CFGZeroStar
import nodes
import folder_paths
from .core import *
from .wan_first_last_first_frame_to_video import WanFirstLastFirstFrameToVideo
from .wan_video_vae_decode import WanVideoVaeDecode
from .wan_get_max_image_resolution_by_aspect_ratio import WanGetMaxImageResolutionByAspectRatio
from .wan_video_enhance_a_video import WanVideoEnhanceAVideo

# Try to import optional dependencies for downloading
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False

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
                # 👁️ CLIP VISION SETTINGS
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
                ("high_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.001, "tooltip": "Denoising strength for the first sampling pass. 1.0 = full denoising, lower values preserve more of the input."})),
                ("low_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.001, "tooltip": "Denoising strength for the second sampling pass. Used for refinement when dual samplers are enabled."})),
                ("total_steps", ("INT", {"default": 15, "min": 1, "max": 90, "step":1, "advanced": True, "tooltip": "Total number of sampling steps. More steps generally improve quality but increase generation time."})),
                ("total_steps_high_cfg", ("INT", {"default": 5, "min": 1, "max": 90, "step":1, "advanced": True, "tooltip": "Percentage of total_steps dedicated to the high CFG pass when using dual samplers. Remaining steps use low CFG for refinement."})),
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
                ("use_taesd_preview", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable TAESD preview for Wan2.1 models. Provides fast latent preview during generation."})),  # Proper TAESD implementation for Wan2.1
            ]),
            "optional": OrderedDict([
                ("lora_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of LoRAs to apply to the diffusion model. Each LoRA modifies the model's behavior."})),
                ("prompt_stack", (any_type, {"default": None, "advanced": True, "tooltip": "Stack of prompts to apply to the diffusion model on each chunck generated. If there is less prompts than chunks, the last prompt will be used for the remaining chunks."})),
                ("start_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "Start image for the video generation process."})),
                ("end_image", ("IMAGE", {"default": None, "advanced": True, "tooltip": "End image for the video generation process."})),
            ]),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.22, tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda", use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=2.0, use_block_swap=True, block_swap=35, large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size=WAN_720P, total_video_seconds=1, clip_vision_model=NONE, clip_vision_strength=1.0, use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0, lora_stack=None, start_image=None, start_image_clip_vision_enabled=True, end_image=None, end_image_clip_vision_enabled=True, video_enhance_enabled=True, use_cfg_zero_star=True, apply_color_match=True, use_taesd_preview=True, causvid_lora=NONE, high_cfg_causvid_strength=1.0, low_cfg_causvid_strength=1.0, high_denoise=1.0, low_denoise=1.0, total_video_chunks=1):

        #variables
        output_image = None
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

        # Initialize SageAttention
        sage_attention = self.initialize_sage_attention(use_sage_attention, sage_attention_mode)

        # Initialize Model Shift
        model_shift = self.initialize_model_shift(use_shift, shift)

        output_image, = self.postprocess(model, vae, clip, clip_type, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, SLG_blocks, SLG_start_percent, SLG_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, use_taesd_preview, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks)

        return (output_image,)

    def postprocess(self, model, vae, clip, clip_type, positive, negative, sage_attention, sage_attention_mode, model_shift, shift, use_shift, wanBlockSwap, use_block_swap, block_swap, tea_cache, use_tea_cache, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent, clip_vision_model, clip_vision_strength, start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled, large_image_side, wan_model_size, total_video_seconds, image_generation_mode, use_dual_samplers, high_cfg, low_cfg, high_denoise, low_denoise, total_steps, total_steps_high_cfg, noise_seed, video_enhance_enabled, use_cfg_zero_star, apply_color_match, use_taesd_preview, lora_stack, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, total_video_chunks):

        output_to_terminal_successful("Generation started...")

        output_image = None
        working_model = model.clone()
        k_sampler = nodes.KSamplerAdvanced()
        text_encode = nodes.CLIPTextEncode()
        wan_image_to_video = WanFirstLastFirstFrameToVideo()
        wan_video_vae_decode = WanVideoVaeDecode()
        wan_max_resolution = WanGetMaxImageResolutionByAspectRatio()
        CLIPVisionLoader = nodes.CLIPVisionLoader()
        CLIPVisionEncoder = nodes.CLIPVisionEncode()
        clip_vision = None        
        resizer = ImageResizeKJv2()
        image_width = 512
        image_height = 512
        in_latent = None
        out_latent = None
        clip_vision_start_image = None
        clip_vision_end_image = None
        total_frames = (total_video_seconds * 16) + 1
        lora_loader = nodes.LoraLoader()
        wanVideoEnhanceAVideo = WanVideoEnhanceAVideo()
        cfgZeroStar = CFGZeroStar()
        colorMatch = ColorMatch()

        # Enable TAESD preview if requested
        if (use_taesd_preview):
            output_to_terminal_successful("Setting up TAESD override system...")
            try:
                from .taesd_override import initialize_taesd_override
                if initialize_taesd_override():
                    output_to_terminal_successful("TAESD override system activated successfully")
                else:
                    output_to_terminal_error("TAESD override system failed to initialize")
            except Exception as e:
                output_to_terminal_error(f"TAESD override setup failed: {e}")
                output_to_terminal_error("Continuing with default ComfyUI preview system...")
        else:
            output_to_terminal_successful("TAESD override is disabled, using default ComfyUI preview system...")

        # Process LoRA stack
        working_model, clip = self.process_lora_stack(lora_stack, working_model, clip)

        # Apply Model Patch Torch Settings
        working_model = self.apply_model_patch_torch_settings(working_model)

        # Load CLIP Vision Model
        clip_vision = self.load_clip_vision_model(clip_vision_model, CLIPVisionLoader)

        # Process start and end images
        start_image, image_width, image_height, clip_vision_start_image, end_image, clip_vision_end_image = self.process_start_and_end_images(
            start_image, start_image_clip_vision_enabled, end_image, end_image_clip_vision_enabled,
            clip_vision, resizer, wan_max_resolution, CLIPVisionEncoder, large_image_side, wan_model_size,
            image_generation_mode
        )

        # Apply Sage Attention
        working_model = self.apply_sage_attention(sage_attention, working_model, sage_attention_mode)

        # Apply TeaCache and SLG
        working_model = self.apply_tea_cache_and_slg(tea_cache, use_tea_cache, working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device, slg_wanvideo, use_SLG, slg_wanvideo_blocks_string, slg_wanvideo_start_percent, slg_wanvideo_end_percent)
            
        # Apply Model Shift
        working_model = self.apply_model_shift(model_shift, use_shift, working_model, shift)

        # Apply Block Swap
        working_model = self.apply_block_swap(use_block_swap, working_model, wanBlockSwap, block_swap)

        # Apply Video Enhance
        working_model = self.apply_video_enhance(video_enhance_enabled, working_model, wanVideoEnhanceAVideo, total_frames)

        # Apply CFG Zero Star
        working_model = self.apply_cfg_zero_star(use_cfg_zero_star, working_model, cfgZeroStar)

        # Generate video chunks sequentially
        images_chunck = []
        for chunk_index in range(total_video_chunks):
            output_to_terminal_successful(f"Generating video chunk {chunk_index + 1}/{total_video_chunks}...")
            
            output_to_terminal_successful("Encoding Positive CLIP text...")
            temp_positive_clip, = text_encode.encode(clip, positive)

            output_to_terminal_successful("Encoding Negative CLIP text...")
            temp_negative_clip, = text_encode.encode(clip, negative)

            output_to_terminal_successful("Wan Image to Video started...")
            temp_positive_clip, temp_negative_clip, in_latent, = wan_image_to_video.encode(temp_positive_clip, temp_negative_clip, vae, image_width, image_height, total_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, 0.5, image_generation_mode)

            if (use_dual_samplers):
                # Apply dual sampler processing
                out_latent = self.apply_dual_sampler_processing(working_model, k_sampler, lora_loader, clip, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, noise_seed, total_steps, high_cfg, low_cfg, temp_positive_clip, temp_negative_clip, in_latent, total_steps_high_cfg, high_denoise, low_denoise)
            else:
                # Apply single sampler processing
                out_latent = self.apply_single_sampler_processing(working_model, k_sampler, lora_loader, clip, causvid_lora, high_cfg_causvid_strength, noise_seed, total_steps, high_cfg, temp_positive_clip, temp_negative_clip, in_latent, high_denoise)

            output_to_terminal_successful("Vae Decode started...")
            output_image, = wan_video_vae_decode.decode(out_latent, vae, 0, image_generation_mode)

            # Apply color match
            output_image = self.apply_color_match(start_image, output_image, apply_color_match, colorMatch)
            images_chunck.append(output_image)

        # Merge all video chunks in sequence
        if len(images_chunck) > 1:
            output_to_terminal_successful(f"Merging {len(images_chunck)} video chunks in sequence...")
            import torch
            # Concatenate all video chunks along the frame dimension (assuming batch, frames, height, width, channels)
            output_image = torch.cat(images_chunck, dim=1)
            output_to_terminal_successful("Video chunks merged successfully")
        elif len(images_chunck) == 1:
            output_image = images_chunck[0]
        else:
            output_to_terminal_error("No video chunks generated")

        return (output_image,)
   
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

    def load_taehv_model(self):
        """
        Load TAEHV model for potential fallback use in TAESD creation.
        
        This is kept for potential weight adaptation but not used directly in the preview system.
        The actual preview system now uses proper TAESD architecture.
        """
        try:
            import folder_paths
            # Try relative import first (when running as ComfyUI custom node)
            try:
                from .taehv_simple import TAEHV
            except ImportError:
                # Fallback to absolute import (when running standalone)
                from taehv_simple import TAEHV
            
            # Look for TAEHV models (prefer pth files from GitHub repository)
            vae_approx_files = folder_paths.get_filename_list("vae_approx")
            output_to_terminal_successful(f"Checking vae_approx folder, found {len(vae_approx_files)} files")
            
            # Priority order: pth files from GitHub repository
            taehv_candidates = []
            for f in vae_approx_files:
                if any(keyword in f.lower() for keyword in ['taehv', 'taew2_1', 'wan2.1', 'hunyuan', 'taecvx', 'taeos1_3']):
                    if f.endswith('.pth'):
                        taehv_candidates.insert(0, f)  # Prioritize pth files
                    elif f.endswith('.safetensors'):
                        taehv_candidates.append(f)  # Fallback to safetensors if available
            
            output_to_terminal_successful(f"Found {len(taehv_candidates)} TAEHV candidates: {taehv_candidates}")
            
            if not taehv_candidates:
                output_to_terminal_error("No TAEHV models found in vae_approx folder")
                output_to_terminal_error("Download models like 'taew2_1.pth' to ComfyUI/models/vae_approx/")
                return None
            
            # Try to load each candidate
            for model_name in taehv_candidates:
                try:
                    model_path = folder_paths.get_full_path("vae_approx", model_name)
                    output_to_terminal_successful(f"Loading TAEHV model: {model_name}")
                    
                    # Load state dict
                    import torch
                    from comfy.utils import load_torch_file
                    state_dict = load_torch_file(model_path, safe_load=True)
                    
                    # Create TAEHV model using the official implementation
                    taehv_model = TAEHV(state_dict=state_dict)
                    
                    # Move to device
                    import comfy.model_management as mm
                    device = mm.unet_offload_device()
                    taehv_model.to(device=device, dtype=torch.float16)
                    taehv_model.eval()
                    
                    output_to_terminal_successful(f"TAEHV model {model_name} loaded successfully on {device}")
                    return taehv_model
                    
                except Exception as e:
                    output_to_terminal_error(f"Failed to load {model_name}: {e}")
                    continue
            
            output_to_terminal_error("No compatible TAEHV models could be loaded")
            return None
            
        except ImportError as e:
            output_to_terminal_error(f"Failed to import TAEHV: {e}")
            return None
        except Exception as e:
            output_to_terminal_error(f"TAEHV loading error: {e}")
            return None

    def test_internet_connectivity(self):
        """Test internet connectivity by checking GitHub availability."""
        test_urls = [
            "https://github.com",
            "https://raw.githubusercontent.com",
            "https://github.com/madebyollin/taehv"
        ]
        
        for url in test_urls:
            try:
                if REQUESTS_AVAILABLE:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True
                elif URLLIB_AVAILABLE:
                    import urllib.request
                    import urllib.error
                    urllib.request.urlopen(url, timeout=5)
                    return True
            except Exception:
                continue
        
        return False

    def _test_url_availability(self, url):
        """Test if a URL is accessible."""
        try:
            if REQUESTS_AVAILABLE:
                response = requests.head(url, timeout=10)
                return response.status_code == 200
            elif URLLIB_AVAILABLE:
                import urllib.request
                import urllib.error
                urllib.request.urlopen(url, timeout=10)
                return True
        except Exception as e:
            output_to_terminal_error(f"URL test failed for {url}: {e}")
            return False
        return False

    def download_taehv_models(self):
        """
        Download TAEHV models automatically from official repositories.
        
        Downloads models from the official TAEHV repository for Wan2.1 preview support.
        Returns True if at least one model was downloaded successfully, False otherwise.
        """
        if not (REQUESTS_AVAILABLE or URLLIB_AVAILABLE):
            output_to_terminal_error("Neither 'requests' nor 'urllib' available for downloading")
            output_to_terminal_error("Manual installation:")
            output_to_terminal_error("1. Download models from https://github.com/madebyollin/taehv/")
            output_to_terminal_error("2. Place them in ComfyUI/models/vae_approx/")
            return False
            
        try:
            # Get vae_approx directory
            vae_approx_paths = folder_paths.get_folder_paths("vae_approx")
            if not vae_approx_paths:
                output_to_terminal_error("vae_approx folder not found in ComfyUI")
                return False
            
            vae_approx_dir = vae_approx_paths[0]
            os.makedirs(vae_approx_dir, exist_ok=True)
            
            # Official TAEHV models from madebyollin/taehv repository
            models = [
                {
                    "name": "taew2_1.pth",
                    "url": "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth",
                    "description": "TAEHV for Wan 2.1"                    
                },
                {
                    "name": "taehv.pth",
                    "url": "https://github.com/madebyollin/taehv/raw/main/taehv.pth",
                    "description": "TAEHV for Hunyuan Video"
                },
                {
                    "name": "taecvx.pth", 
                    "url": "https://github.com/madebyollin/taehv/raw/main/taecvx.pth",
                    "description": "TAEHV for CogVideoX"
                },
                {
                    "name": "taeos1_3.pth", 
                    "url": "https://github.com/madebyollin/taehv/raw/main/taeos1_3.pth",
                    "description": "TAEHV for Open-Sora 1.3"
                }
            ]
            
            # Also download official TAESD models for reference
            taesd_models = [
                {
                    "name": "taesd_decoder.pth",
                    "url": "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth",
                    "description": "TAESD decoder for SD1/2"
                },
                {
                    "name": "taesdxl_decoder.pth",
                    "url": "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
                    "description": "TAESD decoder for SDXL"
                },
                {
                    "name": "taesd3_decoder.pth",
                    "url": "https://github.com/madebyollin/taesd/raw/main/taesd3_decoder.pth",
                    "description": "TAESD decoder for SD3"
                },
                {
                    "name": "taef1_decoder.pth",
                    "url": "https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth",
                    "description": "TAESD decoder for FLUX.1"
                }
            ]
            
            all_models = models + taesd_models
            
            output_to_terminal_successful("Downloading official TAEHV and TAESD models...")
            output_to_terminal_successful(f"Target directory: {vae_approx_dir}")
            output_to_terminal_successful(f"Available download methods: requests={REQUESTS_AVAILABLE}, urllib={URLLIB_AVAILABLE}")
            
            # Test connectivity
            test_url = models[0]["url"]
            output_to_terminal_successful(f"Testing connectivity to: {test_url}")
            if not self._test_url_availability(test_url):
                output_to_terminal_error("Cannot connect to GitHub. Check your internet connection.")
                return False
            else:
                output_to_terminal_successful("GitHub connectivity confirmed")
            
            success_count = 0
            
            for model in all_models:
                filepath = os.path.join(vae_approx_dir, model["name"])
                
                # Skip if already exists
                if os.path.exists(filepath):
                    output_to_terminal_successful(f"{model['name']} already exists, skipping")
                    success_count += 1
                    continue
                
                try:
                    output_to_terminal_successful(f"Downloading {model['name']}...")
                    
                    # Try requests first, fallback to urllib
                    if REQUESTS_AVAILABLE:
                        self._download_file_with_requests(model["url"], filepath, model["description"])
                    else:
                        self._download_file_with_urllib(model["url"], filepath, model["description"])
                        
                    output_to_terminal_successful(f"{model['name']} downloaded successfully")
                    success_count += 1
                    
                except Exception as e:
                    output_to_terminal_error(f"Failed to download {model['name']}: {e}")
            
            output_to_terminal_successful(f"Download complete: {success_count}/{len(all_models)} models")
            
            if success_count > 0:
                output_to_terminal_successful("Official TAEHV and TAESD models are now available!")
                return True
            else:
                output_to_terminal_error("No models were downloaded successfully.")
                return False
                
        except Exception as e:
            output_to_terminal_error(f"Download error: {e}")
            output_to_terminal_error("Manual installation:")
            output_to_terminal_error("1. Download models from https://github.com/madebyollin/taehv/ and https://github.com/madebyollin/taesd/")
            output_to_terminal_error("2. Place them in ComfyUI/models/vae_approx/")
            return False

    def _download_file_with_requests(self, url, filepath, description="Downloading"):
        """Download a file using requests with progress tracking."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Use a simple progress indicator since tqdm might not work well in ComfyUI console
        output_to_terminal_successful(f"{description} - {total_size // (1024*1024)} MB")
        
        with open(filepath, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator every 10MB
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100)
                        output_to_terminal_successful(f"Progress: {progress:.1f}%")

    def _download_file_with_urllib(self, url, filepath, description="Downloading"):
        """Download a file using urllib as fallback."""
        import urllib.request
        import urllib.error
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                progress = (downloaded / total_size * 100)
                output_to_terminal_successful(f"Progress: {progress:.1f}%")
        
        try:
            output_to_terminal_successful(f"{description} - Starting download with urllib...")
            urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        except urllib.error.URLError as e:
            raise Exception(f"urllib download failed: {e}")
        except Exception as e:
            raise Exception(f"Download failed: {e}")
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
                    working_model, clip = lora_loader.load_lora(working_model, clip, lora_name, model_strength, clip_strength)
                else:
                    output_to_terminal_error(f"Skipping LoRA {lora_count}/{len(lora_stack)}: No valid LoRA name")
            
            output_to_terminal_successful(f"Successfully applied {len(lora_stack)} LoRAs to the model")
        else:
            output_to_terminal_successful("No LoRA stack provided, skipping LoRA application")
            
        return working_model, clip

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
        if (clip_vision_model != NONE and self.clip_vision_model is None and self.clip_vision_model != clip_vision_model):
            output_to_terminal_successful("Loading clip vision model...")
            clip_vision, = CLIPVisionLoader.load_clip(clip_vision_model)
            self.clip_vision_model = clip_vision_model
            self.clip_vision = clip_vision
            return self.clip_vision
        elif (self.clip_vision is not None):
            output_to_terminal_successful("Loading clip vision model...")
            return self.clip_vision
        else:
            output_to_terminal_error("No clip vision model selected, skipping...")
            return None

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

    def apply_dual_sampler_processing(self, working_model, k_sampler, lora_loader, clip, causvid_lora, high_cfg_causvid_strength, low_cfg_causvid_strength, noise_seed, total_steps, high_cfg, low_cfg, temp_positive_clip, temp_negative_clip, in_latent, total_steps_high_cfg, high_denoise, low_denoise):
        """
        Apply dual sampler processing with high and low CFG models.
        
        Args:
            working_model: The base model to clone for dual sampling
            k_sampler: The KSampler instance
            lora_loader: The LoRA loader instance
            clip: The CLIP model
            causvid_lora: The CausVid LoRA to apply
            high_cfg_causvid_strength (float): LoRA strength for high CFG model
            low_cfg_causvid_strength (float): LoRA strength for low CFG model
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
        model_high_cfg = working_model.clone()
        model_low_cfg = working_model.clone()

        stop_steps = int(total_steps_high_cfg / 100 * total_steps)

        if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
            output_to_terminal_successful(f"Applying CausVid LoRA for High CFG with strength: {high_cfg_causvid_strength}")
            model_high_cfg,_, = lora_loader.load_lora(model_high_cfg, clip, causvid_lora, causvid_lora, high_cfg_causvid_strength)

        output_to_terminal_successful("High CFG KSampler started...")
        out_latent, = k_sampler.sample(model_high_cfg, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, stop_steps, "enabled", high_denoise)

        if (causvid_lora != NONE and low_cfg_causvid_strength > 0.0):
            output_to_terminal_successful(f"Applying CausVid LoRA for Low CFG with strength: {low_cfg_causvid_strength}")
            model_low_cfg,_, = lora_loader.load_lora(model_low_cfg, clip, causvid_lora, causvid_lora, low_cfg_causvid_strength)

        output_to_terminal_successful("Low CFG KSampler started...")
        out_latent, = k_sampler.sample(model_low_cfg, "disable", noise_seed, total_steps, low_cfg, "lcm", "simple", temp_positive_clip, temp_negative_clip, out_latent, stop_steps, 1000, "disable", low_denoise)
        
        return out_latent

    def apply_single_sampler_processing(self, working_model, k_sampler, lora_loader, clip, causvid_lora, high_cfg_causvid_strength, noise_seed, total_steps, high_cfg, temp_positive_clip, temp_negative_clip, in_latent, high_denoise):
        """
        Apply single sampler processing with optional CausVid LoRA.
        
        Args:
            working_model: The model to use for sampling
            k_sampler: The KSampler instance
            lora_loader: The LoRA loader instance
            clip: The CLIP model
            causvid_lora: The CausVid LoRA to apply
            high_cfg_causvid_strength (float): LoRA strength for the model
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
        if (causvid_lora != NONE and high_cfg_causvid_strength > 0.0):
            output_to_terminal_successful(f"Applying CausVid LoRA with strength: {high_cfg_causvid_strength}")
            working_model,_, = lora_loader.load_lora(working_model, clip, causvid_lora, causvid_lora, high_cfg_causvid_strength)

        output_to_terminal_successful("KSampler started...")
        out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, 1000, "disable", high_denoise)
        
        return out_latent
    
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
