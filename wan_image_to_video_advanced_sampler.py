import os
from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
import nodes
import folder_paths
from .core import *
from .wan_first_last_first_frame_to_video import WanFirstLastFirstFrameToVideo
from .wan_video_vae_decode import WanVideoVaeDecode
from .wan_get_max_image_resolution_by_aspect_ratio import WanGetMaxImageResolutionByAspectRatio

# Import external custom nodes using the centralized import function
imported_nodes = {}
teacache_imports = import_nodes(["teacache"], ["TeaCache"])
# Import nodes from different custom node packages
kjnodes_imports = import_nodes(["comfyui-kjnodes"], ["SkipLayerGuidanceWanVideo", "PathchSageAttentionKJ", "ImageResizeKJv2"])
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

class WanImageToVideoAdvancedSampler:
    @classmethod
    def INPUT_TYPES(s):
        
        clip_names = [NONE] + folder_paths.get_filename_list("text_encoders")
        gguf_model_names = [NONE] + folder_paths.get_filename_list("unet_gguf")
        diffusion_models_names = [NONE] + folder_paths.get_filename_list("diffusion_models")
        vae_names = [NONE] + folder_paths.get_filename_list("vae")
        clip_vision_models = [NONE] + folder_paths.get_filename_list("clip_vision")        

        return {
            "required": OrderedDict([
                ("GGUF", (gguf_model_names, {"default": NONE, "advanced": True})),
                ("Diffusor", (diffusion_models_names, {"default": NONE, "advanced": True})),
                ("Diffusor_weight_dtype", (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True})),
                ("Use_Model_Type", (MODEL_TYPE_LIST, {"default": MODEL_GGUF, "advanced": True})),
                ("positive", ("STRING", {"default": None, "advanced": True})),
                ("negative", ("STRING", {"default": None, "advanced": True})),
                ("clip", (clip_names, {"default": None, "advanced": True})),
                ("clip_type", (CLIP_TYPE_LIST, {"default": CLIP_WAN, "advanced": True})),
                ("clip_device", (CLIP_DEVICE_LIST, {"default": CLIP_DEVICE_DEFAULT, "advanced": True})),
                ("vae", (vae_names, {"default": NONE, "advanced": True})),
                ("use_tea_cache", ("BOOLEAN", {"default": True, "advanced": True})),
                ("tea_cache_model_type", (["flux", "ltxv", "lumina_2", "hunyuan_video", "hidream_i1_dev", "hidream_i1_full", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode", "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode"], {"default": "wan2.1_i2v_720p_14B", "tooltip": "Supported diffusion model."})),
                ("tea_cache_rel_l1_thresh", ("FLOAT", {"default": 0.22, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})),
                ("tea_cache_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."})),
                ("tea_cache_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."})),
                ("tea_cache_cache_device", (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside"})),
                ("use_SLG", ("BOOLEAN", {"default": True, "advanced": True})),
                ("SLG_blocks", ("STRING", {"default": "10", "multiline": False, "tooltip": "Number of blocks to process in each step. You can comma separate the blocks like 8, 9, 10"})),
                ("SLG_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001})),
                ("SLG_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001})),
                ("use_sage_attention", ("BOOLEAN", {"default": True, "advanced": True})),
                ("sage_attention_mode", (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": "auto", "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."})),
                ("use_shift", ("BOOLEAN", {"default": True, "advanced": True})),
                ("shift", ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step":0.01})),
                ("use_block_swap", ("BOOLEAN", {"default": True, "advanced": True})),
                ("block_swap", ("INT", {"default": 35, "min": 1, "max": 40, "step":1})),
                ("large_image_side", ("INT", {"default": 832, "min": 2.0, "max": 1200, "step":2, "advanced": True, "tooltip": "The larger side of the image to resize to. The smaller side will be resized proportionally."})),
                ("image_generation_mode", (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE})),
                ("wan_model_size", (WAN_MODELS, {"default": "wan2.1_i2v_720p_14B_ret_mode", "tooltip": "The model type to use for the diffusion process."})),
                ("total_video_seconds", ("INT", {"default": 1, "min": 1, "max": 5, "step":1, "advanced": True, "tooltip": "The total duration of the video in seconds."})),
                ("clip_vision_model", (clip_vision_models, {"default": NONE, "advanced": True})),
                ("clip_vision_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01})),
                ("start_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the start image. If disabled, the start image will be used as a static frame."})),
                ("end_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the end image. If disabled, the end image will be used as a static frame."})),                
                ("use_dual_samplers", ("BOOLEAN", {"default": True, "advanced": True})),
                ("high_cfg", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01})),
                ("low_cfg", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01})),
                ("total_steps", ("INT", {"default": 15, "min": 1, "max": 90, "step":1, "advanced": True,})),
                ("total_steps_high_cfg", ("INT", {"default": 5, "min": 1, "max": 90, "step":1, "advanced": True,})),
                ("noise_seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
                ("use_TAEHV_preview", ("BOOLEAN", {"default": True, "advanced": True})),                
            ]),
            "optional": OrderedDict([
                ("lora_stack", ("LORA_STACK", {"default": None, "advanced": True})),
                ("start_image", ("IMAGE", {"default": None, "advanced": True})),
                ("end_image", ("IMAGE", {"default": None, "advanced": True})),
            ]),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative,
            clip, clip_type, clip_device, vae, use_tea_cache,
            tea_cache_model_type="wan2.1_i2v_720p_14B", tea_cache_rel_l1_thresh=0.22,
            tea_cache_start_percent=0.2, tea_cache_end_percent=0.8, tea_cache_cache_device="cuda",
            use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8,
            use_sage_attention=True, sage_attention_mode="auto", use_shift=True, shift=2.0,
            use_block_swap=True, block_swap=35,
            large_image_side=832, image_generation_mode=START_IMAGE, wan_model_size="wan2.1_i2v_720p_14B_ret_mode",
            total_video_seconds=1, clip_vision_model=NONE, clip_vision_strength=1.0,
            use_dual_samplers=True, high_cfg=1.0, low_cfg=1.0, total_steps=15, total_steps_high_cfg=5, noise_seed=0,
            lora_stack=None, start_image=None, start_image_clip_vision_enabled=True,
            end_image=None, end_image_clip_vision_enabled=True, use_TAEHV_preview=True):

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
            output_to_terminal_error("TeaCache not available - continuing without caching optimization")

        if (SageAttention is not None) and use_sage_attention:
            if sage_attention_mode == "disabled":
                sage_attention = None
                output_to_terminal_successful("SageAttention disabled")
            else:
                sage_attention = SageAttention()
                output_to_terminal_successful(f"SageAttention enabled with mode: {sage_attention_mode}")
        else:
            sage_attention = None
            output_to_terminal_successful("SageAttention disabled")

        if (ModelSamplingSD3 is not None) and use_shift:
            model_shift = ModelSamplingSD3()
            model_shift.shift = shift
            output_to_terminal_successful(f"Model Shift enabled with shift: {shift}")
        else:
            model_shift = None
            output_to_terminal_successful("Model Shift disabled")

        output_image = self.postprocess(
            model, 
            vae, 
            clip,
            clip_type,
            positive, 
            negative, 

            sage_attention, 
            sage_attention_mode,

            model_shift,
            shift,
            use_shift,

            wanBlockSwap,
            use_block_swap,
            block_swap,

            tea_cache,
            use_tea_cache,
            tea_cache_model_type, 
            tea_cache_rel_l1_thresh, 
            tea_cache_start_percent, 
            tea_cache_end_percent,             
            tea_cache_cache_device,

            slg_wanvideo,
            use_SLG,
            SLG_blocks,
            SLG_start_percent,
            SLG_end_percent,

            clip_vision_model,
            clip_vision_strength,
            start_image,
            start_image_clip_vision_enabled,
            end_image,
            end_image_clip_vision_enabled,
            large_image_side,
            wan_model_size,
            total_video_seconds,
            image_generation_mode,
            use_dual_samplers,
            high_cfg,
            low_cfg,
            total_steps,
            total_steps_high_cfg,
            noise_seed,
            use_TAEHV_preview
        )

        return (output_image,)


    def postprocess(self, 
                    model,
                    vae,
                    clip,
                    clip_type,
                    positive,
                    negative,

                    sage_attention,
                    sage_attention_mode,

                    model_shift,
                    shift,
                    use_shift,

                    wanBlockSwap,
                    use_block_swap,
                    block_swap,

                    tea_cache,
                    use_tea_cache,
                    tea_cache_model_type,
                    tea_cache_rel_l1_thresh,
                    tea_cache_start_percent,
                    tea_cache_end_percent,
                    tea_cache_cache_device,

                    slg_wanvideo,
                    use_SLG,
                    slg_wanvideo_blocks_string,
                    slg_wanvideo_start_percent,
                    slg_wanvideo_end_percent,

                    clip_vision_model,
                    clip_vision_strength,
                    start_image,
                    start_image_clip_vision_enabled,
                    end_image,
                    end_image_clip_vision_enabled,
                    large_image_side,
                    wan_model_size,
                    total_video_seconds,
                    image_generation_mode,
                    use_dual_samplers,
                    high_cfg,
                    low_cfg,
                    total_steps,
                    total_steps_high_cfg,
                    noise_seed,
                    use_TAEHV_preview
    ):

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

        # lora_loader = nodes.LoraLoader()

        if (use_TAEHV_preview):
            output_to_terminal_successful("Setting up TAESD for Wan2.1...")
            self.setup_taesd_preview(clip_type, working_model)
        else:
            output_to_terminal_error("TAESD for Wan2.1 is disabled, using default settings...")

        if (clip_vision_model != NONE):
            output_to_terminal_successful("Loading clip vision model...")
            clip_vision, = CLIPVisionLoader.load_clip(clip_vision_model)
        else:
            output_to_terminal_error("No clip vision model selected, skipping...")

        if (start_image is not None):
            output_to_terminal_successful("Resizing Start Image...")
            start_image, image_width, image_height = resizer.resize(start_image, large_image_side, large_image_side, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            tmp_width, tmp_height, = wan_max_resolution.run(wan_model_size, start_image)
            tmpTotalPixels = tmp_width * tmp_height
            imageTotalPixels = image_width * image_height
            if (tmpTotalPixels < imageTotalPixels):
                image_width = tmp_width
                image_height = tmp_height
                start_image, image_width, image_height = resizer.resize(start_image, image_width, image_height, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")

            output_to_terminal_successful(f"Start Image final size: {image_width}x{image_height}")

            if (start_image_clip_vision_enabled) and (clip_vision is not None):
                output_to_terminal_successful("Encoding CLIP Vision for Start Image...")
                clip_vision_start_image, = CLIPVisionEncoder.encode(clip_vision, start_image, "center")
        else:
            output_to_terminal_error("Start Image is not provided, skipping...")

        if (end_image is not None):
            output_to_terminal_successful("Resizing End Image...")
            end_image, image_width, image_height = resizer.resize(end_image, large_image_side, large_image_side, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            tmp_width, tmp_height, = wan_max_resolution.run(wan_model_size, end_image)
            tmpTotalPixels = tmp_width * tmp_height
            imageTotalPixels = image_width * image_height
            if (tmpTotalPixels < imageTotalPixels):
                image_width = tmp_width
                image_height = tmp_height
                end_image, image_width, image_height = resizer.resize(end_image, image_width, image_height, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")

            output_to_terminal_successful(f"End Image final size: {image_width}x{image_height}")

            if (end_image_clip_vision_enabled) and (clip_vision is not None):
                output_to_terminal_successful("Encoding CLIP Vision for End Image...")
                clip_vision_end_image, = CLIPVisionEncoder.encode(clip_vision, end_image, "center")
        else:
            output_to_terminal_error("End Image is not provided, skipping...")

        if (sage_attention is not None):
            output_to_terminal_successful("Applying Sage Attention...")
            working_model, = sage_attention.patch(working_model, sage_attention_mode)
        else:
            output_to_terminal_error("Sage Attention disabled, skipping...")

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
            
        
        if (model_shift is not None and use_shift):
            output_to_terminal_successful("Applying Model Shift...")
            working_model, = model_shift.patch(working_model, shift)
        else:
            output_to_terminal_error("Model Shift disabled, skipping...")

        if (use_block_swap):
            output_to_terminal_successful("Setting block swap...")
            working_model, = wanBlockSwap.set_callback(working_model, block_swap, True, True, True)
        else:
            output_to_terminal_error("Block swap disabled, skipping...")

        output_to_terminal_successful("Encoding Positive CLIP text...")
        temp_positive_clip, = text_encode.encode(clip, positive)

        output_to_terminal_successful("Encoding Negative CLIP text...")
        temp_negative_clip, = text_encode.encode(clip, negative)

        output_to_terminal_successful("Wan Image to Video started...")
        temp_positive_clip, temp_negative_clip, in_latent, = wan_image_to_video.encode(temp_positive_clip, temp_negative_clip, vae, image_width, image_height, total_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, 0.5, image_generation_mode)

        if (use_dual_samplers):
            output_to_terminal_successful("High CFG KSampler started...")
            out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, total_steps_high_cfg, "enabled", 1)

            output_to_terminal_successful("Low CFG KSampler started...")
            out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, low_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, out_latent, total_steps_high_cfg, 1000, "enabled", 1)
        else:
            output_to_terminal_successful("KSampler started...")
            out_latent, = k_sampler.sample(working_model, "enable", noise_seed, total_steps, high_cfg, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 0, 1000, "enabled", 1)

        output_to_terminal_successful("Vae Decode started...")
        output_image = wan_video_vae_decode.decode(out_latent, vae, 0, START_IMAGE)

        return output_image
   
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

    def setup_taesd_preview(self, clip_type, model):
        """
        TAESD preview setup for Wan2.1 models using TAEHV integration.
        
        This method sets up proper TAESD preview support for Wan2.1 models by:
        1. Loading TAEHV (Tiny AutoEncoder for Hunyuan Video) for Wan2.1 models
        2. Auto-downloading TAEHV models if they don't exist
        3. Installing TAEHV as the preview decoder in the latent format
        4. Falling back to RGB previews if TAEHV is not available
        
        TAEHV is specifically designed for Wan2.1's 16-channel latent space
        and provides proper video preview functionality.
        """
        # Only handle Wan2.1 models, let KSampler deal with everything else
        if clip_type != CLIP_WAN:
            return
            
        # Check if this is a Wan2.1 model
        if hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
            latent_format = model.model.latent_format
            if (hasattr(latent_format, 'latent_channels') and 
                latent_format.latent_channels == 16 and
                hasattr(latent_format, 'latent_dimensions') and 
                latent_format.latent_dimensions == 3):
                
                # Try to load and install TAEHV for Wan2.1 models
                taehv_model = self.load_taehv_model()
                if taehv_model is not None:
                    # Install TAEHV previewer
                    from .taehv_preview_simple import install_taehv_previewer
                    success = install_taehv_previewer(model, taehv_model)
                    
                    if success:
                        output_to_terminal_successful("TAEHV preview system installed for Wan2.1 models")
                    else:
                        output_to_terminal_error("Failed to install TAEHV preview system")
                        latent_format.taesd_decoder_name = None  # Fallback to RGB
                else:
                    # Try to download TAEHV models automatically
                    output_to_terminal_successful("TAEHV models not found, attempting automatic download...")
                    if self.download_taehv_models():
                        # Try loading again after download
                        taehv_model = self.load_taehv_model()
                        if taehv_model is not None:
                            from .taehv_preview_simple import install_taehv_previewer
                            success = install_taehv_previewer(model, taehv_model)
                            if success:
                                output_to_terminal_successful("TAEHV models downloaded and preview system installed for Wan2.1 models")
                            else:
                                output_to_terminal_error("Failed to install TAEHV preview system after download")
                                latent_format.taesd_decoder_name = None  # Fallback to RGB
                        else:
                            output_to_terminal_error("Failed to load TAEHV models after download")
                            latent_format.taesd_decoder_name = None  # Fallback to RGB
                    else:
                        # Fallback: disable TAESD and use RGB previews
                        latent_format.taesd_decoder_name = None
                        output_to_terminal_successful("TAEHV download failed - using RGB fallback previews for Wan2.1 models")
        return

    def load_taehv_model(self):
        """
        Load TAEHV model for Wan2.1 video previews.
        
        Searches for TAEHV models in the vae_approx folder and loads the first compatible one.
        Returns the loaded TAEHV model or None if not found.
        """
        try:
            import folder_paths
            from .taehv_simple import TAEHV
            
            # Look for TAEHV models (prefer safetensors like WanVideoWrapper)
            vae_approx_files = folder_paths.get_filename_list("vae_approx")
            
            # Priority order: safetensors first, then other formats
            taehv_candidates = []
            for f in vae_approx_files:
                if any(keyword in f.lower() for keyword in ['taehv', 'taew2_1', 'wan2.1', 'hunyuan']):
                    if f.endswith('.safetensors'):
                        taehv_candidates.insert(0, f)  # Prioritize safetensors
                    else:
                        taehv_candidates.append(f)
            
            if not taehv_candidates:
                output_to_terminal_error("No TAEHV models found in vae_approx folder")
                output_to_terminal_error("Download models like 'taew2_1.safetensors' to ComfyUI/models/vae_approx/")
                return None
            
            # Try to load each candidate (prioritizing safetensors)
            for model_name in taehv_candidates:
                try:
                    model_path = folder_paths.get_full_path("vae_approx", model_name)
                    output_to_terminal_successful(f"Loading TAEHV model: {model_name}")
                    
                    # Load state dict
                    import torch
                    from comfy.utils import load_torch_file
                    state_dict = load_torch_file(model_path, safe_load=True)
                    
                    # Create TAEHV model using simplified approach
                    taehv_model = TAEHV(state_dict)
                    
                    # Move to device
                    import comfy.model_management as mm
                    device = mm.unet_offload_device()
                    taehv_model.to(device=device, dtype=torch.float16)
                    taehv_model.eval()
                    
                    output_to_terminal_successful(f"✓ TAEHV model {model_name} loaded successfully on {device}")
                    return taehv_model
                    
                except Exception as e:
                    output_to_terminal_error(f"✗ Failed to load {model_name}: {e}")
                    continue
            
            output_to_terminal_error("❌ No compatible TAEHV models could be loaded")
            return None
            
        except ImportError as e:
            output_to_terminal_error(f"Failed to import required modules for TAEHV: {e}")
            return None
        except Exception as e:
            output_to_terminal_error(f"Failed to load TAEHV model: {e}")
            return None

    def _convert_taehv_state_dict(self, source_state_dict, target_state_dict):
        """
        Convert TAEHV state dict from different model formats to match our architecture.
        
        Handles cases where the source model has different layer structures or dimensions.
        Returns converted state dict or None if conversion fails.
        """
        try:
            import torch
            converted_dict = {}
            
            output_to_terminal_successful("Attempting TAEHV state dict conversion...")
            
            # Check if source uses 2D convolutions that need to be converted to 3D
            needs_3d_conversion = False
            for key, tensor in source_state_dict.items():
                if 'conv' in key and 'weight' in key and tensor.dim() == 4:
                    needs_3d_conversion = True
                    break
            
            if needs_3d_conversion:
                output_to_terminal_successful("Converting 2D convolutions to 3D for video processing...")
                
                # Convert 2D conv weights to 3D by adding temporal dimension
                for key, tensor in source_state_dict.items():
                    if 'conv' in key and 'weight' in key and tensor.dim() == 4:
                        # Convert [out_ch, in_ch, h, w] to [out_ch, in_ch, t, h, w] with t=3
                        converted_tensor = tensor.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3.0
                        converted_dict[key] = converted_tensor
                    else:
                        converted_dict[key] = tensor
            else:
                converted_dict = source_state_dict.copy()
            
            # Try to match as many keys as possible
            final_dict = {}
            matched_keys = 0
            total_target_keys = len(target_state_dict)
            
            for target_key in target_state_dict.keys():
                if target_key in converted_dict:
                    source_tensor = converted_dict[target_key]
                    target_shape = target_state_dict[target_key].shape
                    
                    if source_tensor.shape == target_shape:
                        final_dict[target_key] = source_tensor
                        matched_keys += 1
                    else:
                        # Try to reshape or adapt if possible
                        if source_tensor.numel() == target_state_dict[target_key].numel():
                            try:
                                final_dict[target_key] = source_tensor.reshape(target_shape)
                                matched_keys += 1
                            except:
                                output_to_terminal_error(f"Could not reshape {target_key}: {source_tensor.shape} -> {target_shape}")
                        else:
                            output_to_terminal_error(f"Size mismatch for {target_key}: {source_tensor.shape} vs {target_shape}")
            
            match_percentage = (matched_keys / total_target_keys) * 100
            output_to_terminal_successful(f"State dict conversion: {matched_keys}/{total_target_keys} keys matched ({match_percentage:.1f}%)")
            
            # Require at least 50% of keys to match for a successful conversion
            if match_percentage >= 50:
                output_to_terminal_successful("State dict conversion successful - using converted model")
                return final_dict
            else:
                output_to_terminal_error(f"State dict conversion failed - only {match_percentage:.1f}% keys matched")
                return None
                
        except Exception as e:
            output_to_terminal_error(f"State dict conversion error: {e}")
            return None

    def download_taehv_models(self):
        """
        Download TAEHV models automatically if they don't exist.
        
        Downloads TAEHV models from HuggingFace to the vae_approx directory.
        Returns True if at least one model was downloaded successfully, False otherwise.
        """
        try:
            import requests
            from tqdm import tqdm
            
            # Get vae_approx directory
            vae_approx_paths = folder_paths.get_folder_paths("vae_approx")
            if not vae_approx_paths:
                output_to_terminal_error("vae_approx folder not found in ComfyUI")
                return False
            
            vae_approx_dir = vae_approx_paths[0]
            os.makedirs(vae_approx_dir, exist_ok=True)
            
            # Define TAEHV models to download (prioritize safetensors)
            models = [
                {
                    "name": "taew2_1.safetensors",
                    "url": "https://huggingface.co/madebyollin/taehv/resolve/main/taew2_1.safetensors",
                    "description": "TAEHV for Wan 2.1 (safetensors)"
                },
                {
                    "name": "taehv.safetensors",
                    "url": "https://huggingface.co/madebyollin/taehv/resolve/main/taehv.safetensors",
                    "description": "TAEHV for Hunyuan Video (safetensors)"
                },
                {
                    "name": "taew2_1.pth", 
                    "url": "https://huggingface.co/madebyollin/taehv/resolve/main/taew2_1.pth",
                    "description": "TAEHV for Wan 2.1 (pth)"
                }
            ]
            
            output_to_terminal_successful("Downloading TAEHV models for Wan2.1 preview support...")
            output_to_terminal_successful(f"Target directory: {vae_approx_dir}")
            
            success_count = 0
            
            for model in models:
                filepath = os.path.join(vae_approx_dir, model["name"])
                
                # Skip if already exists
                if os.path.exists(filepath):
                    output_to_terminal_successful(f"✓ {model['name']} already exists, skipping")
                    success_count += 1
                    continue
                
                try:
                    output_to_terminal_successful(f"Downloading {model['name']}...")
                    self._download_file_with_progress(model["url"], filepath, model["description"])
                    output_to_terminal_successful(f"✓ {model['name']} downloaded successfully")
                    success_count += 1
                    
                except Exception as e:
                    output_to_terminal_error(f"✗ Failed to download {model['name']}: {e}")
            
            output_to_terminal_successful(f"Download complete: {success_count}/{len(models)} models")
            
            if success_count > 0:
                output_to_terminal_successful("🎉 TAEHV models are now available for Wan2.1 preview support!")
                return True
            else:
                output_to_terminal_error("❌ No models were downloaded successfully.")
                return False
                
        except ImportError as e:
            output_to_terminal_error(f"Required modules not available for downloading: {e}")
            output_to_terminal_error("Please install manually or run: pip install requests tqdm")
            return False
        except Exception as e:
            output_to_terminal_error(f"Download error: {e}")
            output_to_terminal_error("Manual installation:")
            output_to_terminal_error("1. Download TAEHV models from https://huggingface.co/madebyollin/taehv")
            output_to_terminal_error("2. Place them in ComfyUI/models/vae_approx/")
            return False

    def _download_file_with_progress(self, url, filepath, description="Downloading"):
        """Download a file with progress tracking."""
        import requests
        from tqdm import tqdm
        
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
                    if downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        output_to_terminal_successful(f"Progress: {progress:.1f}%")
