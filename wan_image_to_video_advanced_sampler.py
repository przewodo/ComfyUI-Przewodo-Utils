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
kjnodes_imports = import_nodes(["comfyui-kjnodes", "nodes", "model_optimization_nodes"], ["SkipLayerGuidanceWanVideo", "PathchSageAttentionKJ", "ImageResizeKJv2"])
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
                ("tea_cache_model_type", (["flux", "ltxv", "lumina_2", "hunyuan_video", "hidream_i1_dev", "hidream_i1_full", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode", "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode"], {"default": "wan2.1_i2v_720p_14B_ret_mode", "tooltip": "Supported diffusion model."})),
                ("tea_cache_rel_l1_thresh", ("FLOAT", {"default": 0.22, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})),
                ("tea_cache_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."})),
                ("tea_cache_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."})),
                ("tea_cache_cache_device", (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside"})),
                ("use_SLG", ("BOOLEAN", {"default": True, "advanced": True})),
                ("SLG_blocks", ("STRING", {"default": "10", "multiline": False, "tooltip": "Number of blocks to process in each step. You can comma separate the blocks like 8, 9, 10"})),
                ("SLG_start_percent", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001})),
                ("SLG_end_percent", ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001})),
                ("use_sage_attention", ("BOOLEAN", {"default": True, "advanced": True})),
                ("sage_attention_mode", (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."})),
                ("use_shift", ("BOOLEAN", {"default": True, "advanced": True})),
                ("shift", ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step":0.01})),
                ("large_image_side", ("INT", {"default": 832, "min": 2.0, "max": 1200, "step":2, "advanced": True, "tooltip": "The larger side of the image to resize to. The smaller side will be resized proportionally."})),
                ("image_generation_mode", (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE})),
                ("wan_model_size", (WAN_MODELS, {"default": "wan2.1_i2v_720p_14B_ret_mode", "tooltip": "The model type to use for the diffusion process."})),
                ("total_video_seconds", ("INT", {"default": 1, "min": 1, "max": 5, "step":1, "advanced": True, "tooltip": "The total duration of the video in seconds."})),
                ("clip_vision_model", (clip_vision_models, {"default": NONE, "advanced": True})),
                ("clip_vision_strength", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01})),
            ]),
            "optional": OrderedDict([
                ("lora_stack", ("LORA_STACK", {"default": None, "advanced": True})),
                ("start_image", ("IMAGE", {"default": None, "advanced": True})),
                ("start_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the start image. If disabled, the start image will be used as a static frame."})),
                ("end_image", ("IMAGE", {"default": None, "advanced": True})),
                ("end_image_clip_vision_enabled", ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Enable CLIP vision for the end image. If disabled, the end image will be used as a static frame."})),                
            ]),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative,
            clip, clip_type, clip_device, vae, use_tea_cache,
            tea_cache_model_type="wan2.1_i2v_720p_14B_ret_mode", tea_cache_rel_l1_thresh=0.4,
            tea_cache_start_percent=0.0, tea_cache_end_percent=1.0, tea_cache_cache_device="cuda",
            use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8,
            lora_stack=None, use_sage_attention=True, sage_attention_mode="auto", use_shift=False, shift=2,
            start_image=None, start_image_clip_vision_enabled=True,
            end_image=None, end_image_clip_vision_enabled=True, large_image_side=832, wan_model_size=WAN_480P,
            clip_vision_model=NONE, total_video_seconds=1, clip_vision_strength=1.0):

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
        block_swap = WanVideoBlockSwap()

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

            block_swap,          

            tea_cache, 
            tea_cache_model_type, 
            tea_cache_rel_l1_thresh, 
            tea_cache_start_percent, 
            tea_cache_end_percent,             
            tea_cache_cache_device,

            slg_wanvideo,
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
            image_generation_mode
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

                    block_swap,

                    tea_cache,
                    tea_cache_model_type,
                    tea_cache_rel_l1_thresh,
                    tea_cache_start_percent,
                    tea_cache_end_percent,
                    tea_cache_cache_device,

                    slg_wanvideo,
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
                    image_generation_mode
    ):

        output_to_terminal_successful("Generation Started started...")

        output_image = None
        working_model = model.clone()
        k_sampler_high = nodes.KSamplerAdvanced()
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

        output_to_terminal_successful("Setting up TAESD for Wan2.1...")
        self.setup_taesd_preview(clip_type, working_model)

        if (clip_vision_model != NONE):
            output_to_terminal_successful("Loading clip vision model...")
            clip_vision, = CLIPVisionLoader.load_clip(clip_vision_model)

        if (start_image is not None):
            start_image, image_width, image_height = resizer.resize(start_image, large_image_side, large_image_side, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            image_width, image_height, = wan_max_resolution.run(wan_model_size, start_image)
            start_image, image_width, image_height = resizer.resize(start_image, image_width, image_height, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            if (start_image_clip_vision_enabled) and (clip_vision is not None):
                output_to_terminal_successful("Encoding CLIP Vision for Start Image...")
                clip_vision_start_image, = CLIPVisionEncoder.encode(clip_vision, start_image, "center")

        if (end_image is not None):
            end_image, image_width, image_height = resizer.resize(end_image, large_image_side, large_image_side, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            image_width, image_height, = wan_max_resolution.run(wan_model_size, end_image)
            end_image, image_width, image_height = resizer.resize(end_image, image_width, image_height, "resize", "lanczos", 2, "0, 0, 0", "center", "cpu")
            if (end_image_clip_vision_enabled) and (clip_vision is not None):
                output_to_terminal_successful("Encoding CLIP Vision for End Image...")
                clip_vision_end_image, = CLIPVisionEncoder.encode(clip_vision, end_image, "center")

        if (sage_attention is not None):
            output_to_terminal_successful("Applying Sage Attention...")
            working_model, = sage_attention.patch(working_model, sage_attention_mode)

        if (tea_cache is not None):
            output_to_terminal_successful("Applying TeaCache...")
            working_model, = tea_cache.apply_teacache(working_model, tea_cache_model_type, tea_cache_rel_l1_thresh, tea_cache_start_percent, tea_cache_end_percent, tea_cache_cache_device)

            if (slg_wanvideo is not None) and (slg_wanvideo_blocks_string is not None) and (slg_wanvideo_blocks_string.strip() != ""):
                output_to_terminal_successful(f"Applying Skip Layer Guidance with blocks: {slg_wanvideo_blocks_string}...")
                working_model, = slg_wanvideo.slg(working_model, slg_wanvideo_start_percent, slg_wanvideo_end_percent, slg_wanvideo_blocks_string)
        
        if (model_shift is not None):
            output_to_terminal_successful("Applying Model Shift...")
            working_model, = model_shift.patch(working_model, shift)

        output_to_terminal_successful("Seetting block swap...")
        working_model, = block_swap.set_callback(working_model, 35, True, True, True)

        output_to_terminal_successful("Encoding Positive CLIP text...")
        temp_positive_clip, = text_encode.encode(clip, positive)

        output_to_terminal_successful("Encoding Negative CLIP text...")
        temp_negative_clip, = text_encode.encode(clip, negative)

        output_to_terminal_successful("Wan Image to Video started...")
        temp_positive_clip, temp_negative_clip, in_latent, = wan_image_to_video.encode(temp_positive_clip, temp_negative_clip, vae, image_width, image_height, total_frames, start_image, end_image, clip_vision_start_image, clip_vision_end_image, 0, 0, clip_vision_strength, 0.5, image_generation_mode)

        output_to_terminal_successful("High CFG KSampler started...")
        out_latent, = k_sampler_high.sample(working_model, "enable", 123456789, 15, 1.2, "uni_pc", "simple", temp_positive_clip, temp_negative_clip, in_latent, 8, 1000, "enabled", 1)

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
                output_to_terminal_successful(f"GGUF model '{GGUF}' loaded successfully using UnetLoaderGGUF")
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
                output_to_terminal_successful(f"Diffusion model '{Diffusor}' loaded successfully using UNETLoader")
                return model
            else:
                output_to_terminal_error("No Diffusion model specified")
                raise ValueError("No Diffusion model specified")
        else:
            output_to_terminal_error("Invalid model type selected. Please choose either GGUF or Diffusion model.")
            raise ValueError("Invalid model type selected. Please choose either GGUF or Diffusion model.")

    def setup_taesd_preview(self, clip_type, model):
        """
        Set up TAESD preview for Wan2.1 models based on CLIP type.
        Only applies taew2_1 models for CLIP_WAN, otherwise lets KSampler handle it.
        """
        # Only handle Wan2.1 models, let KSampler deal with everything else
        if clip_type != CLIP_WAN:
            return
            
        try:
            models_dir = folder_paths.models_dir
            vae_approx_dir = os.path.join(models_dir, "vae_approx")
            
            # Create vae_approx directory if it doesn't exist
            if not os.path.exists(vae_approx_dir):
                os.makedirs(vae_approx_dir)
                output_to_terminal_successful("Created vae_approx directory")
            
            # Look for Wan2.1 TAESD models and ensure they're in vae_approx
            wan_taesd_files = ["taew2_1.pth", "taew2_1.safetensors"]
            
            for file_name in wan_taesd_files:
                model_path = os.path.join(models_dir, file_name)
                target_path = os.path.join(vae_approx_dir, file_name)
                
                if os.path.exists(model_path):
                    if not os.path.exists(target_path):
                        try:
                            # Try to create symbolic link first (more efficient)
                            os.symlink(model_path, target_path)
                            output_to_terminal_successful(f"Created symlink for Wan2.1 TAESD: {file_name}")
                        except OSError:
                            # If symlink fails, copy the file
                            import shutil
                            shutil.copy2(model_path, target_path)
                            output_to_terminal_successful(f"Copied Wan2.1 TAESD to vae_approx: {file_name}")
                    
                    # Dynamically set the TAESD decoder name on the model's latent format
                    if hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
                        # Check if this is a Wan21 latent format
                        latent_format = model.model.latent_format
                        if (hasattr(latent_format, 'latent_channels') and 
                            latent_format.latent_channels == 16 and
                            hasattr(latent_format, 'latent_dimensions') and 
                            latent_format.latent_dimensions == 3 and
                            (latent_format.taesd_decoder_name is None or latent_format.taesd_decoder_name == "")):
                            
                            # Set the TAESD decoder name for Wan2.1 models
                            latent_format.taesd_decoder_name = "taew2_1"
                            output_to_terminal_successful("Set Wan2.1 TAESD decoder name on model - previews enabled")
                    return
            
            output_to_terminal_error("No Wan2.1 TAESD models found - previews will use fallback RGB method")
                        
        except Exception as e:
            output_to_terminal_error(f"Failed to setup Wan2.1 TAESD preview: {str(e)}")
