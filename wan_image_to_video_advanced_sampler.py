from collections import OrderedDict
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
import nodes
import folder_paths
from .core import *
from .wan_first_last_first_frame_to_video import WanFirstLastFirstFrameToVideo
from .wan_video_vae_decode import WanVideoVaeDecode

# Import external custom nodes using the centralized import function
imported_nodes = {}

# Import TeaCache from teacache custom node
teacache_imports = import_nodes(["teacache"], ["TeaCache"])

# Import SkipLayerGuidanceWanVideo from comfyui-kjnodes custom node
kjnodes_imports = import_nodes(["comfyui-kjnodes", "nodes", "model_optimization_nodes"], ["SkipLayerGuidanceWanVideo", "PathchSageAttentionKJ"])

# Import UnetLoaderGGUF from ComfyUI-GGUF custom node
gguf_imports = import_nodes(["ComfyUI-GGUF"], ["UnetLoaderGGUF"])


imported_nodes.update(teacache_imports)
imported_nodes.update(kjnodes_imports)
imported_nodes.update(gguf_imports)

TeaCache = imported_nodes.get("TeaCache")
SkipLayerGuidanceWanVideo = imported_nodes.get("SkipLayerGuidanceWanVideo")
UnetLoaderGGUF = imported_nodes.get("UnetLoaderGGUF")
SageAttention = imported_nodes.get("PathchSageAttentionKJ")

class WanImageToVideoAdvancedSampler:
    @classmethod
    def INPUT_TYPES(s):
        
        clip_names = [NONE] + folder_paths.get_filename_list("text_encoders")
        gguf_model_names = [NONE] + folder_paths.get_filename_list("unet_gguf")
        diffusion_models_names = [NONE] + folder_paths.get_filename_list("diffusion_models")
        vae_names = [NONE] + folder_paths.get_filename_list("vae")

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
            ]),
            "optional": OrderedDict([
                ("lora_stack", ("LORA_STACK", {"default": None, "advanced": True})),
            ]),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Diffusor_weight_dtype, Use_Model_Type, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache,
            tea_cache_model_type="wan2.1_i2v_720p_14B_ret_mode", tea_cache_rel_l1_thresh=0.4, tea_cache_start_percent=0.0, tea_cache_end_percent=1.0, tea_cache_cache_device="cuda",
            use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, lora_stack=None, use_sage_attention=True, sage_attention_mode="auto", use_shift=False, shift=2):
        
        #variables
        output_image = None
        model = self.load_model(GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype)
        output_to_terminal_successful("Loading VAE...")
        vae = nodes.VAELoader().load_vae(vae)

        output_to_terminal_successful("Loading CLIP...")
        clip, = nodes.CLIPLoader().load_clip(clip_name=clip, type=clip_type, device=clip_device)
        tea_cache = None
        sage_attention = None
        slg_wanvideo = None
        model_shift = None

        # Create TeaCache node if available
        if TeaCache is not None and use_tea_cache:
            tea_cache = TeaCache()
            tea_cache.model_type = tea_cache_model_type
            tea_cache.rel_l1_thresh = tea_cache_rel_l1_thresh
            tea_cache.start_percent = tea_cache_start_percent
            tea_cache.end_percent = tea_cache_end_percent
            tea_cache.cache_device = tea_cache_cache_device
            output_to_terminal_successful(f"TeaCache enabled")

            if (SkipLayerGuidanceWanVideo is not None) and use_SLG:
                slg_wanvideo = SkipLayerGuidanceWanVideo()
                slg_wanvideo.blocks = [int(x.strip()) for x in SLG_blocks.split(",") if x.strip().isdigit()]
                slg_wanvideo.start_percent = SLG_start_percent
                slg_wanvideo.end_percent = SLG_end_percent
                output_to_terminal_successful(f"SkipLayerGuidanceWanVideo enabled with blocks: {slg_wanvideo.blocks}")
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

        output_image = self.postprocess(model, vae, clip, positive, negative, tea_cache, sage_attention, slg_wanvideo, model_shift)
        
        return (output_image,)

    def postprocess(self, model, vae, clip, positive, negative, tea_cache, sage_attention, slg_wanvideo, model_shift):
        output_image = None
        k_sampler_high = nodes.KSamplerAdvanced()
        positive_clip = nodes.CLIPTextEncode()
        negative_clip = nodes.CLIPTextEncode()
        wan_image_to_video = WanFirstLastFirstFrameToVideo()
        wan_video_vae_decode = WanVideoVaeDecode()
        working_model = model.clone()
        in_latent = None
        out_latent = None

        output_to_terminal_successful("Postprocessing started...")
        output_to_terminal_successful("Encoding Positive CLIP text...")
        temp_positive_clip, = positive_clip.encode_text(positive, clip)

        output_to_terminal_successful("Encoding Negative CLIP text...")
        temp_negative_clip, = negative_clip.encode_text(negative, clip)

        output_to_terminal_successful("Wan Image to Video started...")
        in_latent, = wan_image_to_video.encode(temp_positive_clip, temp_negative_clip, vae, 512, 512, (16 * 2) + 1, None, None, None, None, 0, 0, 1, 0.5, START_IMAGE)

        output_to_terminal_successful("High KSampler started...")
        out_latent, = k_sampler_high.sample(working_model, "enable", 123456789, 15, 1.2, "uni_pc", "simple", positive_clip, negative_clip, in_latent, 8, 1000, "enabled", 1)

        output_to_terminal_successful("Vae Decode started...")
        output_image = wan_image_to_video.decode(out_latent, vae, START_IMAGE)

        return output_image
   
    def load_model(self, GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype):
        """
        Load the model based on the selected type.
        """
        if Use_Model_Type == MODEL_GGUF:
            if UnetLoaderGGUF is not None and GGUF != NONE:
                # Use UnetLoaderGGUF to load the GGUF model
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
