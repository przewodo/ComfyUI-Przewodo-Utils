from collections import OrderedDict
import sys
import os
from comfy.ldm.wan import vae
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.clip_vision
import folder_paths
from .core import *

# Import external custom nodes using the centralized import function
imported_nodes = {}

# Import TeaCache from teacache custom node
teacache_imports = import_nodes(["teacache"], ["TeaCache"])

# Import SkipLayerGuidanceWanVideo from comfyui-kjnodes custom node
kjnodes_imports = import_nodes(["comfyui-kjnodes", "nodes", "model_optimization_nodes"], ["SkipLayerGuidanceWanVideo"])

# Import UnetLoaderGGUF from ComfyUI-GGUF custom node
gguf_imports = import_nodes(["ComfyUI-GGUF"], ["UnetLoaderGGUF"])



imported_nodes.update(teacache_imports)
imported_nodes.update(kjnodes_imports)
imported_nodes.update(gguf_imports)

TeaCache = imported_nodes.get("TeaCache")
SkipLayerGuidanceWanVideo = imported_nodes.get("SkipLayerGuidanceWanVideo")
UnetLoaderGGUF = imported_nodes.get("UnetLoaderGGUF")

class WanImageToVideoAdvancedSampler:
    @classmethod
    def INPUT_TYPES(s):
        
        clip_names = [NONE] + folder_paths.get_filename_list("text_encoders")
        gguf_model_names = [NONE] + folder_paths.get_filename_list("unet")
        diffusion_models_names = [NONE] + folder_paths.get_filename_list("diffusion_models")
        vae_names = [NONE] + folder_paths.get_filename_list("vae")

        return {
            "required": OrderedDict([
                ("GGUF", (gguf_model_names, {"default": NONE, "advanced": True})),
                ("Diffusor", (diffusion_models_names, {"default": NONE, "advanced": True})),
                ("Diffusor_weight_dtype", (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True})),
                ("Use Model Type", (MODEL_TYPE_LIST, {"default": MODEL_GGUF, "advanced": True})),
                ("lora_stack", ("LORA_STACK", {"default": None, "advanced": True})),
                ("positive", ("CONDITIONING", {"default": None, "advanced": True})),
                ("negative", ("CONDITIONING", {"default": None, "advanced": True})),
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

            ])
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Use_Model_Type, lora_stack, positive, negative, clip, clip_type, clip_device, vae,
            use_tea_cache, tea_cache_model_type="wan2.1_i2v_720p_14B_ret_mode", tea_cache_rel_l1_thresh=0.4, tea_cache_start_percent=0.0, tea_cache_end_percent=1.0, tea_cache_cache_device="cuda",
            use_SLG=True, SLG_blocks="10", SLG_start_percent=0.2, SLG_end_percent=0.8, Diffusor_weight_dtype="default"):
        output_image = None
        ## load the model

        model = self.load_model(GGUF, Diffusor, Use_Model_Type, Diffusor_weight_dtype)

        loadClip = nodes.CLIPLoader()
        # Load the CLIP model with the specified clip name and device
        loaded_clip, = loadClip.load_clip(clip_name=clip, type=clip_type, device=clip_device)

        # Create TeaCache node if available
        if TeaCache is not None and use_tea_cache:
            tea_cache = TeaCache(
                model_type=tea_cache_model_type,
                rel_l1_thresh=tea_cache_rel_l1_thresh,
                start_percent=tea_cache_start_percent,
                end_percent=tea_cache_end_percent,
                cache_device=tea_cache_cache_device
            )
        else:
            tea_cache = None
            output_to_terminal_error("TeaCache not available - continuing without caching optimization")
        
        return (output_image,)
    
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
