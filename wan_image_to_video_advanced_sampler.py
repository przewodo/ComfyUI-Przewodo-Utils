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

# Import TeaCache from the teacache custom node
try:
    teacache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "custom_nodes", "teacache")
    output_to_terminal("TeaCache Path: " + teacache_path)
    
    # Import using importlib to avoid conflicts with existing 'nodes' module
    import importlib.util
    teacache_nodes_file = os.path.join(teacache_path, "nodes.py")
    
    if os.path.exists(teacache_nodes_file):
        spec = importlib.util.spec_from_file_location("teacache_nodes", teacache_nodes_file)
        teacache_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(teacache_module)
        TeaCache = teacache_module.TeaCache
        output_to_terminal_successful("TeaCache imported successfully!")
    else:
        raise ImportError(f"TeaCache nodes.py not found at {teacache_nodes_file}")
        
except (ImportError, AttributeError) as e:
    output_to_terminal_error(f"Warning: TeaCache not found ({e}). Please install the teacache custom node.")
    TeaCache = None


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
                ("tea_cache_rel_l1_thresh", ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})),
                ("tea_cache_start_percent", ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."})),
                ("tea_cache_end_percent", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."})),
                ("tea_cache_cache_device", (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside"})),
            ])
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, GGUF, Diffusor, Use_Model_Type, lora_stack, positive, negative, clip, clip_type, clip_device, vae, use_tea_cache,
            tea_cache_model_type="wan2.1_i2v_720p_14B_ret_mode", tea_cache_rel_l1_thresh=0.4, tea_cache_start_percent=0.0, tea_cache_end_percent=1.0, tea_cache_cache_device="cuda"):
        loadClip = nodes.CLIPLoader()
        # Load the CLIP model with the specified clip name and device
        loaded_clip, _ = loadClip.load_clip(clip_name=clip, type=clip_type, device=clip_device)

        # Create TeaCache node if available
        if TeaCache is not None and use_tea_cache:
            tea_cache = TeaCache()
        else:
            tea_cache = None
            print("TeaCache not available - continuing without caching optimization")
        
        return ("IMAGE",), 
