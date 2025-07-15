import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import importlib.util
import os
from collections import OrderedDict
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
import folder_paths
from comfy.samplers import KSampler, SCHEDULER_HANDLERS, SCHEDULER_NAMES, SAMPLER_NAMES


class WanImageToVideoAdvancedSampler:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = ["None"] + [x for x in folder_paths.get_filename_list("unet_gguf")]
        dequant_dtype = ["default", "target", "float32", "float16", "bfloat16"]
        patch_dtype = ["default", "target", "float32", "float16", "bfloat16"]
        diffusion_models = ["None"] + [x for x in folder_paths.get_filename_list("diffusion_models")]
        models_types = ["Diffusion", "GGUF"]
        weight_dtype = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
        vae_names = ["None"] + [x for x in folder_paths.get_filename_list("vae")]

        return {
            "optional": OrderedDict([
                ("gguf_model", (unet_names, )),
                ("dequant_dtype", (dequant_dtype, {"default": "default"})),
                ("patch_dtype", (patch_dtype, {"default": "default"})),
                ("patch_on_device", ("BOOLEAN", {"default": False})),

                ("diff_model", (diffusion_models, )),
                ("weight_dtype", (weight_dtype, {"default": "default"})),
                ("model_type", (models_types, {"default": "Diffusion"})),
                ("vae", (vae_names, {"default": "None"})),

#                ("positive", ("CONDITIONING", )),
#                ("negative", ("CONDITIONING", )),
#                ("vae", ("VAE", )),
#                ("width", ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16})),
#                ("height", ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16})),
#                ("length", ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4})),
#                ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096})),
#                ("clip_vision_start_image", ("CLIP_VISION_OUTPUT", )),
#                ("clip_vision_end_image", ("CLIP_VISION_OUTPUT", )),
#                ("start_image", ("IMAGE", )),
#                ("end_image", ("IMAGE", )),
            ])
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self):
#        model = None
#
#        if gguf_model and gguf_model != "None":
#            # Use UnetLoaderGGUFAdvanced from ComfyUI-GGUF/nodes.py
#            unet_loader = UnetLoaderGGUFAdvanced()
#            model = unet_loader.load_unet_model(
#                gguf_model,
#                dequant_dtype=dequant_dtype,
#                patch_dtype=patch_dtype,
#                patch_on_device=patch_on_device
#            )
#
#            if not model:
#                raise ValueError(f"Failed to load GGUF model: {gguf_model}")
#            
#        elif diff_model and diff_model != "None":
#            model_tuple = self.load_diff_model(diff_model, weight_dtype)
#            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
#            if not model:
#                raise ValueError(f"Failed to load Diffusion model: {diff_model}")
#
#        return None
#
#    def load_diff_model(self, unet_name, weight_dtype):
#        if unet_name == "None":
#            return (None,)
#
#        model_options = {}
#        if weight_dtype == "fp8_e4m3fn":
#            model_options["dtype"] = torch.float8_e4m3fn
#        elif weight_dtype == "fp8_e4m3fn_fast":
#            model_options["dtype"] = torch.float8_e4m3fn
#            model_options["fp8_optimizations"] = True
#        elif weight_dtype == "fp8_e5m2":
#            model_options["dtype"] = torch.float8_e5m2
#
#        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
#        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (None, None, None,), 
