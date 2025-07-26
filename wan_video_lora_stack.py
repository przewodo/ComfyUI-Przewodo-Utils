import folder_paths
from .core import *

class WanVideoLoraStack:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ([NONE] + folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "optional": {
                "previous_lora": (any_type,),
            },
        }

    RETURN_TYPES = ("lora",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "PrzewodoUtils/Wan"

    def load_lora(self, lora_name, strength_model, strength_clip, previous_lora=None):
        if (previous_lora is None):
            previous_lora = []

        if strength_model == 0 and strength_clip == 0:
            return (previous_lora,)
        
        if (previous_lora is None):
            previous_lora = []

        previous_lora.append({"name": lora_name, "strength_model": strength_model, "strength_clip": strength_clip})

        return (previous_lora,)