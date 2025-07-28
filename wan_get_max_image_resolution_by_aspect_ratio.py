import math
import sys
import os
from .core import *
class WanGetMaxImageResolutionByAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (WAN_MODELS, {"tooltip": "WAN model type that determines maximum resolution constraints and pixel limits for optimal processing"}),
                "image": ("IMAGE", {"tooltip": "Input image to analyze for calculating optimal width and height based on aspect ratio and model constraints"})
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("Width", "Height",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, model_type, image):
        if model_type not in WAN_MODELS_CONFIG:
            raise ValueError(f"Unknown model_type: {model_type}")

        max_side = WAN_MODELS_CONFIG[model_type]['max_side']
        max_pixels = WAN_MODELS_CONFIG[model_type]['max_pixels']

        width = image.shape[2]
        height = image.shape[1]

        # Step 1: Scale so that the largest side is equal to max_side
        if width >= height:
            scale = max_side / width
        else:
            scale = max_side / height

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        total_pixels = new_width * new_height

        # Step 2: If pixel count exceeds limit, scale down further
        if total_pixels > max_pixels:
            reduction_scale = math.sqrt(max_pixels / total_pixels)
            new_width = int(round(new_width * reduction_scale))
            new_height = int(round(new_height * reduction_scale))

        return (new_width, new_height,)