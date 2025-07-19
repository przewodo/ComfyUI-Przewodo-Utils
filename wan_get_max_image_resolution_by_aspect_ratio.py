import math
import sys
import os
class WanGetMaxImageResolutionByAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["Wan 480p","Wan 720p"],),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("Width","Height",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, model_type, image):
        # Define per-model target largest side and max pixel count
        config = {
            'Wan 480p': {'max_side': 832, 'max_pixels': 832 * 480},
            'Wan 720p': {'max_side': 1280, 'max_pixels': 1280 * 720}
        }

        if model_type not in config:
            raise ValueError(f"Unknown model_type: {model_type}")

        max_side = config[model_type]['max_side']
        max_pixels = config[model_type]['max_pixels']

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

        return (new_width, new_height)