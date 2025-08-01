import math
from .core import *

class ImageSizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["SD","SDXL", WAN_480P, WAN_720P, "Flux Kontext", "Flux 1D"], {"tooltip": "Model type that determines the base resolution and total pixel count. Each model has optimized dimensions: SD (512x512), SDXL (1024x1024), Video 480p (832x480), Video 720p (1280x720), Flux Kontext (1024x1024), Flux 1D (1536x1536)."}),
                "aspect_ratio_width": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number",
                    "tooltip": "Width component of the desired aspect ratio. Combined with aspect_ratio_height to calculate the final dimensions while maintaining the model's total pixel count."
                }),
                "aspect_ratio_height": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number",
                    "tooltip": "Height component of the desired aspect ratio. Combined with aspect_ratio_width to calculate the final dimensions while maintaining the model's total pixel count."
                })
            }
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("Width", "Height")

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, model_type, aspect_ratio_width, aspect_ratio_height):
        # Define the total pixel counts for SD and SDXL
        total_pixels = {
            'SD': 512 * 512,
            'SDXL': 1024 * 1024,
            WAN_480P: 832 * 480,
            WAN_720P: 1280 * 720,
            'Flux Kontext': 1024 * 1024,
            'Flux 1D': 1536 * 1536
        }
        
        # Calculate the number of total pixels based on model type
        pixels = total_pixels.get(model_type, 0)
        
        # Calculate the aspect ratio decimal
        aspect_ratio_decimal = aspect_ratio_width / aspect_ratio_height
        
        # Calculate width and height
        width = math.sqrt(pixels * aspect_ratio_decimal)
        height = pixels / width
        max_side = max(width, height)

        # Step 1: Scale so that the largest side is equal to max_side
        if width >= height:
            scale = max_side / width
        else:
            scale = max_side / height

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        total_pixels = new_width * new_height

        # Step 2: If pixel count exceeds limit, scale down further
        if total_pixels > pixels:
            reduction_scale = math.sqrt(pixels / total_pixels)
            new_width = int(round(new_width * reduction_scale))
            new_height = int(round(new_height * reduction_scale))
        
        # Return the width and height as a tuple of integers
        return (int(round(width)), int(round(height)),)