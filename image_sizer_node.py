import math
from .core import *

class ImageSizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": ([SD, SDXL, WAN_480P, WAN_720P, FLUX_KONTEXT, FLUX_1D, QWEN_IMAGE, LTX2_480P, LTX2_720P, LTX2_1080P], {"tooltip": "Model type that determines the base resolution and total pixel count. Each model has optimized dimensions: SD (512x512), SDXL (1024x1024), Video 480p (832x480), Video 720p (1280x720), Flux Kontext (1024x1024), Flux 1D (1536x1536)."}),
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
                }),
                "image_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Scale factor to apply to the final image dimensions."
                })
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, model_type, aspect_ratio_width, aspect_ratio_height, image_scale):
        # Use the shared calculation function
        width, height = self._calculate_dimensions(model_type, aspect_ratio_width, aspect_ratio_height, image_scale)
        
        # Return using ComfyUI's UI result pattern with dimensions for JavaScript extension
        return {
            "ui": {"dimensions": [{"width": width, "height": height}]}, 
            "result": (width, height)
        }
    
    @staticmethod
    def _calculate_dimensions(model_type, aspect_ratio_width, aspect_ratio_height, image_scale):
        """Calculate the final image dimensions based on input parameters"""
        # Define the total pixel counts for each model type
        total_pixels = {
            SD: 512 * 512,
            SDXL: 1024 * 1024,
            WAN_480P: 832 * 480,
            WAN_720P: 1280 * 720,
            FLUX_KONTEXT: 1024 * 1024,
            FLUX_1D: 1536 * 1536,
            QWEN_IMAGE: 3584 * 3584,
            LTX2_480P: 854 * 854,
            LTX2_720P: 1280 * 1280,
            LTX2_1080P: 1920 * 1920
        }
        
        # Calculate the number of total pixels based on model type
        pixels = total_pixels.get(model_type, 0)
        pixels = int(pixels * image_scale)
        
        # Calculate the aspect ratio decimal
        aspect_ratio_decimal = aspect_ratio_width / aspect_ratio_height
        
        # Calculate width and height
        width = math.sqrt(pixels * aspect_ratio_decimal)
        height = pixels / width
        
        # QWEN_IMAGE: dimensions divisible by 28
        if model_type == QWEN_IMAGE:
            final_width = int(round(width))
            final_height = int(round(height))
            final_width = round(final_width / 28) * 28
            final_height = round(final_height / 28) * 28
            if final_width == 0:
                final_width = 28
            if final_height == 0:
                final_height = 28
            return (final_width, final_height)

        # LTX2 models: set smaller side to base, calculate larger side
        ltx2_bases = {
            LTX2_480P: 480,
            LTX2_720P: 720,
            LTX2_1080P: 1080
        }
        if model_type in ltx2_bases:
            base = ltx2_bases[model_type]
            if aspect_ratio_width == aspect_ratio_height:
                final_width = final_height = base
            elif aspect_ratio_width > aspect_ratio_height:
                final_height = base
                final_width = int(round(base * (aspect_ratio_width / aspect_ratio_height)))
                # Ensure largest side is divisible by 2
                if final_width % 2 != 0:
                    final_width += 1
            else:
                final_width = base
                final_height = int(round(base * (aspect_ratio_height / aspect_ratio_width)))
                # Ensure largest side is divisible by 2
                if final_height % 2 != 0:
                    final_height += 1
            return (final_width, final_height)

        # Other model types: default calculation
        return (int(round(width)), int(round(height)))