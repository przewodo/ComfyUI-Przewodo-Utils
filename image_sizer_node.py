import math
from .core import *

class ImageSizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": ([SD, SDXL, WAN_480P, WAN_720P, FLUX_KONTEXT, FLUX_1D, QWEN_IMAGE], {"tooltip": "Model type that determines the base resolution and total pixel count. Each model has optimized dimensions: SD (512x512), SDXL (1024x1024), Video 480p (832x480), Video 720p (1280x720), Flux Kontext (1024x1024), Flux 1D (1536x1536)."}),
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
            QWEN_IMAGE: 3584 * 3584
        }
        
        # Calculate the number of total pixels based on model type
        pixels = total_pixels.get(model_type, 0)
        pixels = int(pixels * image_scale)
        
        # Calculate the aspect ratio decimal
        aspect_ratio_decimal = aspect_ratio_width / aspect_ratio_height
        
        # Calculate width and height
        width = math.sqrt(pixels * aspect_ratio_decimal)
        height = pixels / width
        
        # Step 3: If model type is QWEN_IMAGE, ensure dimensions are divisible by 28
        if model_type == QWEN_IMAGE:
            # Make sure both width and height are divisible by 28 while maintaining proportions
            final_width = int(round(width))
            final_height = int(round(height))
            
            # Round to nearest multiple of 28
            final_width = round(final_width / 28) * 28
            final_height = round(final_height / 28) * 28
            
            # Ensure we don't get zero dimensions
            if final_width == 0:
                final_width = 28
            if final_height == 0:
                final_height = 28
                
            return (final_width, final_height)
        
        # Return the width and height as a tuple of integers for other model types
        return (int(round(width)), int(round(height)))