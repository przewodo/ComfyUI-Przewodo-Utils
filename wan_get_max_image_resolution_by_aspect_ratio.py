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
                "image": ("IMAGE",),
                "resolution_divisible_by": ("INT",{ "default": 2, "step":1, "display": "number"})
            }
        }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("Width","Height",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, model_type, image, resolution_divisible_by):
        total_pixels = {
            'Wan 480p': 832 * 480,
            'Wan 720p': 1280 * 720
        }
    
        # Calculate the number of total pixels based on model type
        pixels = total_pixels.get(model_type, 0)
        
        width = image.shape[2]
        height = image.shape[1]       
        
        out_width = resolution_divisible_by * int(round(((width / height) * math.sqrt(pixels / (width / height))) / resolution_divisible_by))
        out_height = resolution_divisible_by * int(round((math.sqrt(pixels / (width / height))) /resolution_divisible_by))
                
        return (out_width, out_height)