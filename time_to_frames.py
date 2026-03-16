import math
from .core import *

class TimeToFrames:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seconds": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number",
                    "tooltip": "The duration in seconds for which to calculate frames."
                }),
                "interpolation_scale": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number",
                    "tooltip": "Scale factor for interpolation."
                }),
                "framerate": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "The frame rate at which to calculate frames."
                })
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("total_frames", "total_interpolated_frames", "interpolated_framerate")

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, seconds, interpolation_scale, framerate):
        # Calculate total frames
        total_frames = int(seconds * framerate) + 1
        
        # Return using ComfyUI's UI result pattern with dimensions for JavaScript extension
        return (total_frames, int(total_frames * interpolation_scale), framerate * interpolation_scale)