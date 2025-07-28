import math

class ImageScaleFactor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to calculate scale factor for. The node will analyze the image dimensions to determine scaling."}),
                "final_size": ("INT",{
                    "default": 1024,
                    "min": 1,
                    "step":1,
                    "display": "number",
                    "tooltip": "Target size for the larger side of the image. The scale factor will be calculated to resize the larger dimension to this value while maintaining aspect ratio."
                })
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("scale_factor",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, image, final_size):
        # Get dimensions from the image tensor
        # Assuming image shape is (B, H, W, C) or (1, H, W, C)
        height = image.shape[1]
        width = image.shape[2]

        # Determine current larger side
        current_larger = max(width, height)

        # Calculate scale factor
        scale_factor = 1.1
        scale_factor = float(final_size) / float(current_larger)

        return (scale_factor,)