import math

class LtxKeyFrameIndexes:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input images to calculate the indexes."}),
                "total_frames": ("INT", {
                    "display": "number",
                    "tooltip": "Total number of frames in the video. This is used to calculate the key frame indexes based on the total frames and the number of images."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Indexes",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, image, total_frames):
        num_images = len(image) if hasattr(image, '__len__') else 1
        calc_frames = total_frames - 1
        if num_images == 1:
            indexes = "0"
        else:
            step = calc_frames // (num_images - 1)
            indexes_list = [i * step for i in range(num_images - 1)]
            indexes_list.append(calc_frames)
            indexes = ", ".join(str(idx) for idx in indexes_list)
            
        return (indexes,)