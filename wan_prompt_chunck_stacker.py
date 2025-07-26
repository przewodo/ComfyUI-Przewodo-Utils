import nodes
from .core import *

class WanPromptChunkStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "previous_prompt": (any_type, {"default": None,} ),
                "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                "chunk_index_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "The starting index for this chunk video to be used."}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, previous_prompt=None, positive_prompt="", negative_prompt="", chunk_index_start=0):
        if (previous_prompt is None):
            previous_prompt = []

        previous_prompt.append([positive_prompt, negative_prompt, chunk_index_start])

        return (previous_prompt,)