import nodes
from .core import *

class WanPromptChunkStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "previous_prompt": (any_type, {"default": None, "tooltip": "Previous prompt stack to append to. If None, creates a new stack. Used for chaining multiple prompt chunks together"}),
                "lora_stack": (any_type, {"default": None, "advanced": True, "tooltip": "Stack of LoRAs to apply to the diffusion model. Each LoRA modifies the model's behavior."}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Positive prompt text for this chunk. Describes what you want to generate in the video segment"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Negative prompt text for this chunk. Describes what you want to avoid in the video segment"}),
                "chunk_index_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "The starting index for this chunk video to be used."}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, previous_prompt=None, lora_stack=None, positive_prompt="", negative_prompt="", chunk_index_start=0):
        if (previous_prompt is None):
            previous_prompt = []

        previous_prompt.append([positive_prompt, negative_prompt, chunk_index_start, lora_stack])

        return (previous_prompt,)