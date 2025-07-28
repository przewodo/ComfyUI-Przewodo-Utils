from .core import *

class WanVideoGenerationModeSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE, "tooltip": "Video generation pattern: start only, end only, start->end, end->start, or start->end->start frame sequences"}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("Generation Mode",)
    FUNCTION = "encode"

    CATEGORY = "PrzewodoUtils/Wan"

    def encode(self, generation_mode=START_IMAGE):
        return (generation_mode,)