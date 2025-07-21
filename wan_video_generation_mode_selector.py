from .core import START_IMAGE, END_IMAGE, START_END_IMAGE, END_TO_START_IMAGE, START_TO_END_TO_START_IMAGE, WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, any

class WanVideoGenerationModeSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("Generation Mode",)
    FUNCTION = "encode"

    CATEGORY = "PrzewodoUtils/Wan"

    def encode(self, generation_mode=START_IMAGE):
        return (generation_mode)