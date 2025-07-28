import nodes
from .core import *

class WanModelTypeSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (WAN_MODELS, {"tooltip": "Select WAN model type to pass through. Used for connecting model type selection to other WAN nodes"}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("Model Type",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, model_type):

        return (model_type,)