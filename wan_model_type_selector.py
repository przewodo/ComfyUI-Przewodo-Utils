import nodes
from .core import *

class WanModelTypeSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (WAN_MODELS,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("Model Type",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils/Wan"

    def run(self, model_type):

        return (model_type,)