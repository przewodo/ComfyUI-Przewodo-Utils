import math
import sys
import os
from .core import any_type

class FloatIfElse:
    @classmethod
    def INPUT_TYPES(cls, **kwargs):

        optional = {
            "input_true": ("FLOAT", {"default": 0.0, "label": "Input True", "tooltip": "Float value to return when conditioning is True."}),
            "input_false": ("FLOAT", {"default": 0.0, "label": "Input False", "tooltip": "Float value to return when conditioning is False."}),
        }

        required = {
            "conditioning": ("BOOLEAN", {"default": True, "tooltip": "Boolean condition that determines which float value to return. True returns input_true, False returns input_false."}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input_true=0.0, input_false=0.0, conditioning=True, **kwargs):
        if conditioning is True:
            return (input_true,)
        else:
            return (input_false,)