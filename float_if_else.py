import math
import sys
import os
from .core import any

class FloatIfElse:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always trigger recalculation/redraw when comparison_type changes
        return True

    @classmethod
    def INPUT_TYPES(cls, **kwargs):

        optional = {
            "input_true": ("FLOAT", {"default": 0.0, "label": "Input True"}),
            "input_false": ("FLOAT", {"default": 0.0, "label": "Input False"}),
        }

        required = {
            "conditioning": ("BOOLEAN", {"default": True}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = ("FLOAT")
    RETURN_NAMES = ("Output")
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input_true=0.0, input_false=0.0, conditioning=True, **kwargs):
        if conditioning is True:
            return (input_true,)
        else:
            return (input_false,)