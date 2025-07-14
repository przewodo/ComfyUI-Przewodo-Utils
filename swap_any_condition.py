import math
import sys
import os
from .core import any

class SwapAnyCondition:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always trigger recalculation/redraw when comparison_type changes
        return True

    @classmethod
    def INPUT_TYPES(cls, **kwargs):

        optional = {
            "input_a": (any),
            "input_b": (any),
        }

        required = {
            "conditioning": ("BOOLEAN", {"default": True}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("*",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(
        self,
        input_a = None,
        input_b = None,
        conditioning=True,
        **kwargs
    ):
        if conditioning is False:
            return (input_b,)
        else:
            return (input_a,)
        
        return (input_a,)