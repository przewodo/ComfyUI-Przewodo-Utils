import math
import sys
import os
from .core import COMPARE_FUNCTIONS, any

class SwapAnyComparison:
    @classmethod
    def IS_CHANGED(cls, comparison_type=None, **kwargs):
        # Always trigger recalculation/redraw when comparison_type changes
        return True

    @classmethod
    def INPUT_TYPES(cls, **kwargs):

        compare_functions = list(COMPARE_FUNCTIONS.keys())

        optional = {
            "input_a": (any),
            "input_b": (any),
        }

        required = {
            "comparison": (compare_functions, {"default": "a == b"}),
            "value_a": (any),
            "value_b": (any),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("*",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(
        self,
        input_a=None,
        input_b=None,
        comparison=None,
        value_a=None,
        value_b=None,
        **kwargs
    ):
        if COMPARE_FUNCTIONS[comparison](value_a, value_b):
            return (input_b,)
        else:
            return (input_a,)
        
        return (input_a,)