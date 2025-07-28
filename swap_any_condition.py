import math
import sys
import os
from .core import any_type

class SwapAnyCondition:
    @classmethod
    def INPUT_TYPES(cls):

        optional = {
            "input_a": (any_type, {"default": None, "tooltip": "First input value. Will be returned as 'input_b' if invert_values is True, otherwise as 'input_a'"}),
            "input_b": (any_type, {"default": None, "tooltip": "Second input value. Will be returned as 'input_a' if invert_values is True, otherwise as 'input_b'"}),
        }

        required = {
            "invert_values": ("BOOLEAN", {"default": True, "tooltip": "When True, swaps input_a and input_b positions in output. When False, returns inputs in original order"}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (any_type, any_type,)
    RETURN_NAMES = ("input_a", "input_b")
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input_a = None, input_b = None, invert_values=True):
        if invert_values is True:
            return (input_b, input_a,)
        else:
            return (input_a, input_b,)