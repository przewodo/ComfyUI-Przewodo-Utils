import math
import sys
import os
from .core import any

class SwapAnyCondition:
    @classmethod
    def INPUT_TYPES(cls):

        optional = {
            "input_a": (any, {"default": None}),
            "input_b": (any, {"default": None}),
        }

        required = {
            "invert_values": ("BOOLEAN", {"default": True}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (any, any,)
    RETURN_NAMES = ("input_a", "input_b")
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input_a = None, input_b = None, invert_values=True):
        if invert_values is True:
            return (input_b, input_a,)
        else:
            return (input_a, input_b,)