import math
import sys
import os
from .core import any_type

class SendFirstValidValue:

    @classmethod
    def INPUT_TYPES(cls):

        optional = {
            "input_a": (any_type, {"default": None }),
            "input_b": (any_type, {"default": None }),
            "input_c": (any_type, {"default": None }),
            "input_d": (any_type, {"default": None }),
            "tick_in": ("INT", {"default": 0}),
        }
        
        return {"optional": optional }

    RETURN_TYPES = (any_type, "INT", "STRING")
    RETURN_NAMES = ("output", "tick", "debug_text")
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"
    OUTPUT_NODE = True

    def run(self, input_a = None, input_b = None, input_c = None, input_d = None, tick_in=0):
        tick_out = tick_in + 1
        debug_text = f"input_a={input_a} | input_b={input_b} | input_c={input_c} | input_d={input_d} | tick={tick_out}"

        if input_a is not None:
            return (input_a, tick_out, debug_text,)

        if input_b is not None:
            return (input_b, tick_out, debug_text,)

        if input_c is not None:
            return (input_c, tick_out, debug_text,)

        if input_d is not None:
            return (input_d, tick_out, debug_text,)
        
        return (None, tick_out, debug_text,)

    @classmethod
    def IS_CHANGED(cls, input_a=None, input_b=None, input_c=None, input_d=None, tick_out=0):
        return (repr(input_a), repr(input_b), repr(input_c), repr(input_d), repr(tick_out))