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
        }
        
        return {"optional": optional }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"
    OUTPUT_NODE = True

    def run(self, input_a = None, input_b = None, input_c = None, input_d = None):
        value = None

        if input_a is not None:
            value = input_a
        elif input_b is not None:
            value = input_b
        elif input_c is not None:
            value = input_c
        elif input_d is not None:
            value = input_d
        
        return (value,)

    @classmethod
    def IS_CHANGED(cls, input_a=None, input_b=None, input_c=None, input_d=None):
        return (repr(input_a), repr(input_b), repr(input_c), repr(input_d))