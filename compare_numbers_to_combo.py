import math
import sys
import os
from .core import COMPARE_FUNCTIONS, any_type

class CompareNumbersToCombo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": ("INT",),
                "b": ("INT",),
                "comparison": (compare_functions, {"default": "a == b"}),
                "string_true": ("STRING", {"default": ""}),
                "string_false": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, a, b, comparison, string_true, string_false):
        if COMPARE_FUNCTIONS[comparison](a, b):
            return (string_true,)
        else:
            return (string_false,)