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
                "a": ("INT", {"tooltip": "First number to compare (left side of comparison)."}),
                "b": ("INT", {"tooltip": "Second number to compare (right side of comparison)."}),
                "comparison": (compare_functions, {"default": "a == b", "tooltip": "Comparison operation to perform between numbers a and b (==, !=, <, >, <=, >=)."}),
                "string_true": ("STRING", {"default": "", "tooltip": "Value to return when the comparison is true. Can be any string or value."}),
                "string_false": ("STRING", {"default": "", "tooltip": "Value to return when the comparison is false. Can be any string or value."}),
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