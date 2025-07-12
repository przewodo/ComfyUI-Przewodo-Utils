import math
import sys
import os

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

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

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("*",)

    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, a, b, comparison, string_true, string_false):
        if COMPARE_FUNCTIONS[comparison](a, b):
            return (string_true,)
        else:
            return (string_false,)