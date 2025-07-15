import math
import sys
import os
from .core import COMPARE_FUNCTIONS, any

class IsInputDisabled:
    @classmethod
    def IS_CHANGED(cls):
        return True

    @classmethod
    def INPUT_TYPES(cls, **kwargs):

        required = {
            "invert_output": ("BOOLEAN", {"default": False, "label": "Invert Output" }),
        }

        optional = {
            "input": (any, {"default": None, "label": "Input", "optional": True}),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("Is Input Disabled",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input=None, invert_output=False):
        if input is None:
            return (invert_output != True,)
        return (invert_output == True,)
