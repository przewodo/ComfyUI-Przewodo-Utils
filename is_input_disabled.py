import math
import sys
import os
from .core import COMPARE_FUNCTIONS, any_type

class IsInputDisabled:
    @classmethod
    def INPUT_TYPES(cls):

        required = {
            "invert_output": ("BOOLEAN", {"default": False, "label": "Invert Output", "tooltip": "Invert the boolean output. When False: returns True if input is disabled (None). When True: returns True if input is enabled (has value)."}),
        }

        optional = {
            "input": (any_type, {"default": None, "label": "Input", "optional": True, "tooltip": "Any input to check if it's disabled/disconnected. Returns True if this input is None or disconnected, False if it has a value."}),
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
