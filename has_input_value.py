import math
import sys
import os
from .core import any_type

class HasInputvalue:
    @classmethod
    def INPUT_TYPES(cls):

        optional = {
            "input": (any_type, {"default": None, "label": "Input", "tooltip": "Any input value to check for existence. Returns True if the input has a value (not None), False if it's None or disconnected."}),
        }

        return {"optional": optional}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("Input", "Has Input Value",)
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(self, input=None):
        if input is None:
            return (False,)
        
        return (True,)