import math
import sys
import os
from .core import any

class HasInputvalue:
    @classmethod
    def INPUT_TYPES(cls):

        optional = {
            "input": (any, {"default": None, "label": "Input"}),
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