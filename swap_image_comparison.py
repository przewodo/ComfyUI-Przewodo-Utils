from .core import COMPARE_FUNCTIONS, any

class SwapImageComparison:
    @classmethod
    def INPUT_TYPES(cls):

        compare_functions = list(COMPARE_FUNCTIONS.keys())

        optional = {
            "input_a": ("IMAGE",),
            "input_b": ("IMAGE",),
        }

        required = {
            "comparison": (compare_functions, {"default": "a == b"}),
            "value_a": (any,),
            "value_b": (any,),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (("IMAGE"), ("IMAGE"),)
    RETURN_NAMES = ("input_a", "input_b")
    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    def run(
        self,
        input_a=None,
        input_b=None,
        comparison=None,
        value_a=None,
        value_b=None,
        **kwargs
    ):
        if COMPARE_FUNCTIONS[comparison](value_a, value_b):
            return (input_b, input_a,)
        else:
            return (input_a, input_b,)
        
        return (input_a, input_b)