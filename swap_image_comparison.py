from .core import COMPARE_FUNCTIONS, any_type

class SwapImageComparison:
    @classmethod
    def INPUT_TYPES(cls):

        compare_functions = list(COMPARE_FUNCTIONS.keys())

        optional = {
            "input_a": ("IMAGE", {"tooltip": "First image input to swap. Will be returned as 'input_b' if comparison is true, otherwise as 'input_a'"}),
            "input_b": ("IMAGE", {"tooltip": "Second image input to swap. Will be returned as 'input_a' if comparison is true, otherwise as 'input_b'"}),
        }

        required = {
            "value_a": (any_type, {"tooltip": "First value for comparison operation. Used as left operand in the comparison"}),
            "value_b": (any_type, {"tooltip": "Second value for comparison operation. Used as right operand in the comparison"}),
            "comparison": (compare_functions, {"default": "a == b", "tooltip": "Comparison operation to evaluate between value_a and value_b. If true, image inputs are swapped"}),
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