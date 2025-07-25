from .core import COMPARE_FUNCTIONS, any_type

class SwapAnyComparison:
    @classmethod
    def INPUT_TYPES(cls):

        compare_functions = list(COMPARE_FUNCTIONS.keys())

        optional = {
            "input_a": (any_type,),
            "input_b": (any_type,),
        }

        required = {
            "comparison": (compare_functions, {"default": "a == b"}),
            "value_a": (any_type,),
            "value_b": (any_type,),
        }

        return {"optional": optional, "required": required}

    RETURN_TYPES = (any_type, any_type,)
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
        
        return (input_a, input_b,)