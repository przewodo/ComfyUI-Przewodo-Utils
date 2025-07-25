from .core import any_type

class AppendToAnyList:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "any_list": (any_type,),
                "any": (any_type,),
            },
        }

    FUNCTION = "run"
    CATEGORY = "PrzewodoUtils"

    RETURN_TYPES = (any_type,)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = (True, False)
    RETURN_NAMES = ("list",)
    
    def run(self, any_list, any):
        
        if (any_list is None) or (not isinstance(any_list, list)):
            any_list = []

        if any is None:
            return (any_list, )
        
        if isinstance(any, list):
            any_list.extend(any)
            return (any_list, )
            
        any_list.append(any)
        return (any_list,)