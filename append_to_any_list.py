from .core import any_type

class AppendToAnyList:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "any_list": (any_type, {"tooltip": "The existing list to append/merge to. Can be empty or contain any type of items."}),
                "any": (any_type, {"tooltip": "Single item to append to the list, or array to merge with the list. If this is a list, all items will be added. If it's a single item, only that item will be appended."}),
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