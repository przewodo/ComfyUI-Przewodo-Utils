try:
    from .compare_numbers_to_combo import *
    from .image_scale_factor import *
    from .image_sizer_node import *
    from .wan_get_max_image_resolution_by_aspect_ratio import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Essential nodes\033[0m")


NODE_CLASS_MAPPINGS = {
    "CompareNumbersToCombo": CompareNumbersToCombo,
    "WanGetMaxImageResolutionByAspectRatio": WanGetMaxImageResolutionByAspectRatio,
    "ImageScaleFactor": ImageScaleFactor,
    "ImageSizer": ImageSizer   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompareNumbersToCombo": "Compare Numbers to Combo",
    "WanGetMaxImageResolutionByAspectRatio": "Wan Get Max Image Resolution By Aspect Ratio",
    "ImageScaleFactor": "Image Scale Factor",    
    "ImageSizer": "Image Sizer"    
}