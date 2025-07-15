try:
    from .compare_numbers_to_combo import *
    from .image_scale_factor import *
    from .image_sizer_node import *
    from .swap_any_condition import *
    from .swap_any_comparison import *
    from .swap_image_comparison import *
    from .wan_get_max_image_resolution_by_aspect_ratio import *
    from .wan_first_last_first_frame_to_video import *
    
except ImportError:
    print("\033[Przewodo Utils: \033[92mFailed to load Essential nodes\033[0m")


NODE_CLASS_MAPPINGS = {
    "przewodo CompareNumbersToCombo": CompareNumbersToCombo,
    "przewodo WanGetMaxImageResolutionByAspectRatio": WanGetMaxImageResolutionByAspectRatio,
    "przewodo ImageScaleFactor": ImageScaleFactor,
    "przewodo ImageSizer": ImageSizer,
    "przewodo SwapAnyCondition": SwapAnyCondition,
    "przewodo SwapAnyComparison": SwapAnyComparison,
    "przewodo SwapImageComparison": SwapImageComparison,
    "przewodo WanFirstLastFirstFrameToVideo": WanFirstLastFirstFrameToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "przewodo CompareNumbersToCombo": "Compare Numbers to Combo",
    "przewodo WanGetMaxImageResolutionByAspectRatio": "Wan Get Max Image Resolution By Aspect Ratio",
    "przewodo ImageScaleFactor": "Image Scale Factor",    
    "przewodo ImageSizer": "Image Sizer",
    "przewodo SwapAnyCondition": "Swap any Two values in a condition",
    "przewodo SwapAnyComparison": "Swap any Two values in a comparison",
    "przewodo SwapImageComparison": "Swap Two Images in a comparison",
    "przewodo WanFirstLastFirstFrameToVideo": "WanFirstLastFirstFrameToVideo",
}