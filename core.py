import math
import sys
import os

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

WAN_480P = 'Wan 480p'
WAN_720P = 'Wan 720p'
WAN_MODELS = [WAN_480P, WAN_720P]
WAN_MODELS_CONFIG = {
    WAN_480P: { 'max_side': 832, 'max_pixels': 832 * 480, 'model_name': WAN_480P},
    WAN_720P: { 'max_side': 1280, 'max_pixels': 1280 * 720, 'model_name': WAN_720P},
}

START_IMAGE = "Start Image"
END_IMAGE = "End Image"
START_END_IMAGE = "Start to End Image"
END_TO_START_IMAGE = "End to Start Image"
START_TO_END_TO_START_IMAGE = "Start to End to Start Image"
WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE = [START_IMAGE, END_IMAGE, START_END_IMAGE, END_TO_START_IMAGE, START_TO_END_TO_START_IMAGE]

BLUE = "\033[1;34m"
CYAN = "\033[36m"
RESET = "\033[0m"