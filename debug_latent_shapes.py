from .core import any
from comfy.ldm.modules import attention as comfy_attention
import logging
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import folder_paths
import comfy.model_management as mm
from comfy.cli_args import args

class DebugLatentShapes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "shape_index": ("INT",)
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("*",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, latent, shape_index):

        latent_shape_len = len(latent["samples"].shape)

        for i in range(latent_shape_len):
            print(f"latent.shape[{i}] = {latent['samples'].shape[i]}")

        return (latent['samples'].shape[shape_index % latent_shape_len],)