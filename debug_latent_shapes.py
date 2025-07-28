from .core import any_type
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
                "latent": ("LATENT", {"tooltip": "The latent tensor to analyze. This node will print all dimensions and return a specific dimension value."}),
                "shape_index": ("INT", {"tooltip": "Index of the dimension to return (0=batch, 1=channels, 2=height, 3=width for typical latents). Uses modulo to prevent index errors."})
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "run"

    CATEGORY = "PrzewodoUtils"

    def run(self, latent, shape_index):

        latent_shape_len = len(latent["samples"].shape)

        for i in range(latent_shape_len):
            print(f"latent.shape[{i}] = {latent['samples'].shape[i]}")

        return (latent['samples'].shape[shape_index % latent_shape_len],)