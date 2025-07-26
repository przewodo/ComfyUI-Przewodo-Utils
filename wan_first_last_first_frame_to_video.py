import sys
import os
from comfy.ldm.wan import vae
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.clip_vision
from .core import *

# Force color support
try:
    import colorama
    colorama.init(autoreset=True, convert=True, strip=False)
except ImportError:
    pass

class WanFirstLastFirstFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "first_end_frame_shift": ("INT", {"default": 0, "min": 0, "max": 80, "step": 1}),
                "first_end_frame_denoise": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.0001}),
                "fill_denoise": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE}),
                "clip_vision_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "PrzewodoUtils/Wan"

    def encode(self, positive, negative, vae, width, height, length, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None, first_end_frame_shift=3, first_end_frame_denoise=0, clip_vision_strength=1.0, fill_denoise=0.5, generation_mode=START_IMAGE):
        
        batch_size = 1
        total_shift = (first_end_frame_shift * 4)
        total_length = length + total_shift

        output_to_terminal_successful(f"Generating {length} frames with a padding of {total_shift}. Total Frames: {total_length}")

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:total_length].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-total_length:].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

        latent = torch.zeros([batch_size, 16, ((total_length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        image = torch.ones((total_length, height, width, 3)) * fill_denoise
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None or end_image is not None:
            start_shift = (total_shift // 2) + 1 if first_end_frame_shift != 0 else 0
            end_shift = (total_shift // 2) + 1 if first_end_frame_shift != 0 else 0
            middle_start = (total_length // 2) - ((total_length // 6) // 2)
            middle_end = (total_length // 2) + ((total_length // 6) // 2)

            if (generation_mode == START_TO_END_TO_START_IMAGE and start_image is not None and end_image is not None):
                output_to_terminal_successful("Generating start -> end -> start frame sequence")

                # Fix first section (start frames)
                image[0:middle_start] = start_image
                mask[:, :, 0:middle_start] = first_end_frame_denoise
                output_to_terminal_successful(f"Start sequence: frames 0-{middle_start - 1} ({middle_start} frames)")

                # Fix the middle frame (the "end" frame)
                image[middle_start:middle_end] = end_image
                mask[:, :, middle_start:middle_end] = first_end_frame_denoise
                output_to_terminal_successful(f"Middle sequence: frames {middle_start}-{middle_end - 1} ({middle_end - middle_start} frames)")

                # Fix last section (return to start frames)
                image[middle_end:total_length - middle_end] = start_image
                mask[:, :, middle_end:total_length - middle_end] = first_end_frame_denoise
                output_to_terminal_successful(f"End sequence: frames {middle_end}-{total_length - 1} ({total_length - middle_end} frames)")

            elif (generation_mode == START_END_IMAGE and start_image is not None and end_image is not None):
                output_to_terminal_successful("Generating start -> end frame sequence")
                # Fix first frame
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0]] = first_end_frame_denoise

                # Fix last frame (cycle closure)
                image[-end_image.shape[0]:] = end_image
                mask[:, :, -end_image.shape[0]:] = first_end_frame_denoise

            elif (generation_mode == END_TO_START_IMAGE and start_image is not None and end_image is not None):
                output_to_terminal_successful("Generating end -> start frame sequence")
                # Fix first frame
                image[:end_image.shape[0]] = end_image
                mask[:, :, :end_image.shape[0]] = first_end_frame_denoise

                # Fix last frame (cycle closure)
                image[-start_image.shape[0]:] = start_image
                mask[:, :, -start_image.shape[0]:] = first_end_frame_denoise

            elif (generation_mode == START_IMAGE and start_image is not None):
                output_to_terminal_successful("Generating start frame sequence")
                # Fix first frame
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0]] = first_end_frame_denoise

            elif (generation_mode == END_IMAGE and end_image is not None):
                output_to_terminal_successful("Generating end frame sequence")
                # Fix last frame (cycle closure)
                image[-end_image.shape[0]:] = end_image
                mask[:, :, -end_image.shape[0]:] = first_end_frame_denoise

            # Force the first frame to not be denoised
            if first_end_frame_denoise > 0:
                mask[:, :, start_shift:start_shift + 1] = 0

            # Force the middle frame to not be denoised
            if first_end_frame_denoise > 0:
                mask[:, :, (total_length // 2):(total_length // 2) + 1] = 0

            # Force the last frame to not be denoised
            if first_end_frame_denoise > 0:
                mask[:, :, total_length - end_shift:total_length - end_shift + 1] = 0

            output_to_terminal_successful(f"First KeyFrame: {start_shift} ({(start_shift) - (start_shift - 1)} frames)")
            output_to_terminal_successful(f"Middle KeyFrame: {(total_length // 2)} ({(total_length // 2) - ((total_length // 2) - 1)} frames)")
            output_to_terminal_successful(f"End KeyFrame: {total_length - end_shift} ({(total_length - end_shift + 1) - (total_length - end_shift)} frames)")

        concat_latent_image = vae.encode_tiled(image[:,:,:,:3], 512, 512, 64, 64, 8)
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if (clip_vision_start_image is not None or clip_vision_end_image is not None):
            if (generation_mode == START_TO_END_TO_START_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                output_to_terminal_successful("Running clipvision for start -> end -> start sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                # Strengthen CLIP vision influence
                start_hidden = start_hidden * clip_vision_strength
                end_hidden = end_hidden * clip_vision_strength

                # New sequence: start → end → start
                states = torch.cat([start_hidden, end_hidden, start_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == START_END_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                output_to_terminal_successful("Running clipvision for start -> end sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                # Strengthen CLIP vision influence
                start_hidden = start_hidden * clip_vision_strength
                end_hidden = end_hidden * clip_vision_strength

                # New sequence: start → end
                states = torch.cat([start_hidden, end_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == END_TO_START_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                output_to_terminal_successful("Running clipvision for end -> start sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states 

                # Strengthen CLIP vision influence
                start_hidden = start_hidden * clip_vision_strength
                end_hidden = end_hidden * clip_vision_strength

                # New sequence: end → start
                states = torch.cat([end_hidden, start_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == START_IMAGE and clip_vision_start_image is not None):
                output_to_terminal_successful("Running clipvision for start sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states

                start_hidden = start_hidden * clip_vision_strength

                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = start_hidden

            elif (generation_mode == END_IMAGE and clip_vision_end_image is not None):
                output_to_terminal_successful("Running clipvision for end sequence")
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                end_hidden = end_hidden * clip_vision_strength

                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = end_hidden

            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)
