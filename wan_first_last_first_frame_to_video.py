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
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for video generation"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for video generation"}),
                "vae": ("VAE", {"tooltip": "VAE model for encoding/decoding video frames"}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 2, "tooltip": "Width of the generated video in pixels (must be divisible by 2)"}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 2, "tooltip": "Height of the generated video in pixels (must be divisible by 2)"}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "Number of frames in the generated video (step of 4 recommended for optimal performance)"}),
                "first_end_frame_shift": ("INT", {"default": 0, "min": 0, "max": 80, "step": 1, "tooltip": "Frame shift offset for first and end frames positioning in the sequence"}),
                "first_end_frame_denoise": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.0001, "tooltip": "Denoising strength for first and end frames (0=no denoising, 1=full denoising)"}),
                "fill_denoise": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01, "tooltip": "Denoising strength for intermediate frames between keyframes"}),
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE, "tooltip": "Video generation pattern: start only, end only, start->end, end->start, or start->end->start"}),
                "clip_vision_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Strength multiplier for CLIP vision influence on video generation"}),
            },
            "optional": {
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision encoding of the start image for enhanced conditioning"}),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision encoding of the end image for enhanced conditioning"}),
                "start_image": ("IMAGE", {"tooltip": "Starting image/frame for the video sequence"}),
                "end_image": ("IMAGE", {"tooltip": "Ending image/frame for the video sequence"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MASK")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "PrzewodoUtils/Wan"

    def encode(self, positive_high, negative_high, positive_low, negative_low, vae, width, height, length, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None, first_end_frame_shift=3, first_end_frame_denoise=0, clip_vision_strength=1.0, fill_denoise=0.5, generation_mode=START_IMAGE):

        latent, concat_latent_image, mask = self.make_latent_and_mask(
            vae, width, height, length, first_end_frame_shift, 
            first_end_frame_denoise, fill_denoise, generation_mode, 
            start_image, end_image
        )

        positive_high, negative_high, positive_low, negative_low = self.make_conditioning_and_clipvision(
            positive_high, negative_high, positive_low, negative_low,
            concat_latent_image, mask, clip_vision_start_image, clip_vision_end_image,
            clip_vision_strength, generation_mode
        )

        return (positive_high, negative_high, positive_low, negative_low, latent, mask)

    def make_latent_and_mask(self, vae, width, height, length, first_end_frame_shift, first_end_frame_denoise, fill_denoise, generation_mode, start_image=None, end_image=None):
        """
        Creates latent tensors, image tensors, and masks for video generation based on the specified mode.
        
        Args:
            vae: VAE model for encoding
            width: Video width in pixels
            height: Video height in pixels
            length: Number of frames
            first_end_frame_shift: Frame shift offset
            first_end_frame_denoise: Denoising strength for keyframes
            fill_denoise: Denoising strength for intermediate frames
            generation_mode: Video generation pattern
            start_image: Optional start image
            end_image: Optional end image
            
        Returns:
            tuple: (latent, concat_latent_image, mask)
        """
        batch_size = 1
        total_shift = (first_end_frame_shift * 4)
        total_length = length + total_shift
        
        latent = torch.zeros([batch_size, 16, ((total_length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        image = torch.ones((total_length, height, width, 3)) * fill_denoise
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        output_to_terminal_successful(f"Generating {length} frames with a padding of {total_shift}. Total Frames: {total_length}")

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
                output_to_terminal_successful(f"First KeyFrame: {start_shift} ({(start_shift) - (start_shift - 1)} frames)")
                output_to_terminal_successful(f"Middle KeyFrame: {(total_length // 2)} ({(total_length // 2) - ((total_length // 2) - 1)} frames)")
                output_to_terminal_successful(f"End KeyFrame: {total_length - end_shift} ({(total_length - end_shift + 1) - (total_length - end_shift)} frames)")

            elif (generation_mode == START_END_IMAGE and start_image is not None and end_image is not None):
                output_to_terminal_successful("Generating start -> end frame sequence")
                # Fix first frame
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0]] = first_end_frame_denoise

                # Fix last frame (cycle closure)
                image[-end_image.shape[0]:] = end_image
                mask[:, :, -end_image.shape[0]:] = first_end_frame_denoise
                output_to_terminal_successful(f"First KeyFrame: {start_shift} ({(start_shift) - (start_shift - 1)} frames)")
                output_to_terminal_successful(f"End KeyFrame: {total_length - end_shift} ({(total_length - end_shift + 1) - (total_length - end_shift)} frames)")

            elif (generation_mode == END_TO_START_IMAGE and start_image is not None and end_image is not None):
                output_to_terminal_successful("Generating end -> start frame sequence")
                # Fix first frame
                image[:end_image.shape[0]] = end_image
                mask[:, :, :end_image.shape[0]] = first_end_frame_denoise

                # Fix last frame (cycle closure)
                image[-start_image.shape[0]:] = start_image
                mask[:, :, -start_image.shape[0]:] = first_end_frame_denoise
                output_to_terminal_successful(f"First KeyFrame: {start_shift} ({(start_shift) - (start_shift - 1)} frames)")
                output_to_terminal_successful(f"End KeyFrame: {total_length - end_shift} ({(total_length - end_shift + 1) - (total_length - end_shift)} frames)")

            elif (generation_mode == START_IMAGE and start_image is not None):
                output_to_terminal_successful("Generating start frame sequence")
                # Fix first frame
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0]] = first_end_frame_denoise
                output_to_terminal_successful(f"First KeyFrame: {start_shift} ({(start_shift) - (start_shift - 1)} frames)")

            elif (generation_mode == TEXT_TO_VIDEO):
                output_to_terminal_successful("Generating text to video sequence")

            # Force the first frame to not be denoised
            if first_end_frame_denoise > 0 and generation_mode != TEXT_TO_VIDEO:
                mask[:, :, start_shift:start_shift + 1] = 0

            # Force the middle frame to not be denoised
            if first_end_frame_denoise > 0 and generation_mode != TEXT_TO_VIDEO:
                mask[:, :, (total_length // 2):(total_length // 2) + 1] = 0

            # Force the last frame to not be denoised
            if first_end_frame_denoise > 0 and generation_mode != TEXT_TO_VIDEO:
                mask[:, :, total_length - end_shift:total_length - end_shift + 1] = 0


        concat_latent_image = vae.encode_tiled(image[:,:,:,:3], 512, 512, 64, 64, 8)
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        out_latent = {}
        out_latent["samples"] = latent

        return out_latent, concat_latent_image, mask

    def make_conditioning_and_clipvision(self, positive_high, negative_high, positive_low, negative_low, concat_latent_image, mask, clip_vision_start_image, clip_vision_end_image, clip_vision_strength, generation_mode):
        """
        Sets up conditioning values and processes CLIP vision outputs for video generation.
        
        Args:
            positive_high: High-level positive conditioning
            negative_high: High-level negative conditioning
            positive_low: Low-level positive conditioning
            negative_low: Low-level negative conditioning
            concat_latent_image: Encoded latent image tensor
            mask: Mask tensor for conditioning
            clip_vision_start_image: Optional CLIP vision encoding of start image
            clip_vision_end_image: Optional CLIP vision encoding of end image
            clip_vision_strength: Strength multiplier for CLIP vision influence
            generation_mode: Video generation pattern
            
        Returns:
            tuple: (positive_high, negative_high, positive_low, negative_low)
        """
        # Set conditioning values for latent and mask
        positive_high = node_helpers.conditioning_set_values(positive_high, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative_high = node_helpers.conditioning_set_values(negative_high, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        # Only process low conditioning if they are not None (dual sampler mode)
        if positive_low is not None:
            positive_low = node_helpers.conditioning_set_values(positive_low, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        if negative_low is not None:
            negative_low = node_helpers.conditioning_set_values(negative_low, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        clip_vision_output = None

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

            elif (generation_mode == TEXT_TO_VIDEO and clip_vision_start_image is not None):
                output_to_terminal_successful("Running clipvision for text to video sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states

                start_hidden = start_hidden * clip_vision_strength

                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = start_hidden

            if clip_vision_output is not None:
                positive_high = node_helpers.conditioning_set_values(positive_high, {"clip_vision_output": clip_vision_output})
                negative_high = node_helpers.conditioning_set_values(negative_high, {"clip_vision_output": clip_vision_output})
                # Only process low conditioning if they are not None (dual sampler mode)
                if positive_low is not None:
                    positive_low = node_helpers.conditioning_set_values(positive_low, {"clip_vision_output": clip_vision_output})
                if negative_low is not None:
                    negative_low = node_helpers.conditioning_set_values(negative_low, {"clip_vision_output": clip_vision_output})

        return positive_high, negative_high, positive_low, negative_low