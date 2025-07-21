from shapely import length
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
from .core import START_IMAGE, END_IMAGE, START_END_IMAGE, END_TO_START_IMAGE, START_TO_END_TO_START_IMAGE, WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE

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
                "first_end_frame_shift": ("INT", {"default": 3, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "first_end_frame_denoise": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.0001}),
                "fill_denoise": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
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

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None, first_end_frame_shift=3, first_end_frame_denoise=0, fill_denoise=0.5, generation_mode=START_IMAGE):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * fill_denoise
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None or end_image is not None:
            if (generation_mode == START_TO_END_TO_START_IMAGE and start_image is not None and end_image is not None):
                # Fix first frame
                image[first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = start_image
                mask[:, :, first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = 0

                # Fix the middle frame (the "end" frame)
                middle = length // 2
                image[middle:middle + end_image.shape[0]] = end_image
                mask[:, :, middle:middle + end_image.shape[0]] = first_end_frame_denoise

                # Fix last frame (cycle closure)
                image[:-first_end_frame_shift - start_image.shape[0]:] = start_image
                mask[:, :, :-first_end_frame_shift - start_image.shape[0]] = 0

            elif (generation_mode == START_END_IMAGE and start_image is not None and end_image is not None):
                # Fix first frame
                image[first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = start_image
                mask[:, :, first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = 0

                # Fix last frame
                image[-first_end_frame_shift - end_image.shape[0]:] = end_image
                mask[:, :, -first_end_frame_shift - end_image.shape[0]:] = 0

            elif (generation_mode == END_TO_START_IMAGE and start_image is not None and end_image is not None):
                # Fix first frame
                image[first_end_frame_shift:first_end_frame_shift + end_image.shape[0]] = end_image
                mask[:, :, first_end_frame_shift:first_end_frame_shift + end_image.shape[0]] = 0

                # Fix last frame
                image[-first_end_frame_shift - start_image.shape[0]:] = start_image
                mask[:, :, -first_end_frame_shift - start_image.shape[0]:] = 0

            elif (generation_mode == START_IMAGE and start_image is not None):
                image[first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = start_image
                mask[:, :, first_end_frame_shift:first_end_frame_shift + start_image.shape[0]] = 0

            elif (generation_mode == END_IMAGE and end_image is not None):
                image[-first_end_frame_shift:-first_end_frame_shift - end_image.shape[0]:] = end_image
                mask[:, :, -first_end_frame_shift:-first_end_frame_shift - end_image.shape[0]] = 0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if (clip_vision_start_image is not None or clip_vision_end_image is not None):
            if (generation_mode == START_TO_END_TO_START_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                print(f"Running clipvision for start -> end -> start sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                # New sequence: start → end → start
                states = torch.cat([start_hidden, end_hidden, start_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == START_END_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                print(f"Running clipvision for start -> end sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                # New sequence: start → end
                states = torch.cat([start_hidden, end_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == END_TO_START_IMAGE and clip_vision_start_image is not None and clip_vision_end_image is not None):
                print(f"Running clipvision for end -> start sequence")
                start_hidden = clip_vision_start_image.penultimate_hidden_states
                end_hidden = clip_vision_end_image.penultimate_hidden_states

                # New sequence: end → start
                states = torch.cat([end_hidden, start_hidden], dim=-2)
                
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            elif (generation_mode == START_IMAGE and clip_vision_start_image is not None):
                print("Running clipvision for start sequence")
                clip_vision_output = clip_vision_start_image

            elif (generation_mode == END_IMAGE and clip_vision_end_image is not None):
                print("Running clipvision for end sequence")
                clip_vision_output = clip_vision_end_image

            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)