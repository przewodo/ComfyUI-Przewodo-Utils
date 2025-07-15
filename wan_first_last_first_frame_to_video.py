import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision

class WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                             "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if (start_image is not None) and (end_image is not None):
            half = length // 2
            for i in range(half):
                alpha = i / max(half - 1, 1)
                image[i] = (1 - alpha) * start_image + alpha * end_image
            for i in range(half, length):
                alpha = (i - half) / max(length - half - 1, 1)
                image[i] = (1 - alpha) * end_image + alpha * start_image
            # Fix first frame
            mask[:, :, :1 + 3] = 0.0
            # Fix the middle frame (the "end" frame)
            middle = length // 2
            mask[:, :, middle:middle + 1] = 0.0
            # Fix last frame (cycle closure)
            mask[:, :, -1:] = 0.0        
        elif (start_image is not None) and end_image is None:
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0] + 3] = 0.0

#        if start_image is not None:
#            image[:start_image.shape[0]] = start_image
#            mask[:, :, :start_image.shape[0] + 3] = 0.0
#
#        if end_image is not None:
#            image[-end_image.shape[0]:] = end_image
#            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if (clip_vision_start_image is not None) and (clip_vision_end_image is not None):
            start_states = clip_vision_start_image.penultimate_hidden_states
            end_states = clip_vision_end_image.penultimate_hidden_states

            # Make sure both are at least 2D (frames or sequence, tokens, dim)
            # If your states are (tokens, dim), unsqueeze(0) makes it (1, tokens, dim)
            if start_states.ndim == 2:
                start_states = start_states.unsqueeze(0)
            if end_states.ndim == 2:
                end_states = end_states.unsqueeze(0)

            # Now interpolate for every frame
            half = length // 2
            clip_states = []
            for i in range(half):
                alpha = i / max(half - 1, 1)
                state = (1 - alpha) * start_states + alpha * end_states
                clip_states.append(state)
            for i in range(half, length):
                alpha = (i - half) / max(length - half - 1, 1)
                state = (1 - alpha) * end_states + alpha * start_states
                clip_states.append(state)
            # Stack to (length, tokens, dim)
            clip_states = torch.cat(clip_states, dim=0)

            # Create new output object for conditioning
            clip_vision_output = comfy.clip_vision.Output()
            clip_vision_output.penultimate_hidden_states = clip_states

        elif (clip_vision_start_image is not None) and (clip_vision_end_image is None):
            clip_vision_output = clip_vision_start_image
            
        elif (clip_vision_start_image is None) and (clip_vision_end_image is not None):
            clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)