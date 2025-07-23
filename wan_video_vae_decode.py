import nodes
from .core import START_IMAGE, END_IMAGE, START_END_IMAGE, END_TO_START_IMAGE, START_TO_END_TO_START_IMAGE, WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, CYAN, RESET
class WanVideoVaeDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", ),
                "vae": ("VAE", ),
                "first_end_frame_shift": ("INT", {"default": 3, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "generation_mode": (WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE, {"default": START_IMAGE}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "encode"

    CATEGORY = "PrzewodoUtils/Wan"

    def encode(self, latent, vae, first_end_frame_shift, generation_mode):

        out_images = None
        if (first_end_frame_shift == 0):
            print(f"{RESET+CYAN}" f"Decoding without first_end_frame_shift" f"{RESET}")
            out_images = vae.decode(latent["samples"])

        total_shift = (first_end_frame_shift * 16)    
        
        if (generation_mode == START_TO_END_TO_START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start -> end -> start frame sequence" f"{RESET}")
            # Remove first_end_frame_shift frames from beginning and end
            samples = latent["samples"]
            if samples.shape[2] > total_shift:  # Ensure we don't remove more frames than available
                new_samples = samples[:, :, (total_shift // 2):-(total_shift // 2), :, :]
                new_latent = {"samples": new_samples}
            else:
                new_latent = latent  # Keep original if not enough frames to remove
            out_images = vae.decode(new_latent["samples"])

        elif (generation_mode == START_END_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start -> end frame sequence" f"{RESET}")
            # Remove first_end_frame_shift frames from beginning and end
            samples = latent["samples"]
            if samples.shape[2] > total_shift:
                new_samples = samples[:, :, (total_shift // 2):-(total_shift // 2), :, :]
                new_latent = {"samples": new_samples}
            else:
                new_latent = latent
            out_images = vae.decode(new_latent["samples"])

        elif (generation_mode == END_TO_START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding end -> start frame sequence" f"{RESET}")
            # Remove first_end_frame_shift frames from beginning and end
            samples = latent["samples"]
            if samples.shape[2] > total_shift:
                new_samples = samples[:, :, (total_shift // 2):-(total_shift // 2), :, :]
                new_latent = {"samples": new_samples}
            else:
                new_latent = latent
            out_images = vae.decode(new_latent["samples"])

        elif (generation_mode == START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start frame sequence" f"{RESET}")
            # Remove 2*first_end_frame_shift frames from beginning only
            samples = latent["samples"]
            frames_to_remove = total_shift
            if samples.shape[2] > frames_to_remove:
                new_samples = samples[:, :, frames_to_remove:, :, :]
                new_latent = {"samples": new_samples}
            else:
                new_latent = latent
            out_images = vae.decode(new_latent["samples"])

        elif (generation_mode == END_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding end frame sequence" f"{RESET}")
            # Remove 2*first_end_frame_shift frames from end only
            samples = latent["samples"]
            frames_to_remove = total_shift
            if samples.shape[2] > frames_to_remove:
                new_samples = samples[:, :, :-frames_to_remove, :, :]
                new_latent = {"samples": new_samples}
            else:
                new_latent = latent
            out_images = vae.decode(new_latent["samples"])

        return (out_images,)