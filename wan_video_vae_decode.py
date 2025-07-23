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
        samples = latent["samples"]
        out_images = vae.decode(samples)
        if len(out_images.shape) == 5:
            out_images = out_images.reshape(-1, out_images.shape[-3], out_images.shape[-2], out_images.shape[-1])

        total_shift = (first_end_frame_shift * 4)
        start_shift = (total_shift // 2)
        end_shift = (total_shift // 2) + 1

        if (generation_mode == START_TO_END_TO_START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start -> end -> start frame sequence" f"{RESET}")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if out_images.shape[0] > (start_shift + end_shift) and (start_shift + end_shift) > 0:
                out_images = out_images[start_shift:-end_shift]

        elif (generation_mode == START_END_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start -> end frame sequence" f"{RESET}")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if out_images.shape[0] > (start_shift + end_shift) and (start_shift + end_shift) > 0:
                out_images = out_images[start_shift:-end_shift]

        elif (generation_mode == END_TO_START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding end -> start frame sequence" f"{RESET}")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if out_images.shape[0] > (start_shift + end_shift) and (start_shift + end_shift) > 0:
                out_images = out_images[start_shift:-end_shift]

        elif (generation_mode == START_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding start frame sequence" f"{RESET}")
            # Remove total_shift frames from beginning only
            if out_images.shape[0] > (total_shift) and (total_shift) > 0:
                out_images = out_images[total_shift:]

        elif (generation_mode == END_IMAGE):
            print(f"{RESET+CYAN}" f"Decoding end frame sequence" f"{RESET}")
            if out_images.shape[0] > (total_shift) and (total_shift) > 0:
                out_images = out_images[:-total_shift]

        if len(out_images.shape) == 5:
            out_images = out_images.reshape(-1, out_images.shape[-3], out_images.shape[-2], out_images.shape[-1])

        return (out_images,)