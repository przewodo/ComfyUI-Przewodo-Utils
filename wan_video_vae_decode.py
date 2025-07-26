import nodes
import torch
from .core import *
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
    FUNCTION = "decode"

    CATEGORY = "PrzewodoUtils/Wan"

    def decode(self, latent, vae, first_end_frame_shift, generation_mode):

        out_images = self.vae_decode(vae, latent, 512, 64, 64, 8)
        
        # Process VAE output to correct format [0, 1] range
        out_images = torch.clamp((out_images + 1.0) / 2.0, min=0.0, max=1.0)

        total_shift = (first_end_frame_shift * 4)
        start_shift = (total_shift // 2)
        end_shift = (total_shift // 2)

        if (generation_mode == START_TO_END_TO_START_IMAGE):
            output_to_terminal_successful("Decoding start -> end -> start frame sequence")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if (start_shift + end_shift) > 0:
                output_to_terminal_successful(f"Removing first {start_shift} and last {end_shift + 1} frames")
                out_images = out_images[start_shift:-end_shift - 1]
            else:
                out_images = out_images[:-1]

        elif (generation_mode == START_END_IMAGE):
            output_to_terminal_successful("Decoding start -> end frame sequence")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if (start_shift + end_shift) > 0:
                output_to_terminal_successful(f"Removing first {start_shift} and last {end_shift + 1} frames")
                out_images = out_images[start_shift:-end_shift - 1]
            else:
                out_images = out_images[:-1]

        elif (generation_mode == END_TO_START_IMAGE):
            output_to_terminal_successful("Decoding end -> start frame sequence")
            # Remove first start_shift frames and last end_shift frames from decoded images
            if (start_shift + end_shift) > 0:
                output_to_terminal_successful(f"Removing first {start_shift} and last {end_shift + 1} frames")
                out_images = out_images[start_shift:-end_shift - 1]
            else:
                out_images = out_images[:-1]

        elif (generation_mode == START_IMAGE):
            output_to_terminal_successful("Decoding start frame sequence")
            # Remove total_shift frames from beginning only
            if (total_shift) > 0:
                output_to_terminal_successful(f"Removing first {total_shift + 1} frames")
                out_images = out_images[total_shift:-1]
            else:
                out_images = out_images[:-1]

        elif (generation_mode == END_IMAGE):
            output_to_terminal_successful("Decoding end frame sequence")
            if (total_shift) > 0:
                output_to_terminal_successful(f"Removing last {total_shift + 1} frames")
                out_images = out_images[:-total_shift - 1]
            else:
                out_images = out_images[:-1]

        return (out_images,)
    
    def vae_decode(self, vae, latent, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(latent["samples"], tile_size // compression, tile_size // compression, overlap // compression, temporal_size, temporal_overlap)

        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images    