#!/usr/bin/env python3
"""
Simplified TAEHV Preview Integration for ComfyUI

Based on WanVideoWrapper's approach for TAEHV preview integration.
"""
import torch
import comfy.model_management as mm
from PIL import Image


def preview_to_image(latent_image):
    """Convert latent tensor to PIL Image"""
    latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                        .mul(0xFF)  # to 0..255
                        )
    if mm.directml_enabled:
        latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
    latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=mm.device_supports_non_blocking(latent_image.device))

    return Image.fromarray(latents_ubyte.numpy())


class LatentPreviewer:
    """Base class for latent previewers - ComfyUI compatible"""
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, 512)


class TAEHVPreviewer(LatentPreviewer):
    """TAEHV-based latent previewer for Wan2.1 video models"""
    
    def __init__(self, taehv_model):
        self.taehv = taehv_model
    
    def decode_latent_to_preview(self, x0):
        """Decode latent to preview image using TAEHV"""
        try:
            # Handle different input shapes
            if x0.dim() == 5:
                # Video latent [B, C, T, H, W] - take first frame
                x0 = x0[0, :, 0]  # [C, H, W]
            elif x0.dim() == 4:
                # Batch of frames [B, C, H, W] or [C, T, H, W]
                x0 = x0[0]  # Take first item
                if x0.shape[0] > 16:  # If first dim is temporal, take first frame
                    x0 = x0[0]
            
            # Ensure we have [C, H, W]
            if x0.dim() != 3:
                return None
            
            # Add batch and temporal dimensions: [C, H, W] -> [1, C, 1, H, W]
            x0 = x0.unsqueeze(0).unsqueeze(2)
            
            # Decode using TAEHV
            decoded = self.taehv.decode_video(x0, parallel=False, show_progress_bar=False)
            
            if decoded and len(decoded) > 0:
                # Extract first frame: [1, 3, T, H, W] -> [H, W, 3]
                frame = decoded[0][0, :, 0].permute(1, 2, 0)
                return preview_to_image(frame)
            
            return None
            
        except Exception as e:
            print(f"TAEHV preview error: {e}")
            return None


def install_taehv_previewer(model, taehv_model):
    """Install TAEHV as the preview decoder for a model"""
    try:
        if not hasattr(model, 'model') or not hasattr(model.model, 'latent_format'):
            return False
        
        latent_format = model.model.latent_format
        
        # Verify this is a Wan2.1 model (16-channel, 3D latents)
        if (hasattr(latent_format, 'latent_channels') and 
            latent_format.latent_channels == 16 and
            hasattr(latent_format, 'latent_dimensions') and 
            latent_format.latent_dimensions == 3):
            
            # Install TAEHV previewer
            previewer = TAEHVPreviewer(taehv_model)
            
            # Set the previewer in the latent format
            if hasattr(latent_format, 'taesd_decoder_name'):
                latent_format.taesd_decoder_name = previewer
            
            # Also try to set it directly as the previewer
            if hasattr(latent_format, 'previewer'):
                latent_format.previewer = previewer
            
            print(f"✓ TAEHV previewer installed for 16-channel video model")
            return True
        
        return False
        
    except Exception as e:
        print(f"Failed to install TAEHV previewer: {e}")
        return False
