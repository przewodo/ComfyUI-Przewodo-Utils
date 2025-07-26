"""
TAEHV Latent Preview Integration for ComfyUI

This module provides integration between TAEHV (Tiny AutoEncoder for Hunyuan Video)
and ComfyUI's latent preview system for Wan2.1 models.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


class TAEHVPreviewer:
    """
    TAEHV-based latent previewer for Wan2.1 models.
    
    This class provides the interface needed by ComfyUI's preview system
    to generate previews using TAEHV instead of standard TAESD.
    """
    
    def __init__(self, taehv_model):
        """
        Initialize the TAEHV previewer.
        
        Args:
            taehv_model: Loaded TAEHV model instance
        """
        self.taehv = taehv_model
        self.device = next(taehv_model.parameters()).device
        self.dtype = next(taehv_model.parameters()).dtype
    
    def decode_latent_to_preview(self, x0):
        """
        Decode latent tensor to preview image(s).
        
        Args:
            x0: Latent tensor with shape [B, C, T, H, W] where C=16 for Wan2.1
            
        Returns:
            Preview image tensor with shape [T, H, W, C] where C=3 (RGB)
        """
        try:
            # Ensure input is on correct device and dtype
            if x0.device != self.device:
                x0 = x0.to(self.device)
            if x0.dtype != self.dtype:
                x0 = x0.to(self.dtype)
            
            # Ensure input has correct shape [B, C, T, H, W]
            if x0.dim() == 4:  # [B, C, H, W] -> add time dimension
                x0 = x0.unsqueeze(2)  # [B, C, 1, H, W]
            elif x0.dim() == 5:  # [B, C, T, H, W] - correct shape
                pass
            else:
                raise ValueError(f"Unexpected latent shape: {x0.shape}")
            
            # Decode using TAEHV
            with torch.no_grad():
                # TAEHV expects NTCHW format, we have BCTHW
                # Rearrange: [B, C, T, H, W] -> [B, T, C, H, W]
                x_input = x0.permute(0, 2, 1, 3, 4)
                
                # Decode video frames
                decoded = self.taehv.decode_video(
                    x_input, 
                    parallel=False,  # Use sequential for memory efficiency
                    show_progress_bar=False  # No progress for previews
                )
                
                # decoded shape: [B, T, C, H, W] with C=3 (RGB)
                # Convert to [T, H, W, C] format expected by ComfyUI
                if decoded.dim() == 5:
                    # Take first batch, rearrange to [T, C, H, W] then [T, H, W, C]
                    preview = decoded[0].permute(0, 2, 3, 1)  # [T, H, W, C]
                else:
                    # Single frame case
                    preview = decoded.permute(0, 2, 3, 1)  # [1, H, W, C]
                
                # Clamp values to [0, 1] range
                preview = torch.clamp(preview, 0.0, 1.0)
                
                # If single frame, remove time dimension
                if preview.shape[0] == 1:
                    preview = preview[0]  # [H, W, C]
                
                return preview.cpu().float()
                
        except Exception as e:
            # Fallback to a simple RGB conversion if TAEHV fails
            print(f"TAEHV preview failed, using fallback: {e}")
            return self._fallback_preview(x0)
    
    def _fallback_preview(self, x0):
        """
        Fallback preview method using simple latent-to-RGB conversion.
        
        Args:
            x0: Latent tensor
            
        Returns:
            RGB preview tensor
        """
        # Simple RGB conversion as fallback
        # Take first 3 channels of latent and normalize
        if x0.dim() == 5:  # [B, C, T, H, W]
            rgb_latent = x0[0, :3, 0]  # [C, H, W] - first batch, first 3 channels, first frame
        elif x0.dim() == 4:  # [B, C, H, W]
            rgb_latent = x0[0, :3]  # [C, H, W]
        else:
            rgb_latent = x0[:3]  # [C, H, W]
        
        # Normalize to [0, 1] range
        rgb_latent = (rgb_latent - rgb_latent.min()) / (rgb_latent.max() - rgb_latent.min() + 1e-8)
        
        # Convert to [H, W, C] format
        preview = rgb_latent.permute(1, 2, 0).cpu().float()
        
        return preview


def install_taehv_previewer(model, taehv_model):
    """
    Install TAEHV previewer into the model's latent format.
    
    Args:
        model: ComfyUI model object
        taehv_model: Loaded TAEHV model instance
    """
    if hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
        latent_format = model.model.latent_format
        
        # Create TAEHV previewer
        previewer = TAEHVPreviewer(taehv_model)
        
        # Install custom decode method
        latent_format.taehv_previewer = previewer
        
        # Override the taesd decode method if it exists
        if hasattr(latent_format, 'decode_latent_to_preview'):
            latent_format._original_decode_latent_to_preview = latent_format.decode_latent_to_preview
        
        latent_format.decode_latent_to_preview = previewer.decode_latent_to_preview
        
        return True
    
    return False


def preview_to_image(latent_image):
    """
    Convert latent preview tensor to PIL Image.
    
    Args:
        latent_image: Tensor with shape [H, W, C] or [T, H, W, C]
        
    Returns:
        PIL Image
    """
    if isinstance(latent_image, torch.Tensor):
        latent_image = latent_image.detach().cpu().numpy()
    
    # Handle different shapes
    if latent_image.ndim == 4:  # [T, H, W, C] - take first frame
        latent_image = latent_image[0]
    elif latent_image.ndim == 3:  # [H, W, C] - single frame
        pass
    else:
        raise ValueError(f"Unexpected image shape: {latent_image.shape}")
    
    # Ensure values are in [0, 1] range
    latent_image = np.clip(latent_image, 0.0, 1.0)
    
    # Convert to uint8
    latent_image = (latent_image * 255).astype(np.uint8)
    
    return Image.fromarray(latent_image)
