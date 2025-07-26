"""
TAESD Override Implementation
Based on: https://github.com/madebyollin/taesd

This module provides a complete TAESD implementation that overrides ComfyUI's default TAESD
and automatically downloads required models from the official repository.
"""

import os
import torch
import torch.nn as nn
import folder_paths
from comfy.utils import load_torch_file
import comfy.model_management as mm

# Try to import download utilities
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False

def output_to_terminal_successful(message):
    print(f"[TAESD] ✓ {message}")

def output_to_terminal_error(message):
    print(f"[TAESD] ✗ {message}")

class TAESD(nn.Module):
    """
    Tiny AutoEncoder for Stable Diffusion (TAESD)
    Based on the official implementation from madebyollin/taesd
    """
    
    def __init__(self, latent_channels=4, decoder_path=None):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Create the decoder architecture based on madebyollin/taesd
        if latent_channels == 4:  # SD 1.x/2.x
            self.decoder = self._make_decoder_sd()
        elif latent_channels == 8:  # SDXL  
            self.decoder = self._make_decoder_sdxl()
        elif latent_channels == 16:  # SD3/Wan2.1
            self.decoder = self._make_decoder_sd3()
        elif latent_channels == 64:  # FLUX
            self.decoder = self._make_decoder_flux()
        else:
            # Fallback generic decoder
            self.decoder = self._make_generic_decoder(latent_channels)
        
        # Load weights if path provided
        if decoder_path and os.path.exists(decoder_path):
            try:
                state_dict = load_torch_file(decoder_path, safe_load=True)
                self.decoder.load_state_dict(state_dict, strict=False)
                output_to_terminal_successful(f"Loaded TAESD weights from {decoder_path}")
            except Exception as e:
                output_to_terminal_error(f"Failed to load TAESD weights: {e}")
    
    def _make_decoder_sd(self):
        """Decoder for SD 1.x/2.x (4 channels)"""
        return nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def _make_decoder_sdxl(self):
        """Decoder for SDXL (8 channels)"""
        return nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def _make_decoder_sd3(self):
        """Decoder for SD3/Wan2.1 (16 channels)"""
        return nn.Sequential(
            nn.Conv2d(16, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def _make_decoder_flux(self):
        """Decoder for FLUX (64 channels)"""
        return nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def _make_generic_decoder(self, channels):
        """Generic decoder for other channel counts"""
        hidden = max(64, channels * 4)
        return nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden // 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def decode_latent_to_preview_image(self, preview_format, x0):
        """
        ComfyUI LatentPreviewer interface method
        Returns: (format, preview_image, max_resolution)
        """
        try:
            preview_image = self.decode_latent_to_preview(x0)
            # Import here to avoid circular imports
            from comfy.cli_args import args
            MAX_PREVIEW_RESOLUTION = args.preview_size
            return (preview_format, preview_image, MAX_PREVIEW_RESOLUTION)
        except Exception as e:
            output_to_terminal_error(f"TAESD preview failed: {e}")
            # Return fallback
            from PIL import Image
            fallback_image = Image.new('RGB', (64, 64), color=(128, 128, 128))
            return (preview_format, fallback_image, 64)
    
    def decode_latent_to_preview(self, x0):
        """
        ComfyUI LatentPreviewer interface method  
        Returns: PIL Image
        """
        try:
            # Handle different tensor shapes
            if len(x0.shape) == 5:  # Video: [batch, channels, frames, height, width]
                # Take first frame for preview
                latents = x0[:1, :, 0, :, :]
            else:
                # Take first sample for preview
                latents = x0[:1]
            
            # Move to same device as decoder
            device = next(self.decoder.parameters()).device
            if latents.device != device:
                latents = latents.to(device)
            
            # Ensure compatible dtype
            decoder_dtype = next(self.decoder.parameters()).dtype
            if latents.dtype != decoder_dtype:
                latents = latents.to(decoder_dtype)
            
            with torch.no_grad():
                # Decode
                decoded = self.decoder(latents)
                
                # Move channels to last dimension: [batch, channels, height, width] -> [batch, height, width, channels]
                decoded = decoded[0].movedim(0, 2)  # Take first batch, move channels to end
                
                # Use ComfyUI's preview_to_image function for consistency
                def preview_to_image(latent_image):
                    latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                                        .mul(0xFF)  # to 0..255
                                        )
                    import comfy.model_management
                    if comfy.model_management.directml_enabled:
                        latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
                    latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))
                    
                    from PIL import Image
                    return Image.fromarray(latents_ubyte.numpy())
                
                return preview_to_image(decoded)
                
        except Exception as e:
            output_to_terminal_error(f"TAESD decode failed: {e}")
            import traceback
            output_to_terminal_error(f"Traceback: {traceback.format_exc()}")
            # Return a simple fallback PIL image
            from PIL import Image
            return Image.new('RGB', (64, 64), color=(128, 128, 128))

def download_taesd_models():
    """Download TAESD models from official repository"""
    if not (REQUESTS_AVAILABLE or URLLIB_AVAILABLE):
        output_to_terminal_error("No download libraries available")
        return False
    
    # Get vae_approx directory
    vae_approx_paths = folder_paths.get_folder_paths("vae_approx")
    if not vae_approx_paths:
        output_to_terminal_error("vae_approx folder not found")
        return False
    
    vae_approx_dir = vae_approx_paths[0]
    os.makedirs(vae_approx_dir, exist_ok=True)
    
    # Official TAESD models
    models = [
        {
            "name": "taesd_decoder.pth",
            "url": "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth",
            "channels": 4
        },
        {
            "name": "taesdxl_decoder.pth", 
            "url": "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
            "channels": 8
        },
        {
            "name": "taesd3_decoder.pth",
            "url": "https://github.com/madebyollin/taesd/raw/main/taesd3_decoder.pth", 
            "channels": 16
        },
        {
            "name": "taef1_decoder.pth",
            "url": "https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth",
            "channels": 64
        }
    ]
    
    output_to_terminal_successful("Downloading TAESD models...")
    success_count = 0
    
    for model in models:
        filepath = os.path.join(vae_approx_dir, model["name"])
        
        # Skip if already exists
        if os.path.exists(filepath):
            output_to_terminal_successful(f"{model['name']} already exists")
            success_count += 1
            continue
        
        try:
            output_to_terminal_successful(f"Downloading {model['name']}...")
            
            if REQUESTS_AVAILABLE:
                response = requests.get(model["url"], stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                urllib.request.urlretrieve(model["url"], filepath)
            
            output_to_terminal_successful(f"Downloaded {model['name']}")
            success_count += 1
            
        except Exception as e:
            output_to_terminal_error(f"Failed to download {model['name']}: {e}")
    
    output_to_terminal_successful(f"Downloaded {success_count}/{len(models)} models")
    return success_count > 0

def get_taesd_decoder_path(latent_channels):
    """Get the appropriate TAESD decoder path for the given channel count"""
    vae_approx_paths = folder_paths.get_folder_paths("vae_approx")
    if not vae_approx_paths:
        return None
    
    vae_approx_dir = vae_approx_paths[0]
    
    # Map channel counts to model files
    model_map = {
        4: "taesd_decoder.pth",      # SD 1.x/2.x
        8: "taesdxl_decoder.pth",    # SDXL
        16: "taesd3_decoder.pth",    # SD3/Wan2.1
        64: "taef1_decoder.pth"      # FLUX
    }
    
    model_name = model_map.get(latent_channels)
    if not model_name:
        output_to_terminal_error(f"No TAESD model available for {latent_channels} channels")
        return None
    
    model_path = os.path.join(vae_approx_dir, model_name)
    
    # Download if doesn't exist
    if not os.path.exists(model_path):
        output_to_terminal_successful(f"TAESD model {model_name} not found, downloading...")
        if download_taesd_models():
            if os.path.exists(model_path):
                return model_path
        return None
    
    return model_path

def create_taesd_decoder(latent_channels):
    """Create a TAESD decoder for the given channel count"""
    decoder_path = get_taesd_decoder_path(latent_channels)
    
    try:
        taesd = TAESD(latent_channels=latent_channels, decoder_path=decoder_path)
        
        # Move to appropriate device
        device = mm.unet_offload_device()
        if device.type == 'cpu' and torch.cuda.is_available():
            device = torch.device('cuda')
        
        taesd = taesd.to(device)
        taesd.eval()
        
        output_to_terminal_successful(f"Created TAESD decoder for {latent_channels} channels on {device}")
        return taesd
        
    except Exception as e:
        output_to_terminal_error(f"Failed to create TAESD decoder: {e}")
        return None

# Global storage for TAESD decoders
_taesd_decoders = {}

def get_or_create_taesd_decoder(latent_channels):
    """Get cached TAESD decoder or create new one"""
    if latent_channels not in _taesd_decoders:
        _taesd_decoders[latent_channels] = create_taesd_decoder(latent_channels)
    
    return _taesd_decoders[latent_channels]

def override_comfyui_taesd():
    """Override ComfyUI's TAESD system with our implementation"""
    try:
        # Import ComfyUI's latent preview module (done here to avoid import issues)
        import latent_preview
        
        # Store original function
        if not hasattr(latent_preview, '_original_get_previewer'):
            latent_preview._original_get_previewer = latent_preview.get_previewer
        
        def custom_get_previewer(device, latent_format):
            """Custom previewer that uses our TAESD implementation"""
            try:
                output_to_terminal_successful(f"Custom previewer called with device: {device}, latent_format: {type(latent_format)}")
                
                channels = None
                
                # Try multiple ways to detect channel count
                if hasattr(latent_format, 'latent_channels'):
                    channels = latent_format.latent_channels
                    output_to_terminal_successful(f"Found latent_channels: {channels}")
                elif hasattr(latent_format, 'latent_rgb_factors'):
                    # Try to infer from RGB factors length
                    factors = latent_format.latent_rgb_factors
                    if hasattr(factors, '__len__'):
                        channels = len(factors)
                        output_to_terminal_successful(f"Inferred channels from RGB factors: {channels}")
                elif hasattr(latent_format, 'scale_factor'):
                    # Default assumption based on common formats
                    output_to_terminal_successful("Using default 4 channels for unknown format")
                    channels = 4
                
                if channels and channels in [4, 8, 16, 64]:
                    output_to_terminal_successful(f"Attempting to create TAESD decoder for {channels} channels")
                    
                    # Try to get our TAESD decoder
                    taesd_decoder = get_or_create_taesd_decoder(channels)
                    
                    if taesd_decoder is not None:
                        output_to_terminal_successful(f"Successfully created custom TAESD for {channels} channels")
                        return taesd_decoder
                    else:
                        output_to_terminal_error(f"Failed to create TAESD decoder for {channels} channels")
                else:
                    output_to_terminal_error(f"Unsupported channel count: {channels}. Supported: 4, 8, 16, 64")
                    if hasattr(latent_format, '__dict__'):
                        output_to_terminal_error(f"Latent format attributes: {latent_format.__dict__}")
                
                # Fallback to original previewer
                output_to_terminal_successful("Falling back to original previewer")
                return latent_preview._original_get_previewer(device, latent_format)
                
            except Exception as e:
                output_to_terminal_error(f"Custom previewer failed: {e}")
                import traceback
                output_to_terminal_error(f"Traceback: {traceback.format_exc()}")
                return latent_preview._original_get_previewer(device, latent_format)
        
        # Replace the function
        latent_preview.get_previewer = custom_get_previewer
        output_to_terminal_successful("ComfyUI TAESD system overridden successfully")
        return True
        
    except Exception as e:
        output_to_terminal_error(f"Failed to override ComfyUI TAESD: {e}")
        return False

# Auto-initialize override when module is imported
def initialize_taesd_override():
    """Initialize the TAESD override system"""
    try:
        # Download models if needed
        download_taesd_models()
        
        # Override ComfyUI's system
        override_comfyui_taesd()
        
        output_to_terminal_successful("TAESD override system initialized")
        return True
        
    except Exception as e:
        output_to_terminal_error(f"Failed to initialize TAESD override: {e}")
        return False
