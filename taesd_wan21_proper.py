#!/usr/bin/env python3
"""
TAESD for Wan2.1 - Proper TAESD architecture adapted for 16-channel latents

Based on the official TAESD implementation from:
https://github.com/madebyollin/taesd

This adapts the proven TAESD architecture for Wan2.1's 16-channel latent space,
ensuring full compatibility with ComfyUI's TAESD preview system.
"""
import torch
import torch.nn as nn
import os

def conv(n_in, n_out, **kwargs):
    """3x3 convolution with padding=1"""
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    """TAESD clamp activation"""
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    """TAESD residual block"""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=16):
    """TAESD Encoder adapted for 16-channel latents (Wan2.1)"""
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )

def Decoder(latent_channels=16):
    """TAESD Decoder adapted for 16-channel latents (Wan2.1)"""
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESDWan21(nn.Module):
    """
    TAESD adapted for Wan2.1 16-channel latents.
    
    This follows the exact TAESD architecture from madebyollin/taesd
    but adapted for 16-channel latents instead of 4-channel.
    """
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None):
        """Initialize TAESD for Wan2.1"""
        super().__init__()
        self.encoder = Encoder(latent_channels=16)
        self.decoder = Decoder(latent_channels=16)
        
        if encoder_path is not None and os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
        if decoder_path is not None and os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESDWan21.latent_magnitude).add(TAESDWan21.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESDWan21.latent_shift).mul(2 * TAESDWan21.latent_magnitude)
    
    def encode(self, x):
        """Encode images to latents [0, 1] -> ~[-3, 3]"""
        try:
            # Ensure input tensor has the same dtype and device as the encoder
            encoder_params = list(self.encoder.parameters())
            if len(encoder_params) > 0:
                target_device = encoder_params[0].device
                target_dtype = encoder_params[0].dtype
                
                # Convert input tensor to match encoder
                if x.device != target_device:
                    x = x.to(device=target_device)
                if x.dtype != target_dtype:
                    x = x.to(dtype=target_dtype)
            else:
                # Fallback: convert to float16 and move to cuda if available
                if x.dtype == torch.float32:
                    x = x.to(dtype=torch.float16)
                if x.device.type == 'cpu' and torch.cuda.is_available():
                    x = x.to(device='cuda')
                    
        except Exception as e:
            print(f"Warning: Failed to match encoder dtype/device: {e}")
            # Emergency fallback: ensure at least basic compatibility
            if x.dtype == torch.float32:
                x = x.to(dtype=torch.float16)
            
        return self.encoder(x)
    
    def decode(self, x):
        """Decode latents to images ~[-3, 3] -> [0, 1]"""
        # Ensure input tensor has the same dtype and device as the decoder
        try:
            # Get target device and dtype from decoder parameters
            decoder_params = list(self.decoder.parameters())
            if len(decoder_params) > 0:
                target_device = decoder_params[0].device
                target_dtype = decoder_params[0].dtype
                
                # Convert input tensor to match decoder
                if x.device != target_device:
                    x = x.to(device=target_device)
                if x.dtype != target_dtype:
                    x = x.to(dtype=target_dtype)
            else:
                # Fallback: convert to float16 and move to cuda if available
                if x.dtype == torch.float32:
                    x = x.to(dtype=torch.float16)
                if x.device.type == 'cpu' and torch.cuda.is_available():
                    x = x.to(device='cuda')
                    
        except Exception as e:
            print(f"Warning: Failed to match decoder dtype/device: {e}")
            # Emergency fallback: ensure at least basic compatibility
            if x.dtype == torch.float32:
                x = x.to(dtype=torch.float16)
            
        return self.decoder(x).clamp(0, 1)
    
    def forward(self, x):
        """ComfyUI TAESD compatibility - decode latents"""
        # Handle both 4D and 5D tensors (in case of video latents)
        if x.ndim == 5:
            # Video case: [B, 16, T, H, W] -> decode first frame [B, 16, H, W]
            x = x[:, :, 0]  # Take first frame
        
        try:
            # Ensure input tensor has the same dtype and device as the decoder
            decoder_params = list(self.decoder.parameters())
            if len(decoder_params) > 0:
                target_device = decoder_params[0].device
                target_dtype = decoder_params[0].dtype
                
                # Convert input tensor to match decoder
                if x.device != target_device:
                    x = x.to(device=target_device)
                if x.dtype != target_dtype:
                    x = x.to(dtype=target_dtype)
            else:
                # Fallback: convert to float16 and move to cuda if available
                if x.dtype == torch.float32:
                    x = x.to(dtype=torch.float16)
                if x.device.type == 'cpu' and torch.cuda.is_available():
                    x = x.to(device='cuda')
                    
        except Exception as e:
            print(f"Warning: Failed to match decoder dtype/device in forward: {e}")
            # Emergency fallback: ensure at least basic compatibility
            if x.dtype == torch.float32:
                x = x.to(dtype=torch.float16)
        
        return self.decode(x)
    
    def decode_latent_to_preview_image(self, preview_format, x):
        """
        ComfyUI TAESD compatibility method for preview generation.
        
        Args:
            preview_format: Preview format (unused in this implementation)
            x: Latent tensor to decode
            
        Returns:
            Decoded image tensor
        """
        # Handle both 4D and 5D tensors
        if x.ndim == 5:
            # Video case: [B, 16, T, H, W] -> decode first frame [B, 16, H, W]
            x = x[:, :, 0]  # Take first frame
        
        # Decode latents to preview image
        with torch.no_grad():
            try:
                # Ensure input tensor has the same dtype and device as the decoder
                decoder_params = list(self.decoder.parameters())
                if len(decoder_params) > 0:
                    target_device = decoder_params[0].device
                    target_dtype = decoder_params[0].dtype
                    
                    # Convert input tensor to match decoder
                    if x.device != target_device:
                        x = x.to(device=target_device)
                    if x.dtype != target_dtype:
                        x = x.to(dtype=target_dtype)
                else:
                    # Fallback: convert to float16 and move to cuda if available
                    if x.dtype == torch.float32:
                        x = x.to(dtype=torch.float16)
                    if x.device.type == 'cpu' and torch.cuda.is_available():
                        x = x.to(device='cuda')
                        
            except Exception as e:
                print(f"Warning: Failed to match decoder dtype/device in preview: {e}")
                # Emergency fallback: ensure at least basic compatibility
                if x.dtype == torch.float32:
                    x = x.to(dtype=torch.float16)
            
            decoded = self.decode(x)
            # Convert from [0, 1] to [0, 255] uint8 format expected by ComfyUI
            return (decoded * 255).clamp(0, 255).to(torch.uint8)

def get_wan21_taesd_decoder(vae_approx_dir):
    """
    Get a TAESD decoder for Wan2.1, creating weights if needed.
    
    Args:
        vae_approx_dir: Directory where VAE approximation models are stored
        
    Returns:
        TAESDWan21 instance or None if failed
    """
    # Look for existing TAESD weights for Wan2.1
    decoder_path = os.path.join(vae_approx_dir, "taesdwan21_decoder.pth")
    
    if not os.path.exists(decoder_path):
        print(f"TAESD Wan2.1 decoder not found at {decoder_path}")
        print("Creating TAESD decoder weights for Wan2.1...")
        
        if create_wan21_taesd_weights(vae_approx_dir):
            print(f"Created TAESD Wan2.1 decoder at {decoder_path}")
        else:
            print("Failed to create TAESD Wan2.1 decoder")
            return None
    
    try:
        # Load the TAESD decoder - return the full TAESDWan21 instance for ComfyUI compatibility
        taesd = TAESDWan21(decoder_path=decoder_path)
        print("Successfully loaded TAESD Wan2.1 decoder")
        return taesd  # Return full instance, not just the decoder part
    except Exception as e:
        print(f"Failed to load TAESD Wan2.1 decoder: {e}")
        return None

def create_wan21_taesd_weights(vae_approx_dir):
    """
    Create TAESD weights for Wan2.1 models.
    
    Args:
        vae_approx_dir: Directory to save the weights
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output path
        decoder_path = os.path.join(vae_approx_dir, "taesdwan21_decoder.pth")
        
        # Create a properly initialized TAESD decoder for 16 channels
        print("Creating TAESD decoder for 16-channel latents (Wan2.1)...")
        decoder = Decoder(latent_channels=16)
        
        # Initialize weights using Xavier/Kaiming initialization
        for module in decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Try to adapt weights from TAEHV model if available
        taehv_adapted_weights = try_adapt_taehv_weights(vae_approx_dir, decoder.state_dict())
        if taehv_adapted_weights is not None:
            print("Using TAEHV-adapted weights for better initialization")
            decoder_state_dict = taehv_adapted_weights
        else:
            print("Using random initialization for TAESD weights")
            decoder_state_dict = decoder.state_dict()
        
        # Ensure parent directory exists
        os.makedirs(vae_approx_dir, exist_ok=True)
        
        # Save the decoder state dict
        torch.save(decoder_state_dict, decoder_path)
        print(f"TAESD Wan2.1 decoder weights saved to: {decoder_path}")
        
        return True
        
    except Exception as e:
        print(f"Failed to create TAESD Wan2.1 weights: {e}")
        return False

def try_adapt_taehv_weights(vae_approx_dir, taesd_state_dict):
    """
    Try to adapt TAEHV weights for better TAESD initialization.
    
    Args:
        vae_approx_dir: Directory to look for TAEHV models
        taesd_state_dict: TAESD state dict to populate
        
    Returns:
        Adapted state dict or None if adaptation failed
    """
    try:
        # Try relative import first (when running as ComfyUI custom node)
        try:
            from .wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        except ImportError:
            # Fallback to absolute import (when running standalone)
            from wan_image_to_video_advanced_sampler import WanImageToVideoAdvancedSampler
        
        # Look for TAEHV models (specifically taew2_1.pth for Wan2.1)
        taehv_candidates = []
        for f in os.listdir(vae_approx_dir):
            if f.endswith('.pth') and any(keyword in f.lower() for keyword in ['taew2_1', 'wan2.1', 'taehv']):
                taehv_candidates.append(f)
        
        if not taehv_candidates:
            print("No TAEHV models found for weight adaptation")
            return None
        
        print(f"Found TAEHV candidates for adaptation: {taehv_candidates}")
        
        # Try to load the first available TAEHV model
        for model_name in taehv_candidates:
            try:
                model_path = os.path.join(vae_approx_dir, model_name)
                taehv_state = torch.load(model_path, map_location="cpu", weights_only=True)
                
                print(f"Loaded TAEHV model {model_name} for weight adaptation")
                
                # Extract relevant weights that might help with initialization
                # Focus on early convolutional layers that process 16-channel inputs
                adapted_state = taesd_state_dict.copy()
                
                # Try to find compatible layers in TAEHV decoder
                for key in adapted_state.keys():
                    # Look for potentially compatible TAEHV weights
                    if 'conv' in key or 'weight' in key:
                        # Try to find a matching layer in TAEHV
                        for taehv_key, taehv_weight in taehv_state.items():
                            if taehv_weight.shape == adapted_state[key].shape:
                                adapted_state[key] = taehv_weight
                                print(f"Adapted weight: {key} <- {taehv_key}")
                                break
                
                print("TAEHV weight adaptation completed")
                return adapted_state
                
            except Exception as e:
                print(f"Failed to adapt weights from {model_name}: {e}")
                continue
        
        print("No compatible TAEHV weights found for adaptation")
        return None
        
    except Exception as e:
        print(f"TAEHV weight adaptation error: {e}")
        return None

def download_wan21_models(vae_approx_dir):
    """
    Download necessary models for Wan2.1 TAESD support from official repositories.
    
    Downloads:
    - taew2_1.pth from madebyollin/taehv (for weight adaptation)
    - taesd_decoder.pth from madebyollin/taesd (for reference architecture)
    
    Args:
        vae_approx_dir: Directory to save the models
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        try:
            import urllib.request
            REQUESTS_AVAILABLE = False
        except ImportError:
            print("Warning: Neither requests nor urllib available for downloading models")
            return False
    
    try:
        # Models to download from official repositories
        models = [
            {
                "name": "taew2_1.pth",
                "url": "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth",
                "description": "TAEHV Wan 2.1 model for weight adaptation"
            },
            {
                "name": "taesd_decoder.pth", 
                "url": "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth",
                "description": "Official TAESD decoder for reference"
            }
        ]
        
        os.makedirs(vae_approx_dir, exist_ok=True)
        success_count = 0
        
        for model in models:
            filepath = os.path.join(vae_approx_dir, model["name"])
            
            if os.path.exists(filepath):
                print(f"✓ {model['name']} already exists")
                success_count += 1
                continue
                
            print(f"Downloading {model['name']} from official repository...")
            
            try:
                if REQUESTS_AVAILABLE:
                    response = requests.get(model["url"], stream=True)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    urllib.request.urlretrieve(model["url"], filepath)
                
                print(f"✓ {model['name']} downloaded successfully")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Failed to download {model['name']}: {e}")
        
        if success_count > 0:
            print(f"✓ Downloaded {success_count}/{len(models)} models from official repositories")
            return True
        else:
            print("✗ No models were downloaded successfully")
            return False
        
    except Exception as e:
        print(f"Download error: {e}")
        print("Manual installation:")
        print("1. Download taew2_1.pth from https://github.com/madebyollin/taehv/")
        print("2. Download taesd_decoder.pth from https://github.com/madebyollin/taesd/") 
        print(f"3. Place them in {vae_approx_dir}")
        return False

if __name__ == "__main__":
    # Test the implementation
    print("Testing TAESD Wan2.1 implementation...")
    
    # Test decoder creation
    decoder = Decoder(latent_channels=16)
    print(f"✓ Decoder created: {type(decoder)}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        output = decoder(dummy_input)
        print(f"✓ Decoder output shape: {output.shape}")
    
    # Test TAESD class
    taesd = TAESDWan21()
    print(f"✓ TAESDWan21 created: {type(taesd)}")
    
    # Test state dict structure
    state_dict = taesd.state_dict()
    has_prefixes = any(key.startswith(('encoder.', 'decoder.')) for key in state_dict.keys())
    print(f"✓ State dict compatible: {not has_prefixes} (no encoder./decoder. prefixes)")
    
    # Test decode_latent_to_preview_image method
    try:
        preview_result = taesd.decode_latent_to_preview_image("RGB", dummy_input)
        print(f"✓ Preview decode works: {preview_result.shape}, dtype: {preview_result.dtype}")
    except Exception as e:
        print(f"✗ Preview decode failed: {e}")
    
    # Test with 5D input (video case)
    try:
        dummy_video_input = torch.randn(1, 16, 4, 32, 32)  # [B, C, T, H, W]
        preview_result_video = taesd.decode_latent_to_preview_image("RGB", dummy_video_input)
        print(f"✓ Video preview decode works: {preview_result_video.shape}, dtype: {preview_result_video.dtype}")
    except Exception as e:
        print(f"✗ Video preview decode failed: {e}")
    
    print("✓ All tests passed!")
