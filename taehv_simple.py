#!/usr/bin/env python3
"""
Simplified TAEHV implementation based on WanVideoWrapper's approach

This implementation follows the pattern used by ComfyUI-WanVideoWrapper
and is designed to work with actual TAEHV model files (like taew2_1.safetensors).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAEHV(nn.Module):
    """
    Simplified Tiny AutoEncoder for Hunyuan Video (TAEHV)
    
    Compatible with WanVideoWrapper's calling convention and actual model files.
    """
    
    def __init__(self, state_dict):
        super().__init__()
        
        # Build model from state dict
        self._build_from_state_dict(state_dict)
        
        # Load the weights
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"TAEHV missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"TAEHV unexpected keys: {len(unexpected_keys)}")
    
    def _build_from_state_dict(self, state_dict):
        """Build model architecture from state dict keys"""
        
        # Analyze state dict structure
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
        
        # Build encoder
        self.encoder = self._build_conv_layers(encoder_keys, state_dict, 'encoder')
        
        # Build decoder
        self.decoder = self._build_conv_layers(decoder_keys, state_dict, 'decoder')
        
        print(f"TAEHV built with {len(self.encoder)} encoder layers and {len(self.decoder)} decoder layers")
    
    def _build_conv_layers(self, layer_keys, state_dict, prefix):
        """Build conv layers from state dict"""
        layers = nn.ModuleDict()
        
        # Extract layer structure
        layer_info = {}
        for key in layer_keys:
            if '.weight' in key:
                # Parse key like "encoder.0.conv.weight" 
                parts = key.split('.')
                if len(parts) >= 4 and parts[0] == prefix:
                    layer_idx = parts[1]
                    layer_type = parts[2]
                    
                    if layer_idx not in layer_info:
                        layer_info[layer_idx] = {}
                    
                    weight = state_dict[key]
                    if layer_type == 'conv':
                        layer_info[layer_idx]['conv'] = {
                            'weight_shape': weight.shape,
                            'is_3d': weight.dim() == 5
                        }
        
        # Build layers in order
        for idx in sorted(layer_info.keys(), key=int):
            info = layer_info[idx]
            
            if 'conv' in info:
                conv_info = info['conv']
                weight_shape = conv_info['weight_shape']
                out_ch, in_ch = weight_shape[:2]
                
                if conv_info['is_3d']:
                    # 3D conv for video
                    kernel_size = weight_shape[2:]
                    layer = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, 
                                    padding=tuple(k//2 for k in kernel_size))
                else:
                    # 2D conv
                    kernel_size = weight_shape[2:]
                    layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    padding=tuple(k//2 for k in kernel_size))
                
                layers[idx] = layer
        
        return layers
    
    def _apply_layers(self, x, layers):
        """Apply layers sequentially with activations"""
        layer_items = sorted(layers.items(), key=lambda item: int(item[0]))
        
        for i, (idx, layer) in enumerate(layer_items):
            x = layer(x)
            # Apply activation except for last layer
            if i < len(layer_items) - 1:
                x = F.relu(x, inplace=True)
        
        return x
    
    def encode_video(self, x):
        """Encode video to latent space"""
        return self._apply_layers(x, self.encoder)
    
    def decode_video(self, latent, parallel=False, show_progress_bar=False):
        """
        Decode latent to video - compatible with WanVideoWrapper
        
        Args:
            latent: Latent tensor 
            parallel: Ignored for compatibility
            show_progress_bar: Ignored for compatibility
            
        Returns:
            List containing decoded video tensor
        """
        # Ensure correct input shape
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)  # Add batch dim
        
        # Apply decoder layers
        x = self._apply_layers(latent, self.decoder)
        
        # Clamp to valid range
        x = torch.tanh(x)
        
        return [x]
    
    def forward(self, x):
        """Standard forward pass"""
        latent = self.encode_video(x)
        decoded = self.decode_video(latent)
        return decoded[0]
