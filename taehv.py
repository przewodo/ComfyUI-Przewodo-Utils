#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video https://github.com/madebyollin/taehv
(DNN for encoding / decoding videos to Hunyuan Video's latent space)

Adapted for ComfyUI-Przewodo-Utils to provide proper TAESD preview support for Wan2.1 models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, kernel_size=3, **kwargs):
    return nn.Conv3d(n_in, n_out, kernel_size=kernel_size, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x * 0.5) * 2

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = conv(n_in, n_out, kernel_size=3, padding=1)
        self.conv2 = conv(n_out, n_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, n_out)
        self.norm2 = nn.GroupNorm(32, n_out)
        self.skip = nn.Identity() if n_in == n_out else conv(n_in, n_out, kernel_size=1)

    def forward(self, x):
        h = x
        h = self.norm1(self.conv1(h))
        h = F.relu(h, inplace=True)
        h = self.norm2(self.conv2(h))
        h = F.relu(h, inplace=True)
        return h + self.skip(x)

class TPool(nn.Module):
    def __init__(self, n_channels, factor):
        super().__init__()
        self.conv = conv(n_channels, n_channels, kernel_size=3, padding=1, stride=(factor, 1, 1))

    def forward(self, x):
        return self.conv(x)

class TGrow(nn.Module):
    def __init__(self, n_channels, factor):
        super().__init__()
        self.n_channels = n_channels
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = x.repeat_interleave(self.factor, dim=0)
        x = x.reshape(b, t * self.factor, c, h, w).permute(0, 2, 1, 3, 4)
        return x

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """Apply model with memory blocks for efficient processing."""
    if parallel:
        return model(x)
    
    # Sequential processing for memory efficiency
    results = []
    for i in tqdm(range(x.size(1)), desc="Processing frames", disable=not show_progress_bar):
        frame = x[:, i:i+1]
        result = model(frame)
        results.append(result)
    
    return torch.cat(results, dim=1)

class TAEHV(nn.Module):
    latent_channels = 16
    image_channels = 3
    
    def __init__(self, state_dict, parallel=False, decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True)):
        """Initialize pretrained TAEHV from the given checkpoint.

        Args:
            state_dict: Model state dict to load
            parallel: if True, process all frames at once (faster but more memory)
            decoder_time_upscale: whether temporal upsampling is enabled for each block
            decoder_space_upscale: whether spatial upsampling is enabled for each block
        """
        super().__init__()
        self.encoder = nn.Sequential(
            conv(TAEHV.image_channels, 64), nn.ReLU(inplace=True),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, TAEHV.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(TAEHV.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], TAEHV.image_channels),
        )
        if state_dict is not None:
            self.load_state_dict(self.patch_tgrow_layers(state_dict))
        self.dtype = torch.float16
        self.parallel = parallel

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to handle dimension mismatches."""
        for name, module in self.named_modules():
            if isinstance(module, TGrow):
                weight_key = f"{name}.weight"
                if weight_key in sd and sd[weight_key].shape != getattr(module, 'weight', torch.empty(0)).shape:
                    # Handle dimension mismatch for TGrow layers
                    del sd[weight_key]
        return sd

    def encode_video(self, x, parallel=False, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
            show_progress_bar: whether to show progress bar
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        return apply_model_with_memblocks(self.encoder, x, self.parallel or parallel, show_progress_bar)

    def decode_video(self, x, parallel=False, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=16) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
            show_progress_bar: whether to show progress bar
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        x = apply_model_with_memblocks(self.decoder, x, self.parallel or parallel, show_progress_bar)
        return x[:, self.frames_to_trim:]

    def forward(self, x):
        return self.decode_video(x)
