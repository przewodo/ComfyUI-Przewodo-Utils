# TAEHV Integration for Wan2.1 Models

This implementation provides proper TAESD preview support for Wan2.1 models using TAEHV (Tiny AutoEncoder for Hunyuan Video).

## What is TAEHV?

TAEHV is a specialized tiny autoencoder designed specifically for video models like Wan2.1. Unlike standard TAESD which works with 4-channel latents from image models, TAEHV is designed for 16-channel latents used by video models.

## Features

- **Proper Video Previews**: TAEHV provides accurate previews for Wan2.1 models during generation
- **Memory Efficient**: Sequential frame processing to minimize VRAM usage
- **Automatic Detection**: Automatically detects Wan2.1 models and enables TAEHV
- **Fallback Support**: Falls back to RGB previews if TAEHV is not available

## Installation

### Method 1: Automatic Download (Recommended)

Run the download helper script:

```python
python download_taehv.py
```

This will automatically download the required TAEHV models to the correct location.

### Method 2: Manual Installation

1. Download TAEHV models from [HuggingFace](https://huggingface.co/madebyollin/taehv):
   - `taehv.pth` - For Hunyuan Video models
   - `taew2_1.pth` - For Wan 2.1 models

2. Place them in your ComfyUI `models/vae_approx/` folder

3. Restart ComfyUI

## Usage

Once installed, TAEHV integration works automatically:

1. Load a Wan2.1 model using the advanced sampler node
2. The system automatically detects the model type
3. TAEHV is loaded and configured for preview generation
4. You'll see proper video previews during generation instead of garbled RGB fallbacks

## Technical Details

### Architecture Compatibility

- **Standard TAESD**: 4-channel latents (SD 1.x, SDXL)
- **TAEHV**: 16-channel latents (Wan2.1, Hunyuan Video)

### Model Detection

The system identifies Wan2.1 models by checking:
- `latent_channels == 16`
- `latent_dimensions == 3` (3D latents for video)
- `clip_type == CLIP_WAN`

### Preview Process

1. **Model Loading**: TAEHV model is loaded from `vae_approx` folder
2. **Integration**: Custom preview decoder is installed in the latent format
3. **Generation**: During sampling, latents are decoded using TAEHV
4. **Display**: Proper video frames are shown as previews

## File Structure

```
ComfyUI-Przewodo-Utils/
├── taehv.py                    # TAEHV model implementation
├── taehv_preview.py           # Preview system integration
├── download_taehv.py          # Download helper script
└── wan_image_to_video_advanced_sampler.py  # Main node with TAEHV integration
```

## Troubleshooting

### No TAEHV Models Found

If you see: `"No TAEHV models found in vae_approx folder"`

**Solution**: Download TAEHV models using the download script or manually place them in `ComfyUI/models/vae_approx/`

### TAEHV Loading Failed

If TAEHV fails to load:

1. Check that the model files are not corrupted
2. Ensure sufficient VRAM is available
3. Check the ComfyUI console for detailed error messages

### RGB Fallback Previews

If you see: `"TAEHV not available - using RGB fallback previews"`

This means TAEHV couldn't be loaded, but the node will still work with basic RGB previews.

## Performance Notes

- **Sequential Processing**: TAEHV uses sequential frame processing for memory efficiency
- **Optimized Settings**: Temporal upscaling is disabled for faster preview generation
- **Device Management**: TAEHV is automatically moved to the appropriate device

## Credits

- **TAEHV**: Based on the implementation by [madebyollin](https://github.com/madebyollin/taehv)
- **WanVideoWrapper**: Inspired by the TAEHV integration in [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## License

This implementation follows the same license as the original TAEHV project.
