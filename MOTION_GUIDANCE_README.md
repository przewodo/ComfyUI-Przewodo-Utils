# Wan2.1 Motion-Guided Video Chunking System

## Overview

This document describes the enhanced video chunking system for Wan2.1 Image-to-Video generation that provides seamless transitions between chunks using advanced motion analysis and temporal consistency techniques.

## Key Features

### 1. Motion Analysis & Prediction
- **Optical Flow Estimation**: Analyzes motion between consecutive frames in overlap regions
- **Trajectory Prediction**: Predicts future motion based on historical flow patterns
- **Motion Confidence Scoring**: Evaluates motion prediction reliability

### 2. Latent Space Warping
- **Flow-Based Warping**: Uses optical flow to warp latent representations for continuity
- **Occlusion Handling**: Detects and handles areas where motion prediction fails
- **Temporal Position Encoding**: Provides chunk context information to the model

### 3. Progressive Blending
- **Gaussian Spatial Masks**: Creates smooth spatial transitions using configurable sigma
- **Motion-Weighted Blending**: Applies different blending strategies based on motion confidence
- **Adaptive Falloff**: Gradually reduces overlap influence across transition frames

### 4. Quality Preservation
- **Color Consistency**: Progressive color matching during generation process
- **Feature Continuity**: Maintains visual feature consistency across chunk boundaries
- **Noise Handling**: Proper Wan2.1-specific noise injection (0.5 multiplier)

## Technical Implementation

### Core Function: `guide_next_chunk()`

```python
def guide_next_chunk(self, previous_latent, previous_frames, overlap_frames, 
                    blend_strength, motion_weight, mask_sigma, step_gain, 
                    vae, chunk_index, chunk_frames, image_height, image_width, 
                    reference_image=None):
```

**Parameters:**
- `previous_latent`: Latent representation from previous chunk
- `overlap_frames`: Number of overlapping frames (typically 8-16)
- `blend_strength`: Alpha blending strength (0.0-1.0, default 0.8)
- `motion_weight`: Motion prediction weight (0.2-0.35 typical, default 0.3)
- `mask_sigma`: Gaussian mask sigma (0.25-0.5 typical, default 0.35)
- `step_gain`: Global motion step gain (default 0.5)

### Motion Analysis Pipeline

1. **Extract Overlap Region**
   ```python
   prev_overlap_latent = previous_latent["samples"][:, :, -overlap_frames_in_latent_space:, :, :]
   overlap_images = vae.decode(prev_overlap_latent)
   ```

2. **Estimate Optical Flow**
   - Uses Lucas-Kanade approximation for gradient-based flow estimation
   - Converts frames to grayscale for flow computation
   - Calculates spatial and temporal gradients

3. **Predict Motion Trajectory**
   - Linear extrapolation with acceleration analysis
   - Confidence decay for extended predictions
   - Motion scaling with step gain parameter

4. **Apply Motion-Weighted Blending**
   - Grid sampling for motion-based warping
   - Gaussian spatial masking for smooth transitions
   - Progressive blend strength application

## Parameter Configuration

### Optimal Settings by Use Case

#### Standard Video Generation (5-second chunks)
```python
frames_overlap_chunks = 16                    # More overlap for smoother transitions
frames_overlap_chunks_blend = 0.8            # Strong blending for continuity
frames_overlap_chunks_motion_weight = 0.3    # Balanced motion guidance
frames_overlap_chunks_mask_sigma = 0.35      # Medium spatial smoothing
frames_overlap_chunks_step_gain = 0.5        # Conservative motion prediction
```

#### Fast Motion Scenarios
```python
frames_overlap_chunks = 24                    # Extended overlap for complex motion
frames_overlap_chunks_blend = 0.6            # Reduced to preserve detail
frames_overlap_chunks_motion_weight = 0.4    # Stronger motion guidance
frames_overlap_chunks_mask_sigma = 0.25      # Tighter spatial focus
frames_overlap_chunks_step_gain = 0.7        # More aggressive prediction
```

#### Slow/Static Scenes
```python
frames_overlap_chunks = 8                     # Minimal overlap sufficient
frames_overlap_chunks_blend = 0.9            # High blending for smoothness
frames_overlap_chunks_motion_weight = 0.2    # Light motion guidance
frames_overlap_chunks_mask_sigma = 0.5       # Wider spatial smoothing
frames_overlap_chunks_step_gain = 0.3        # Conservative prediction
```

## Integration with Existing Workflow

### Wan2.1 Specific Requirements

1. **Latent Space**: Uses 4×8×8 compression ratio from Wan-VAE
2. **Noise Initialization**: Empty latent multiplied by 0.5 (Wan2.1 requirement)
3. **Mask Configuration**: 1s for generation, 0s for keyframes
4. **CLIP Latent Injection**: Image conditioning for first/overlap frames

### Memory Management

The system includes comprehensive memory management:
- **Tensor Optimization**: Optimizes memory layout for efficiency
- **Garbage Collection**: Strategic cleanup between chunks
- **CUDA Cache Management**: Prevents memory accumulation
- **Fallback Handling**: Safe fallbacks when motion analysis fails

## Troubleshooting

### Common Issues & Solutions

#### Poor Chunk Transitions
- **Increase overlap_frames**: More frames provide better motion analysis
- **Adjust blend_strength**: Higher values for smoother transitions
- **Tune motion_weight**: Balance between motion guidance and original content

#### Color Inconsistency
- **Verify reference_image**: Ensure consistent reference across chunks
- **Check color matching**: Apply progressive color matching during generation
- **Monitor blend_strength**: Too high values may cause color drift

#### Performance Issues
- **Reduce overlap_frames**: Fewer frames = faster processing
- **Lower mask_sigma**: Smaller masks = less computation
- **Disable motion guidance**: Fallback to simple chunking if needed

#### Motion Artifacts
- **Reduce step_gain**: Less aggressive motion prediction
- **Lower motion_weight**: Reduce motion guidance influence
- **Increase mask_sigma**: Smoother spatial blending

### Debug Output

The system provides comprehensive logging:
```
[INFO] Applying motion-guided chunk transition for chunk 2
[INFO] Motion-guided preparation complete for chunk 2
[INFO] Guided Input Latent Shape: torch.Size([1, 16, 21, 104, 104])
[INFO] Applied motion guidance to conditioning for chunk 2
```

### Fallback Behavior

When motion guidance fails, the system automatically falls back to:
1. Simple overlap-based chunking
2. Basic color matching
3. Standard Wan2.1 noise injection
4. Warning messages in console

## Advanced Usage

### Custom Motion Analysis

For specialized use cases, you can extend the motion analysis:

```python
def custom_optical_flow(self, frames):
    # Integrate with external optical flow libraries
    # e.g., OpenCV's Lucas-Kanade, Farneback, or deep learning models
    pass
```

### Prompt Transition Integration

The system works with prompt stacks for smooth prompt transitions:

```python
# Prompt stack format: [positive, negative, start_chunk, lora_stack]
prompt_stack = [
    ["walking slowly", "blurry, bad quality", 0, None],
    ["running fast", "blurry, bad quality", 2, None],
    ["jumping high", "blurry, bad quality", 4, None]
]
```

### Motion-Aware LoRA Application

LoRAs are applied with motion consideration:
- Higher motion areas may benefit from motion-specific LoRAs
- Prompt transitions can include LoRA changes for motion enhancement

## Future Enhancements

Potential improvements for future versions:
1. **Deep Flow Networks**: Replace gradient-based flow with neural networks
2. **Attention-Based Warping**: Use transformer attention for motion prediction
3. **Multi-Scale Analysis**: Analyze motion at multiple temporal scales
4. **Semantic Motion Understanding**: Motion-aware semantic guidance
5. **Hardware Acceleration**: Optimized CUDA kernels for flow computation

## Conclusion

The motion-guided chunking system provides significant improvements in temporal consistency for long-form Wan2.1 video generation. By leveraging optical flow analysis, progressive blending, and motion-weighted guidance, it achieves seamless transitions between video chunks while maintaining the quality and characteristics of the Wan2.1 model.

For optimal results, experiment with the parameter settings based on your specific content type and motion characteristics.
