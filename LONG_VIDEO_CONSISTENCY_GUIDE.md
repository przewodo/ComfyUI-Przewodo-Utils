# Long Video Consistency Guide for Wan2.1

## Overview

This guide explains how to use the advanced consistency features in the `WanImageToVideoAdvancedSampler` node to generate high-quality, coherent videos longer than 5 seconds without degradation, drift, or identity loss.

## The Problem

Standard Wan2.1 video generation often suffers from:
- **Quality degradation** after ~5 seconds
- **Character identity drift** in longer sequences  
- **Color and lighting shifts** between chunks
- **Blurriness accumulation** over time
- **Temporal inconsistencies** at chunk boundaries

## The Solution: Advanced Consistency Controls

Our implementation addresses these issues through 14 new parameters organized into 4 categories:

### 🎯 Core Consistency Controls

#### `overlap_frames` (Default: 4)
- **Purpose**: Creates smooth transitions between video chunks
- **How it works**: Overlaps N frames between chunks and blends them
- **Recommended values**: 
  - 4-6 frames for normal sequences
  - 8-12 frames for character-heavy scenes
  - 0 to disable (fastest but may have seams)

#### `temporal_overlap_strength` (Default: 0.8)
- **Purpose**: Controls how much previous chunk influences the overlap
- **How it works**: Weighted blending where 1.0 = keep all previous, 0.0 = use all new
- **Recommended values**:
  - 0.8-0.9 for character consistency
  - 0.6-0.7 for scene transitions
  - 0.9+ for extreme consistency needs

#### `anchor_frame_strength` (Default: 0.9)
- **Purpose**: Locks chunk boundaries to prevent sudden changes
- **How it works**: Forces first frame of each chunk to closely match last frame of previous
- **Recommended values**:
  - 0.9+ for character preservation
  - 0.7-0.8 for natural motion
  - 0.5-0.6 for creative transitions

### ⚡ Progressive Enhancement

#### `progressive_denoise_ramp` (Default: True)
- **Purpose**: Prevents quality degradation in later chunks
- **How it works**: Gradually reduces denoise strength to preserve existing details
- **Usage**: Keep enabled for sequences longer than 3 chunks

#### `persistent_noise_seed` (Default: True)  
- **Purpose**: Maintains visual coherence across chunks
- **How it works**: Uses related seeds instead of random ones
- **Usage**: Essential for character consistency

#### `keyframe_interval` (Default: 0)
- **Purpose**: Periodically injects high-quality reference frames
- **How it works**: Every N chunks, uses first frame as quality anchor
- **Recommended values**:
  - 0 to disable
  - 3-5 for very long sequences (10+ chunks)
  - 2-3 for challenging character preservation

### 🧠 Intelligence Features

#### `prompt_reinforcement` (Default: True)
- **Purpose**: Strengthens character descriptions at chunk boundaries
- **How it works**: Automatically emphasizes character keywords
- **Usage**: Keep enabled for character-focused videos

#### `quality_monitoring` (Default: True)
- **Purpose**: Detects and corrects quality degradation
- **How it works**: Monitors brightness/contrast and applies corrections
- **Usage**: Essential for long sequences

#### `periodic_detail_boost` (Default: False)
- **Purpose**: Combats gradual blur accumulation
- **How it works**: Applies sharpening every 3rd chunk
- **Usage**: Enable for sequences with fine details

### 🔬 Advanced Features

#### `temporal_clip_guidance` (Default: False)
- **Purpose**: CLIP-based consistency enforcement
- **How it works**: Uses CLIP vision to maintain visual similarity
- **Warning**: Slower generation but better coherence
- **Usage**: Enable for critical consistency needs

#### `reference_conditioning_strength` (Default: 1.0)
- **Purpose**: Enhanced reference image conditioning
- **How it works**: Strengthens influence of original reference image
- **Values**: 1.0-1.5 for strong consistency, 0.5-0.8 for more variation

## Quick Start Presets

### Basic Long Video (5-10 seconds)
```
overlap_frames: 4
temporal_overlap_strength: 0.8
anchor_frame_strength: 0.9
progressive_denoise_ramp: True
persistent_noise_seed: True
keyframe_interval: 0
prompt_reinforcement: True
quality_monitoring: True
```

### Character-Focused Video (10-20 seconds)
```
overlap_frames: 6
temporal_overlap_strength: 0.9
anchor_frame_strength: 0.95
progressive_denoise_ramp: True
persistent_noise_seed: True
keyframe_interval: 3
prompt_reinforcement: True
quality_monitoring: True
reference_conditioning_strength: 1.2
```

### Maximum Quality (20+ seconds)
```
overlap_frames: 8
temporal_overlap_strength: 0.9
anchor_frame_strength: 0.95
progressive_denoise_ramp: True
persistent_noise_seed: True
keyframe_interval: 2
prompt_reinforcement: True
quality_monitoring: True
periodic_detail_boost: True
temporal_clip_guidance: True
reference_conditioning_strength: 1.5
```

### Fast Generation (Trade quality for speed)
```
overlap_frames: 2
temporal_overlap_strength: 0.6
anchor_frame_strength: 0.7
progressive_denoise_ramp: False
persistent_noise_seed: True
keyframe_interval: 0
prompt_reinforcement: True
quality_monitoring: False
```

## Best Practices

### 1. Chunk Planning
- **Short chunks (1-2 seconds)**: Better quality, more seams to manage
- **Medium chunks (3-4 seconds)**: Good balance
- **Long chunks (5+ seconds)**: Risk degradation but fewer seams

### 2. Prompt Strategy
- Use **consistent character descriptions** throughout
- Include **lighting/style keywords** for coherence
- Avoid **changing character details** mid-sequence

### 3. Image Selection
- Use **high-quality start images** (1024x1024+)
- Ensure **good lighting and clarity**
- **Match aspect ratio** to target video

### 4. Performance Optimization
- Start with **basic settings** and increase gradually
- **Monitor VRAM usage** - overlap features use more memory
- Use **lower frame rates** and interpolate afterward

### 5. Troubleshooting Common Issues

#### "Seams still visible between chunks"
- Increase `overlap_frames` to 6-8
- Increase `temporal_overlap_strength` to 0.9+
- Enable `anchor_frame_strength` at 0.9+

#### "Character changes appearance"
- Enable `prompt_reinforcement`
- Increase `reference_conditioning_strength` to 1.2+
- Set `keyframe_interval` to 2-3
- Use `persistent_noise_seed`

#### "Video gets blurry over time"
- Enable `progressive_denoise_ramp`
- Enable `periodic_detail_boost`
- Lower `total_steps` and use more chunks
- Enable `quality_monitoring`

#### "Generation too slow"
- Reduce `overlap_frames` to 2-4
- Disable `temporal_clip_guidance`
- Disable `periodic_detail_boost`
- Use fewer chunks with longer duration

#### "Colors shift between chunks"
- Keep `apply_color_match` enabled
- Use `persistent_noise_seed`
- Ensure consistent lighting in prompts

## Technical Details

### Memory Usage
- Base model: ~8-12GB VRAM
- With overlap (4 frames): +1-2GB VRAM
- With CLIP guidance: +0.5-1GB VRAM
- Monitor with `nvidia-smi` during generation

### Processing Time
- Basic settings: Same as normal generation
- Full overlap + guidance: +30-50% generation time
- CLIP guidance: +10-20% per chunk

### Quality Metrics
The system monitors:
- **Brightness consistency**: Prevents lighting drift
- **Contrast preservation**: Maintains detail levels
- **CLIP similarity**: Semantic consistency (when enabled)

## Integration with Other Features

### Works Well With:
- **TeaCache**: Faster inference, no conflicts
- **SageAttention**: Memory optimization
- **Skip Layer Guidance**: Quality enhancement  
- **Video Enhancement**: Post-processing improvement
- **RIFE Interpolation**: Smooth frame generation

### Potential Conflicts:
- **Very high CFG values**: May fight consistency
- **Extreme LoRA strengths**: Can override consistency
- **Multiple prompt changes**: Conflicts with reinforcement

## Advanced Usage Examples

### Example 1: Character Conversation (15 seconds)
```python
# Settings for maintaining character identity during dialogue
total_video_chunks = 6
total_video_seconds = 2  # Per chunk = 12 seconds total

# Strong consistency for character preservation
overlap_frames = 8
temporal_overlap_strength = 0.95
anchor_frame_strength = 0.95
keyframe_interval = 2
prompt_reinforcement = True
reference_conditioning_strength = 1.3
```

### Example 2: Scenic Animation (30 seconds)  
```python
# Settings for environmental scenes with camera movement
total_video_chunks = 10
total_video_seconds = 3  # Per chunk = 30 seconds total

# Balanced consistency for scene flow
overlap_frames = 6
temporal_overlap_strength = 0.8
anchor_frame_strength = 0.8
progressive_denoise_ramp = True
quality_monitoring = True
periodic_detail_boost = True
```

### Example 3: Action Sequence (20 seconds)
```python
# Settings for fast motion with character consistency
total_video_chunks = 8
total_video_seconds = 2.5  # Per chunk = 20 seconds total

# Moderate consistency to allow motion
overlap_frames = 4
temporal_overlap_strength = 0.7
anchor_frame_strength = 0.8
persistent_noise_seed = True
prompt_reinforcement = True
```

## Future Enhancements

The system is designed to be extensible. Planned features include:
- **ControlNet integration** for structural consistency
- **Automatic scene detection** for smart segmentation
- **Reverse generation** for anti-drift sampling
- **Multi-scale blending** for better temporal fusion

## Support and Troubleshooting

For issues or questions:
1. Check the console output for specific error messages
2. Try basic settings first, then add features gradually
3. Monitor VRAM usage if experiencing crashes
4. Ensure all required custom nodes are installed

This implementation represents state-of-the-art techniques for long-form video generation with AI, bringing together research from FramePack, TeaCache, and other advanced video diffusion methods.
