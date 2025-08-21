import torch

class TemporalFrameBuffer:
    """
    Advanced frame buffer for maintaining temporal continuity across chunks
    """
    def __init__(self, buffer_size=32, overlap_frames=16):
        self.buffer_size = buffer_size
        self.overlap_frames = overlap_frames
        self.frame_buffer = None
        self.latent_buffer = None
        self.motion_buffer = None
        self.style_features = None  # Store style features for consistency
        self.color_stats = None     # Store color statistics for matching
        
    def initialize_buffer(self, initial_frames, initial_latent=None):
        """Initialize buffer with first chunk frames"""
        self.frame_buffer = initial_frames.clone()
        if initial_latent is not None:
            self.latent_buffer = initial_latent.clone()
        self._update_motion_buffer()
    
    def add_frames(self, new_frames, new_latent=None):
        """Add new frames to buffer, maintaining size limit"""
        if self.frame_buffer is None:
            self.initialize_buffer(new_frames, new_latent)
            return
            
        # Concatenate new frames
        self.frame_buffer = torch.cat([self.frame_buffer, new_frames], dim=0)
        
        if new_latent is not None and self.latent_buffer is not None:
            self.latent_buffer = torch.cat([self.latent_buffer, new_latent], dim=2)
        
        # Trim buffer to maximum size
        if self.frame_buffer.shape[0] > self.buffer_size:
            excess = self.frame_buffer.shape[0] - self.buffer_size
            self.frame_buffer = self.frame_buffer[excess:]
            
            if self.latent_buffer is not None:
                # Latent buffer trimming (accounting for temporal compression)
                latent_excess = excess // 4  # VAE temporal compression
                if latent_excess > 0:
                    self.latent_buffer = self.latent_buffer[:, :, latent_excess:]
        
        self._update_motion_buffer()
        self.update_style_features()  # Update style features when adding frames
    
    def get_overlap_frames(self):
        """Get frames for overlapping with next chunk"""
        if self.frame_buffer is None:
            return None
        
        overlap_length = min(self.overlap_frames, self.frame_buffer.shape[0])
        return self.frame_buffer[-overlap_length:].clone()
    
    def get_overlap_latent(self):
        """Get latent representation of overlap frames"""
        if self.latent_buffer is None:
            return None
            
        overlap_latent_length = min(self.overlap_frames // 4, self.latent_buffer.shape[2])
        return self.latent_buffer[:, :, -overlap_latent_length:].clone()
    
    def _update_motion_buffer(self):
        """Update motion information for temporal consistency"""
        if self.frame_buffer is None or self.frame_buffer.shape[0] < 2:
            return
            
        # Calculate optical flow/motion vectors for last few frames
        recent_frames = self.frame_buffer[-8:]  # Last 8 frames for motion analysis
        motion_vectors = []
        
        for i in range(len(recent_frames) - 1):
            frame_a = recent_frames[i]
            frame_b = recent_frames[i + 1]
            
            # Simple motion estimation (difference-based)
            motion = frame_b - frame_a
            motion_vectors.append(motion)
        
        if motion_vectors:
            self.motion_buffer = torch.stack(motion_vectors, dim=0)
    
    def get_motion_prediction(self):
        """Get predicted motion for next frames"""
        if self.motion_buffer is None:
            return None
            
        # Average recent motion for prediction
        avg_motion = self.motion_buffer.mean(dim=0)
        return avg_motion
    
    def clear_buffer(self):
        """Clear all buffers"""
        self.frame_buffer = None
        self.latent_buffer = None
        self.motion_buffer = None
        self.style_features = None
        self.color_stats = None
    
    def extract_style_features(self, frames):
        """Extract style features from frames for consistency"""
        if frames is None or frames.shape[0] == 0:
            return None
        
        # Use last few frames for style extraction
        style_frames = frames[-min(4, frames.shape[0]):]
        
        # Calculate color statistics
        color_mean = style_frames.mean(dim=[0, 1, 2], keepdim=True)
        color_std = style_frames.std(dim=[0, 1, 2], keepdim=True)
        
        # Calculate luminance
        luminance = (style_frames[:, :, :, 0] * 0.299 + 
                    style_frames[:, :, :, 1] * 0.587 + 
                    style_frames[:, :, :, 2] * 0.114)
        luma_mean = luminance.mean()
        luma_std = luminance.std()
        
        # Extract edge characteristics (simple gradient)
        dx = torch.diff(style_frames, dim=2)
        dy = torch.diff(style_frames, dim=1)
        edge_strength = (dx[:, :-1, :, :].abs() + dy[:, :, :-1, :].abs()).mean()
        
        self.style_features = {
            'color_mean': color_mean,
            'color_std': color_std,
            'luma_mean': luma_mean,
            'luma_std': luma_std,
            'edge_strength': edge_strength
        }
        
        # Simple color statistics for quick access
        self.color_stats = {
            'mean': color_mean,
            'std': color_std
        }
        
        return self.style_features
    
    def get_style_features(self):
        """Get current style features"""
        return self.style_features
    
    def get_color_stats(self):
        """Get current color statistics"""
        return self.color_stats
    
    def update_style_features(self):
        """Update style features from current buffer"""
        if self.frame_buffer is not None:
            self.extract_style_features(self.frame_buffer)