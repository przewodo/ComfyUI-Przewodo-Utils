from comfy.ldm.modules import attention as comfy_attention
import logging
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import folder_paths
import comfy.model_management as mm
from comfy.cli_args import args

class WanVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the enhance effect"}),
                "length": ("INT", {"tooltip": "Number of frames in the video", "default": 16, "min": 1, "max": 1000}),
                "width": ("INT",),
                "height": ("INT",),
           }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enhance"
    CATEGORY = "PrzewodoUtils/Wan"
    EXPERIMENTAL = True

    def enhance(self, model, weight, length, width, height):
        if weight == 0:
            return (model,)
        
        latent = torch.zeros([length, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
       
        num_frames = latent.shape[2]

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["enhance_weight"] = weight
        diffusion_model = model_clone.get_model_object("diffusion_model")

        compile_settings = getattr(model.model, "compile_settings", None)
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanAttentionPatch(num_frames, weight).__get__(block.self_attn, block.__class__)
            if compile_settings is not None:
                patched_attn = torch.compile(patched_attn, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])
            
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.self_attn.forward", patched_attn)
            
        return (model_clone,)
    
import types
class WanAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight
        
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return modified_wan_self_attention_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)
    
from comfy.ldm.flux.math import apply_rope
def modified_wan_self_attention_forward(self, x, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)

    q, k = apply_rope(q, k, freqs)

    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight)

    x = comfy.ldm.modules.attention.optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        v,
        heads=self.num_heads,
    )

    x = self.o(x)

    x *= feta_scores

    return x

from einops import rearrange
def get_feta_scores(query, key, num_frames, enhance_weight):
    img_q, img_k = query, key #torch.Size([2, 9216, 12, 128])
    
    _, ST, num_heads, head_dim = img_q.shape
    spatial_dim = ST / num_frames
    spatial_dim = int(spatial_dim)

    query_image = rearrange(
        img_q, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )
    key_image = rearrange(
        img_k, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )

    return feta_score(query_image, key_image, head_dim, num_frames, enhance_weight)

def feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores