from .node_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Initialize TAESD override if available
try:
    from .taesd_override import initialize_taesd_override
    # Don't auto-initialize here, let the node control it
    print("[ComfyUI-Przewodo-Utils] TAESD override module loaded")
except ImportError:
    print("[ComfyUI-Przewodo-Utils] TAESD override module not available")
except Exception as e:
    print(f"[ComfyUI-Przewodo-Utils] TAESD override load error: {e}")
