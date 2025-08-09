"""
ComfyUI-Przewodo-Utils: Advanced video generation and utility nodes for ComfyUI.

This package provides specialized nodes for video generation, model management,
and various utility functions for ComfyUI workflows.
"""

# WEB_DIRECTORY is the ComfyUI nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web"

try:
    from .node_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Handle import errors gracefully during testing or when dependencies aren't available
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .__version__ import VERSION as __version__
except ImportError:
    __version__ = "1.0.0"  # Fallback version

try:
    from .core import output_to_terminal
except ImportError:
    __version__ = "1.0.0"  # Fallback version

output_to_terminal(f"Version {__version__}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
