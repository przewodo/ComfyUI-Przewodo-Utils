#!/usr/bin/env python3
"""
Test script for Przewodo UTILS color functions.
Run this to verify that colors work both during ComfyUI startup and after it finishes loading.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our core module
from core import (
    output_to_terminal, 
    output_to_terminal_successful, 
    output_to_terminal_error,
    debug_color_support,
    test_all_colors
)

def main():
    print("=" * 70)
    print("PRZEWODO UTILS COLOR TEST SCRIPT")
    print("=" * 70)
    
    print("\nTesting basic output functions:")
    output_to_terminal("Testing regular output message")
    output_to_terminal_successful("Testing success message")
    output_to_terminal_error("Testing error message")
    
    print("\nRunning comprehensive color test:")
    test_all_colors()
    
    print("Test completed!")
    print("If you can see colors above, the color system is working correctly.")
    print("If not, you may need to:")
    print("1. Set FORCE_COLOR=1 environment variable")
    print("2. Check if your terminal supports ANSI colors")
    print("3. Try running ComfyUI in a different terminal")

if __name__ == "__main__":
    main()
