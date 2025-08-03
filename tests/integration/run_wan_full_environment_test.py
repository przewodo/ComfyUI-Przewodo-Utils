#!/usr/bin/env python3
"""
Test runner for WAN Image to Video Advanced Sampler in full ComfyUI environment.

This script sets up the complete ComfyUI environment and runs comprehensive
integration tests for the WanImageToVideoAdvancedSampler node.

Usage:
    python run_wan_full_environment_test.py

Requirements:
    - Full ComfyUI installation with all dependencies
    - WAN models in the models directory
    - Test images in the input directory
"""

import sys
import os
from pathlib import Path
import subprocess
import json
import time

def setup_comfyui_environment():
    """Set up the ComfyUI environment paths and imports."""
    
    # Get the ComfyUI root directory
    script_dir = Path(__file__).parent
    comfyui_root = script_dir.parent.parent.parent.parent
    
    print(f"ComfyUI root: {comfyui_root}")
    
    # Add ComfyUI to Python path
    sys.path.insert(0, str(comfyui_root))
    
    # Add custom nodes to path
    custom_nodes_dir = comfyui_root / "custom_nodes" / "ComfyUI-Przewodo-Utils"
    sys.path.insert(0, str(custom_nodes_dir))
    
    try:
        # Try to import ComfyUI core modules
        import folder_paths
        import comfy.model_management as mm
        
        print(f"✓ ComfyUI core modules imported successfully")
        print(f"  - Models directory: {folder_paths.models_dir}")
        print(f"  - Torch device: {mm.get_torch_device()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import ComfyUI modules: {e}")
        print("Make sure ComfyUI is properly installed and this script is run from the correct location")
        return False

def check_requirements():
    """Check if all requirements for testing are met."""
    
    print("\n🔍 Checking test requirements...")
    
    requirements_met = True
    
    try:
        import folder_paths
        
        # Check for WAN models
        gguf_models = folder_paths.get_filename_list("unet")
        wan_gguf = [m for m in gguf_models if "wan" in m.lower() and "i2v" in m.lower()]
        
        if wan_gguf:
            print(f"✓ Found {len(wan_gguf)} WAN GGUF models: {wan_gguf[:3]}")
        else:
            print(f"⚠ No WAN GGUF models found - some tests may be skipped")
        
        # Check for VAE models
        vae_models = folder_paths.get_filename_list("vae")
        wan_vae = [v for v in vae_models if "wan" in v.lower()]
        
        if wan_vae:
            print(f"✓ Found {len(wan_vae)} WAN VAE models: {wan_vae[:3]}")
        else:
            print(f"⚠ No WAN VAE models found - using default VAE")
        
        # Check for CLIP models
        clip_models = folder_paths.get_filename_list("clip")
        t5_clips = [c for c in clip_models if "t5" in c.lower()]
        
        if t5_clips:
            print(f"✓ Found {len(t5_clips)} T5 CLIP models: {t5_clips[:3]}")
        else:
            print(f"⚠ No T5 CLIP models found - some features may not work")
        
        # Check for test images
        input_dir = Path(folder_paths.base_path) / "input"
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        
        if len(image_files) >= 2:
            print(f"✓ Found {len(image_files)} test images in input directory")
        else:
            print(f"⚠ Need at least 2 images in {input_dir} for testing")
            print("  Please add some test images to the ComfyUI input directory")
        
        # Check for LoRA models
        lora_models = folder_paths.get_filename_list("loras")
        wan_loras = [l for l in lora_models if "wan" in l.lower() or "i2v" in l.lower()]
        
        if wan_loras:
            print(f"✓ Found {len(wan_loras)} WAN LoRA models: {wan_loras[:3]}")
        else:
            print(f"⚠ No WAN LoRA models found - LoRA tests may be skipped")
        
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        requirements_met = False
    
    return requirements_met

def run_integration_tests():
    """Run the integration tests using pytest."""
    
    print("\n🧪 Running WAN sampler integration tests...")
    
    # Get the test file path
    test_file = Path(__file__).parent / "test_wan_sampler_full_environment.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v", "-s",
        "--tb=short",
        "-m", "not slow"  # Run fast tests first
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Fast integration tests passed!")
            
            # Ask if user wants to run slow tests
            response = input("\n🐌 Run slow tests (actual model inference)? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                print("\n🧪 Running slow integration tests...")
                
                slow_cmd = cmd[:-2] + ["-m", "slow"]  # Remove "not slow" and add "slow"
                slow_result = subprocess.run(slow_cmd, capture_output=False, text=True)
                
                if slow_result.returncode == 0:
                    print("\n✅ All integration tests passed!")
                    return True
                else:
                    print("\n❌ Some slow tests failed")
                    return False
            else:
                print("\n✅ Fast tests completed successfully")
                return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Test execution failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test runner function."""
    
    print("🚀 WAN Image to Video Advanced Sampler - Full Environment Test")
    print("=" * 60)
    
    # Step 1: Set up ComfyUI environment
    if not setup_comfyui_environment():
        print("\n❌ Failed to set up ComfyUI environment")
        return 1
    
    # Step 2: Check requirements
    if not check_requirements():
        print("\n⚠ Some requirements not met, but continuing with tests...")
    
    # Step 3: Run integration tests
    if run_integration_tests():
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Tests failed or were skipped")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
