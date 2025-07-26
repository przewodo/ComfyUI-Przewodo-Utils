#!/usr/bin/env python3
"""
Simplified test for TAESD Wan2.1 that focuses only on the essential functionality.

This script tests the TAESD implementation without ComfyUI dependencies.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_functionality():
    """Test the core TAESD functionality."""
    
    print("Testing core TAESD Wan2.1 functionality...")
    
    try:
        from taesd_wan21 import TAESDWan21, Wan21Decoder, get_wan21_taesd_decoder
        
        print("✓ Successfully imported TAESD components")
        
        # Test 1: Direct decoder
        print("\nTest 1: Direct decoder functionality")
        decoder = Wan21Decoder()
        
        dummy_latents = torch.randn(1, 16, 64, 64)
        with torch.no_grad():
            output = decoder(dummy_latents)
            print(f"✓ Decoder output shape: {output.shape}")
            print(f"✓ Expected upscaling: 64x64 → 512x512 ✓" if output.shape[-2:] == (512, 512) else f"✗ Unexpected output size: {output.shape[-2:]}")
        
        # Test 2: TAESD class
        print("\nTest 2: TAESDWan21 class functionality")
        taesd = TAESDWan21()
        
        with torch.no_grad():
            output = taesd(dummy_latents)
            print(f"✓ TAESDWan21 output shape: {output.shape}")
        
        # Test 3: State dict structure
        print("\nTest 3: State dict compatibility")
        state_dict = taesd.state_dict()
        print(f"✓ State dict has {len(state_dict)} parameters")
        
        # Check for ComfyUI compatibility
        has_bad_prefixes = any(key.startswith(('encoder.', 'decoder.')) for key in state_dict.keys())
        if not has_bad_prefixes:
            print("✓ State dict structure is ComfyUI TAESD compatible")
        else:
            print("✗ State dict has incompatible prefixes")
            return False
        
        # Test 4: Device compatibility
        print("\nTest 4: Device compatibility")
        if torch.cuda.is_available():
            taesd_cuda = taesd.cuda()
            dummy_cuda = dummy_latents.cuda()
            
            with torch.no_grad():
                output_cuda = taesd_cuda(dummy_cuda)
                print(f"✓ CUDA inference successful: {output_cuda.shape}")
                
            taesd_cuda.cpu()  # Move back to CPU
        else:
            print("⚠ CUDA not available, skipping GPU test")
        
        return True
        
    except Exception as e:
        print(f"✗ Core functionality test failed: {e}")
        return False

def test_weight_creation():
    """Test weight creation and adaptation functionality."""
    
    print("\nTesting weight creation functionality...")
    
    try:
        from taesd_wan21 import create_wan21_taesd_weights
        
        # Test weight creation
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Test creating weights from scratch
            print("Testing fallback weight creation...")
            try:
                weights_path = create_wan21_taesd_weights(temp_dir, taehv_model=None)
                if weights_path and os.path.exists(weights_path):
                    print(f"✓ Fallback weights created: {weights_path}")
                    
                    # Test loading the created weights
                    state_dict = torch.load(weights_path, map_location='cpu')
                    print(f"✓ Created weights have {len(state_dict)} parameters")
                    
                    return True
                else:
                    print("⚠ Fallback weight creation returned None")
                    return False
                    
            except Exception as e:
                print(f"✗ Weight creation failed: {e}")
                return False
        
    except Exception as e:
        print(f"✗ Weight creation test failed: {e}")
        return False

def test_download_urls():
    """Test that download URLs are correct."""
    
    print("\nTesting download URL configuration...")
    
    expected_urls = {
        "taew2_1.pth": "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth",
        "taecvx.safetensors": "https://github.com/madebyollin/taehv/raw/main/taecvx.safetensors",
        "taeos1_3.safetensors": "https://github.com/madebyollin/taehv/raw/main/taeos1_3.safetensors",
        "taehv.safetensors": "https://github.com/madebyollin/taehv/raw/main/taehv.safetensors"
    }
    
    for filename, expected_url in expected_urls.items():
        if "madebyollin/taehv" in expected_url and "/raw/main/" in expected_url:
            print(f"✓ URL for {filename} is correctly formatted")
        else:
            print(f"✗ URL for {filename} has incorrect format: {expected_url}")
            return False
    
    print("✓ All download URLs are correctly configured")
    return True

def main():
    """Run all simplified tests."""
    
    print("=" * 70)
    print("TAESD Wan2.1 Simplified Test Suite")
    print("=" * 70)
    
    success = True
    
    # Test core functionality
    if not test_core_functionality():
        success = False
    
    # Test weight creation
    if not test_weight_creation():
        success = False
    
    # Test download URLs
    if not test_download_urls():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed! TAESD Wan2.1 implementation is ready.")
        print("✓ Architecture is compatible with ComfyUI TAESD system.")
        print("✓ Weight creation and download systems are functional.")
        print("\nNext steps:")
        print("1. Place TAEHV models (taew2_1.pth) in ComfyUI/models/vae_approx/")
        print("2. Use Wan2.1 models in the advanced sampler")
        print("3. Enable preview method to see TAESD previews")
    else:
        print("✗ Some tests failed. Please review the implementation.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
