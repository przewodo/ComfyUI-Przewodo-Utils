#!/usr/bin/env python3
"""
Simple test for TAESD Wan2.1 decoder architecture.

This script tests just the decoder architecture without file loading.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_decoder_architecture():
    """Test the TAESD decoder architecture directly."""
    
    print("Testing TAESD Wan2.1 decoder architecture...")
    
    try:
        # Import the decoder function directly
        from taesd_wan21 import Wan21Decoder
        
        # Create a decoder directly
        print("Creating TAESD decoder for Wan2.1...")
        decoder = Wan21Decoder()
        print(f"✓ Decoder created successfully: {type(decoder)}")
        
        # Test with dummy 16-channel latents
        print("Testing with dummy 16-channel latents...")
        
        # Create dummy latent tensor (batch_size=1, channels=16, height=64, width=64)
        dummy_latents = torch.randn(1, 16, 64, 64)
        print(f"✓ Dummy latents shape: {dummy_latents.shape}")
        
        # Set model to eval mode
        decoder.eval()
        
        # Test inference
        with torch.no_grad():
            try:
                output = decoder(dummy_latents)
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Output dtype: {output.dtype}")
                print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                # Verify output dimensions (should be 3-channel RGB)
                if output.shape[1] == 3:
                    print("✓ Output has correct 3-channel RGB format")
                else:
                    print(f"✗ Output has {output.shape[1]} channels, expected 3")
                    return False
                
                # Check if output is in reasonable range for images
                if -1.5 <= output.min().item() <= 1.5 and -1.5 <= output.max().item() <= 1.5:
                    print("✓ Output values are in reasonable range for normalized images")
                else:
                    print(f"⚠ Output values may be outside expected range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                return True
                
            except Exception as e:
                print(f"✗ Inference failed: {e}")
                return False
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_state_dict_structure():
    """Test that the state dict structure is correct."""
    
    print("\nTesting state dict structure...")
    
    try:
        from taesd_wan21 import Wan21Decoder
        
        decoder = Wan21Decoder()
        state_dict = decoder.state_dict()
        
        print(f"✓ State dict has {len(state_dict)} parameters")
        
        # Check key structure
        sample_keys = list(state_dict.keys())[:10]  # First 10 keys
        print(f"Sample keys: {sample_keys}")
        
        # Verify no encoder/decoder prefixes (ComfyUI TAESD expects flat structure)
        has_prefixes = any(key.startswith(('encoder.', 'decoder.')) for key in state_dict.keys())
        if not has_prefixes:
            print("✓ State dict has flat structure (no encoder./decoder. prefixes)")
        else:
            print("✗ State dict has encoder./decoder. prefixes (incompatible with ComfyUI TAESD)")
            return False
        
        # Check for sequential layer structure
        has_sequential = any('.' in key and key.split('.')[0].isdigit() for key in state_dict.keys())
        if has_sequential:
            print("✓ State dict has sequential layer structure")
        else:
            print("⚠ State dict may not have expected sequential structure")
        
        return True
        
    except Exception as e:
        print(f"✗ State dict test failed: {e}")
        return False

def test_taesd_class():
    """Test the TAESDWan21 class."""
    
    print("\nTesting TAESDWan21 class...")
    
    try:
        from taesd_wan21 import TAESDWan21
        
        # Create instance
        taesd = TAESDWan21()
        print(f"✓ TAESDWan21 created successfully: {type(taesd)}")
        
        # Test decoder access
        decoder = taesd.decoder
        print(f"✓ Decoder accessible: {type(decoder)}")
        
        # Test with dummy input
        dummy_latents = torch.randn(1, 16, 32, 32)
        with torch.no_grad():
            output = taesd(dummy_latents)
            print(f"✓ TAESDWan21 inference successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ TAESDWan21 test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("=" * 60)
    print("TAESD Wan2.1 Architecture Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test decoder architecture
    if not test_decoder_architecture():
        success = False
    
    # Test state dict structure
    if not test_state_dict_structure():
        success = False
    
    # Test TAESD class
    if not test_taesd_class():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! TAESD Wan2.1 architecture is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()
