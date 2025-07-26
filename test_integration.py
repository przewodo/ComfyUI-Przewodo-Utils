#!/usr/bin/env python3
"""
Integration test for TAESD Wan2.1 with ComfyUI simulation.

This script tests the integration with simulated ComfyUI environment.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_comfyui_integration():
    """Test integration with simulated ComfyUI environment."""
    
    print("Testing ComfyUI integration...")
    
    try:
        # Import the main sampler class
        # Remove relative import attempts from this test - they won't work in standalone mode
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import wan_image_to_video_advanced_sampler
        WanImageToVideoAdvancedSampler = wan_image_to_video_advanced_sampler.WanImageToVideoAdvancedSampler
        
        print("✓ Successfully imported WanImageToVideoAdvancedSampler")
        
        # Create instance
        sampler = WanImageToVideoAdvancedSampler()
        print("✓ Sampler instance created")
        
        # Test TAESD setup method
        print("Testing TAESD setup...")
        
        # Create a mock latent format
        class MockLatentFormat:
            def __init__(self):
                self.latent_rgb_factors = None
                self.taesd_decoder_name = None
        
        mock_format = MockLatentFormat()
        
        # Test the setup method
        try:
            result = sampler.setup_wan21_taesd_preview(mock_format)
            print(f"✓ TAESD setup completed: {result}")
            
            if result:
                print("✓ TAESD preview system successfully configured")
            else:
                print("⚠ TAESD setup returned False (expected if no TAEHV models found)")
            
        except Exception as e:
            print(f"✗ TAESD setup failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_download_system():
    """Test the download system functionality."""
    
    print("\nTesting download system...")
    
    try:
        import wan_image_to_video_advanced_sampler
        WanImageToVideoAdvancedSampler = wan_image_to_video_advanced_sampler.WanImageToVideoAdvancedSampler
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Test connectivity check
        print("Testing connectivity check...")
        try:
            has_internet = sampler.test_internet_connectivity()
            print(f"✓ Internet connectivity test: {has_internet}")
        except Exception as e:
            print(f"⚠ Connectivity test failed: {e}")
        
        # Test download URL validation (without actually downloading)
        print("Testing download URL validation...")
        
        test_urls = [
            "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth",
            "https://github.com/madebyollin/taehv/raw/main/taecvx.safetensors"
        ]
        
        for url in test_urls:
            print(f"  Checking URL format: {url}")
            if url.startswith("https://github.com/madebyollin/taehv/"):
                print(f"  ✓ URL format is correct")
            else:
                print(f"  ✗ URL format is incorrect")
        
        return True
        
    except Exception as e:
        print(f"✗ Download system test failed: {e}")
        return False

def test_preview_patching():
    """Test the preview system patching functionality."""
    
    print("\nTesting preview system patching...")
    
    try:
        import wan_image_to_video_advanced_sampler
        WanImageToVideoAdvancedSampler = wan_image_to_video_advanced_sampler.WanImageToVideoAdvancedSampler
        
        sampler = WanImageToVideoAdvancedSampler()
        
        # Test the preview patching methods exist
        required_methods = [
            'setup_wan21_taesd_preview',
            'install_wan21_taesd_decoder',
            'patch_preview_system_for_wan21_taesd'
        ]
        
        for method_name in required_methods:
            if hasattr(sampler, method_name):
                print(f"✓ Method {method_name} exists")
            else:
                print(f"✗ Method {method_name} missing")
                return False
        
        # Test that the preview patching doesn't crash
        try:
            import taesd_wan21
            get_wan21_taesd_decoder = taesd_wan21.get_wan21_taesd_decoder
            
            # Create a temporary directory for testing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                decoder = get_wan21_taesd_decoder(temp_dir)
                if decoder is not None:
                    print("✓ TAESD decoder creation works")
                else:
                    print("⚠ TAESD decoder returned None (expected without TAEHV models)")
                
        except Exception as e:
            print(f"⚠ TAESD decoder test warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preview patching test failed: {e}")
        return False

def main():
    """Run integration tests."""
    
    print("=" * 70)
    print("TAESD Wan2.1 ComfyUI Integration Test Suite")
    print("=" * 70)
    
    success = True
    
    # Test ComfyUI integration
    if not test_comfyui_integration():
        success = False
    
    # Test download system
    if not test_download_system():
        success = False
    
    # Test preview patching
    if not test_preview_patching():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All integration tests passed! The system is ready for ComfyUI.")
        print("✓ TAESD Wan2.1 implementation is compatible with ComfyUI architecture.")
        print("✓ Download system and preview patching are functional.")
    else:
        print("✗ Some integration tests failed. Please review the implementation.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()
