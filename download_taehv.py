"""
TAEHV Model Download Helper

This script helps users download TAEHV models for Wan2.1 preview support.
Run this script to automatically download the required TAEHV models.
"""

import os
import requests
import folder_paths
from tqdm import tqdm


def download_file(url, filepath, description="Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_taehv_models():
    """Download TAEHV models to the appropriate ComfyUI directory."""
    
    # Get vae_approx directory
    vae_approx_paths = folder_paths.get_folder_paths("vae_approx")
    if not vae_approx_paths:
        print("Error: vae_approx folder not found in ComfyUI")
        return False
    
    vae_approx_dir = vae_approx_paths[0]
    os.makedirs(vae_approx_dir, exist_ok=True)
    
    # Define TAEHV models to download
    models = [
        {
            "name": "taehv.pth",
            "url": "https://huggingface.co/madebyollin/taehv/resolve/main/taehv.pth",
            "description": "TAEHV for Hunyuan Video"
        },
        {
            "name": "taew2_1.pth", 
            "url": "https://huggingface.co/madebyollin/taehv/resolve/main/taew2_1.pth",
            "description": "TAEHV for Wan 2.1"
        }
    ]
    
    print("Downloading TAEHV models for Wan2.1 preview support...")
    print(f"Target directory: {vae_approx_dir}")
    
    success_count = 0
    
    for model in models:
        filepath = os.path.join(vae_approx_dir, model["name"])
        
        # Skip if already exists
        if os.path.exists(filepath):
            print(f"✓ {model['name']} already exists, skipping")
            success_count += 1
            continue
        
        try:
            print(f"\nDownloading {model['name']}...")
            download_file(model["url"], filepath, model["description"])
            print(f"✓ {model['name']} downloaded successfully")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed to download {model['name']}: {e}")
    
    print(f"\nDownload complete: {success_count}/{len(models)} models")
    
    if success_count > 0:
        print("\n🎉 TAEHV models are now available for Wan2.1 preview support!")
        print("Restart ComfyUI to use the new preview functionality.")
        return True
    else:
        print("\n❌ No models were downloaded successfully.")
        return False


if __name__ == "__main__":
    try:
        download_taehv_models()
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual installation:")
        print("1. Download TAEHV models from https://huggingface.co/madebyollin/taehv")
        print("2. Place them in ComfyUI/models/vae_approx/")
        print("3. Restart ComfyUI")
