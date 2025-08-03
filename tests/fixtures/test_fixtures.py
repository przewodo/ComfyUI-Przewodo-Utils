"""
Test fixtures and sample data for ComfyUI-Przewodo-Utils tests.
"""
import torch
import json
from pathlib import Path

def create_sample_image_tensor(height=512, width=512, channels=3, batch_size=1):
    """Create a sample image tensor for testing."""
    return torch.randn(batch_size, height, width, channels)

def create_sample_latent_tensor(height=64, width=64, channels=4, batch_size=1):
    """Create a sample latent tensor for testing."""
    return torch.randn(batch_size, channels, height, width)

def create_sample_video_tensor(frames=16, height=512, width=512, channels=3, batch_size=1):
    """Create a sample video tensor for testing."""
    return torch.randn(batch_size, frames, height, width, channels)

def create_sample_lora_stack():
    """Create a sample LoRA stack for testing."""
    return [
        ["style_lora.safetensors", 0.8, 0.8],
        ["detail_lora.safetensors", 0.6, 0.6],
        ["cinematic_lora.safetensors", 1.0, 1.0],
    ]

def create_sample_prompt_stack():
    """Create a sample prompt stack for testing."""
    return [
        ("A beautiful landscape in spring", "blurry, low quality", 0, None),
        ("A beautiful landscape in summer", "blurry, low quality", 1, None),
        ("A beautiful landscape in autumn", "blurry, low quality", 2, None),
    ]

def create_basic_workflow_json():
    """Create a basic workflow JSON for testing."""
    return {
        "last_node_id": 3,
        "last_link_id": 2,
        "nodes": [
            {
                "id": 1,
                "type": "WanImageToVideoAdvancedSampler",
                "pos": [100, 100],
                "size": {"0": 400, "1": 600},
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [
                    {"name": "start_image", "type": "IMAGE", "link": 1}
                ],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [2]},
                    {"name": "FPS", "type": "FLOAT", "links": None}
                ],
                "properties": {
                    "Node name for S&R": "WanImageToVideoAdvancedSampler"
                },
                "widgets_values": [
                    "test_model.gguf",  # GGUF_High
                    "None",             # GGUF_Low
                    "A beautiful video", # positive
                    "blurry",           # negative
                    "test_clip.safetensors", # clip
                    "wan",              # clip_type
                    "auto",             # clip_device
                    "test_vae.safetensors",  # vae
                    True,               # use_tea_cache
                    15,                 # total_steps
                    42,                 # noise_seed
                ]
            }
        ],
        "links": [
            [1, 0, 0, 1, 0, "IMAGE"],
            [2, 1, 0, 2, 0, "IMAGE"]
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }

def save_test_fixture(data, filename, fixtures_dir="tests/fixtures"):
    """Save test fixture data to file."""
    fixtures_path = Path(fixtures_dir)
    fixtures_path.mkdir(parents=True, exist_ok=True)
    
    file_path = fixtures_path / filename
    
    if filename.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif filename.endswith('.pt'):
        torch.save(data, file_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    return file_path

def load_test_fixture(filename, fixtures_dir="tests/fixtures"):
    """Load test fixture data from file."""
    fixtures_path = Path(fixtures_dir)
    file_path = fixtures_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test fixture not found: {file_path}")
    
    if filename.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif filename.endswith('.pt'):
        return torch.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

# Create some standard test fixtures
if __name__ == "__main__":
    # Create sample test data
    fixtures_dir = Path(__file__).parent
    
    # Save sample tensors
    save_test_fixture(
        create_sample_image_tensor(), 
        "sample_image.pt", 
        fixtures_dir
    )
    
    save_test_fixture(
        create_sample_latent_tensor(), 
        "sample_latent.pt", 
        fixtures_dir
    )
    
    save_test_fixture(
        create_sample_video_tensor(), 
        "sample_video.pt", 
        fixtures_dir
    )
    
    # Save sample stacks
    save_test_fixture(
        create_sample_lora_stack(), 
        "sample_lora_stack.json", 
        fixtures_dir
    )
    
    save_test_fixture(
        create_sample_prompt_stack(), 
        "sample_prompt_stack.json", 
        fixtures_dir
    )
    
    # Save sample workflow
    save_test_fixture(
        create_basic_workflow_json(), 
        "basic_workflow.json", 
        fixtures_dir
    )
    
    print("Test fixtures created successfully!")
