import sys
from unittest.mock import MagicMock

# Mock ComfyUI modules BEFORE importing nodes
sys.modules["folder_paths"] = MagicMock()
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.sd"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()
sys.modules["comfy.model_patcher"] = MagicMock()
sys.modules["comfy.clip_vision"] = MagicMock()

# Setup mock values
sys.modules["comfy.samplers"].KSampler.SAMPLERS = ["euler"]
sys.modules["comfy.samplers"].KSampler.SCHEDULERS = ["normal"]

import pytest
import torch
from pathlib import Path

# Add repo root to sys.path
current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from comfyui_longcat.nodes import (
    LongCatCheckpointLoader,
    TextEncodeLongCatImage,
    TextEncodeLongCatImageEdit,
    VAEEncodeLongCat,
    VAEDecodeLongCat,
    LongCatSampler,
    LongCatImageSizeScale,
)

def test_loader():
    # Mock folder_paths
    sys.modules["folder_paths"].get_full_path.return_value = "dummy_path.safetensors"
    sys.modules["comfy.utils"].load_torch_file.return_value = {}
    
    loader = LongCatCheckpointLoader()
    model, clip, vae = loader.load_checkpoint("dummy.safetensors")
    
    assert model is not None
    # CLIP/VAE are None in placeholder implementation
    
def test_scale():
    scaler = LongCatImageSizeScale()
    image = torch.zeros((1, 100, 100, 3))
    
    # Mock common_upscale
    sys.modules["comfy.utils"].common_upscale.return_value = torch.zeros((1, 3, 1024, 1024))
    
    scaled, w, h = scaler.scale(image, 1024*1024, "bicubic")
    assert w == 1024
    assert h == 1024

def test_sampler():
    sampler = LongCatSampler()
    # Just check if it calls common_ksampler
    sys.modules["comfy.samplers"].common_ksampler.return_value = ({"samples": torch.zeros(1)},)
    
    res = sampler.sample(MagicMock(), [], [], {}, 0, 1, 1.0, "euler", "normal", 1.0)
    assert res is not None
