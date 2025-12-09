import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the directory containing this file to sys.path to allow importing longcat_image
# which is located in the same directory
current_file_path = Path(__file__).resolve()
node_dir = current_file_path.parent
if str(node_dir) not in sys.path:
    sys.path.insert(0, str(node_dir))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

# ComfyUI imports
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.samplers
import comfy.model_patcher
import comfy.clip_vision

from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline, LongCatImagePipeline
from longcat_image.pipelines.pipeline_longcat_image_edit import calculate_dimensions
from longcat_image.utils.model_utils import prepare_pos_ids, calculate_shift, retrieve_timesteps


# Helpers for device/dtype
def _select_device() -> torch.device:
    return comfy.model_management.get_torch_device()

def _select_dtype(device: torch.device) -> torch.dtype:
    # ComfyUI usually handles this, but we can default to fp16 for inference
    return torch.float16

def _torch_image_to_pil_list(image: torch.Tensor) -> List[Image.Image]:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.clamp(0, 1).detach().cpu()
    pil_images = []
    for img in image:
        np_img = (img.numpy() * 255.0).round().astype(np.uint8)
        pil_images.append(Image.fromarray(np_img))
    return pil_images

def _pil_list_to_torch(images: List[Image.Image]) -> torch.Tensor:
    tensors = []
    for img in images:
        np_img = np.array(img, copy=False)
        if np_img.ndim == 2:
            np_img = np.repeat(np_img[..., None], 3, axis=2)
        tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)
        tensors.append(tensor)
    if not tensors:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    return torch.stack(tensors, dim=0)

def _round_to_multiple(value: int, multiple: int = 16) -> int:
    return int(math.ceil(value / multiple) * multiple)


# --- Loaders & Wrappers ---

class LongCatCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "longcat"

    def load_checkpoint(self, ckpt_name: str):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Load state dict
        sd = comfy.utils.load_torch_file(ckpt_path)
        
        # Load Model (Transformer)
        # We assume the checkpoint contains the transformer weights or is a diffusers dump
        # If it's a single file, we need to know the prefix or structure.
        # For now, let's assume it's a standard Diffusers-style checkpoint loaded as a dict.
        # If it's a folder, comfy.utils.load_torch_file might fail or return dict of files?
        # folder_paths usually returns file paths.
        
        # Initialize Transformer
        # We need config. If not in checkpoint, we might need default config.
        # Assuming standard LongCat config for now.
        transformer = LongCatImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            num_layers=28,
            num_single_layers=10, # Example default, adjust as needed
            attention_head_dim=72,
            num_attention_heads=16,
            joint_attention_dim=768, # Example default
            pooled_projection_dim=768, # Example default
            axes_dims_rope=[16, 56, 56],
        )
        
        # Load weights into transformer
        # We might need to filter keys if the checkpoint contains CLIP/VAE too.
        # This is a simplification. Real implementation needs robust key mapping.
        # transformer.load_state_dict(sd, strict=False) 
        
        # Wrap in ModelPatcher
        model = comfy.model_patcher.ModelPatcher(transformer, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        
        # Load CLIP
        # Using Comfy's CLIP loader logic if possible, or creating one.
        # clip = comfy.sd.load_clip(ckpt_path) # This might work if it's a supported format
        
        # Load VAE
        # vae = comfy.sd.load_vae(ckpt_path)
        
        # Placeholder for actual loading logic which depends on file format
        # For this refactor, we assume we return standard objects.
        
        return (model, None, None) # Placeholder


# --- Encoders ---

class TextEncodeLongCatImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
                "enable_prompt_rewrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "longcat"

    def encode(self, clip: Any, prompt: str, enable_prompt_rewrite: bool):
        # clip is now a comfy.sd.CLIP object
        # We can use standard encoding or custom logic
        
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        # Pack into CONDITIONING
        return ([[cond, {"pooled_output": pooled}]],)


class TextEncodeLongCatImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "longcat"

    def encode(self, clip: Any, vae: Any, image1: torch.Tensor, prompt: str, image2: Optional[torch.Tensor] = None, image3: Optional[torch.Tensor] = None):
        # Custom logic for edit encoding
        # We need to access the internal tokenizer/encoder of the CLIP object if possible
        # or use standard encoding and add image tokens?
        
        # Placeholder for complex logic
        return ([],)


# --- VAE ---

class VAEEncodeLongCat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "longcat"
    
    def encode(self, vae: Any, image: torch.Tensor):
        return (vae.encode(image[:,:,:,:3]),)


class VAEDecodeLongCat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "longcat"
    
    def decode(self, vae: Any, samples: Dict):
        return (vae.decode(samples["samples"]),)


# --- Sampler ---

class LongCatSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "longcat"

    def sample(self, model: Any, positive: List, negative: List, latent_image: Dict, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, denoise: float):
        # Use ComfyUI's common_ksampler logic
        # This handles the sampling loop using the provided sampler/scheduler
        
        return comfy.samplers.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise
        )


class LongCatImageSizeScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_area": ("INT", {"default": 1024 * 1024, "min": 256 * 256, "max": 4096 * 4096}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale"
    CATEGORY = "longcat"

    def scale(self, image: torch.Tensor, target_area: int, upscale_method: str):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        b, h, w, c = image.shape
        ratio = w / float(h)
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = _round_to_multiple(int(round(width)), 16)
        height = _round_to_multiple(int(round(height)), 16)
        
        # Use ComfyUI's common_upscale
        scaled = comfy.utils.common_upscale(image.permute(0, 3, 1, 2), width, height, upscale_method, "center")
        scaled = scaled.permute(0, 2, 3, 1)
        
        return (scaled, int(width), int(height))
