import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any, Optional
import math

import folder_paths
import comfy.model_management
import comfy.utils
import comfy.sd
import comfy.model_patcher

# Add current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from longcat_image.pipelines.pipeline_longcat_image_edit import LongCatImageEditPipeline
from longcat_image.pipelines.pipeline_longcat_image import LongCatImagePipeline
from longcat_image.models.longcat_image_dit import LongCatImageTransformer2DModel
from longcat_image.dataset.data_utils import MULTI_ASPECT_RATIO_1024, MULTI_ASPECT_RATIO_512, MULTI_ASPECT_RATIO_256
from longcat_image.utils.model_utils import calculate_shift, retrieve_timesteps, prepare_pos_ids

def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    # image: [B, H, W, C]
    batch_count = image.size(0)
    pil_images = []
    for i in range(batch_count):
        img = image[i].cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img))
    return pil_images

def pil2tensor(images: List[Image.Image]) -> torch.Tensor:
    tensors = []
    for img in images:
        # Convert to numpy, normalize to 0-1
        np_img = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(np_img))
    # Stack to [B, H, W, C]
    if len(tensors) > 0:
        return torch.stack(tensors, dim=0)
    else:
        return torch.empty(0)

class TextEncodeLongCatImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LongCat"

    def encode(self, clip, prompt):
        tokens = clip.tokenize(prompt)
        prompt_embeds = clip.encode_from_tokens(tokens, return_pooled=False)
        
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        
        text_ids = prepare_pos_ids(modality_id=0,
                                   type='text',
                                   start=(0, 0),
                                   num_token=prompt_embeds.shape[1]).to(device, dtype=dtype)
            
        conditioning = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }
        
        return ([[prompt_embeds, conditioning]],)

class TextEncodeLongCatImageEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LongCat"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        tokens = clip.tokenize(prompt)
        prompt_embeds = clip.encode_from_tokens(tokens, return_pooled=False)
        
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        
        text_ids = prepare_pos_ids(modality_id=0,
                                   type='text',
                                   start=(0, 0),
                                   num_token=prompt_embeds.shape[1]).to(device, dtype=dtype)
        
        # Collect images
        images = []
        if image1 is not None: images.append(image1)
        if image2 is not None: images.append(image2)
        if image3 is not None: images.append(image3)
        
        if len(images) > 0 and vae is None:
            raise ValueError("VAE is required when images are provided.")

        conditioning = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "images": images if len(images) > 0 else None,
            "vae": vae
        }
        
        return ([[prompt_embeds, conditioning]],)

class LoadLongCatModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LongCat"

    def load_model(self, unet_name):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Config from LongCat-Image-Edit/transformer/config.json
        config = {
            "patch_size": 1,
            "in_channels": 64,
            "num_layers": 10,
            "num_single_layers": 20,
            "attention_head_dim": 128,
            "num_attention_heads": 24,
            "joint_attention_dim": 3584,
            "pooled_projection_dim": 3584,
            "axes_dims_rope": [16, 56, 56],
        }
        
        # Instantiate model
        model = LongCatImageTransformer2DModel(**config)
        
        # Load weights
        sd = comfy.utils.load_torch_file(unet_path)
        
        # Handle potential key mismatches
        # If loading from a file that was part of a diffusers directory, keys might be correct.
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 or len(u) > 0:
            print(f"LongCat Model Load: Missing {len(m)} keys, Unexpected {len(u)} keys")
            
        # Wrap for ComfyUI
        class LongCatInnerModel:
            def __init__(self, diffusion_model):
                self.diffusion_model = diffusion_model
                self.latent_format = None 
                self.manual_cast_dtype = None
            
            def get_dtype(self):
                return self.diffusion_model.dtype
            
            def is_adm(self):
                return False
                
        inner_model = LongCatInnerModel(model)
        patcher = comfy.model_patcher.ModelPatcher(inner_model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        return (patcher,)

class LongCatSizePicker:
    @classmethod
    def INPUT_TYPES(s):
        # Flatten the dictionaries to a list of strings "resolution (aspect_ratio)"
        # or just use the keys/values.
        # The user said "size: a list of avaliable size(from data_utils.py)"
        
        # Let's combine all sizes
        sizes = []
        for name, mapping in [("1024", MULTI_ASPECT_RATIO_1024), ("512", MULTI_ASPECT_RATIO_512), ("256", MULTI_ASPECT_RATIO_256)]:
            for ratio, (h, w) in mapping.items():
                sizes.append(f"{name} - {ratio} ({int(h)}x{int(w)})")
        
        return {
            "required": {
                "size": (sizes,),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "pick_size"
    CATEGORY = "LongCat"

    def pick_size(self, size):
        # Format: "1024 - 1.0 (1024x1024)"
        try:
            part = size.split("(")[1].split(")")[0]
            h_str, w_str = part.split("x")
            height = int(h_str)
            width = int(w_str)
        except:
            height = 1024
            width = 1024
            
        # Create empty latent
        # Standard ComfyUI latent is [1, 4, H//8, W//8]
        latent = torch.zeros([1, 4, height // 8, width // 8])
        
        return ({"samples": latent}, width, height)

class LongCatImageResizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["keep proportion", "stretch", "fill/crop", "pad"],),
                "interpolation": (["nearest", "bilinear", "bicubic", "lanczos"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "LongCat"

    def resize(self, image, method, interpolation):
        # image is [B, H, W, C]
        pil_images = tensor2pil(image)
        out_images = []
        
        # Determine interpolation
        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        interp = interp_map.get(interpolation, Image.BICUBIC)
        
        final_w, final_h = 0, 0
        
        for img in pil_images:
            w, h = img.size
            
            # Find nearest supported resolution
            all_sizes = []
            for mapping in [MULTI_ASPECT_RATIO_1024, MULTI_ASPECT_RATIO_512, MULTI_ASPECT_RATIO_256]:
                for _, (th, tw) in mapping.items():
                    all_sizes.append((int(tw), int(th))) 
            
            # Find nearest size
            best_size = None
            min_dist = float('inf')
            
            for tw, th in all_sizes:
                dist = (tw - w)**2 + (th - h)**2
                if dist < min_dist:
                    min_dist = dist
                    best_size = (tw, th)
            
            target_w, target_h = best_size
            final_w, final_h = target_w, target_h
            
            if method == "stretch":
                resized = img.resize((target_w, target_h), interp)
                out_images.append(resized)
            elif method == "keep proportion":
                # Find size with closest aspect ratio
                best_ar_size = None
                min_ar_diff = float('inf')
                current_ar = w / h
                
                for tw, th in all_sizes:
                    ar = tw / th
                    diff = abs(ar - current_ar)
                    if diff < min_ar_diff:
                        min_ar_diff = diff
                        best_ar_size = (tw, th)
                
                target_w, target_h = best_ar_size
                final_w, final_h = target_w, target_h
                
                resized = img.resize((target_w, target_h), interp)
                out_images.append(resized)
                
            elif method == "pad":
                ratio = min(target_w/w, target_h/h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                resized = img.resize((new_w, new_h), interp)
                
                new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                new_img.paste(resized, ((target_w - new_w)//2, (target_h - new_h)//2))
                out_images.append(new_img)
                
            elif method == "fill/crop":
                ratio = max(target_w/w, target_h/h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                resized = img.resize((new_w, new_h), interp)
                
                left = (new_w - target_w)//2
                top = (new_h - target_h)//2
                cropped = resized.crop((left, top, left + target_w, top + target_h))
                out_images.append(cropped)

        return (pil2tensor(out_images), final_w, final_h)

class LongCatSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "cfg_norm": ("BOOLEAN", {"default": True}),
                "cfg_renorm_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_after_generate": (["fixed", "increment", "decrement", "random"], {"default": "fixed"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "LongCat"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, cfg_norm, cfg_renorm_min, control_after_generate):
        device = comfy.model_management.get_torch_device()
        transformer = model.model.diffusion_model
        transformer.to(device)
        
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        
        sched = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
        )
        
        samples = latent_image["samples"]
        batch_size, _, h, w = samples.shape
        height = h * 8
        width = w * 8
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        def _pack_latents(latents, batch_size, num_channels_latents, height, width):
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
            return latents

        def _unpack_latents(latents, height, width, vae_scale_factor):
            batch_size, num_patches, channels = latents.shape
            height = 2 * (int(height) // (vae_scale_factor * 2))
            width = 2 * (int(width) // (vae_scale_factor * 2))
            latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
            return latents

        num_channels_latents = 16
        vae_scale_factor = 8
        pixel_step = vae_scale_factor * 2
        height = int(height / pixel_step) * pixel_step
        width = int(width / pixel_step) * pixel_step
        
        latents = torch.randn((batch_size, num_channels_latents, height, width), generator=generator, device=device, dtype=transformer.dtype)
        latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)
        
        latent_image_ids = prepare_pos_ids(modality_id=1,
                                               type='image',
                                               start=(512, 512),
                                               height=height//2,
                                               width=width//2).to(device, dtype=torch.float64)

        # Unpack conditioning
        # positive is [[cond, dict]]
        pos_cond = positive[0][0]
        pos_dict = positive[0][1]
        
        neg_cond = negative[0][0]
        neg_dict = negative[0][1]

        prompt_embeds = pos_cond.to(device, dtype=transformer.dtype)
        text_ids = pos_dict["text_ids"].to(device, dtype=torch.float64)
        
        neg_prompt_embeds = neg_cond.to(device, dtype=transformer.dtype)
        
        image_latents = None
        image_latents_ids = None
        
        if "images" in pos_dict and pos_dict["images"] is not None:
            images = pos_dict["images"]
            vae = pos_dict["vae"]
            
            from diffusers.image_processor import VaeImageProcessor
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
            
            pil_imgs = tensor2pil(images[0])
            
            processed_imgs = image_processor.resize(pil_imgs, height, width)
            processed_imgs = image_processor.preprocess(processed_imgs, height, width).to(device=device, dtype=vae.dtype)
            
            with torch.no_grad():
                encoded = vae.encode(processed_imgs).latent_dist.mode()
                encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
                encoded = encoded.to(dtype=transformer.dtype)
                
                image_latents = _pack_latents(encoded, encoded.shape[0], num_channels_latents, height, width)
                
                image_latents_ids = prepare_pos_ids(modality_id=2,
                                           type='image',
                                           start=(prompt_embeds.shape[1], prompt_embeds.shape[1]),
                                           height=height//2,
                                           width=width//2).to(device, dtype=torch.float64)
        
        if cfg > 1.0:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
        
        if image_latents is not None:
            latent_image_ids = torch.cat([latent_image_ids, image_latents_ids], dim=0)

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            256,
            4096,
            0.5,
            1.15,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            sched,
            steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        with torch.inference_mode():
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                latent_model_input = torch.cat([latent_model_input] * 2) if cfg > 1.0 else latent_model_input
                
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                if image_latents is not None:
                    noise_pred = noise_pred[:, :image_seq_len]

                if cfg > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                    
                    if cfg_norm:
                        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        noise_pred = noise_pred * scale

                latents = sched.step(noise_pred, t, latents, return_dict=False)[0]

        unpacked = _unpack_latents(latents, height, width, vae_scale_factor)
        
        return ({"samples": unpacked},)

NODE_CLASS_MAPPINGS = {
    "TextEncodeLongCatImage": TextEncodeLongCatImage,
    "TextEncodeLongCatImageEdit": TextEncodeLongCatImageEdit,
    "LongCatSizePicker": LongCatSizePicker,
    "LongCatImageResizer": LongCatImageResizer,
    "LongCatSampler": LongCatSampler,
    "LoadLongCatModel": LoadLongCatModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeLongCatImage": "Text Encode LongCat Image",
    "TextEncodeLongCatImageEdit": "Text Encode LongCat Image Edit",
    "LongCatSizePicker": "LongCat Size Picker",
    "LongCatImageResizer": "LongCat Image Resizer",
    "LongCatSampler": "LongCat Sampler",
    "LoadLongCatModel": "Load LongCat Model",
}