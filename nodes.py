import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any, Optional
import math

from transformers import AutoTokenizer, AutoModel, AutoProcessor, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

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
from longcat_image.pipelines.pipeline_longcat_image import (
    LongCatImagePipeline,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_ZH,
    get_prompt_language,
)
from longcat_image.models.longcat_image_dit import LongCatImageTransformer2DModel
from longcat_image.dataset.data_utils import MULTI_ASPECT_RATIO_1024, MULTI_ASPECT_RATIO_512, MULTI_ASPECT_RATIO_256
from longcat_image.utils.model_utils import calculate_shift, retrieve_timesteps, prepare_pos_ids, split_quotation

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


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_device(device_str: str):
    if not device_str or device_str == "auto":
        return comfy.model_management.get_torch_device()

    lowered = device_str.lower()

    if lowered.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Requested CUDA device but CUDA is not available; falling back to default device.")
            return comfy.model_management.get_torch_device()
        return torch.device(device_str)

    if lowered == "mps":
        if not torch.backends.mps.is_available():
            print("Requested MPS device but MPS is not available; falling back to default device.")
            return comfy.model_management.get_torch_device()
        return torch.device("mps")

    if lowered == "cpu":
        return torch.device("cpu")

    # Fallback for any other torch-recognized device string
    return torch.device(device_str)


def list_available_devices():
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
        for idx in range(torch.cuda.device_count()):
            devices.append(f"cuda:{idx}")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    # Deduplicate while preserving order
    deduped = []
    for dev in devices:
        if dev not in deduped:
            deduped.append(dev)
    return deduped


class LongCatCLIPWrapper:
    def __init__(
        self,
        text_encoder,
        tokenizer,
        text_processor,
        image_encoder=None,
        feature_extractor=None,
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.image_processor_vl = getattr(text_processor, "image_processor", None)

        self.prompt_template_encode_prefix = '<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n'
        self.prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        self.prompt_template_encode_start_idx = 36
        self.prompt_template_encode_end_idx = 5
        self.default_sample_size = 128
        self.max_tokenizer_len = 512

        self.image_token = "<|image_pad|>"
        self.edit_prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        self.edit_prompt_template_encode_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        self.edit_prompt_template_encode_start_idx = 67
        self.edit_prompt_template_encode_end_idx = 5

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.text_encoder.parameters()).dtype

    def rewrite_prompt(self, prompt: str) -> str:
        language = get_prompt_language(prompt)
        if language == 'zh':
            question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{prompt}\n改写后的prompt为："
        else:
            question = SYSTEM_PROMPT_EN + f"\nUser Input: {prompt}\nRewritten prompt:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.text_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        generated_ids = self.text_encoder.generate(
            **inputs, max_new_tokens=self.max_tokenizer_len
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    def _encode_tokens(self, prompt: str, prefix: str, suffix: str):
        prompt = prompt.strip('"') if prompt.startswith('"') and prompt.endswith('"') else prompt
        all_tokens = []
        for clean_prompt_sub, matched in split_quotation(prompt):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)['input_ids']
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)

        all_tokens = all_tokens[: self.max_tokenizer_len]
        text_tokens_and_mask = self.tokenizer.pad(
            {'input_ids': [all_tokens]},
            max_length=self.max_tokenizer_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
        suffix_tokens = self.tokenizer(suffix, add_special_tokens=False)['input_ids']
        prefix_tokens_mask = torch.tensor(
            [1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype
        )
        suffix_tokens_mask = torch.tensor(
            [1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype
        )

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat(
            (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1
        )

        return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)

    @torch.inference_mode()
    def encode_text(self, prompts: Union[str, List[str]], rewrite: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt = prompts[0]
        if rewrite:
            prompt = self.rewrite_prompt(prompt)

        input_ids, attention_mask = self._encode_tokens(
            prompt,
            self.prompt_template_encode_prefix,
            self.prompt_template_encode_suffix,
        )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[
            :, self.prompt_template_encode_start_idx: -self.prompt_template_encode_end_idx, :
        ]

        text_ids = prepare_pos_ids(
            modality_id=0,
            type='text',
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(self.device, dtype=torch.float64)

        return prompt_embeds, text_ids

    @torch.inference_mode()
    def encode_edit(self, images: List[Image.Image], prompts: Union[str, List[str]]):
        if self.image_processor_vl is None:
            raise ValueError("Loaded LongCat CLIP does not contain a vision-language processor; please use the edit checkpoint.")

        if isinstance(prompts, str):
            prompts = [prompts]
        prompt = prompts[0]

        raw_vl_input = self.image_processor_vl(images=images, return_tensors="pt")
        pixel_values = raw_vl_input['pixel_values']
        image_grid_thw = raw_vl_input['image_grid_thw']

        prompt = prompt.strip('"') if prompt.startswith('"') and prompt.endswith('"') else prompt

        text = self.edit_prompt_template_encode_prefix
        num_images = len(images)
        if num_images > 1:
            single_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            if single_image_placeholder in text:
                multi_image_placeholder = single_image_placeholder * num_images
                text = text.replace(single_image_placeholder, multi_image_placeholder)

        merge_length = self.image_processor_vl.merge_size ** 2
        image_idx = 0
        while self.image_token in text:
            if image_idx < len(image_grid_thw):
                grid = image_grid_thw[image_idx]
                num_image_tokens = grid.prod() // merge_length
                text = text.replace(self.image_token, "<|placeholder|>" * int(num_image_tokens), 1)
                image_idx += 1
            else:
                break
        text = text.replace("<|placeholder|>", self.image_token)

        input_ids, attention_mask = self._encode_tokens(
            prompt,
            text,
            self.edit_prompt_template_encode_suffix,
        )

        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[
            :, self.edit_prompt_template_encode_start_idx: -self.edit_prompt_template_encode_end_idx, :
        ]

        text_ids = prepare_pos_ids(
            modality_id=0,
            type='text',
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(self.device, dtype=torch.float64)

        return prompt_embeds, text_ids


class LongCatVAEWrapper:
    def __init__(self, vae: AutoencoderKL):
        self.vae = vae
        self.scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        self.shift_factor = getattr(vae.config, "shift_factor", 0.0)
        self.latent_channels = getattr(vae.config, "latent_channels", 16)
        self.vae_scale_factor = 2 ** (len(getattr(vae.config, "block_out_channels", [1])) - 1)

    def to(self, device, dtype=None):
        target_dtype = dtype if dtype is not None else None
        if target_dtype is None:
            self.vae = self.vae.to(device)
        else:
            self.vae = self.vae.to(device=device, dtype=target_dtype)
        return self

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Expect image_tensor shape [B, H, W, C] in range 0..1
        pixel_values = image_tensor.permute(0, 3, 1, 2)
        posterior = self.vae.encode(pixel_values).latent_dist
        latents = posterior.mode()
        latents = (latents - self.shift_factor) * self.scaling_factor
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents / self.scaling_factor) + self.shift_factor
        decoded = self.vae.decode(latents, return_dict=False)[0]
        decoded = decoded.permute(0, 2, 3, 1)
        return decoded

    def get_dtype(self):
        return self.vae.dtype


class LongCatCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (list_available_devices(),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "LongCat"

    def load_clip(self, model_path: str, dtype: str, device: str):
        if not model_path:
            raise ValueError("Please provide a LongCat checkpoint directory for the CLIP loader.")

        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        target_device = _resolve_device(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", trust_remote_code=True
        )
        text_processor = AutoProcessor.from_pretrained(
            model_path, subfolder="tokenizer", trust_remote_code=True
        )
        text_encoder = AutoModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch_dtype, trust_remote_code=True
        ).to(target_device)

        image_encoder = None
        feature_extractor = None
        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path, subfolder="image_encoder", torch_dtype=torch_dtype, trust_remote_code=True
            ).to(target_device)
            feature_extractor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="feature_extractor", trust_remote_code=True
            )
        except Exception:
            image_encoder = None
            feature_extractor = None

        clip = LongCatCLIPWrapper(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_processor=text_processor,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        return (clip,)


class LongCatVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (list_available_devices(),),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "LongCat"

    def load_vae(self, model_path: str, dtype: str, device: str):
        if not model_path:
            raise ValueError("Please provide a LongCat checkpoint directory for the VAE loader.")

        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        target_device = _resolve_device(device)
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch_dtype
        ).to(target_device)
        wrapper = LongCatVAEWrapper(vae)
        return (wrapper,)

class TextEncodeLongCatImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "rewrite_prompt": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LongCat"

    def encode(self, clip: LongCatCLIPWrapper, prompt: str, rewrite_prompt: bool = True):
        prompt_embeds, text_ids = clip.encode_text(prompt, rewrite=rewrite_prompt)

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

    def encode(self, clip: LongCatCLIPWrapper, prompt, vae=None, image1=None, image2=None, image3=None):
        images = []
        if image1 is not None:
            images.append(image1)
        if image2 is not None:
            images.append(image2)
        if image3 is not None:
            images.append(image3)

        if images and vae is None:
            raise ValueError("VAE is required when images are provided.")

        if images:
            pil_images: List[Image.Image] = []
            for img in images:
                pil_images.extend(tensor2pil(img))
            prompt_embeds, text_ids = clip.encode_edit(pil_images, prompt)
        else:
            prompt_embeds, text_ids = clip.encode_text(prompt, rewrite=False)

        conditioning = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "images": images if len(images) > 0 else None,
            "vae": vae,
        }

        return ([[prompt_embeds, conditioning]],)

class LoadLongCatModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (list_available_devices(),),
                "gpu_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LongCat"

    def load_model(self, model_path: str, dtype: str, device: str, gpu_only: bool):
        if not model_path:
            raise ValueError("Please provide a LongCat checkpoint directory or transformer weight file.")

        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        target_device = _resolve_device(device)
        scheduler = None

        if os.path.isdir(model_path):
            transformer = LongCatImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                use_safetensors=True,
            ).to(target_device)
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    model_path, subfolder="scheduler"
                )
            except Exception:
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=3.0,
                    use_dynamic_shifting=True,
                    base_shift=0.5,
                    max_shift=1.15,
                    base_image_seq_len=256,
                    max_image_seq_len=4096,
                )
        else:
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
            transformer = LongCatImageTransformer2DModel(**config).to(target_device)
            sd = comfy.utils.load_torch_file(model_path)
            missing, unexpected = transformer.load_state_dict(sd, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"LongCat Model Load: Missing {len(missing)} keys, Unexpected {len(unexpected)} keys"
                )
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                use_dynamic_shifting=True,
                base_shift=0.5,
                max_shift=1.15,
                base_image_seq_len=256,
                max_image_seq_len=4096,
            )

        class LongCatInnerModel:
            def __init__(self, diffusion_model, scheduler):
                self.diffusion_model = diffusion_model
                self.latent_format = None
                self.manual_cast_dtype = None
                self.longcat_scheduler = scheduler

            def get_dtype(self):
                return self.diffusion_model.dtype

            def is_adm(self):
                return False

            def to(self, device):
                self.diffusion_model.to(device)
                return self

            def state_dict(self, *args, **kwargs):
                return self.diffusion_model.state_dict(*args, **kwargs)

            def load_state_dict(self, *args, **kwargs):
                return self.diffusion_model.load_state_dict(*args, **kwargs)

            def parameters(self, *args, **kwargs):
                return self.diffusion_model.parameters(*args, **kwargs)

            def modules(self, *args, **kwargs):
                return self.diffusion_model.modules(*args, **kwargs)

            def named_modules(self, *args, **kwargs):
                return self.diffusion_model.named_modules(*args, **kwargs)

        inner_model = LongCatInnerModel(transformer, scheduler)

        load_device = target_device
        if gpu_only:
            offload_device = load_device
        else:
            offload_device = comfy.model_management.unet_offload_device()

        patcher = comfy.model_patcher.ModelPatcher(
            inner_model, load_device=load_device, offload_device=offload_device
        )
        patcher.longcat_scheduler = scheduler
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
        comfy.model_management.load_model_gpu(model)
        transformer = model.model.diffusion_model
        device = comfy.model_management.get_torch_device()
        sched = getattr(model, "longcat_scheduler", None)
        if sched is None and hasattr(model, "model"):
            sched = getattr(model.model, "longcat_scheduler", None)
        if sched is None:
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

        def _pack_latents(latents, batch, num_channels_latents, h_px, w_px):
            latents = latents.view(batch, num_channels_latents, h_px // 2, 2, w_px // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch, (h_px // 2) * (w_px // 2), num_channels_latents * 4)
            return latents

        def _unpack_latents(latents, h_px, w_px, vae_scale_factor):
            batch, num_patches, channels = latents.shape
            h_px = 2 * (int(h_px) // (vae_scale_factor * 2))
            w_px = 2 * (int(w_px) // (vae_scale_factor * 2))
            latents = latents.view(batch, h_px // 2, w_px // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(batch, channels // (2 * 2), h_px, w_px)
            return latents

        # Unpack conditioning
        pos_cond = positive[0][0]
        pos_dict = positive[0][1]
        neg_cond = negative[0][0]
        neg_dict = negative[0][1]

        prompt_embeds = pos_cond.to(device, dtype=transformer.dtype)
        neg_prompt_embeds = neg_cond.to(device, dtype=transformer.dtype)

        text_ids = pos_dict["text_ids"].to(device, dtype=torch.float64)
        neg_text_ids = neg_dict.get("text_ids", text_ids).to(device, dtype=torch.float64)

        text_len = prompt_embeds.shape[1]
        neg_text_len = neg_prompt_embeds.shape[1]
        target_len = max(text_len, neg_text_len)

        if prompt_embeds.shape[1] != target_len:
            pad = torch.zeros(
                prompt_embeds.shape[0],
                target_len - prompt_embeds.shape[1],
                prompt_embeds.shape[2],
                device=device,
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

        if neg_prompt_embeds.shape[1] != target_len:
            pad = torch.zeros(
                neg_prompt_embeds.shape[0],
                target_len - neg_prompt_embeds.shape[1],
                neg_prompt_embeds.shape[2],
                device=device,
                dtype=neg_prompt_embeds.dtype,
            )
            neg_prompt_embeds = torch.cat([neg_prompt_embeds, pad], dim=1)

        images = pos_dict.get("images") or neg_dict.get("images")
        vae = pos_dict.get("vae") or neg_dict.get("vae")

        if images is not None and not isinstance(vae, LongCatVAEWrapper):
            raise ValueError("LongCat VAE Loader output is required when conditioning on images.")

        num_channels_latents = 16
        vae_scale_factor = 8
        if isinstance(vae, LongCatVAEWrapper):
            num_channels_latents = vae.latent_channels
            vae_scale_factor = vae.vae_scale_factor

        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = torch.randn(
            (batch_size, num_channels_latents, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=transformer.dtype,
        )
        latents = _pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

        start_pos = max(target_len, 512)
        latent_image_ids = prepare_pos_ids(
            modality_id=1,
            type='image',
            start=(start_pos, start_pos),
            height=latent_height // 2,
            width=latent_width // 2,
        ).to(device, dtype=torch.float64)

        image_latents = None
        image_latents_ids = None

        if images is not None:
            stacked_images = torch.cat(images, dim=0) if len(images) > 1 else images[0]
            stacked_images = stacked_images.to(device=device, dtype=transformer.dtype)
            stacked_images_chw = stacked_images.permute(0, 3, 1, 2)
            stacked_images_chw = torch.nn.functional.interpolate(
                stacked_images_chw, size=(height, width), mode="bicubic", align_corners=False
            )
            stacked_images = stacked_images_chw.permute(0, 2, 3, 1)
            encoded = vae.encode(stacked_images)
            encoded = encoded.to(device=device, dtype=transformer.dtype)

            image_latents = _pack_latents(
                encoded, encoded.shape[0], num_channels_latents, latent_height, latent_width
            )

            image_latents_ids = prepare_pos_ids(
                modality_id=2,
                type="image",
                start=(target_len, target_len),
                height=latent_height // 2,
                width=latent_width // 2,
            ).to(device, dtype=torch.float64)

        if cfg > 1.0:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

        if image_latents is not None:
            latent_image_ids = torch.cat([latent_image_ids, image_latents_ids], dim=0)

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            sched.config.get("base_image_seq_len", 256) if hasattr(sched, "config") else 256,
            sched.config.get("max_image_seq_len", 4096) if hasattr(sched, "config") else 4096,
            sched.config.get("base_shift", 0.5) if hasattr(sched, "config") else 0.5,
            sched.config.get("max_shift", 1.15) if hasattr(sched, "config") else 1.15,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            sched,
            steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        with torch.inference_mode():
            for _, t in enumerate(timesteps):
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

        unpacked = _unpack_latents(latents, latent_height, latent_width, vae_scale_factor)

        return ({"samples": unpacked},)

NODE_CLASS_MAPPINGS = {
    "LongCatCLIPLoader": LongCatCLIPLoader,
    "LongCatVAELoader": LongCatVAELoader,
    "TextEncodeLongCatImage": TextEncodeLongCatImage,
    "TextEncodeLongCatImageEdit": TextEncodeLongCatImageEdit,
    "LongCatSizePicker": LongCatSizePicker,
    "LongCatImageResizer": LongCatImageResizer,
    "LongCatSampler": LongCatSampler,
    "LoadLongCatModel": LoadLongCatModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatCLIPLoader": "LongCat CLIP Loader",
    "LongCatVAELoader": "LongCat VAE Loader",
    "TextEncodeLongCatImage": "Text Encode LongCat Image",
    "TextEncodeLongCatImageEdit": "Text Encode LongCat Image Edit",
    "LongCatSizePicker": "LongCat Size Picker",
    "LongCatImageResizer": "LongCat Image Resizer",
    "LongCatSampler": "LongCat Sampler",
    "LoadLongCatModel": "Load LongCat Model",
}