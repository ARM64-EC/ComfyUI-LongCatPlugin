# ComfyUI LongCat Plugin

[简体中文](README_CN.md)

## Overview
ComfyUI nodes wrapping LongCat image generation and editing pipelines (diffusers-based). Includes text-to-image and multi-image edit flows with prompt rewriting, aspect-aware sizing, and latent decoding for ComfyUI.

## Features
- **LongCatCheckpointLoader**: Loads the LongCat model components (Model, CLIP, VAE).
- **TextEncodeLongCatImage / TextEncodeLongCatImageEdit**: Encodes text prompts and reference images (for editing) into conditioning.
- **VAEEncodeLongCat / VAEDecodeLongCat**: Encodes images to latents and decodes latents to images using the LongCat VAE.
- **LongCatSampler**: Handles the sampling process (denoising) using the LongCat transformer.
- **LongCatImageSizeScale**: Scales images to a target pixel area and rounds dimensions to multiples of 16.

## Installation
1.  **Copy Folder**: Copy the `comfyui_longcat` folder into your ComfyUI `custom_nodes` directory.
    *   Example: `ComfyUI/custom_nodes/comfyui_longcat`
2.  **Install Dependencies**: Ensure your ComfyUI environment has the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: `requirements.txt` is in the root of this repo, or check `setup.py` for list).
3.  **Model Weights**: Place LongCat model weights in a directory accessible to ComfyUI.

## Usage in ComfyUI
- Put this plugin folder into ComfyUI `custom_nodes`, then restart.
- Nodes appear under category `longcat`.
- **Workflow**:
    1. Load model with `LongCatCheckpointLoader`.
    2. Connect `CLIP` to `TextEncodeLongCatImage` (T2I) or `TextEncodeLongCatImageEdit` (Edit).
    3. Connect `VAE` to `VAEEncodeLongCat` (if using image input) and `VAEDecodeLongCat`.
    4. Connect `MODEL`, `CONDITIONING` (Positive/Negative), and `LATENT` to `LongCatSampler`.
- Set `model_path` to your downloaded LongCat checkpoint; on CUDA you may enable `cpu_offload` to save VRAM.

## Development
- Tests: `pytest tests/test_nodes.py`
- Key modules: `nodes.py`, `longcat_image/pipelines/*`, `longcat_image/models/longcat_image_dit.py`.

## To-Do
- [ ] Add robust fallback when prompt rewriting model lacks `generate`.
- [ ] Guard `LongCatImageEditPipeline` for `image=None` misuse.
- [ ] Expand docs with example ComfyUI workflows and screenshots.
- [ ] Provide sample model download script and config guidance.
- [ ] Add more automated tests (pipeline smoke tests, dtype/device matrix).

## Credits
- LongCat base model & pipelines: original LongCat project (LongCat team).
- **diffusers** (Hugging Face) for pipeline scaffolding.
- **transformers** (Hugging Face) for text and vision encoders.
- **accelerate** for optional CPU/GPU offload helpers.
- **PyTorch** for core tensor and model runtime.
- **Pillow (PIL)** and **NumPy** for image/tensor conversions.
- OpenAI / DeepSeek API usage scaffold in `misc/prompt_rewrite_api.py`.
