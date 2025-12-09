# ComfyUI LongCat 插件

[English](README.md)

## 插件简介
基于 Hugging Face diffusers 的 LongCat 文生图与图像编辑管线，对应的 ComfyUI 节点集。支持提示词改写、按比例自动计算分辨率、潜变量解码，方便在 ComfyUI 中直接使用。

## 功能特性
- **LongCatCheckpointLoader**：加载 LongCat 模型组件（Model, CLIP, VAE）。
- **TextEncodeLongCatImage / TextEncodeLongCatImageEdit**：将文本提示词和参考图（编辑模式）编码为条件。
- **VAEEncodeLongCat / VAEDecodeLongCat**：使用 LongCat VAE 将图像编码为潜变量，或将潜变量解码为图像。
- **LongCatSampler**：使用 LongCat Transformer 进行采样（去噪）。
- **LongCatImageSizeScale**：按目标像素面积缩放，并将宽高对齐到 16 的倍数。

## 已实现（当前可用）
- Transformer: `LongCatImageTransformer2DModel` 已实现并用于管线。
- 管线：
    - `LongCatImagePipeline`（文生图）：实现了提示词改写、tokenizer、潜变量打包、去噪循环与 VAE 解码等功能。
    - `LongCatImageEditPipeline`（图像编辑）：实现了视觉-语言提示处理、图像潜变量与编辑专用去噪逻辑。
- 节点（ComfyUI）：
    - `LongCatCheckpointLoader`：实现了基本的 transformer 加载（注意：CLIP/VAE 的完全加载仍为占位/脚手架实现，需要改进以支持多种模型格式）。
    - `TextEncodeLongCatImage`：T2I 文本编码实现（基于 CLIP tokenizer/encoder）。
    - `TextEncodeLongCatImageEdit`：编辑模式的文本+图像编码（部分实现，尚需完善对 CLIP/VL 的完整支持）。
    - `VAEEncodeLongCat` 与 `VAEDecodeLongCat`：VAE 编码/解码封装。
    - `LongCatSampler`：封装 ComfyUI 的采样器。
    - `LongCatImageSizeScale`：图像缩放节点。
- 工具函数：`longcat_image/utils/model_utils.py` 包含 `pack_latents`/`unpack_latents`、`prepare_pos_ids`、`split_quotation`、`retrieve_timesteps`、`optimized_scale` 等函数。
- 训练示例：包含 LoRA、SFT、DPO、编辑训练脚本和示例配置（位于 `train_examples/`）。
- 测试：提供简单的节点级单元测试（`tests/test_nodes.py`）。

## 安装
1.  **复制文件夹**：将 `comfyui_longcat` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录下。
    *   例如：`ComfyUI/custom_nodes/comfyui_longcat`
2.  **安装依赖**：确保 ComfyUI 环境中安装了必要的包：
    ```bash
    pip install -r requirements.txt
    ```
    （注：`requirements.txt` 位于本仓库根目录，或查看 `setup.py` 中的列表）。
3.  **模型权重**：将 LongCat 模型权重放入 ComfyUI 可访问的目录。

## 在 ComfyUI 中使用
- 将本插件目录放入 ComfyUI 的 `custom_nodes`，重启后生效。
- 节点位于分类 `longcat`。
- **工作流**：
    1. 使用 `LongCatCheckpointLoader` 加载模型。
    2. 连接 `CLIP` 到 `TextEncodeLongCatImage` (文生图) 或 `TextEncodeLongCatImageEdit` (编辑)。
    3. 连接 `VAE` 到 `VAEEncodeLongCat` (如有图像输入) 和 `VAEDecodeLongCat`。
    4. 连接 `MODEL`、`CONDITIONING` (正向/负向) 和 `LATENT` 到 `LongCatSampler`。
- `model_path` 指向已下载的 LongCat 权重；在 CUDA 上可开启 `cpu_offload` 以节省显存。

## 开发
- 测试：`pytest tests/test_nodes.py`
- 关键模块：`nodes.py`、`longcat_image/pipelines/*`、`longcat_image/models/longcat_image_dit.py`。

## 待办
### 已完成 / 进行中
- 实现了核心 transformer 与用于文生图/编辑的管线。
- 实现了基本的节点封装（加载器、编码器、VAE、采样器、大小缩放）。
- 提供了针对 LoRA、SFT、DPO 和编辑训练流程的示例脚本。

### 计划 / 路线图
- [ ] 完善 `LongCatCheckpointLoader` 对多种权重格式的兼容：
    - 支持 Diffusers 风格的 checkpoint（subfolder: transformer, vae, tokenizer, scheduler 等）
    - 支持单文件 safetensors/checkpoint 并正确映射模型组件
    - 正确加载 CLIP 与 VAE，并支持自动的设备与 offload 配置
- [ ] 完成 `TextEncodeLongCatImageEdit` 的实现（支持多图输入、VL 合并和 placeholder 替换）。
- [ ] 补充文档：一步步的 ComfyUI 使用流程、示例截图和模型准备脚本。
- [ ] 增加自动化管线测试（冒烟测试、不同 dtype/device 组合）。
- [ ] 为大权重文件启用 Git LFS 并给出示例与说明。
- [ ] 增加 GitHub Actions CI（PR 检查，lint 和运行测试）。
- [ ] 添加 `pre-commit` 配置以避免误提交缓存和二进制文件。
- [ ] 补充示例说明如何运行 LoRA / SFT / DPO 训练脚本以及如何将训练产物应用到 ComfyUI 节点。

如果您希望，我可以继续：完善 `LongCatCheckpointLoader` 和 `TextEncodeLongCatImageEdit`，或为仓库添加 CI 配置与 Git LFS 支持。

## 致谢
- LongCat 基础模型与管线：原 LongCat 项目（LongCat 团队）。
- **diffusers**（Hugging Face）提供管线框架。
- **transformers**（Hugging Face）用于文本与视觉编码器。
- **accelerate** 提供可选的 CPU/GPU offload。
- **PyTorch** 提供核心张量与运行时。
- **Pillow (PIL)** 与 **NumPy** 用于图像/张量转换。
- `misc/prompt_rewrite_api.py` 中的 OpenAI/DeepSeek API 使用脚手架。
