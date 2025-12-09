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

## 状态

### 已实现（仓库中）
- ComfyUI 集成：实现了若干节点和管线，命名空间 `longcat` 中可用。
- 节点与管线：`LongCatCheckpointLoader`、`TextEncodeLongCatImage`、`TextEncodeLongCatImageEdit`、`VAEEncodeLongCat`、`VAEDecodeLongCat`、`LongCatSampler`、`LongCatImageSizeScale`。
- 基础推理脚本：`scripts/inference_t2i.py`、`scripts/inference_edit.py`。
- 模型实现：`longcat_image/models/longcat_image_dit.py` 与 `longcat_image/pipelines/` 下的管线代码。
- 工具类：`longcat_image/utils/*`，以及用于 accelerate 的分布式/加速辅助函数。
- 测试：基本单元测试位于 `tests/test_nodes.py`。
- 训练示例：`train_examples/` 提供 LoRA、SFT、DPO、编辑相关训练的示例脚本与配置。

### 仓库中的训练示例
- `train_examples/lora/`：LoRA 示例训练脚本及配置。
- `train_examples/sft/`：SFT（监督微调）示例脚本与配置。
- `train_examples/dpo/`：DPO（偏好优化）示例脚本与配置。
- `train_examples/edit/`：针对编辑任务的训练示例。

### 规划 / 待办
- [ ] 当提示词改写模型缺少 `generate` 时增加可靠回退。
- [ ] 为 `LongCatImageEditPipeline` 增加 `image=None` 误用保护。
- [ ] 扩展文档，提供完整的 ComfyUI 流程示例、截图与训练教程（LoRA、SFT、DPO）。
- [ ] 提供模型下载脚本和配置指南（包括 Hugging Face Hub 的示例）。
- [ ] 增加全面的自动化测试（管线冒烟测试、dtype/device 组合、示例输入）。
- [ ] 提供可运行的 LoRA / SFT / DPO 训练配方，并补充详细的训练步骤文档。
- [ ] 为模型权重等大文件启用 Git LFS 并说明如何发布权重。
- [ ] 添加 CI（GitHub Actions）用于运行测试与检查。
- [ ] 添加 pre-commit 钩子和统一格式化（Black / isort / flake8）。
- [ ] 提升发布与打包自动化（wheel / setup.py，CI 自动发布）。

## 致谢
- LongCat 基础模型与管线：原 LongCat 项目（LongCat 团队）。
- **diffusers**（Hugging Face）提供管线框架。
- **transformers**（Hugging Face）用于文本与视觉编码器。
- **accelerate** 提供可选的 CPU/GPU offload。
- **PyTorch** 提供核心张量与运行时。
- **Pillow (PIL)** 与 **NumPy** 用于图像/张量转换。
- `misc/prompt_rewrite_api.py` 中的 OpenAI/DeepSeek API 使用脚手架。
