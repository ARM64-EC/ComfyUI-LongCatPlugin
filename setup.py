from pathlib import Path
from setuptools import find_packages, setup


this_dir = Path(__file__).parent
readme = (this_dir / "README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else "LongCat ComfyUI nodes and pipelines."

setup(
    name="comfyui-longcat",
    version="0.1.0",
    description="ComfyUI nodes for LongCat image generation and editing pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LongCat Team",
    packages=find_packages(exclude=("tests", "train_examples")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.41.0",
        "diffusers>=0.30.0",
        "accelerate>=0.27.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "openai>=1.12.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
