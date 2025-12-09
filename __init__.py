from .nodes import (
    LongCatCheckpointLoader,
    TextEncodeLongCatImage,
    TextEncodeLongCatImageEdit,
    VAEEncodeLongCat,
    VAEDecodeLongCat,
    LongCatSampler,
    LongCatImageSizeScale,
)

NODE_CLASS_MAPPINGS = {
    "LongCatCheckpointLoader": LongCatCheckpointLoader,
    "TextEncodeLongCatImage": TextEncodeLongCatImage,
    "TextEncodeLongCatImageEdit": TextEncodeLongCatImageEdit,
    "VAEEncodeLongCat": VAEEncodeLongCat,
    "VAEDecodeLongCat": VAEDecodeLongCat,
    "LongCatSampler": LongCatSampler,
    "LongCatImageSizeScale": LongCatImageSizeScale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatCheckpointLoader": "LongCat Checkpoint Loader",
    "TextEncodeLongCatImage": "LongCat Text Encode (T2I)",
    "TextEncodeLongCatImageEdit": "LongCat Text Encode (Edit)",
    "VAEEncodeLongCat": "LongCat VAE Encode",
    "VAEDecodeLongCat": "LongCat VAE Decode",
    "LongCatSampler": "LongCat Sampler",
    "LongCatImageSizeScale": "LongCat Image Size Scale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
