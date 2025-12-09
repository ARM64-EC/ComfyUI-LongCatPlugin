from .nodes import (
    TextEncodeLongCatImage,
    TextEncodeLongCatImageEdit,
    LongCatSizePicker,
    LongCatImageResizer,
    LongCatSampler,
)

NODE_CLASS_MAPPINGS = {
    "TextEncodeLongCatImage": TextEncodeLongCatImage,
    "TextEncodeLongCatImageEdit": TextEncodeLongCatImageEdit,
    "LongCatSizePicker": LongCatSizePicker,
    "LongCatImageResizer": LongCatImageResizer,
    "LongCatSampler": LongCatSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeLongCatImage": "Text Encode LongCat Image",
    "TextEncodeLongCatImageEdit": "Text Encode LongCat Image Edit",
    "LongCatSizePicker": "LongCat Size Picker",
    "LongCatImageResizer": "LongCat Image Resizer",
    "LongCatSampler": "LongCat Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
