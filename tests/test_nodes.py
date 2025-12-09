import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import os

# Mock ComfyUI modules
sys.modules["folder_paths"] = MagicMock()
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.sd"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()
sys.modules["comfy.model_patcher"] = MagicMock()
sys.modules["comfy.clip_vision"] = MagicMock()

# Setup mock values for ComfyUI
sys.modules["comfy.samplers"].KSampler.SAMPLERS = ["euler"]
sys.modules["comfy.samplers"].KSampler.SCHEDULERS = ["normal"]
sys.modules["folder_paths"].get_filename_list.return_value = ["model.safetensors"]
sys.modules["folder_paths"].get_full_path.return_value = "/path/to/model.safetensors"
sys.modules["comfy.model_management"].get_torch_device.return_value = "cpu"

# Mock longcat_image modules
sys.modules["longcat_image"] = MagicMock()
sys.modules["longcat_image.pipelines"] = MagicMock()
sys.modules["longcat_image.pipelines.pipeline_longcat_image_edit"] = MagicMock()
sys.modules["longcat_image.pipelines.pipeline_longcat_image"] = MagicMock()
sys.modules["longcat_image.dataset"] = MagicMock()
sys.modules["longcat_image.dataset.data_utils"] = MagicMock()
sys.modules["longcat_image.utils"] = MagicMock()
sys.modules["longcat_image.utils.model_utils"] = MagicMock()

# Setup specific attributes that are imported in nodes.py
sys.modules["longcat_image.dataset.data_utils"].MULTI_ASPECT_RATIO_1024 = {'1.0': [1024., 1024.]}
sys.modules["longcat_image.dataset.data_utils"].MULTI_ASPECT_RATIO_512 = {'1.0': [512., 512.]}
sys.modules["longcat_image.dataset.data_utils"].MULTI_ASPECT_RATIO_256 = {'1.0': [256., 256.]}

sys.modules["longcat_image.utils.model_utils"].calculate_shift = MagicMock()
sys.modules["longcat_image.utils.model_utils"].retrieve_timesteps = MagicMock()
sys.modules["longcat_image.utils.model_utils"].prepare_pos_ids = MagicMock()

# Mock diffusers
sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers.schedulers"] = MagicMock()
sys.modules["diffusers.image_processor"] = MagicMock()

# Add parent directory to path to import nodes
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import nodes after mocking
from nodes import (
    LongCatPipelineLoader,
    TextEncodeLongCatImage,
    TextEncodeLongCatImageEdit,
    LongCatSizePicker,
    LongCatImageResizer,
    LongCatSampler
)

class TestLongCatNodes(unittest.TestCase):
    
    def setUp(self):
        pass

    @patch("nodes.LongCatImageEditPipeline")
    def test_pipeline_loader(self, mock_pipeline_cls):
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.transformer = MagicMock()
        mock_pipeline.vae = MagicMock()
        mock_pipeline_cls.from_single_file.return_value = mock_pipeline
        
        loader = LongCatPipelineLoader()
        model, clip, vae = loader.load_pipeline("model.safetensors", "fp16")
        
        self.assertEqual(model, mock_pipeline.transformer)
        self.assertEqual(clip, mock_pipeline)
        self.assertEqual(vae, mock_pipeline.vae)
        mock_pipeline_cls.from_single_file.assert_called_once()

    def test_text_encode(self):
        encoder = TextEncodeLongCatImage()
        mock_clip = MagicMock()
        mock_clip.text_encoder.dtype = torch.float32
        mock_clip.encode_prompt.return_value = (torch.zeros(1, 77, 768), torch.zeros(1, 77))
        
        conditioning, = encoder.encode(mock_clip, "a cat")
        
        self.assertIn("prompt_embeds", conditioning)
        self.assertIn("text_ids", conditioning)
        mock_clip.encode_prompt.assert_called_once()

    def test_text_encode_edit(self):
        encoder = TextEncodeLongCatImageEdit()
        mock_clip = MagicMock()
        mock_clip.text_encoder.dtype = torch.float32
        mock_clip.encode_prompt.return_value = (torch.zeros(1, 77, 768), torch.zeros(1, 77))
        
        # Test without images
        conditioning, = encoder.encode(mock_clip, "a cat")
        self.assertIn("prompt_embeds", conditioning)
        self.assertIsNone(conditioning["images"])
        
        # Test with images
        mock_vae = MagicMock()
        image = torch.zeros(1, 512, 512, 3)
        conditioning, = encoder.encode(mock_clip, "a cat", vae=mock_vae, image1=image)
        self.assertIsNotNone(conditioning["images"])
        self.assertEqual(len(conditioning["images"]), 1)
        self.assertEqual(conditioning["vae"], mock_vae)

    def test_size_picker(self):
        picker = LongCatSizePicker()
        # Mock input size string
        size_str = "1024 - 1.0 (1024x1024)"
        
        res = picker.pick_size(size_str)
        latent_dict, width, height = res
        
        self.assertEqual(width, 1024)
        self.assertEqual(height, 1024)
        self.assertEqual(latent_dict["samples"].shape, (1, 4, 128, 128)) # 1024/8 = 128

    def test_image_resizer(self):
        resizer = LongCatImageResizer()
        # Input image 100x100
        image = torch.zeros(1, 100, 100, 3)
        
        # We mocked data_utils, so nearest size to 100x100 in our mock is 256x256 (from MULTI_ASPECT_RATIO_256)
        
        res_image, w, h = resizer.resize(image, "stretch", "nearest")
        
        self.assertEqual(w, 256)
        self.assertEqual(h, 256)
        self.assertEqual(res_image.shape, (1, 256, 256, 3))

    @patch("nodes.comfy.model_management.get_torch_device", return_value="cpu")
    @patch("nodes.retrieve_timesteps")
    @patch("nodes.calculate_shift")
    @patch("nodes.prepare_pos_ids")
    def test_sampler(self, mock_prepare_pos_ids, mock_calculate_shift, mock_retrieve_timesteps, mock_get_device):
        sampler = LongCatSampler()
        
        mock_model = MagicMock()
        mock_model.dtype = torch.float32
        # Mock transformer call
        mock_model.return_value = (torch.zeros(1, 16, 64, 64),)
        
        positive = {
            "prompt_embeds": torch.zeros(1, 10, 768),
            "text_ids": torch.zeros(1, 10)
        }
        negative = {
            "prompt_embeds": torch.zeros(1, 10, 768),
            "text_ids": torch.zeros(1, 10)
        }
        latent_image = {"samples": torch.zeros(1, 4, 64, 64)} # 512x512
        
        # Mock helpers
        mock_prepare_pos_ids.return_value = torch.zeros(1, 10, 3)
        mock_calculate_shift.return_value = 1.0
        mock_retrieve_timesteps.return_value = (torch.tensor([1.0]), 1)
        
        # Mock scheduler step
        mock_scheduler = MagicMock()
        # Return a 3D tensor to simulate packed latents
        # Shape: [1, num_patches, channels]
        # For 512x512 image, latent is 64x64. Packed 2x2 -> 32x32 patches = 1024 patches.
        # Channels = 16 * 4 = 64.
        mock_scheduler.step.return_value = (torch.zeros(1, 1024, 64),) 
        
        # We need to patch FlowMatchEulerDiscreteScheduler inside nodes.py
        with patch("diffusers.schedulers.FlowMatchEulerDiscreteScheduler", return_value=mock_scheduler):
             res = sampler.sample(
                model=mock_model,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                cfg_norm=True,
                cfg_renorm_min=0.0,
                control_after_generate="fixed"
            )
             
        self.assertIn("samples", res[0])

if __name__ == '__main__':
    unittest.main()
