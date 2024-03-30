import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.datasets import HFDPODataset, HFDPODatasetPreComputeEmbs


class TestHFDPODataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDPODataset(
            dataset="tests/testdata/dataset", csv="metadata_dpo.csv")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a dog"
        assert len(data["img"]) == 2
        assert isinstance(data["img"][0], Image.Image)
        assert data["img"][0].width == 400


class TestHFDPODatasetPreComputeEmbsPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDPODatasetPreComputeEmbs(
            dataset="tests/testdata/dataset", csv="metadata_dpo.csv",
            model="hf-internal-testing/tiny-stable-diffusion-torch",
            tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                        subfolder="tokenizer"),
            text_encoder=dict(type=CLIPTextModel.from_pretrained,
                        subfolder="text_encoder"),
            device="cpu")
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (77, 32)
        assert len(data["img"]) == 2
        assert isinstance(data["img"][0], Image.Image)
        assert data["img"][0].width == 400
