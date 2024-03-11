from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import ImageHubDreamBoothDataset


class TestImageHubDreamBoothDataset(RunnerTestCase):

    def test_dataset(self):
        dataset = ImageHubDreamBoothDataset(
            dataset="ImagenHub/DreamBooth_Concepts",
            subject="dog")
        assert len(dataset) == 5

        data = dataset[0]
        assert data["text"] == "A zwx dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 512
