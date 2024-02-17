from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.datasets import HFDataset


class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDataset(
            dataset="tests/testdata/dataset", image_column="file_name")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

        dataset = HFDataset(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            csv="metadata2.csv")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a cat"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
