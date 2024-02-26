
import pytest

from diffengine.datasets import ConcatDataset, HFDataset


class TestConcatDataset:

    def setup_method(self):
        self.dataset_a = HFDataset(
            dataset="tests/testdata/dataset", image_column="file_name")
        self.dataset_b = HFDataset(
            dataset="tests/testdata/dataset", image_column="file_name",
            csv="metadata2.csv")

        # test init
        self.cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, self.dataset_b])

    def test_init(self):
        with pytest.raises(TypeError):
            ConcatDataset(datasets=[0])

    def test_full_init(self):
        # test init with lazy_init=True
        self.cat_datasets.full_init()
        assert len(self.cat_datasets) == 2

    def test_length(self):
        assert len(self.cat_datasets) == (
            len(self.dataset_a) + len(self.dataset_b))

    def test_getitem(self):
        assert self.cat_datasets[0]["text"] == self.dataset_a[0]["text"]
        assert self.cat_datasets[0]["text"] != self.dataset_b[0]["text"]
