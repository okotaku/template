# flake8: noqa: ANN204
# Copyright (c) OpenMMLab. All rights reserved.

import bisect
from unittest import TestCase

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from diffengine.datasets.samplers import MultiSourceSampler


class DummyDataset(Dataset):

    def __init__(self, length, flag):
        self.length = length
        self.flag = flag
        self.shapes = np.random.default_rng().random((length, 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.shapes[idx]

    def get_data_info(self, idx):
        return dict(
            width=self.shapes[idx][0],
            height=self.shapes[idx][1],
            flag=self.flag)


class DummyConcatDataset(ConcatDataset):

    def _get_ori_dataset_idx(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[
            dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_data_info(self, idx: int):
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)


class TestMultiSourceSampler(TestCase):

    def setUp(self):
        self.length_a = 100
        self.dataset_a = DummyDataset(self.length_a, flag="a")
        self.length_b = 1000
        self.dataset_b = DummyDataset(self.length_b, flag="b")
        self.dataset = DummyConcatDataset([self.dataset_a, self.dataset_b])

    def test_multi_source_sampler(self):
        # test dataset is not ConcatDataset
        with self.assertRaises(AssertionError):  # noqa: PT027
            MultiSourceSampler(
                self.dataset_a, batch_size=5, source_ratio=[1, 4])
        # test invalid batch_size
        with self.assertRaises(AssertionError):  # noqa: PT027
            MultiSourceSampler(
                self.dataset_a, batch_size=-5, source_ratio=[1, 4])
        # test source_ratio longer then dataset
        with self.assertRaises(AssertionError):  # noqa: PT027
            MultiSourceSampler(
                self.dataset, batch_size=5, source_ratio=[1, 2, 4])
        sampler = MultiSourceSampler(
            self.dataset, batch_size=5, source_ratio=[1, 4])
        sampler = iter(sampler)  # type: ignore[assignment]
        flags = []
        for _ in range(100):
            idx = next(sampler)
            flags.append(self.dataset.get_data_info(idx)["flag"])
        flags_gt = ["a", "b", "b", "b", "b"] * 20
        self.assertEqual(flags, flags_gt)
