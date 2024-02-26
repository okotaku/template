# flake8: noqa: C901,N814
from collections.abc import Callable, Sequence

from mmengine.dataset import Compose
from mmengine.dataset import ConcatDataset as MMENGINE_CONCATDATASET
from mmengine.registry import DATASETS
from torch.utils.data import Dataset


class ConcatDataset(MMENGINE_CONCATDATASET):
    """Concat dataset.

    Refer to ``mmocr.datasets.dataset_wrapper.ConcatDataset``.

    A wrapper of concatenated dataset.
    Same as ``torch.utils.data.dataset.ConcatDataset``.

    Note:
    ----
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
    ----
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        pipeline (list, optional): Processing pipeline to be applied to all
            of the concatenated datasets. Defaults to [].
        verify_meta (bool): Whether to verify the consistency of meta
            information of the concatenated datasets. Defaults to True.
        force_apply (bool): Whether to force apply pipeline to all datasets if
            any of them already has the pipeline configured. Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.

    """

    def __init__(self,
                 datasets: Sequence[Dataset | dict],
                 pipeline: list[dict | Callable] | None = None,
                 *,
                 force_apply: bool = False) -> None:
        if pipeline is None:
            pipeline = []
        self.datasets: list[Dataset] = []

        # Compose dataset
        pipeline_compose = Compose(pipeline)

        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, Dataset):
                self.datasets.append(dataset)
            else:
                msg = ("elements in datasets sequence should be config or"
                       f" `BaseDataset` instance, but got {type(dataset)}")
                raise TypeError(msg)
            if len(pipeline_compose.transforms) > 0:
                if len(self.datasets[-1].pipeline.transforms,
                       ) > 0 and not force_apply:
                    msg = (f"The pipeline of dataset {i} is not empty, "
                           "please set `force_apply` to True.")
                    raise ValueError(msg)
                self.datasets[-1].pipeline = pipeline_compose

    def full_init(self) -> None:
        """Fully initialize the dataset."""
        super(MMENGINE_CONCATDATASET, self).__init__(self.datasets)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset."""
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]
