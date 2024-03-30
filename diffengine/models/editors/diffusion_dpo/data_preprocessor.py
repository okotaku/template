import torch
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor


class DPODataPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor."""

    def forward(
            self,
            data: dict,
            training: bool = False  # noqa
    ) -> dict | list:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
        ----
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
        -------
            dict or list: Data in the same format as the model input.

        """
        data["inputs"]["img"] = torch.cat(
            torch.stack(data["inputs"]["img"]).chunk(2, dim=1))
        # pre-compute text embeddings
        if "prompt_embeds" in data["inputs"]:
            data["inputs"]["prompt_embeds"] = torch.stack(
                data["inputs"]["prompt_embeds"])
        return super().forward(data)
