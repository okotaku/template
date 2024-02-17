import random

from diffengine.datasets.transforms.base import BaseTransform


class RandomTextDrop(BaseTransform):
    """RandomTextDrop. Replace text to empty.

    Args:
    ----
        p (float): probability of the image being flipped.
            Default value is 0.5.
        keys (List[str]): `keys` to apply augmentation from results.

    """

    def __init__(self, p: float = 0.1,
                 keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["text"]
        self.p = p
        self.keys = keys

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        """
        if random.random() < self.p:
            for k in self.keys:
                results[k] = ""
        return results



class AddConstantCaption(BaseTransform):
    """AddConstantCaption.

    Example. "a dog." * constant_caption="in szn style"
        -> "a dog. in szn style"

    Args:
    ----
        constant_caption (str): `constant_caption` to add.
        keys (List[str], optional): `keys` to apply augmentation from results.
            Defaults to None.

    """

    def __init__(self, constant_caption: str,
                 keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["text"]
        self.constant_caption: str = constant_caption
        self.keys = keys

    def transform(self,
                  results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        """
        for k in self.keys:
            results[k] = results[k] + " " + self.constant_caption
        return results
