import json
import random

import numpy as np
import webdataset as wds
from nvidia.dali import fn, pipeline, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from .utils import Dummy


class WebdatasetFilter:
    """Filter webdataset."""

    def __init__(self, min_size: int = 1024, max_pwatermark: float = 0.5,
                 ) -> None:
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def __call__(self, x: dict) -> bool:
        """Call function for webdataset filter."""
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (
                    x_json.get("original_width", 0.0) or 0.0
                    ) >= self.min_size and x_json.get(
                    "original_height", 0,
                ) >= self.min_size
                filter_watermark = (
                    x_json.get("pwatermark", 1.0) or 1.0
                    ) <= self.max_pwatermark
                return filter_size and filter_watermark
            else:  # noqa
                return False
        except Exception:  # noqa: BLE001
            return False


def buffered_shuffle(generator_factory, initial_fill, seed):  # noqa
    """Buffered shuffle generator."""
    def buffered_shuffle_generator():  # noqa
        nonlocal generator_factory, initial_fill, seed
        generator = generator_factory()
        # The buffer size must be positive
        assert(initial_fill > 0)

        # The buffer that will hold the randomized samples
        buffer: list = []

        # The random context for preventing side effects
        random_context = random.Random(seed)

        try:
            while len(buffer) < initial_fill: # Fills in the random buffer
                buffer.append(next(generator))

            # Selects a random sample from the buffer and then fills it back
            # in with a new one
            while True:
                idx = random_context.randint(0, initial_fill-1)

                yield buffer[idx]
                buffer[idx] = None
                buffer[idx] = next(generator)

        except StopIteration:
            # When the generator runs out of the samples flushes our the buffer
            random_context.shuffle(buffer)

            while buffer:
                # Prevents the one sample that was not filled from being
                # duplicated
                if buffer[-1] is not None:
                    yield buffer[-1]
                buffer.pop()
    return buffered_shuffle_generator


def last_batch_padding(generator_factory, batch_size):  # noqa
    """Last batch padding generator."""
    def last_batch_padding_generator():  # noqa
        nonlocal generator_factory, batch_size
        generator = generator_factory()
        in_batch_idx = 0
        last_item = None
        try:
            # Keeps track of the last sample and the sample number mod
            # batch_size
            while True:
                if in_batch_idx >= batch_size:
                    in_batch_idx -= batch_size
                last_item = next(generator)
                in_batch_idx += 1
                yield last_item
        except StopIteration:
            # Repeats the last sample the necessary number of times
            while in_batch_idx < batch_size:
                yield last_item
                in_batch_idx += 1
    return last_batch_padding_generator


def collect_batches(generator_factory, batch_size):  # noqa
    """Collect batches generator."""
    def collect_batches_generator():  # noqa
        nonlocal generator_factory, batch_size
        generator = generator_factory()
        batch = []
        try:
            while True:
                batch.append(next(generator))
                if len(batch) == batch_size:
                    # Converts tuples of samples into tuples of batches of
                    # samples
                    yield tuple(map(list, zip(*batch, strict=False)))
                    batch = []
        except StopIteration:
            if batch is not []:
                # Converts tuples of samples into tuples of batches of samples
                yield tuple(map(list, zip(*batch, strict=False)))
    return collect_batches_generator


def read_webdataset(  # noqa: ANN201
    paths: list[str],
    extensions: tuple | str | None = None,
    initial_fill: int = 256,
    seed: int = 0,
    cycle: str = "quiet",
    min_size: int = 512,
    max_pwatermark: float = 0.5,
    *,
    random_shuffle: bool = False,
    pad_last_batch: bool = False,
    read_ahead: bool = False,
):
    """Read WebDataset."""
    # Parsing the input data
    assert(cycle in {"quiet", "raise", "no"})
    if extensions is None:
        # All supported image formats
        extensions = "jpg;peg;img;image;pbm;pgm;png"
    if isinstance(extensions, str):
        extensions = (extensions,)

    # For later information for batch collection and padding
    max_batch_size = pipeline.Pipeline.current().max_batch_size

    def webdataset_generator():  # noqa
        """WebDataset generator."""
        bytes_np_mapper = (lambda data: np.frombuffer(data, dtype=np.uint8),
                           lambda data: np.fromstring(data, dtype=np.uint8))
        dataset_instance = (wds.WebDataset(paths)
                            .select(WebdatasetFilter(
                                min_size=min_size,
                                max_pwatermark=max_pwatermark))
                            .rename(
                                img="jpg;png;jpeg;webp",
                                text="text;txt;caption",
                                handler=wds.warn_and_continue,
                            )
                            .to_tuple(*extensions)
                            .map_tuple(*bytes_np_mapper))

        yield from dataset_instance

    dataset = webdataset_generator

    # Adding the buffered shuffling
    if random_shuffle:
        dataset = buffered_shuffle(dataset, initial_fill, seed)

    # Adding the batch padding
    if pad_last_batch:
        dataset = last_batch_padding(dataset, max_batch_size)

    # Collecting the data into batches (possibly undefull)
    # Handled by a custom function only when `silent_cycle` is False
    if cycle != "quiet":
        dataset = collect_batches(dataset, max_batch_size)

    # Prefetching the data
    if read_ahead:
        dataset=list(dataset())  # type: ignore[assignment]

    # If `cycle` is "quiet" then batching is handled by the external source
    return fn.external_source(
        source=dataset,
        num_outputs=len(extensions),
        batch=(cycle != "quiet"),
        cycle=cycle,
        dtype=types.UINT8,
    )


@pipeline_def(enable_conditionals=True)
def sd_pipeline(
    img_size: int = 512,
    initial_fill: int = 256,
    seed: int = 0,
    cycle: str = "quiet",
    *,
    random_shuffle: bool = True,
    pad_last_batch: bool = False,
    read_ahead: bool = False) -> tuple:
    """Pipeline for WebDataset."""
    paths = [
        f"data/improved_aesthetics_5plus/images/{str(i).zfill(5)}.tar"
        for i in range(3401)]
    img, text = read_webdataset(paths=paths,
                                 extensions=("img", "text"),
                                 random_shuffle=random_shuffle,
                                 initial_fill=initial_fill,
                                 seed=seed,
                                 pad_last_batch=pad_last_batch,
                                 read_ahead=read_ahead,
                                 cycle=cycle)
    img = fn.decoders.image(img, output_type=types.RGB)
    img = img.gpu()

    rng = fn.random.coin_flip(probability=0.5)

    resized = fn.resize(img, device="gpu", resize_shorter=img_size,
                        interp_type=types.INTERP_LINEAR)
    resized = fn.flip(resized, horizontal=rng)
    output = fn.crop_mirror_normalize(
        resized,
        dtype=types.FLOAT,
        crop=(img_size, img_size),
        device="gpu",
        mean=[0.5 * 255] * 3,
        std=[0.5 * 255] * 3)
    return output, fn.pad(text, fill_value=255)


class DALILAIONIterator(DALIGenericIterator):
    """DALI LAION Web Dataset.

    Args:
    ----
        batch_size (int): Batch size.
        num_workers (int): Number of workers. Defaults to 0.
        output_map (list[str], optional): Output map. Defaults to ["data", "text"].
        img_size (int): Image size. Defaults to 512.
        prob_text_drop (float): Probability of text drop. Defaults to 0.1.

    """

    def __init__(self,
                 batch_size: int,
                 num_workers: int = 0,
                 output_map: list[str] | None = None,
                 img_size: int = 512,
                 prob_text_drop: float = 0.1) -> None:
        if output_map is None:
            output_map = ["data", "text"]

        pipeline = sd_pipeline(
            batch_size=batch_size, num_threads=num_workers, device_id=0,
            img_size=img_size)
        self.dataset = Dummy()
        super().__init__(
            pipeline,
            output_map,
            dynamic_shape=False,
            auto_reset=True,
            prepare_first_batch=False,
            last_batch_policy=LastBatchPolicy.DROP)

        self.prob_text_drop = prob_text_drop

    def __next__(self) -> dict:
        """Next function."""
        data = super().__next__()
        pad_t_value = 255
        return dict(
            inputs=dict(
                img=data[0]["data"],
                text=[
                    t.cpu().numpy()[
                        t!=pad_t_value].tostring().decode("utf-8") if (
                        random.random() >= self.prob_text_drop
                        ) else "" for t in data[0]["text"]]))

    def __len__(self) -> int:
        """Length function."""
        return 10000000
