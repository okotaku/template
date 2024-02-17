from typing import Any

import pytest
from peft import LoHaConfig, LoKrConfig, LoraConfig, OFTConfig

from diffengine.models.archs import create_peft_config


def test_create_peft_config():
    config: dict[str, Any] = dict(
        type="Dummy",
    )
    with pytest.raises(AssertionError, match="Unknown PEFT type"):
        create_peft_config(config)

    config = dict(
        type="LoRA",
        r=4,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoraConfig)
    assert config.r == 4

    config = dict(
        type="LoHa",
        r=8,
        alpha=2,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoHaConfig)
    assert config.r == 8
    assert config.alpha == 2

    config = dict(
        type="LoKr",
        r=8,
        alpha=2,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoKrConfig)
    assert config.r == 8
    assert config.alpha == 2

    config = dict(
        type="OFT",
        r=8,
    )
    config = create_peft_config(config)
    assert isinstance(config, OFTConfig)
    assert config.r == 8
