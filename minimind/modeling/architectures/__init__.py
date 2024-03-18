from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
from jax.sharding import PartitionSpec as PS

from minimind.config import Config

_ARCHITECTURES: Dict[str, Any] = {}  # registry


def register_architecture(cls):
    _ARCHITECTURES[cls.__name__.lower()] = cls
    return cls


class Architecture(ABC, nn.Module):
    """Base class for all architectures."""

    config: Config

    @abstractmethod
    def __call__(self, batch: Dict[str, jax.Array], training: bool) -> Dict[str, jax.Array]:
        pass

    @abstractmethod
    def shard(self, ps: PS) -> Tuple[Architecture, PS]:
        pass


from minimind.modeling.architectures.baymax_transformer import *  # isort:skip
from minimind.modeling.architectures.transformer import *  # isort:skip
from minimind.modeling.architectures.clip import *  # isort:skip
from minimind.modeling.architectures.mamba import *  # isort:skip
from minimind.modeling.architectures.vit import *  # isort:skip
from minimind.modeling.architectures.whisper import *  # isort:skip


def get_architecture(config: Config) -> Architecture:
    assert config.arch.architecture_name, "Arch config must specify 'architecture'."
    return _ARCHITECTURES[config.arch.architecture_name.lower()](config)
