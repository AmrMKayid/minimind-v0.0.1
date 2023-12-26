from typing import Callable, Sequence

import jax
from flax import linen as nn

from minimind.config import Config
from minimind.models import register_model


@register_model
class MLP(nn.Module):
    config: Config
    activation_fn: Callable = nn.relu

    def setup(self):
        layers = []
        self.features = self.config.architecture.layers
        for i, feature in enumerate(self.features):
            layers.append(nn.Dense(feature, name=f"mlp_dense_{i}"))
            if i != len(self.features) - 1:
                layers.append(self.activation_fn)
        self.layers = nn.Sequential(layers)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.layers(inputs)
