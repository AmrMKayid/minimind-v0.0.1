from typing import Sequence

import jax
from flax import linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        layers = []
        for i, feature in enumerate(self.features):
            layers.append(nn.Dense(feature, name=f"mlp_layers_{i}"))
            if i != len(self.features) - 1:
                layers.append(nn.relu)
        self.layers = nn.Sequential(layers)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.layers(inputs)
