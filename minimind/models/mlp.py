from typing import Sequence

import jax
from flax import linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for i, feature in enumerate(self.features):
            x = nn.Dense(feature, name=f"mlp_layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
