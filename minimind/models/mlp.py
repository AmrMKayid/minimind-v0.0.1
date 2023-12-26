from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

from minimind.config import Config
from minimind.models import register_model


@register_model
class MLP(nn.Module):
    config: Config
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    training: bool = True

    def setup(self):
        layers = []
        self.features = self.config.architecture.layers
        self.dropout_rate = self.config.architecture.dropout_rate

        for i, feature in enumerate(self.features):
            layers.append(
                nn.Dense(feature, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=f"mlp_dense_{i}")
            )
            if i != len(self.features) - 1:
                layers.append(self.activation_fn)
                if self.dropout_rate is not None:
                    layers.append(
                        nn.Dropout(
                            rate=self.dropout_rate,
                            deterministic=not self.training,
                            name=f"mlp_dropout_{i}",
                        )
                    )
        self.layers = nn.Sequential(layers)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.layers(inputs)
