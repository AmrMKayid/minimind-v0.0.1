from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

from minimind.config import Config
from minimind.modeling import register_model


@register_model
class MLP(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.architecture.ffn_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.architecture.initializer_range),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.architecture.embedding_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.architecture.initializer_range),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.architecture.ffn_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.architecture.initializer_range),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.architecture.residual_dropout)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x
