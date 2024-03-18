"""Transformer model building blocks."""

from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from minimind.config import Config
from minimind.modeling.architectures import register_architecture
from minimind.modeling.modules.emb import Embedding
from minimind.modeling.modules.transformer_block import TransformerBlock


@register_architecture
class Transformer(nn.Module):
    """Transformer model."""

    config: Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.embedding = Embedding(self.config, name="embedding")
        self.transformer_blocks: list[TransformerBlock] = [
            TransformerBlock(self.config, name=f"transformer_block_{idx}") for idx in range(self.config.arch.n_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(
            use_bias=self.config.arch.use_bias,
            param_dtype=self.param_dtype,
            name="final_layer_norm",
        )

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool,
    ) -> Dict[str, jax.Array]:
        batch = self.embedding(batch, training)

        for idx in range(self.config.arch.n_layers):
            batch = self.transformer_blocks[idx](batch, training)

        x = batch.pop("x")
        x = self.final_layer_norm(x)
        batch.update({"x": x})

        batch = self.embedding(batch, training, attend=True)
        return batch
