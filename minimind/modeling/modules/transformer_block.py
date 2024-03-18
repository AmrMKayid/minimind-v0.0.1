from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from minimind.config import Config
from minimind.modeling.modules.attention import create_mask_fn, self_attention


class TransformerBlock(nn.Module):
    """Transformer block."""

    config: Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.head_dim = self.config.arch.embedding_dim // self.config.arch.n_heads

        self.qkv_proj = nn.DenseGeneral(
            features=(
                self.config.arch.n_heads,
                (3 * self.head_dim + (self.config.arch.ffn_dim // self.config.arch.n_heads)),
            ),
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
        )

        # apply layernorm
        self.layernorm = nn.LayerNorm(use_bias=False, name="layernorm")

        self.output_proj = nn.DenseGeneral(
            features=(self.config.arch.embedding_dim),
            axis=(-2, -1),
            use_bias=False,
            name="output_proj",
            dtype=self.dtype,
        )

        self.dropout = nn.Dropout(rate=self.config.arch.residual_dropout_rate)

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool,
    ) -> Dict[str, jax.Array]:
        inputs = batch.pop("x")  # [batch, seq, emb_dim]
        inputs = jax.lax.with_sharding_constraint(
            inputs, P(self.config.mesh.data_mesh, None, self.config.mesh.tensor_axis)
        )
        attn_mask = create_mask_fn(batch)

        ln_inputs = self.layernorm(inputs)  # [batch, seq, emb_dim]

        input_projected = self.qkv_proj(ln_inputs)  # [batch, seq, n_heads, (3 * head_dim + ffn_out_dim)]
        input_projected = jax.lax.with_sharding_constraint(
            input_projected, P(self.config.mesh.data_mesh, None, self.config.mesh.tensor_axis, None)
        )
        q, k, v, ffn_out = jnp.split(
            input_projected,
            [self.head_dim, self.head_dim * 2, self.head_dim * 3],
            axis=-1,
        )

        # Self Attention
        attn_out = jax.named_call(self_attention, name="self_attention")(
            query=q,
            value=v,
            key=k,
            mask=attn_mask,
        )  # [batch, seq, n_heads, head_dim]

        # Fused attn out and ffn red projection
        ffn_out = jax.nn.gelu(ffn_out)  # [batch, seq, n_heads, ffn_dim]
        fused_input = jnp.concatenate((attn_out, ffn_out), axis=-1)
        fused_input = jax.lax.with_sharding_constraint(
            fused_input, P(self.config.mesh.data_mesh, None, self.config.mesh.tensor_axis, None)
        )

        output = self.output_proj(fused_input)  # [batch, seq, emb_dim]

        inputs += self.dropout(output, deterministic=not training)
        inputs = jax.lax.with_sharding_constraint(
            inputs, P(self.config.mesh.data_mesh, None, self.config.mesh.tensor_axis)
        )
        batch.update({"x": inputs})
        return batch
