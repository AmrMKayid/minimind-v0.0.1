from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from minimind.config import Config
from minimind.modeling.modules.attentions import get_attention_fn
from minimind.modeling.modules.attentions.attention import create_mask_fn

class TransformerBlock(nn.Module):
    """Transformer block."""

    config: Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.head_dim = self.config.arch.embedding_dim // self.config.arch.n_heads

        self.layernorm_1 = nn.LayerNorm(name="layernorm_1")

        self.qkv_proj = nn.DenseGeneral(
            features=3 * self.config.arch.embedding_dim,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                ((self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis), self.config.mesh.tensor_axis),
            ),
        )

        self.attention_output = nn.Dense(
            features=self.config.arch.embedding_dim,
            name="attention_output",
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                (self.config.mesh.tensor_axis, (self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis)),
            ),
        )

        self.layernorm_2 = nn.LayerNorm(name="layernorm_2")

        self.ffn_expansion = nn.Dense(
            features=4 * self.config.arch.embedding_dim,
            name="ffn_expansion",
            # kernel_init=nn.with_partitioning(kernel_init, (FSDP_AXIS, TENSOR_AXIS)),
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                ((self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis), self.config.mesh.tensor_axis),
            ),
        )

        self.ffn_reduction = nn.Dense(
            features=self.config.arch.embedding_dim,
            name="ffn_reduction",
            # kernel_init=nn.with_partitioning(kernel_init, (TENSOR_AXIS, FSDP_AXIS)),
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.arch.initializer_range),
                (self.config.mesh.tensor_axis, (self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis)),
            ),
        )

        self.dropout = nn.Dropout(rate=self.config.arch.residual_dropout_rate)

    @staticmethod
    def head_split(x: jax.Array, head_dim: int) -> jax.Array:
        return x.reshape(*x.shape[:2], -1, head_dim)  # [batch, seq, emb_dim/head_dim, head_dim]

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jax.Array],
        training: bool,
    ) -> Dict[str, jax.Array]:
        inputs = batch.pop("x")  # [batch, seq, emb_dim]
        inputs = jax.lax.with_sharding_constraint(
            inputs, P(self.config.mesh.data_mesh, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
        )
        attn_mask = create_mask_fn(batch)

        ln_inputs = self.layernorm_1(inputs)  # [batch, seq, emb_dim]

        input_projected = self.qkv_proj(ln_inputs)  # [batch, seq, 3 * emb_dim]
        input_projected = jax.lax.with_sharding_constraint(
            input_projected,
            P(self.config.mesh.data_mesh, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis),
        )
        q, k, v = jnp.split(
            input_projected,
            [self.config.arch.embedding_dim, 2 * self.config.arch.embedding_dim],
            axis=-1,
        )
        q, k, v = (
            self.head_split(q, self.head_dim),
            self.head_split(k, self.head_dim),
            self.head_split(v, self.head_dim),
        )  # [batch, seq, n_heads, head_dim]

        # Self Attention
        # attention_fn = get_attention_fn("flash_attention") # todo: move to configs
        attention_fn = get_attention_fn(self.config.arch.attention_fn)
        attn_out = jax.named_call(attention_fn, name=attention_fn.__name__)(
            query=q,
            value=v,
            key=k,
            mask=attn_mask,
        )  # [batch, seq, n_heads, head_dim]

        attn_out = attn_out.reshape(attn_out.shape[:-2] + (-1,))  # [batch, seq, emb_dim]
        attn_out = jax.lax.with_sharding_constraint(
            attn_out, P(self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
        )
        attn_out = self.attention_output(attn_out)  # [batch, seq, emb_dim]
        inputs += attn_out
        inputs = jax.lax.with_sharding_constraint(
            attn_out, P(self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
        )

        ln2_inputs = self.layernorm_2(inputs)

        ffn_expanded = self.ffn_expansion(ln2_inputs)
        ffn_expanded = jax.lax.with_sharding_constraint(
            ffn_expanded, P(self.config.mesh.fsdp_axis, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
        )
        ffn_expanded = jax.nn.gelu(ffn_expanded)

        ffn_out = self.ffn_reduction(ffn_expanded)  # [batch, seq, emb_dim]

        inputs += self.dropout(ffn_out, deterministic=not training)
        inputs = jax.lax.with_sharding_constraint(
            inputs, P(self.config.mesh.data_mesh, self.config.mesh.sequence_axis, self.config.mesh.tensor_axis)
        )
        batch.update({"x": inputs})
        return batch


class MTJTransformerBlock(nn.Module):
    """MTJ Transformer block."""

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
        attention_fn = get_attention_fn(self.config.arch.attention_fn)
        attn_out = jax.named_call(attention_fn, name=attention_fn.__name__)(
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


class BlockwiseParallelTransformer(nn.Module):
    pass
