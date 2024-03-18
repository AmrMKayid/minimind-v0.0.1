import math
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from minimind.config import _mesh_cfg


def create_mask_fn(batch: Dict, dtype=jnp.int8):
    """Create mask for attention."""
    if "mask" not in batch:
        return None

    seq_len = batch["mask"].shape[1]

    # Causal mask
    causal_mask = jax.lax.convert_element_type(jnp.tril(jnp.ones((seq_len, seq_len))), dtype)
    # Padding mask
    padding_mask = jax.lax.convert_element_type(batch["mask"][:, :, None], dtype)  # (batch_size, seq_len, 1)
    padding_mask = padding_mask * padding_mask.transpose((0, 2, 1))  # (batch_size, seq_len, seq_len)
    padding_mask = jax.lax.with_sharding_constraint(padding_mask, P(_mesh_cfg.data_mesh, None, None))
    # Combine masks
    mask = jnp.logical_and(causal_mask, padding_mask)
    mask = jax.lax.convert_element_type(mask, dtype)
    mask = jax.lax.with_sharding_constraint(mask[:, None, :, :], P(_mesh_cfg.data_mesh, None, None, None))
    return mask


def self_attention(
    query: jax.Array,
    value: jax.Array,
    key: jax.Array,
    mask: jax.Array = None,
) -> jax.Array:
    """Self attention mechanism."""
    kv_heads = key.shape[-2]
    q_heads, head_dim = query.shape[-2], query.shape[-1]

    if q_heads != kv_heads:
        assert q_heads > kv_heads
        tile_factor = q_heads // kv_heads
        key = jnp.repeat(key, tile_factor, axis=-2)  # [batch, seq, n_heads, head_dim]
        value = jnp.repeat(value, tile_factor, axis=-2)  # [batch, seq, n_heads, head_dim]

    scale = float(1 / math.sqrt(head_dim))

    key = jax.lax.with_sharding_constraint(
        key, P(_mesh_cfg.data_mesh, None, _mesh_cfg.tensor_axis, None)
    )  # [batch, seq, n_heads, head_dim]
    value = jax.lax.with_sharding_constraint(
        value, P(_mesh_cfg.data_mesh, None, _mesh_cfg.tensor_axis, None)
    )  # [batch, seq, num_heads, head_dim]
    query = jax.lax.with_sharding_constraint(
        query, P(_mesh_cfg.data_mesh, None, _mesh_cfg.tensor_axis, None)
    )  # [batch, seq, num_heads, head_dim]

    attention_logits = jnp.einsum("bthd,bThd->bhtT", query, key)  # [batch, num_heads, seq, seq]
    attention_logits = (attention_logits * scale).astype(query.dtype)
    attention_logits = jax.lax.with_sharding_constraint(
        attention_logits, P(_mesh_cfg.data_mesh, _mesh_cfg.tensor_axis, None, None)
    )

    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, jnp.finfo(attention_logits.dtype).min)
        attention_logits = jax.lax.with_sharding_constraint(
            attention_logits, P(_mesh_cfg.data_mesh, _mesh_cfg.tensor_axis, None, None)
        )

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)  # [batch, num_heads, seq, seq]
    attention_weights = attention_weights.astype(value.dtype)
    attention_weights = jax.lax.with_sharding_constraint(
        attention_weights, P(_mesh_cfg.data_mesh, _mesh_cfg.tensor_axis, None, None)
    )

    attention_vec = jnp.einsum(
        "bhtT,bThd->bthd", attention_weights, value
    )  # [batch, seq, num_heads, emb_dim / num_heads]

    attention_vec = jax.lax.with_sharding_constraint(
        attention_vec, P(_mesh_cfg.data_mesh, None, _mesh_cfg.tensor_axis, None)
    )

    return attention_vec


# ------------------------------------------------------ #


class MultiHeadAttention(nn.Module):
    """Implements multi-head attention mechanism as described in "Attention is
    All You Need" by Vaswani et al 2017.

    This module splits the input into multiple heads, applies scaled dot-product attention independently on each head, and then concatenates the results. It allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    Methods:
        setup(): Initializes projection matrices for queries, keys, values, and the output projection.
        __call__(inputs: jnp.ndarray, mask: jnp.ndarray = None): Processes the input tensor through the multi-head self-attention mechanism.
        attention_function(query, key, value, mask=None): Computes the attention scores and applies them to the value vectors.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.wq = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros
        )
        self.wk = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros
        )
        self.wv = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros
        )
        self.output = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros
        )

    def __call__(self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None) -> tuple:
        query = self.wq(inputs)
        key = self.wk(context)
        value = self.wv(context)
        context_vectors, attention = self.attention_function(query, key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention

    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights


class PositionWiseFFN(nn.Module):
    """Implements the position-wise feed-forward network of a transformer
    model.

    This module applies two linear transformations with a gelu activation in between, as per the original transformer model design. It is applied to each position separately and identically.

    Attributes:
        num_hiddens (int): The number of hidden units in the first linear layer.
        num_outputs (int): The number of output units in the second linear layer (usually the same as the model's hidden size).

    Methods:
        setup(): Initializes the two linear layers.
        __call__(X: jnp.ndarray): Applies the position-wise feed-forward network to the input tensor.
    """

    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.num_hiddens, kernel_init=nn.initializers.xavier_uniform())
        self.activation = nn.gelu
        self.dense2 = nn.Dense(self.num_outputs, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(self.activation(self.dense1(X)))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """

    dropout: int

    @nn.compact
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, training=False) -> jnp.ndarray:
        return nn.LayerNorm()(nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
