import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.maps import thread_resources
from jax.experimental.pallas.ops.tpu import flash_attention as flash_attention_pallas
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from minimind.config import _mesh_cfg
from minimind.modeling.modules.attentions import register_attention_fn


@register_attention_fn
def flash_attention(
    query: jax.Array,
    value: jax.Array,
    key: jax.Array,
    mask: jax.Array = None,
    segment_ids: jax.Array = None,
) -> jax.Array:
    """Flash attention mechanism."""
    scale = float(1 / math.sqrt(query.shape[-1]))

    query = jax.lax.with_sharding_constraint(
        jnp.transpose(query, (0, 2, 1, 3)), P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None)
    )
    key = jax.lax.with_sharding_constraint(
        jnp.transpose(key, (0, 2, 1, 3)), P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None)
    )
    value = jax.lax.with_sharding_constraint(
        jnp.transpose(value, (0, 2, 1, 3)), P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None)
    )
    segment_ids = flash_attention_pallas.SegmentIds(segment_ids, segment_ids)

    @partial(
        shard_map,
        mesh=thread_resources.env.physical_mesh,
        in_specs=(
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis),
        ),
        out_specs=P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
        check_rep=False,
    )
    def wrap_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array, seg_ids: jax.Array):
        bs, seq_len = q.shape[0], q.shape[-2]
        return flash_attention_pallas.flash_attention(
            q,
            k,
            v,
            causal=True,
            segment_ids=seg_ids,
            sm_scale=scale,
            block_sizes=flash_attention_pallas.BlockSizes(
                block_q=min(512, seq_len),
                block_k_major=min(512, seq_len),
                block_k=min(512, seq_len),
                block_b=min(2, bs),
                block_q_major_dkv=512,
                block_k_major_dkv=512,
                block_q_dkv=512,
                block_k_dkv=512,
                block_q_dq=1024,
                block_k_dq=256,
                block_k_major_dq=512,
            ),
        )

    attention_vec = wrap_flash_attention(query, key, value, segment_ids)
    attention_vec = jnp.transpose(attention_vec, (0, 2, 1, 3))
    return attention_vec
