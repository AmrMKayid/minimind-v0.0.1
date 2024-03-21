from functools import partial

import jax
from jax.experimental.maps import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from minimind.config import _mesh_cfg
from minimind.modeling.modules.attentions import register_attention_fn
from minimind.modeling.modules.attentions.ring_attention_kernel import ring_attention_standard as rng_attn_kernel


@register_attention_fn
def ring_attention(
    query: jax.Array,
    value: jax.Array,
    key: jax.Array,
    mask: jax.Array = None,
) -> jax.Array:
    """Ring attention mechanism."""

    @partial(
        shard_map,
        mesh=thread_resources.env.physical_mesh,
        in_specs=(
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
            P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
        ),
        out_specs=P(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis, _mesh_cfg.tensor_axis, None),
        check_rep=True,
    )
    def wrap_ring_attention(q: jax.Array, k: jax.Array, v: jax.Array):
        return rng_attn_kernel(
            q=q,
            k=k,
            v=v,
            attn_mask=mask,
            axis_name=_mesh_cfg.sequence_axis,
        )

    attention_vec = wrap_ring_attention(query, key, value)
    return attention_vec
