from typing import Any

import jax
import optax

from minimind.config import Config


def make_optimizer(config: Config, params: Any) -> optax.MultiSteps:
    def _to_decay(p) -> bool:
        # is_embeddings = (
        #     p.shape == params["embedding"]["wte"]["embedding"].value.shape
        #     or p.shape == params["embedding"]["wpe"]["embedding"].value.shape
        # )
        is_embeddings = False
        is_flat = p.ndim < 2

        return not (is_embeddings or is_flat)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.learning_rate,
        warmup_steps=config.optim.warmup_steps,
        decay_steps=config.optim.total_steps,
        end_value=config.optim.lr_min,
    )

    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=schedule if config.optim.lr_decay else config.optim.learning_rate,
        b1=config.optim.beta1,
        b2=config.optim.beta2,
        weight_decay=config.optim.weight_decay,
        mask=jax.tree_util.tree_map(_to_decay, params),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optim.clip_grad_norm),
        optimizer,
    )

    optimizer = optax.MultiSteps(optimizer, every_k_schedule=config.optim.grad_accum_steps)

    return optimizer
