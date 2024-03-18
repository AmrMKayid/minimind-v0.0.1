from typing import Any

import jax
import jmp
import optax
from flax.training.train_state import TrainState as FlaxTrainState


@jax.jit
def count_parameters(params: Any) -> int:
    """Count the total number of parameters in a model's parameter dictionary
    using JAX."""
    return sum(x.size for x in jax.tree_leaves(params))


class TrainState(FlaxTrainState):
    num_params: int
    loss_scale: jmp.LossScale
    skip_infinite: bool = False

    def apply_gradients(
        self,
        *,
        grads,
        skip_infinite: bool = False,
        **kwargs,
    ) -> Any:
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        loss_scale = self.loss_scale
        if skip_infinite:
            grads_finite = jmp.all_finite(grads)
            loss_scale = self.loss_scale.adjust(grads_finite)
            new_params, new_opt_state = jmp.select_tree(
                grads_finite, (new_params, new_opt_state), (self.params, self.opt_state)
            )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            loss_scale=loss_scale,
            **kwargs,
        )

    @property
    def lr(self):
        return self.opt_state.inner_opt_state[1].hyperparams["learning_rate"]

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        **kwargs,
    ):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            num_params=count_parameters(params),
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
