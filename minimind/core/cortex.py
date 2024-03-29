import logging
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from tqdm import tqdm

from minimind.config import Config, _mesh_cfg
from minimind.metrics.cross_entropy import CrossEntropyLoss
from minimind.modeling.architectures import get_architecture
from minimind.modeling.architectures.state import TrainState
from minimind.modeling.optimizers import make_optimizer
from minimind.utils.constants import JAX_DEFAULT_BACKEND


class Cortex:
    """The Cortex class represents the core component of the neural network
    model. It is responsible for initializing the model, training the model,
    and storing the model state.

    Args:
        config (Config): The configuration object containing the model settings.

    Attributes:
        config (Config): The configuration object containing the model settings.
        devices (list): The list of devices used for computation.
        mesh (Mesh): The mesh object representing the distributed computation mesh.
        architecture (Architecture): The architecture object representing the neural network architecture.
        state (TrainState): The train state object representing the current state of the model.

    Methods:
        __init__(self, config: Config) -> None: Initializes the Cortex object.
        initialize_train_state(self) -> None: Initializes the train state of the model.
        train(self, dataset) -> None: Trains the model using the given dataset.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.devices = mesh_utils.create_device_mesh(
            devices=jax.devices(),
            mesh_shape=(
                self.config.mesh.n_data_parallel,
                self.config.mesh.n_fsdp_parallel,
                self.config.mesh.n_sequence_parallel,
                self.config.mesh.n_tensors_parallel,
            ),
            contiguous_submeshes=True,
        )
        logging.info(f"{self.devices=}")

        self.mesh = Mesh(devices=self.devices, axis_names=self.config.mesh.mesh_axis_names)
        logging.info(f"{self.mesh=}")

        self.architecture = get_architecture(self.config)
        self.initialize_train_state()

    def initialize_train_state(self) -> None:
        """Initializes the train state of the model.

        This method initializes the train state of the model by setting up the architecture,
        creating the optimizer, and initializing the model parameters.

        Returns:
            None
        """
        with self.mesh:
            logging.info(f"Initializing architecture with {JAX_DEFAULT_BACKEND=}")
            rng = jax.random.PRNGKey(0)

            # TODO: take batch from dataset directly
            dummy_data = jnp.ones((self.config.data.batch_size, self.config.arch.max_sequence_length), dtype=jnp.int32)
            jnp.ones((self.config.data.batch_size, 224, 224, 3))
            # Text
            batch = {"inputs": dummy_data, "mask": dummy_data, "targets": dummy_data}
            # vit images
            # batch = {
            #     "inputs": dummy_data,
            #     "targets": jax.random.randint(
            #         rng, shape=(self.config.data.batch_size,), minval=0, maxval=self.config.arch.n_outputs - 1
            #     ),
            # }
            # whisper audio
            # dummy_data = jnp.ones(
            #     (self.config.data.batch_size, self.config.arch.max_sequence_length, 8), dtype=jnp.int32
            # )
            # batch = {
            #     "inputs": dummy_data,
            #     # "mask": dummy_data,
            #     "targets": dummy_data,
            # }
            # CLIP
            # batch = {"inputs": dummy_data, "images": dummy_image_data}
            batch = jax.lax.with_sharding_constraint(batch, PS(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis))
            state = self.architecture.init(rng, batch=batch, training=False)
            optimizer = make_optimizer(config=self.config, params=state["params"])
            self.state = TrainState.create(
                apply_fn=self.architecture.apply,
                params=state["params"],
                tx=optimizer,
                loss_scale=jmp.NoOpLossScale(),
            )

            # Compute FLOPs and Summary
            tabulate_fn = nn.tabulate(
                self.architecture,
                rng,
                compute_flops=True,
                compute_vjp_flops=True,
            )

            print(tabulate_fn(batch, False))

    def train(self, dataset) -> None:
        """Trains the model using the given dataset.

        This method trains the model using the given dataset by iterating over the dataset
        and performing training steps for each batch.

        Args:
            dataset: The dataset used for training.

        Returns:
            None
        """

        @jax.jit
        def train_step(
            state: TrainState,
            batch: Dict[str, jax.Array],
        ) -> tuple[TrainState, jax.Array]:
            "Train step for transformer model."

            def loss_fn(params) -> jax.Array:
                """Apply model and compute loss."""
                # TODO: Take the loss function from each ach
                updated_batch = state.apply_fn(
                    variables={"params": params},
                    batch=batch,
                    training=True,
                )
                loss = CrossEntropyLoss()(updated_batch["logits"], batch["targets"])
                loss = jnp.mean(loss)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        with self.mesh:
            ds_iter = iter(dataset)
            pbar = tqdm(range(self.config.minimind.total_steps))
            losses = []
            for step in pbar:
                batch = next(ds_iter)
                batch = jax.lax.with_sharding_constraint(batch, PS(_mesh_cfg.data_mesh, _mesh_cfg.sequence_axis))
                if step == 0:
                    logging.info(f"{batch=}")
                self.state, loss = train_step(self.state, batch=batch)
                losses.append(loss)

                if len(losses) % self.config.minimind.log_interval == 0:
                    avg_loss = np.mean(losses)
                    pbar.set_description(f"{step=} | {avg_loss=}")
