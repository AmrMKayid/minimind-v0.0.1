import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from rich import inspect
from tqdm import tqdm

from minimind.utils import setup_logger

# Set up the logger
logger = setup_logger()

from minimind.config import Config
from minimind.models import MLP


class Cortex:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        logger.info(f"{self.config=}")

        model = MLP([512, 256, 128, 64, 32, 16, 8, 4, 2])
        batch = jnp.ones((32, 10))

        # Model summary
        tabulate_fn = nn.tabulate(
            model,
            jax.random.key(self.config.cortex.PRNGKey),
            compute_flops=True,
            compute_vjp_flops=True,
        )
        print(tabulate_fn(batch))

        # Model init and apply
        variables = model.init(jax.random.key(0), batch)
        output = model.apply(variables, batch)
        logger.info(f"{model=}")
        inspect(model)
        logger.info(f"{output=}")

        env = gym.make(
            id=self.config.environment.environment_name,
            render_mode=self.config.environment.render_mode,
        )
        inspect(env)
        logger.info(f"{env=}")

        observation, info = env.reset(seed=self.config.cortex.seed)
        pbar = tqdm(range(self.config.cortex.total_steps), desc="Running", leave=True)
        for step in pbar:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if step % self.config.cortex.log_interval == 0:
                pbar.set_postfix({"reward": round(reward, 3)})

            if terminated or truncated:
                observation, info = env.reset()
        env.close()
