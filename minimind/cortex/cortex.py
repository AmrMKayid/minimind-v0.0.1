import gymnasium as gym
import jax
import jax.numpy as jnp
from rich import inspect, print
from tqdm import tqdm

from minimind.config import Config
from minimind.models import MLP


class Cortex:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        model = MLP([128, 64, 4])
        batch = jnp.ones((32, 10))
        variables = model.init(jax.random.key(0), batch)
        output = model.apply(variables, batch)
        print(f"{model=}")
        inspect(model)
        print(f"{output=}")

        env = gym.make(
            id=self.config.environment.environment_name,
            render_mode=self.config.environment.render_mode,
        )
        inspect(env)
        print(f"{env=}")

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
