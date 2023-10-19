import gymnasium as gym
from rich import print
from tqdm import tqdm

from minimind.configs import Config


def main() -> None:
    config = Config()
    print(f"{config=}")

    env = gym.make(
        id=config.environment.environment_name,
        render_mode=config.environment.render_mode,
    )
    print(f"{env=}")

    observation, info = env.reset(seed=config.seed)
    pbar = tqdm(range(config.total_steps), desc="Running", leave=True)
    for step in pbar:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if step % config.log_interval == 0:
            pbar.set_postfix({"reward": round(reward, 3)})

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
