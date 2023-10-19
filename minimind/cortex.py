import gymnasium as gym
from rich import print

from minimind.configs import Config


def main() -> None:
    config = Config()
    print(f"{config=}")

    env = gym.make(
        id=config.environment.environment_name,
        render_mode=config.environment.render_mode,
    )

    observation, info = env.reset(seed=config.seed)
    for _ in range(config.total_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
