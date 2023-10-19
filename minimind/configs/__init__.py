from datetime import datetime

from pydantic import BaseModel, Field
from rich import print


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    class Config:
        extra: str = "forbid"
        validate_default: bool = True
        validate_assignment: bool = True


class EnvironmentConfig(BaseConfig):
    """Environment configuration class."""

    environment_name: str = "LunarLander-v2"
    render_mode: str = "human"


class Config(BaseConfig):
    seed: int = 37
    timestamp: datetime = datetime.utcnow()
    total_steps: int = 1000
    log_interval: int = 100
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


if __name__ == "__main__":
    config = Config()
    print(config)
