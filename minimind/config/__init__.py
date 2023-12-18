from datetime import datetime

import yaml
from pydantic import BaseModel, Field
from rich import print


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    class Config:
        extra: str = "forbid"
        validate_default: bool = True
        validate_assignment: bool = True

    @classmethod
    def read_config_from_yaml(cls, file_path: str) -> "BaseConfig":
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        return cls(**yaml_data)


class EnvironmentConfig(BaseConfig):
    """Environment configuration class."""

    environment_name: str = "ALE/Breakout-v5"
    render_mode: str = "human"


class CortexConfig(BaseConfig):
    """Cortex configuration class."""

    seed: int = 37
    timestamp: datetime = datetime.utcnow()
    total_steps: int = 1000
    log_interval: int = 100


class Config(BaseConfig):
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


if __name__ == "__main__":
    config = Config()
    print(config)