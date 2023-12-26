from dataclasses import asdict
from datetime import datetime
from typing import List

import yaml
from pydantic import BaseModel, Field
from rich import print


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    class Config:
        extra: str = "allow"  # TODO: maybe change this later
        validate_default: bool = True
        validate_assignment: bool = True

    @classmethod
    def read_config_from_yaml(cls, file_path: str) -> "BaseConfig":
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        return cls(**yaml_data)


class EnvironmentConfig(BaseConfig):
    """Environment configuration class."""

    id: str = "ALE/Breakout-v5"
    render_mode: str = "human"


class ArchitectureConfig(BaseConfig):
    """Architecture configuration class."""

    architecture_name: str = "mlp"
    layers: List[int] = [32, 16, 8, 4, 2]


class CortexConfig(BaseConfig):
    """Cortex configuration class."""

    PRNGKey: int = 0
    seed: int = 37
    timestamp: datetime = datetime.utcnow()
    total_steps: int = 1000
    log_interval: int = 100


class Config(BaseConfig):
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)


if __name__ == "__main__":
    config = Config()
    print(config)
