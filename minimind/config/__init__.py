import getpass
import os
from datetime import datetime
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    @classmethod
    def read_config_from_yaml(cls, file_path: str) -> "BaseConfig":
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        return cls(**yaml_data)


class BatchProcessorConfig(BaseConfig):
    """Configuration for the TextProcessor."""

    fields: str = "text"
    fields_from_example: str = ""
    subfield_separator: str = " "
    add_bos_token: bool = False
    add_eos_token: bool = False
    prepend_text: str = ""


class DataConfig(BaseConfig):
    """Architecture configuration class."""

    path: str = "allenai/c4"
    name: str = "en"
    split: str = "train"
    streaming: bool = True
    sequence_length: int = 1024
    batch_size: int = 8
    always_start_with_bos: bool = False
    tokenizer: str = "google/byt5-base"
    processor: BatchProcessorConfig = Field(default_factory=BatchProcessorConfig)


class ArchitectureConfig(BaseConfig):
    """Architecture configuration class."""

    architecture_name: str = "transformer"
    embedding_dim: int = 8
    vocab_size: int = 32768
    n_heads: int = 2
    n_layers: int = 2
    n_kv_heads: int = 2
    ffn_dim: int = 32
    max_sequence_length: int = 32
    max_pos_emb_length: int = 512
    positional_embedding_type: str = "learned"
    attention_fn: str = "self_attention"
    norm_type: str = "layer_norm"
    use_qk_norm: bool = False
    residual_dropout_rate: float = 0.0
    initializer_range: float = 0.02
    use_bias: bool = False

    # ViT
    patch_size: tuple = (16, 16)
    n_outputs: int = 5


class MeshConfig(BaseConfig):
    """Mesh class."""

    data_axis: str = "dp"
    fsdp_axis: str = "fsdp"
    sequence_axis: str = "sp"
    tensor_axis: str = "tp"
    n_data_parallel: int = 4
    n_fsdp_parallel: int = 2
    n_sequence_parallel: int = 2
    n_tensors_parallel: int = 1
    data_mesh: tuple[str, str] = (data_axis, fsdp_axis)
    sequence_mesh: tuple[str, str] = (fsdp_axis, sequence_axis)
    mesh_axis_names: tuple[str, str, str, str] = (
        data_axis,
        fsdp_axis,
        tensor_axis,
        sequence_axis,
    )


class MinimindConfig(BaseConfig):
    """Minimind configuration class."""

    PRNGKey: int = 0
    seed: int = 37
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps: int = 1000
    log_interval: int = 10


class OptimConfig(BaseConfig):
    """Optimization configuration class."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    total_steps: int = 1000
    lr_decay: bool = True
    warmup_steps: int = 128
    lr_min: float = 0.0
    clip_grad_norm: float = 1.0
    grad_accum_steps: int = 1


class Config(BaseConfig):
    minimind: MinimindConfig = Field(default_factory=MinimindConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    arch: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)


_mesh_cfg = MeshConfig()

if __name__ == "__main__":
    config = Config()
    print(config)
