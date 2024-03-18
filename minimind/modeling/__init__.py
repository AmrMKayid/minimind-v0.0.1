__all__ = ["MLP"]


from typing import Any, Dict

from minimind.config import Config

_MODELS: Dict[str, Any] = {}


def register_model(cls):
    _MODELS[cls.__name__.lower()] = cls
    return cls


def get_model(config: Config):
    model_name = config.architecture.architecture_name.lower()
    assert (
        model_name in _MODELS
    ), f"Model: {model_name} is not currently supported\nSupported Architectures are: {list(_MODELS.keys())}"
    return _MODELS[model_name](config)


from minimind.modeling.modules.mlp import MLP
