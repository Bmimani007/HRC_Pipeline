"""
Model registry — auto-discovers all models that register themselves.

In a model file:
    from models.registry import register_model
    from models.base import Model

    @register_model("my_model")
    class MyModel(Model):
        ...

In the pipeline:
    from models.registry import build_models
    instances = build_models(config)   # {model_name: instance}
"""
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type
from .base import Model

_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    """Class decorator: registers a Model subclass under `name`."""
    def decorator(cls):
        if not issubclass(cls, Model):
            raise TypeError(f"{cls.__name__} must subclass Model")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def _autodiscover():
    """Import every .py file in this package so decorators run."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in {"base", "registry", "__init__"}:
            continue
        try:
            importlib.import_module(f"models.{module_info.name}")
        except Exception as e:
            print(f"⚠ Could not load model module '{module_info.name}': {e}")


def list_available() -> list:
    """All registered model names."""
    if not _REGISTRY:
        _autodiscover()
    return sorted(_REGISTRY.keys())


def build_models(config: dict, region: str = None) -> Dict[str, Model]:
    """
    Read config.models, instantiate every enabled model.
    If `region` is given, only build models that target that region.
    """
    if not _REGISTRY:
        _autodiscover()

    instances = {}
    for model_name, model_cfg in (config.get("models") or {}).items():
        if not model_cfg.get("enabled", True):
            continue
        if model_name not in _REGISTRY:
            print(f"⚠ Model '{model_name}' is in config but not registered. "
                  f"Available: {list_available()}")
            continue
        # Region filter
        if region is not None:
            allowed = model_cfg.get("regions", [region])
            if region not in allowed:
                continue
        cls = _REGISTRY[model_name]
        instances[model_name] = cls(model_cfg)
    return instances
