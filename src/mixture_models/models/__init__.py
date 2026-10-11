"""Model modules own input restrictions, architecture, and training choices."""
from importlib import import_module

DEFAULT_MODEL = "fixed_length_transformer_mlp"


def get_model_module(name=DEFAULT_MODEL):
    path = name if "." in name else f"mixture_models.models.{name}"
    module = import_module(path)
    for hook in ("training_protocol", "validate_sequences", "collate_records", "build_model", "model_metadata", "build_optimizer", "build_scheduler"):
        if not callable(getattr(module, hook, None)):
            raise ValueError(f"Model module {path} must expose {hook}().")
    return module
