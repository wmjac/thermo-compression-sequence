"""Epoch-boundary checkpoints containing all states needed for continuation."""
from pathlib import Path
import os
import random
import numpy as np
import torch
from .models import get_model_module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, optimizer, scheduler, state, config):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = {"format_version": 1, "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": rng_state(), "training_state": state, "config": config}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_model(path, device):
    payload = load_checkpoint(path)
    config = payload["config"]
    module = get_model_module(config["model_module"])
    if config["model_metadata"] != module.model_metadata():
        raise ValueError(f"Model specification disagrees with checkpoint {path}.")
    model = module.build_model(config["dimension"], config["protocol"]["dtype"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, module, config
