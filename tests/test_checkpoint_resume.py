from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mixture_models.checkpoints import load_checkpoint
from mixture_models.data import MixtureRecord
from mixture_models.models import fixed_length_transformer_mlp as module
from mixture_models.training import StopRequested, train_trial


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    records = []
    for index in range(12):
        concentration = 0.03 + index * 0.003
        sequences = ("A" * 20, "F" * 20)
        records.append(MixtureRecord(str(index), "synthetic", index, str(index), sequences,
            (3, 1), (concentration * 0.75, concentration * 0.25), concentration,
            concentration * 1.1, 300.0, {}))
    protocol = replace(module.training_protocol(), max_epochs=3, warmup_epochs=1,
                       batch_sizes=(4,), learning_rates=(0.001,), restarts=1)
    config = {"trial_id": "resume-check", "model_module": module.__name__, "model_metadata": module.model_metadata(),
              "dimension": 2, "batch_size": 4, "learning_rate": 0.001, "seed": 719, "protocol": asdict(protocol)}
    args = SimpleNamespace(num_workers=0, pin_memory="no", persistent_workers="no", prefetch_factor=4,
                           checkpoint_every=1, plot_every=200, resume="no")
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    train_trial(uninterrupted, records[:9], records[9:], module, protocol, config, args,
                torch.device("cpu"), SimpleNamespace(requested=False))
    with pytest.raises(StopRequested):
        train_trial(resumed, records[:9], records[9:], module, protocol, config, args,
                    torch.device("cpu"), SimpleNamespace(requested=True))
    args.resume = "yes"
    train_trial(resumed, records[:9], records[9:], module, protocol, config, args,
                torch.device("cpu"), SimpleNamespace(requested=False))
    a, b = load_checkpoint(uninterrupted / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert a["training_state"] == b["training_state"]
    for name in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
