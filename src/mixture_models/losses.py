"""All pressure arguments are beta P in nm^-3; concentrations are chains nm^-3."""
import torch


def pressure_loss(prediction, target, ctot):
    return torch.mean(((target - prediction) / ctot).square())
