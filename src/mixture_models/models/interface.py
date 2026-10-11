"""Shared contracts. Scientific default values belong to the model module."""
from dataclasses import dataclass
from typing import Protocol, Sequence
import torch


@dataclass(frozen=True)
class TrainingProtocol:
    batch_sizes: tuple[int, ...]
    learning_rates: tuple[float, ...]
    restarts: int
    max_epochs: int
    warmup_epochs: int
    grad_clip: float
    patience: int
    min_delta: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    amsgrad: bool
    dtype: str
    validation_aggregation: str


class ThermodynamicModel(Protocol):
    """Coordinates are row vectors; densities are in kT nm^-3."""
    def encode_sequences(self, sequences: Sequence[str]) -> torch.Tensor: ...
    def mixture_latent(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor: ...
    def thermodynamics(self, x: torch.Tensor, hessian: bool = False) -> dict[str, torch.Tensor]: ...
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
