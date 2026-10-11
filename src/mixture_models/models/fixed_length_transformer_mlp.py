"""Fixed-length Transformer encoder and MLP free-energy decoder.

Authoritative architecture: wmjac/idpmixture/fe_predictor_tf_linc_meanz.py.
The reference concentration preserves the validated legacy parameterization;
physical model inputs never carry the old concentration unit.
"""
from typing import Sequence
import torch
import torch.nn as nn
from torch.func import vmap, grad, hessian as functional_hessian
from .interface import TrainingProtocol

SEQUENCE_LENGTH = 20
MODEL_ID = "fixed_length_transformer_mlp"
MODEL_VERSION = 1
DECODER_REFERENCE_CONCENTRATION_NM_INV3 = 0.1
AMINO_ACIDS = "MGKTRADEYVLQWFSHNPCI"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}


def training_protocol():
    return TrainingProtocol(
        batch_sizes=(2048,), learning_rates=(1e-3, 5e-3, 1e-2),
        restarts=5, max_epochs=20000, warmup_epochs=100,
        grad_clip=1.0, patience=500, min_delta=1e-4,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        weight_decay=0.0, amsgrad=False, dtype="float32",
        validation_aggregation="batch-mean",
    )


def validate_sequences(sequences):
    for sequence in sequences:
        if not isinstance(sequence, str) or len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(f"{MODEL_ID} requires {SEQUENCE_LENGTH} residues; received {sequence!r}.")
        invalid = set(sequence) - set(AA_TO_INDEX)
        if invalid:
            raise ValueError(f"Unsupported residues {sorted(invalid)} in {sequence!r}.")


def tokenize(sequences, device=None):
    validate_sequences(sequences)
    return torch.tensor([[AA_TO_INDEX[aa] for aa in seq] for seq in sequences],
                        dtype=torch.long, device=device)


def collate_records(records, dtype="float32"):
    """Pad species slots, never sequence length; zero concentrations mask extra slots."""
    n = max(len(record.sequences) for record in records)
    tokens = torch.zeros((len(records), n, SEQUENCE_LENGTH), dtype=torch.long)
    concentrations = torch.zeros((len(records), n, 1), dtype=getattr(torch, dtype))
    for row, record in enumerate(records):
        count = len(record.sequences)
        tokens[row, :count] = tokenize(record.sequences)
        concentrations[row, :count, 0] = torch.tensor(record.concentrations, dtype=getattr(torch, dtype))
    return {"tokens": tokens, "concentrations": concentrations,
            "ctot": concentrations.sum(dim=1),
            "target": torch.tensor([r.pressure for r in records], dtype=getattr(torch, dtype))[:, None]}


def build_optimizer(model, learning_rate, protocol):
    return torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps, weight_decay=protocol.weight_decay, amsgrad=protocol.amsgrad,
    )


def build_scheduler(optimizer, steps_per_epoch, protocol):
    warmup_steps = protocol.warmup_epochs * steps_per_epoch
    def multiplier(step):
        return (step + 1) / warmup_steps if step < warmup_steps else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def model_metadata():
    return {
        "id": MODEL_ID, "version": MODEL_VERSION, "sequence_length": SEQUENCE_LENGTH,
        "amino_acids_by_index": AMINO_ACIDS,
        "encoder": {"embedding_dim": 16, "heads": 2, "feedforward_dim": 48,
                    "layers": 3, "positional_capacity": 50, "dropout": 0.1,
                    "pooling": "mean", "reversal_symmetry": "average_forward_reverse",
                    "activation": "relu", "layer_norm_eps": 1e-5, "norm_first": False},
        "decoder": {"hidden_widths": [128, 64], "activation": "silu",
                    "form": "F(u)=u.T G(u) u; beta_f_ex=F(x/c_ref)*c_ref",
                    "reference_concentration_nm^-3": DECODER_REFERENCE_CONCENTRATION_NM_INV3},
        "physical_units": {"concentration": "nm^-3", "energy": "kT",
                           "pressure": "kT nm^-3", "free_energy_density": "kT nm^-3"},
    }

class ForceFieldEncoder(nn.Module):
    def __init__(
        self,
        zdim: int,
        d_model: int = 16,
        num_heads: int = 2,
        ff_dim: int = 48,
        num_layers: int = 3,
        max_len: int = 50,
        dropout: float = 0.1,
        vocab_size: int = 20,   # assuming 20 unique tokens
    ):
        super().__init__()

        # 1. token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # 2. positional encoding (fixed, added to token embeddings)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # shape [1, L, D]

        # 3. Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # 4. projection head
        self.head = nn.Linear(d_model, zdim)

    @staticmethod
    def _inverse_seq(seqs: torch.Tensor) -> torch.Tensor:
        """Reverse sequences along length dimension (dim=1)."""
        if seqs is None:
            return None
        return torch.flip(seqs, dims=[1])

    def _encode(self, token_batch: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Token -> embedding space → mean-pooled latent."""
        device = self.tok_emb.weight.device
        token_batch = token_batch.to(device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        if token_batch.ndim != 2 or token_batch.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"This model requires sequence length {SEQUENCE_LENGTH}.")
        x = self.tok_emb(token_batch)                               # [B, L, D]
        x = x + self.pos_encoding[:, : x.size(1)]                   # broadcast PE

        x = self.encoder(x, src_key_padding_mask=padding_mask)      # [B, L, D]

        # mean-pool while ignoring padding
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1, keepdim=True)      # [B, 1]
            x = (
                x.masked_fill(padding_mask.unsqueeze(-1), 0)
                 .sum(dim=1) / lengths.clamp(min=1)
            )
        else:
            x = x.mean(dim=1)

        return self.head(x)                                         # [B, zdim]

    def forward(self, token_batch: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        token_batch : LongTensor [B, L]    — padded with an index (e.g. 0)
        padding_mask: BoolTensor [B, L]    — True where padded
        """
        out_fwd = self._encode(token_batch, padding_mask)
        out_rev = self._encode(self._inverse_seq(token_batch), self._inverse_seq(padding_mask))
        return 0.5 * (out_fwd + out_rev)

class MeanFieldDecoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.zdim = zdim
        self.net = nn.Sequential(nn.Linear(zdim, 128), nn.SiLU(),
                                 nn.Linear(128, 64), nn.SiLU(),
                                 nn.Linear(64, zdim * (zdim + 1) // 2))
        iu, ju = torch.triu_indices(zdim, zdim)
        self.register_buffer("iu", iu, persistent=False)
        self.register_buffer("ju", ju, persistent=False)

    def native_free_energy_single(self, u):
        values = self.net(u)
        matrix = values.new_zeros((self.zdim, self.zdim))
        matrix[self.iu, self.ju] = values
        matrix = matrix + matrix.T - torch.diag(torch.diag(matrix))
        return u @ matrix @ u

    def forward(self, x, hessian=False):
        # u is dimensionless, not a concentration expressed in a legacy physical unit.
        inverse_reference = 1.0 / DECODER_REFERENCE_CONCENTRATION_NM_INV3
        u = x * inverse_reference
        fn = self.native_free_energy_single
        native_f = vmap(fn)(u)
        native_gradient = vmap(grad(fn))(u)
        result = {
            "free_energy_excess": native_f[:, None] / inverse_reference,
            "pressure_excess": ((native_gradient * u).sum(dim=1) - native_f)[:, None] / inverse_reference,
            "gradient": native_gradient,
        }
        if hessian:
            result["hessian"] = vmap(functional_hessian(fn))(u) * inverse_reference
        return result


class FixedLengthTransformerMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        # Preserve checkpoint parameter keys from the authoritative architecture.
        self.model1 = ForceFieldEncoder(dimension)
        self.model3 = MeanFieldDecoder(dimension)

    def encode_sequences(self, sequences: Sequence[str]):
        return self.model1(tokenize(sequences, next(self.parameters()).device))

    def mixture_latent(self, inputs):
        tokens, concentrations = inputs["tokens"], inputs["concentrations"]
        if tokens.ndim != 3 or tokens.shape[-1] != SEQUENCE_LENGTH:
            raise ValueError(f"Expected tokens of shape (batch, species, {SEQUENCE_LENGTH}).")
        b, n, length = tokens.shape
        if concentrations.shape != (b, n, 1):
            raise ValueError("Concentrations must have shape (batch, species, 1).")
        z = self.model1(tokens.reshape(b * n, length)).reshape(b, n, self.dimension)
        return (concentrations * z).sum(dim=1)

    def thermodynamics(self, x, hessian=False):
        return self.model3(x, hessian=hessian)

    def forward(self, inputs):
        return self.thermodynamics(self.mixture_latent(inputs))


def build_model(dimension, dtype="float32"):
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Latent dimension must be a positive integer.")
    return FixedLengthTransformerMLP(dimension).to(dtype=getattr(torch, dtype))

