"""Inductive relational GraphSAGE model for transaction risk scoring."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .features import feature_index


class Time2Vec(nn.Module):
    """Learnable time encoding for per-node snapshots of one transaction context."""

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.frequency = nn.Parameter(torch.linspace(1.0, 1.0 + time_dim, time_dim) * 2.0 * torch.pi)
        self.phase = nn.Parameter(torch.zeros(time_dim))

    def forward(self, time_phase: Tensor) -> Tensor:
        return torch.sin(time_phase.unsqueeze(-1) * self.frequency + self.phase)


class RelationalSAGEConv(nn.Module):
    """Mean-aggregating GraphSAGE layer with relation embeddings and edge weights."""

    def __init__(self, dim: int, relation_count: int, dropout: float = 0.0, *, risk_gate: bool = False) -> None:
        super().__init__()
        self.risk_gate_enabled = bool(risk_gate)
        self.risk_feature_index = feature_index("risk_ewma")
        self.self_lin = nn.Linear(dim, dim, bias=False)
        self.neigh_lin = nn.Linear(dim, dim, bias=False)
        self.relation_embedding = nn.Embedding(relation_count, dim)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if self.risk_gate_enabled:
            self.risk_gate = nn.Linear(1, dim)
            nn.init.zeros_(self.risk_gate.weight)
            nn.init.zeros_(self.risk_gate.bias)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_weight: Tensor,
        raw_features: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        if h is not None:
            # ``x`` carries the layer input when available; ``h`` is used by callers
            # that only pass hidden activations. Keeping both paths short here
            # avoids an accidental state alias in the residual connection.
            h_input = h
        else:
            h_input = x
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(h_input)
        else:
            src, dst = edge_index
            messages = self.neigh_lin(h_input[src]) + self.relation_embedding(edge_type)
            messages = messages * edge_weight.clamp(0.05, 1.0).unsqueeze(-1)
            if self.risk_gate_enabled:
                if raw_features is None:
                    source_risk = torch.zeros(h_input.shape[0], 1, device=h_input.device, dtype=h_input.dtype)
                else:
                    source_risk = raw_features[:, self.risk_feature_index].clamp(0.0, 1.0).unsqueeze(-1)
                gate = torch.sigmoid(self.risk_gate(source_risk))
                messages = messages * gate[src]
            aggregated = torch.zeros_like(h_input)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros(h_input.shape[0], device=h_input.device, dtype=h_input.dtype)
            degree.index_add_(0, dst, torch.ones_like(edge_weight))
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        out = self.self_lin(h_input) + aggregated + self.bias
        return self.dropout(torch.relu(self.norm(out)))


class RiskGNN(nn.Module):
    """Two-layer inductive GNN with a transaction-level classification head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        dim = self.config.hidden_dim
        self.type_embedding = nn.Embedding(self.config.type_count, dim)
        self.input_projection = nn.Linear(self.config.input_dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.temporal_encoding = bool(getattr(self.config, "temporal_encoding", False))
        if self.temporal_encoding:
            self.time_encoder = Time2Vec(max(1, self.config.time_dim))
            self.time_projection = nn.Linear(max(1, self.config.time_dim), dim)
        self.layers = nn.ModuleList(
            [
                RelationalSAGEConv(
                    dim,
                    self.config.relation_count,
                    self.config.dropout,
                    risk_gate=self.config.risk_gate,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        node_types: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_weight: Tensor,
        target_index: Tensor,
        *,
        time_phase: Tensor | None = None,
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = self.input_projection(x) + self.type_embedding(node_types)
        h = self.input_dropout(torch.relu(self.input_norm(h)))
        if self.temporal_encoding:
            if time_phase is None:
                time_phase = x.new_zeros(x.shape[0])
            h = h + self.time_projection(self.time_encoder(time_phase.float()))
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_type, edge_weight, x)
        target_embedding = h[target_index]
        logits = self.classifier(target_embedding).squeeze(-1)
        if return_embedding:
            return logits, target_embedding
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(config: ModelConfig | None = None) -> RiskGNN:
    return RiskGNN(config)


def load_model_checkpoint(path: str, device: str = "cpu") -> tuple[RiskGNN, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = RiskGNN(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
