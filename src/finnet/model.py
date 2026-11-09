"""Inductive relational GraphSAGE model for transaction risk scoring."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig


class RelationalSAGEConv(nn.Module):
    """Mean-aggregating GraphSAGE layer with relation embeddings and edge weights."""

    def __init__(self, dim: int, relation_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim, dim, bias=False)
        self.neigh_lin = nn.Linear(dim, dim, bias=False)
        self.relation_embedding = nn.Embedding(relation_count, dim)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(x)
        else:
            src, dst = edge_index
            messages = self.neigh_lin(x[src]) + self.relation_embedding(edge_type)
            messages = messages * edge_weight.clamp(0.05, 1.0).unsqueeze(-1)
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(edge_weight))
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        out = self.self_lin(x) + aggregated + self.bias
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
        self.layers = nn.ModuleList(
            [
                RelationalSAGEConv(dim, self.config.relation_count, self.config.dropout)
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
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = self.input_projection(x) + self.type_embedding(node_types)
        h = self.input_dropout(torch.relu(self.input_norm(h)))
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_type, edge_weight)
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
