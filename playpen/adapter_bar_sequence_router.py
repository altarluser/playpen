from __future__ import annotations

import torch
import torch.nn as nn


class SequenceAdapterRouter(nn.Module):
    """sequence-level Adapter-BAR router over frozen LoRA experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        router_hidden_size: int = 512,
        router_dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.fc1 = nn.Linear(self.hidden_size, int(router_hidden_size))
        self.dropout = nn.Dropout(float(router_dropout))
        self.fc2 = nn.Linear(int(router_hidden_size), self.num_experts)
        act = str(activation).strip().lower()
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        x = self.fc1(pooled_hidden)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)
