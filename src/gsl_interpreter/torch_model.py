from __future__ import annotations

import torch
from torch import nn

from gsl_interpreter import MODEL_FEATURE_SIZE


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = MODEL_FEATURE_SIZE,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(x)
        return self.head(output[:, -1, :])


def build_model(config: dict[str, int | float], num_classes: int) -> SequenceClassifier:
    return SequenceClassifier(
        num_classes=num_classes,
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )


def default_config() -> dict[str, int | float]:
    return {
        "input_size": MODEL_FEATURE_SIZE,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.3,
    }


def best_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
