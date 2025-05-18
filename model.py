import torch
import torch.nn as nn


class MLP(nn.Module):
    """Rede MLP simples para regredir 3 variáveis a partir de 5 entradas."""
    
    def __init__(self,
                 input_dim: int = 5,
                 hidden_dim: int = 64,
                 output_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
