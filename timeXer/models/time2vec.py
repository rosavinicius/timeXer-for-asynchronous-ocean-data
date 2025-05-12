import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super(Time2Vec, self).__init__()
        assert embed_dim > 1
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.k = embed_dim - 1 # Number of periodic features

        # Linear component parameters (omega_0, phi_0 implicitly handled by nn.Linear)
        self.linear_proj = nn.Linear(input_dim, 1, bias=True)

        # Periodic component parameters (omega_i, phi_i implicitly handled by nn.Linear)
        self.periodic_proj = nn.Linear(input_dim, self.k, bias=True)

        # Initialize weights/biases if desired (optional)
        # self.linear_proj.weight.data.uniform_(-1e-4, 1e-4)
        # self.linear_proj.bias.data.uniform_(-1e-4, 1e-4)
        # self.periodic_proj.weight.data.uniform_(-1e-4, 1e-4)
        # self.periodic_proj.bias.data.uniform_(-1e-4, 1e-4)

    def forward(self, tau: torch.Tensor):
        # tau shape: (batch_size, seq_len, input_dim) or similar
        if tau.dim() == 2: # Add feature dimension if input is (batch, seq_len)
            tau = tau.unsqueeze(-1) 

        if tau.shape[-1] != self.input_dim:
             raise ValueError(f"Expected input last dim {self.input_dim}, but got {tau.shape[-1]}")

        time_linear = self.linear_proj(tau)      # (batch, seq_len, 1)
        time_periodic = torch.sin(self.periodic_proj(tau)) # (batch, seq_len, k)

        # Concatenate along the feature dimension
        out = torch.cat([time_linear, time_periodic], dim=-1) # (batch, seq_len, k+1)
        return out