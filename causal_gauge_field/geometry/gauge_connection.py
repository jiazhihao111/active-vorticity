import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GaugeConnection(nn.Module):
    """
    因果补偿规范场 A_μ(h) ∈ Ω¹(M, g)
    
    规范协变导数: D_μ = ∂_μ - iA_μ
    场强(因果曲率): F_μν = [D_μ, D_ν] = ∂_μA_ν - ∂_νA_μ - i[A_μ, A_ν]
    
    破缺=规范固定(选择特定token)，缝合=补偿变换(恢复规范协变性)。
    A_μ由三层因果子群的李代数分量构成。
    """
    def __init__(self, d_model: int, lie_dim: int = 3, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.lie_dim = lie_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.connection_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * lie_dim),
        )
        self.lie_generators = nn.ParameterList([
            nn.Parameter(torch.randn(lie_dim, d_model, d_model) * 0.01)
            for _ in range(3)
        ])
        self.causal_gate = nn.Linear(d_model, 3)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = h.size(0)
        A_flat = self.connection_net(h)
        A = A_flat.view(batch_size, self.lie_dim, self.d_model)
        gate_weights = F.softmax(self.causal_gate(h), dim=-1)
        A_composed = torch.zeros(batch_size, self.d_model, device=h.device)
        for k in range(3):
            w_k = gate_weights[:, k].unsqueeze(-1)
            A_k = A[:, k, :]
            A_composed = A_composed + w_k * A_k
        return A_composed, gate_weights

    def compute_field_strength(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        A_t: Optional[torch.Tensor] = None,
        A_t1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if A_t is None:
            A_t, _ = self.forward(h_t)
        if A_t1 is None:
            A_t1, _ = self.forward(h_t1)
        delta_h = h_t1 - h_t
        delta_A = A_t1 - A_t
        commutator = A_t * delta_A - delta_A * A_t
        F = delta_A - self.scale * commutator
        return F

    def compute_field_strength_norm_sq(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
    ) -> torch.Tensor:
        F = self.compute_field_strength(h_t, h_t1)
        return (F * F).sum(dim=-1)

    def parallel_transport(
        self,
        h_t: torch.Tensor,
        delta_x: torch.Tensor,
        A_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if A_t is None:
            A_t, _ = self.forward(h_t)
        h_t1 = h_t + delta_x - self.scale * A_t * delta_x.norm(dim=-1, keepdim=True)
        return h_t1

    def compute_wilson_loop(
        self,
        hidden_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d = hidden_trajectory.shape
        if seq_len < 2:
            return torch.zeros(batch_size, device=hidden_trajectory.device)
        phase = torch.zeros(batch_size, device=hidden_trajectory.device)
        for t in range(seq_len - 1):
            h_t = hidden_trajectory[:, t, :]
            h_t1 = hidden_trajectory[:, t + 1, :]
            F_norm = self.compute_field_strength_norm_sq(h_t, h_t1)
            phase = phase + F_norm
        return phase / (seq_len - 1)