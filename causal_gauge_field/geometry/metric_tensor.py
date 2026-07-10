import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CausalMetricTensor(nn.Module):
    """
    因果黎曼度量张量 G_μν(h) = δ_μν + η * Tr(F_μλ F^λ_ν)
    
    将四力向量(4维)扩展为d×d对称正定矩阵，在隐状态空间中定义因果距离。
    三层因果约束(物理/叙事/心理)各自贡献一个子度量，加权和构成总度量。
    """
    def __init__(self, d_model: int, num_causal_types: int = 3, eta: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_causal_types = num_causal_types
        self.eta = eta

        self.type_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, d_model * d_model),
            )
            for _ in range(num_causal_types)
        ])
        self.type_gate = nn.Linear(d_model, num_causal_types)
        self.eta_param = nn.Parameter(torch.tensor(eta))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.size(0)
        gate_weights = F.softmax(self.type_gate(h), dim=-1)
        G = torch.eye(self.d_model, device=h.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        for k in range(self.num_causal_types):
            raw = self.type_projections[k](h)
            M_k = raw.view(batch_size, self.d_model, self.d_model)
            M_k = (M_k + M_k.transpose(1, 2)) / 2.0
            w_k = gate_weights[:, k].unsqueeze(-1).unsqueeze(-1)
            G = G + self.eta_param.abs() * w_k * M_k
        G = (G + G.transpose(1, 2)) / 2.0
        eye = torch.eye(self.d_model, device=h.device).unsqueeze(0)
        G = G + 0.01 * eye
        return G

    def compute_geodesic_distance_sq(
        self, h_t: torch.Tensor, h_t1: torch.Tensor, G: torch.Tensor
    ) -> torch.Tensor:
        diff = h_t1 - h_t
        if diff.dim() == 2:
            diff = diff.unsqueeze(1)
        d_g_sq = torch.bmm(diff, torch.bmm(G, diff.transpose(1, 2)))
        return d_g_sq.squeeze(-1).squeeze(-1)

    def compute_geodesic_distance_sq_batch(
        self, h_t: torch.Tensor, h_t1: torch.Tensor
    ) -> torch.Tensor:
        G = self.forward(h_t)
        return self.compute_geodesic_distance_sq(h_t, h_t1, G)

    def get_type_contributions(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_weights = F.softmax(self.type_gate(h), dim=-1)
        type_names = ["physical", "narrative", "psychological"]
        return {type_names[k]: gate_weights[:, k].mean().item() for k in range(self.num_causal_types)}