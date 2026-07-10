import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CausalPotential(nn.Module):
    """因果势函数 V: R^d -> R

    牛顿版本的核心：不假设弯曲空间，而是在平直空间中定义势函数。
    合法转移沿 -∇V 方向（势能下降），非法转移需克服势垒。

    规则2: 因果势函数 — 存在 V(h)，合法转移沿 -∇V 方向
    规则6: 学习即势函数塑形 — 训练过程塑造 V 的景观
    规则7: 推理即惯性行走 — 生成沿势谷的惯性方向前进
    """

    def __init__(self, hidden_dim: int, hidden_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        in_dim = hidden_dim
        for i in range(hidden_layers):
            out_dim = max(hidden_dim // (2 ** i), 16)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)

    def compute_force(self, h: torch.Tensor) -> torch.Tensor:
        """力场 F = -∇V，通过自动微分计算"""
        h_req = h.detach().requires_grad_(True)
        with torch.enable_grad():
            V = self.forward(h_req)
            grad_V = torch.autograd.grad(
                V.sum(), h_req, create_graph=self.training
            )[0]
        return -grad_V

    def compute_potential_barrier(
        self, h_t: torch.Tensor, h_t1_pos: torch.Tensor, h_t1_neg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算正例和负例的势能差

        规则4: 势垒分离 — 非法转移需克服势垒 ΔV > ε
        """
        V_t = self.forward(h_t)
        V_t1_pos = self.forward(h_t1_pos)
        V_t1_neg = self.forward(h_t1_neg)
        delta_V_pos = V_t1_pos - V_t
        delta_V_neg = V_t1_neg - V_t
        return delta_V_pos, delta_V_neg

    def compute_force_alignment(
        self, h_t: torch.Tensor, h_t1: torch.Tensor
    ) -> torch.Tensor:
        """计算转移方向与力场方向的对齐度

        合法转移: h_{t+1} - h_t ≈ F(h_t) · Δt
        对齐度 = cos(转移方向, 力场方向)
        """
        force = self.compute_force(h_t)
        delta_h = h_t1 - h_t
        alignment = F.cosine_similarity(delta_h, force, dim=-1)
        return alignment

    def compute_potential_landscape_stats(
        self, h: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """势函数景观统计：用于可视化和诊断"""
        V = self.forward(h)
        force = self.compute_force(h)
        force_norm = force.norm(dim=-1)
        return {
            "V_mean": V.mean(),
            "V_std": V.std(),
            "V_min": V.min(),
            "V_max": V.max(),
            "force_norm_mean": force_norm.mean(),
            "force_norm_std": force_norm.std(),
        }