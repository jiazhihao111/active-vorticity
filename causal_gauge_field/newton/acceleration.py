import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AccelerationLoss(nn.Module):
    """加速度惩罚损失

    规则3: 加速度惩罚 — 合法轨迹最小化 J = ∫‖ä‖²dt
    在离散序列中: a_t = h_{t+2} - 2h_{t+1} + h_t
    合法轨迹的‖a_t‖²应小于非法轨迹

    这是MVE实验中唯一显著的几何效应（p=0.012）的直接形式化。
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def compute_acceleration(self, hidden: torch.Tensor) -> torch.Tensor:
        """计算轨迹加速度

        Args:
            hidden: [B, T, D] 隐状态序列

        Returns:
            accel: [B, T-2, D] 加速度序列
        """
        if hidden.size(1) < 3:
            return torch.zeros(
                hidden.size(0), 0, hidden.size(2), device=hidden.device
            )
        return hidden[:, 2:, :] - 2 * hidden[:, 1:-1, :] + hidden[:, :-2, :]

    def compute_acceleration_norm_sq(self, hidden: torch.Tensor) -> torch.Tensor:
        """计算加速度范数平方 [B, T-2]"""
        accel = self.compute_acceleration(hidden)
        return (accel ** 2).sum(dim=-1)

    def forward(
        self,
        hidden_pos: torch.Tensor,
        hidden_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """对比加速度损失：合法轨迹加速度 < 非法轨迹加速度

        Args:
            hidden_pos: [B_pos, T, D] 正例隐状态
            hidden_neg: [B_neg, T, D] 负例隐状态

        Returns:
            loss, info_dict
        """
        accel_pos = self.compute_acceleration_norm_sq(hidden_pos)
        accel_neg = self.compute_acceleration_norm_sq(hidden_neg)

        mean_pos = accel_pos.mean()
        mean_neg = accel_neg.mean()

        contrastive = F.relu(mean_pos - mean_neg + self.margin)

        info = {
            "accel_pos_mean": float(mean_pos.item()),
            "accel_neg_mean": float(mean_neg.item()),
            "accel_contrastive": float(contrastive.item()),
        }

        return contrastive, info

    def forward_per_pair(
        self,
        h_t: torch.Tensor,
        h_t1_pos: torch.Tensor,
        h_t1_neg: torch.Tensor,
        h_t2_pos: torch.Tensor,
        h_t2_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """逐对加速度对比

        a_pos = h_{t+2}^pos - 2h_{t+1}^pos + h_t
        a_neg = h_{t+2}^neg - 2h_{t+1}^neg + h_t
        """
        a_pos = h_t2_pos - 2 * h_t1_pos + h_t
        a_neg = h_t2_neg - 2 * h_t1_neg + h_t

        a_pos_sq = (a_pos ** 2).sum(dim=-1)
        a_neg_sq = (a_neg ** 2).sum(dim=-1)

        contrastive = F.relu(a_pos_sq - a_neg_sq + self.margin).mean()

        info = {
            "a_pos_mean": float(a_pos_sq.mean().item()),
            "a_neg_mean": float(a_neg_sq.mean().item()),
        }

        return contrastive, info


class BarrierLoss(nn.Module):
    """势垒损失

    规则4: 势垒分离 — 非法转移需克服势垒 ΔV > ε
    正例: V(h_{t+1}) < V(h_t)（势能下降，沿力场方向）
    负例: V(h_{t+1}) > V(h_t)（势能上升，逆力场方向）
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        V_t: torch.Tensor,
        V_t1_pos: torch.Tensor,
        V_t1_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """势垒对比损失

        Args:
            V_t: [N] 起始点势能
            V_t1_pos: [N] 正例终点势能
            V_t1_neg: [N] 负例终点势能

        Returns:
            loss, info_dict
        """
        delta_V_pos = V_t1_pos - V_t
        delta_V_neg = V_t1_neg - V_t

        pos_downhill = F.relu(delta_V_pos).mean()
        neg_uphill = F.relu(-delta_V_neg + self.margin).mean()

        loss = pos_downhill + neg_uphill

        info = {
            "delta_V_pos_mean": float(delta_V_pos.mean().item()),
            "delta_V_neg_mean": float(delta_V_neg.mean().item()),
            "pos_downhill": float(pos_downhill.item()),
            "neg_uphill_penalty": float(neg_uphill.item()),
        }

        return loss, info