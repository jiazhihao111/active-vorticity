import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def rigid_flexible_layered_loss(
    curv: torch.Tensor,
    layer: torch.Tensor,
    is_neg: torch.Tensor,
    tau: float = 0.5,
    lambda_phys: float = 1.0,
    lambda_flex: float = 1.0,
    margin: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """刚柔分层几何损失 (论文 §3.7 公式 (R) + 铁律八·九).

    输入均为 per-transition 的扁平张量 (N,)（已剔除 padding）:
      curv   : 局部非平坦度代理 (离散代理量, 见 GaugeField.per_transition_connection_norm)
      layer  : 0=物理(刚性)  1=叙事  2=心理 (后两者归为柔性)
      is_neg : 该转移是否对应因果违规 (负例)

    公式 (R) 分层曲率惩罚:
      Λ = λ_phys · ‖F_phys‖²                    (无阈值, 必须为 0)
        + λ_flex · max(0, ‖F_flex‖² − τ)        (带阈值)

    铁律九 (差异化信号, 禁止统一压缩):
      物理负例 → 推离平坦 (penalize curv < margin, 期望不平坦)
      柔性负例 → 推离合法带 (penalize curv < τ, 期望 > τ)
      正例由 (R) 自然拉平(物理)/约束在带内(柔性), 而非所有轨迹一起压平.

    返回 (loss_scalar, info_dict).
    """
    assert curv.dim() == 1
    phys = (layer == 0)
    flex = ~phys
    zero = torch.zeros_like(curv)

    # (R) 分层曲率惩罚: 物理无阈值, 柔性带阈值
    phys_pen = lambda_phys * curv ** 2
    flex_pen = lambda_flex * F.relu(curv ** 2 - tau)
    r_pen = torch.where(phys, phys_pen, flex_pen)

    # 铁律九: 负例推离 (不与正例统一压缩)
    phys_neg = phys & is_neg
    flex_neg = flex & is_neg
    push_phys = torch.where(phys_neg, F.relu(margin - curv), zero)   # 物理违规应不平坦
    push_flex = torch.where(flex_neg, F.relu(tau - curv), zero)      # 柔性违规应超出合法带

    total = r_pen + push_phys + push_flex

    info: Dict[str, float] = {
        "rf_loss": float(total.mean().item()),
        "rf_r_pen": float(r_pen.mean().item()),
        "rf_push_phys": float(push_phys.sum().item()) / max(int(phys_neg.sum().item()), 1),
        "rf_push_flex": float(push_flex.sum().item()) / max(int(flex_neg.sum().item()), 1),
        "n_phys": int(phys.sum().item()),
        "n_flex": int(flex.sum().item()),
        "n_phys_neg": int(phys_neg.sum().item()),
        "n_flex_neg": int(flex_neg.sum().item()),
    }
    return total.mean(), info


class RigidFlexibleLayeredLoss(nn.Module):
    """刚柔分层损失的 nn.Module 包装 (见 rigid_flexible_layered_loss)."""

    def __init__(self, tau: float = 0.5, lambda_phys: float = 1.0,
                 lambda_flex: float = 1.0, margin: float = 0.3):
        super().__init__()
        self.tau = tau
        self.lambda_phys = lambda_phys
        self.lambda_flex = lambda_flex
        self.margin = margin

    def forward(self, curv: torch.Tensor, layer: torch.Tensor, is_neg: torch.Tensor):
        return rigid_flexible_layered_loss(
            curv, layer, is_neg, self.tau, self.lambda_phys,
            self.lambda_flex, self.margin)
