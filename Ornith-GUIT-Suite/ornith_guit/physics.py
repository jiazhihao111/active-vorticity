"""GUIT-TRT 核心潜空间物理计算引擎。

运动方程 (过阻尼 Langevin + 活性驱动):
    m * a + gamma * v = alpha* * v + F_c + xi

派生量:
    P_raw   = (m*a + gamma*v) . v          # 残余力做功
    P_active = alpha* * ||v||^2            # 活性轨道维持力做功
    P_c     = P_raw - P_active             # 约束力做功 (合法轨迹≈0)
    P_c/P_raw                              # 幻觉/相变核心指标 (逐 token)

本模块与 llm_thermodynamics.llm_thermo.core 保持方程一致，
作为 Ornith 套件自包含的物理基座 (不强制依赖外部库)。
"""

from typing import Tuple, List, Optional
import torch
import numpy as np


class ThermoPhysics:
    """LLM 隐状态非平衡态热力学计算引擎。"""

    def __init__(
        self,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        mass: float = 1.0,
    ):
        self.alpha_star = float(alpha_star)
        self.gamma = float(gamma)
        self.mass = float(mass)

    # ---- 运动学 ----------------------------------------------------------
    @staticmethod
    def velocity(h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return h_curr - h_prev

    @staticmethod
    def acceleration(
        h_curr: torch.Tensor, h_prev: torch.Tensor, h_prev2: torch.Tensor
    ) -> torch.Tensor:
        return h_curr - 2.0 * h_prev + h_prev2

    # ---- 动力学功率 ------------------------------------------------------
    def powers(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (P_raw, P_active, P_c)，形状为标量或 [B]。

        h_* 形状允许 [D] 或 [B, D]。
        """
        v = self.velocity(h_curr, h_prev)              # [., D]
        a = self.acceleration(h_curr, h_prev, h_prev2)  # [., D]

        F_res = self.mass * a + self.gamma * v
        P_raw = torch.sum(F_res * v, dim=-1)                       # [.]
        P_active = self.alpha_star * torch.sum(v * v, dim=-1)      # [.]
        P_c = P_raw - P_active
        return P_raw, P_active, P_c

    def pc_ratio(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[float, float]:
        """返回 (P_c/P_raw, v_norm)。逐 token 计算 (GUIT 铁律)。"""
        P_raw, _P_active, P_c = self.powers(h_curr, h_prev, h_prev2)
        pc = float((torch.abs(P_c) / (torch.abs(P_raw) + eps)).item())
        vn = float(torch.norm(self.velocity(h_curr, h_prev), dim=-1).item())
        return pc, vn

    def trajectory_metrics(self, hidden_states: torch.Tensor) -> dict:
        """对整条 [T, D] 轨迹计算汇总指标 (逐 token 平均, 非批量平均)。"""
        if hidden_states.dim() != 2 or hidden_states.size(0) < 4:
            return {"error": "trajectory too short (need T>=4)"}
        T, D = hidden_states.shape
        ratios = []
        vnorms = []
        for t in range(2, T):
            pc, vn = self.pc_ratio(
                hidden_states[t], hidden_states[t - 1], hidden_states[t - 2]
            )
            ratios.append(pc)
            vnorms.append(vn)
        return {
            "mean_pc_ratio": float(np.mean(ratios)),
            "max_pc_ratio": float(np.max(ratios)),
            "std_pc_ratio": float(np.std(ratios)),
            "mean_vel_norm": float(np.mean(vnorms)),
            "alpha_star": self.alpha_star,
            "T": T,
            "D": D,
        }


def calibrate_alpha_star(
    hidden_states_list: List[torch.Tensor],
    gamma: float = 0.01,
    mass: float = 1.0,
) -> float:
    """从一组合法 (pos) 轨迹自动标定 alpha*。

    alpha* = <P_raw> / <P_active>
           = <(m*a + gamma*v).v> / <v.v>

    Args:
        hidden_states_list: 若干条 [T, D] 合法轨迹 (T>=4)
    Returns:
        标定后的 alpha* (float)
    """
    ests = []
    for h in hidden_states_list:
        if h.dim() != 2 or h.size(0) < 4:
            continue
        v = h[1:] - h[:-1]
        a = v[1:] - v[:-1]
        v_f = v[1:]
        if v_f.size(0) < 1:
            continue
        F_res = mass * a + gamma * v_f
        P_raw = (F_res * v_f).sum(dim=-1)
        P_active = (v_f * v_f).sum(dim=-1)
        if float(P_active.abs().mean()) > 1e-10:
            ests.append(float(P_raw.mean().item() / P_active.mean().item()))
    if not ests:
        return 1.41
    return float(np.mean(ests))
