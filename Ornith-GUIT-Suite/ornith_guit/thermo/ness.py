"""NESS 非平衡态定态诊断 + 异常曲率 (论文 §3.4, §3.3, §4.5)。

核心物理量 (逐 token 计算, 禁止批量平均):
  - 熵产生率 σ ≈ α* · <||v||²>/D  (活性功率, ≥0; 论文 σ=α*||v||²/γ_eff,
    因 γ_eff<0 取幅值以保持非负, 见 §3.2 负阻尼悖论 — 仅作定态强度代理)
  - 概率流 J ≈ <||v||>  (稳态环流强度代理; 论文 J_FP=<v·p(v)> 需分布,
    单轨迹取速度幅均值)
  - 一阶矩速度 <v>: NESS 要求 ≈0 (宏观细致平衡未破缺)
  - 速度 CV = std(||v||²)/mean(||v||²) < 0.5 (宏观量稳定)
  - 异常曲率 K_sub = <||a_perp||/||a_parallel||> ≈ 0.01 (局部极度平坦)
"""

from typing import Dict, Optional
import numpy as np
import torch

from ..physics import ThermoPhysics


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def abnormal_curvature(
    hidden_states: torch.Tensor,
    alpha_star: float = 1.41,
    gamma: float = 0.01,
) -> float:
    """异常曲率 K_sub = <||a_perp|| / ||a_parallel||> (论文式6)。

    合法轨迹 a 几乎平行于 v (沿水平分布演化), K_sub≈0.01。
    """
    H = _to_np(hidden_states)
    if H.shape[0] < 4:
        return float("nan")
    Ks = []
    for t in range(2, H.shape[0]):
        v = H[t] - H[t - 1]
        a = H[t] - 2 * H[t - 1] + H[t - 2]
        vn2 = float(v.dot(v)) + 1e-12
        a_par = (a.dot(v) / vn2) * v
        a_perp = a - a_par
        Ks.append(float(np.linalg.norm(a_perp)) / (float(np.linalg.norm(a_par)) + 1e-12))
    return float(np.mean(Ks))


def ness_metrics(
    hidden_states: torch.Tensor,
    alpha_star: float = 1.41,
    gamma: float = 0.01,
) -> Dict:
    """对整条 [T, D] 轨迹计算 NESS 诊断 (逐 token)。"""
    H = _to_np(hidden_states)
    T, D = H.shape
    if T < 4:
        return {"error": "trajectory too short (need T>=4)"}

    vels = H[1:] - H[:-1]                       # [T-1, D]
    speed = np.linalg.norm(vels, axis=-1)       # [T-1]
    speed2 = speed ** 2

    mean_v = vels.mean(axis=0)                  # 一阶矩速度
    first_moment = float(np.linalg.norm(mean_v))
    mean_vel_norm = float(speed.mean())
    cv = float(speed2.std() / (speed2.mean() + 1e-12))

    gamma_eff = gamma - alpha_star             # <0 (负阻尼, 范德波尔机制抵消)
    active_power = alpha_star * float(speed2.mean()) / D   # σ 代理 (≥0)
    J = float(speed.mean())                    # 概率流代理

    K = abnormal_curvature(H, alpha_star, gamma)

    # NESS 判据 (论文 §4.5): σ>0, J>0, <v>≈0, CV<0.5
    moment_ratio = first_moment / (mean_vel_norm + 1e-12)
    is_ness = (active_power > 0) and (J > 0) and (moment_ratio < 0.5) and (cv < 0.5)

    return {
        "alpha_star": alpha_star,
        "gamma_eff": float(gamma_eff),
        "entropy_production_sigma": active_power,
        "probability_flow_J": J,
        "mean_velocity_norm": mean_vel_norm,
        "first_moment_velocity": first_moment,
        "first_moment_ratio": float(moment_ratio),
        "cv_velocity": cv,
        "abnormal_curvature_Ksub": K,
        "is_NESS": bool(is_ness),
        "T": T, "D": D,
    }


class NESSDiagnostics:
    """非平衡态定态诊断器 (封装 ness_metrics)。"""

    def __init__(self, alpha_star: float = 1.41, gamma: float = 0.01):
        self.alpha_star = alpha_star
        self.gamma = gamma

    def diagnose(self, hidden_states: torch.Tensor) -> Dict:
        return ness_metrics(hidden_states, self.alpha_star, self.gamma)

    def compare_regimes(self, trajectories: Dict[str, torch.Tensor]) -> Dict:
        """对多 regime 轨迹批量诊断, 返回结构化对比表。"""
        out = {}
        for name, H in trajectories.items():
            m = self.diagnose(H)
            out[name] = {k: m[k] for k in (
                "entropy_production_sigma", "probability_flow_J",
                "mean_velocity_norm", "first_moment_ratio", "cv_velocity",
                "abnormal_curvature_Ksub", "is_NESS")}
        return out
