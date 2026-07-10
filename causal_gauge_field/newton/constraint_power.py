import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class ConstraintPowerAnalyzer:
    """GUIT-TRT v7.0: 约束力做功分析器

    核心假说:
    - 合法轨迹受理想非完整约束, 约束力F_c⊥v, 不做功 → P(t)≈0
    - 非法轨迹违反约束, 约束力做负功 → P(t)<<0

    P(t) = (m·ḧ + γ·ḣ) · ḣ
    """

    def __init__(self, mass: float = 1.0, friction: float = 0.1):
        self.m = mass
        self.gamma = friction

    def compute_derivatives(self, hidden: torch.Tensor):
        """计算速度和加速度

        Args:
            hidden: [B, T, D] 或 [T, D]

        Returns:
            velocity: [B, T-1, D] 或 [T-1, D]
            acceleration: [B, T-2, D] 或 [T-2, D]
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        velocity = hidden[:, 1:, :] - hidden[:, :-1, :]
        if velocity.size(1) >= 2:
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
        else:
            acceleration = torch.zeros_like(velocity)

        return velocity, acceleration

    def compute_constraint_power(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算约束力做功功率

        P(t) = (m·ḧ + γ·ḣ) · ḣ

        Args:
            hidden: [B, T, D] 或 [T, D]

        Returns:
            P_t: [B, T-2] 功率时间序列
            velocity: [B, T-1, D]
            acceleration: [B, T-2, D]
        """
        velocity, acceleration = self.compute_derivatives(hidden)

        F_res = self.m * acceleration + self.gamma * velocity[:, 1:, :]

        P_t = torch.sum(F_res * velocity[:, 1:, :], dim=-1)

        return P_t, velocity, acceleration

    def compute_velocity_effective_rank(self, velocity: torch.Tensor, threshold: float = 0.95) -> Dict:
        """计算速度场的有效秩（约束流形切空间维度）

        如果合法轨迹的速度向量集中在前k个主成分上 → 存在非完整约束
        """
        if velocity.dim() == 3:
            v_flat = velocity.reshape(-1, velocity.size(-1))
        else:
            v_flat = velocity

        v_np = v_flat.detach().cpu().numpy()
        if v_np.shape[0] < 2 or v_np.shape[1] < 2:
            return {"effective_rank": 0, "total_dim": v_np.shape[1]}

        cov = np.cov(v_np.T)
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        total_var = eigenvalues.sum()
        if total_var < 1e-10:
            return {"effective_rank": 0, "total_dim": v_np.shape[1]}

        cumulative = np.cumsum(eigenvalues) / total_var
        effective_rank = int(np.searchsorted(cumulative, threshold) + 1)

        return {
            "effective_rank": effective_rank,
            "total_dim": v_np.shape[1],
            "top3_variance_ratio": float(eigenvalues[:3].sum() / total_var) if len(eigenvalues) >= 3 else 1.0,
            "eigenvalues_top5": eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
        }

    def full_analysis(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
    ) -> Dict:
        """完整v7.0验证：约束力做功 + 切空间有效秩"""
        pos_P_all = []
        neg_P_all = []
        pos_velocities = []
        neg_velocities = []

        with torch.no_grad():
            for h in pos_stories_hidden:
                if h.size(0) < 4:
                    continue
                P_t, vel, acc = self.compute_constraint_power(h)
                if P_t.numel() > 0:
                    pos_P_all.extend(P_t.flatten().tolist())
                    pos_velocities.append(vel)

            for h in neg_stories_hidden:
                if h.size(0) < 4:
                    continue
                P_t, vel, acc = self.compute_constraint_power(h)
                if P_t.numel() > 0:
                    neg_P_all.extend(P_t.flatten().tolist())
                    neg_velocities.append(vel)

        results = {}

        if pos_P_all and neg_P_all:
            pos_arr = np.array(pos_P_all)
            neg_arr = np.array(neg_P_all)

            pos_mean = np.mean(pos_arr)
            neg_mean = np.mean(neg_arr)
            pos_skew = stats.skew(pos_arr)
            neg_skew = stats.skew(neg_arr)

            t_stat_pos_zero, p_val_pos_zero = stats.ttest_1samp(pos_arr, 0)
            t_stat_diff, p_val_diff = stats.ttest_ind(pos_arr, neg_arr, equal_var=False)
            ks_stat, ks_p = stats.ks_2samp(pos_arr, neg_arr)

            pooled_std = np.sqrt((np.std(pos_arr)**2 + np.std(neg_arr)**2) / 2)
            cohens_d = (pos_mean - neg_mean) / (pooled_std + 1e-10)

            near_zero = abs(pos_mean) < 1e-3 and p_val_pos_zero > 0.05
            neg_lower = neg_mean < pos_mean and p_val_diff < 0.01
            neg_left_skew = neg_skew < -1.0

            criteria = {
                "pos_near_zero": near_zero,
                "neg_significantly_lower": neg_lower,
                "neg_left_skewed": neg_left_skew,
            }

            passed = sum(1 for v in criteria.values() if v)

            results["constraint_power"] = {
                "pos_mean": float(pos_mean),
                "neg_mean": float(neg_mean),
                "pos_std": float(np.std(pos_arr)),
                "neg_std": float(np.std(neg_arr)),
                "pos_skew": float(pos_skew),
                "neg_skew": float(neg_skew),
                "pos_vs_zero_p": float(p_val_pos_zero),
                "pos_vs_neg_p": float(p_val_diff),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "cohens_d": float(cohens_d),
                "criteria": criteria,
                "passed_count": passed,
                "total_criteria": 3,
                "verdict": "STRONG_SUPPORT" if passed >= 2 else ("WEAK_SUPPORT" if passed >= 1 else "OPPOSE"),
            }

        if pos_velocities and neg_velocities:
            pos_vel_flat = torch.cat([v.reshape(-1, v.size(-1)) for v in pos_velocities], dim=0)
            neg_vel_flat = torch.cat([v.reshape(-1, v.size(-1)) for v in neg_velocities], dim=0)
            pos_rank = self.compute_velocity_effective_rank(pos_vel_flat)
            neg_rank = self.compute_velocity_effective_rank(neg_vel_flat)

            pos_lower_rank = pos_rank["effective_rank"] < neg_rank["effective_rank"]

            results["velocity_effective_rank"] = {
                "pos_effective_rank": pos_rank["effective_rank"],
                "neg_effective_rank": neg_rank["effective_rank"],
                "pos_top3_variance_ratio": pos_rank["top3_variance_ratio"],
                "neg_top3_variance_ratio": neg_rank["top3_variance_ratio"],
                "pos_lower_rank": pos_lower_rank,
                "verdict": "SUPPORT" if pos_lower_rank else "OPPOSE",
            }

        support_count = sum(
            1 for k, v in results.items()
            if isinstance(v, dict) and v.get("verdict") in ("SUPPORT", "STRONG_SUPPORT", "WEAK_SUPPORT")
        )
        total_count = len([k for k in results.keys() if isinstance(results[k], dict) and "verdict" in results[k]])

        if support_count >= 2:
            overall = "STRONG_SUPPORT"
        elif support_count >= 1:
            overall = "WEAK_SUPPORT"
        else:
            overall = "OPPOSE"

        results["overall_verdict"] = {
            "support_count": support_count,
            "total_count": total_count,
            "verdict": overall,
            "framework": "v7.0 Lagrangian Constraint Mechanics",
            "mass": self.m,
            "friction": self.gamma,
        }

        return results