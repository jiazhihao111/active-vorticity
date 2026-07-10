import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class ActiveForceAnalyzer:
    """GUIT-TRT v8.0: 带非保守主动力的拉格朗日系统分析器

    v7.0实验结果: P(t) = (m·ḧ + γ·ḣ)·ḣ 显著为正 → 现象C（主动力干扰）
    v8.0核心修正: 引入非保守主动力 F_active，重新定义约束力

    运动方程: m·ḧ + γ·ḣ + ∇V(h) = F_active(h) + ξ(t)
    约束力:   F_c = m·ḧ + γ·ḣ - F_active
    修正功率: P_c(t) = F_c · ḣ = (m·ḧ + γ·ḣ - F_active) · ḣ

    F_active的物理来源:
    - 方案A: 交叉熵损失对隐状态的梯度 ∇_h L_ce (训练驱动力)
    - 方案B: 模型前向传播的隐状态增量 Δh = h_{t+1} - h_t (推理驱动力)
    - 方案C: 势函数力场 -∇V(h) (保守力部分，从v7.0残余力中扣除)

    可证伪预测:
    1. 扣除F_active后, P_c(t) ≈ 0 (合法轨迹在约束流形上惯性滑行)
    2. pos P_c(t) 更接近0, neg P_c(t) 偏离0
    3. F_active与速度方向正相关 (驱动力推动系统前进)
    """

    def __init__(self, mass: float = 1.0, friction: float = 0.1):
        self.m = mass
        self.gamma = friction

    def compute_derivatives(self, hidden: torch.Tensor):
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        velocity = hidden[:, 1:, :] - hidden[:, :-1, :]
        if velocity.size(1) >= 2:
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
        else:
            acceleration = torch.zeros_like(velocity)
        return velocity, acceleration

    def compute_raw_power(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """v7.0原始P(t) = (m·ḧ + γ·ḣ)·ḣ"""
        velocity, acceleration = self.compute_derivatives(hidden)
        F_res = self.m * acceleration + self.gamma * velocity[:, 1:, :]
        P_t = torch.sum(F_res * velocity[:, 1:, :], dim=-1)
        return P_t, velocity, acceleration

    def estimate_active_force_method_A(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """方案A: 用交叉熵损失对隐状态的梯度估计F_active

        F_active ≈ -∇_h L_ce
        物理含义: 训练时交叉熵损失的梯度持续推动隐状态向"下一个token正确"的方向演化
        """
        hidden_req = hidden.detach().requires_grad_(True)
        logits = model.lm_head(model.base_projection.weight @ hidden_req.T).T if hasattr(model, 'base_projection') else model.lm_head(hidden_req)
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().reshape(-1, logits.size(-1)),
            target_ids[:, 1:].contiguous().reshape(-1),
            ignore_index=0,
        )
        grad = torch.autograd.grad(loss, hidden_req)[0]
        return -grad

    def estimate_active_force_method_B(self, velocity: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """方案B: 用速度方向作为F_active的近似（归一化版，保守估计）

        F_active ∝ ḣ / ‖ḣ‖ (归一化速度方向)
        注意: 需从vel[:, 1:, :]截取以与compute_corrected_power中的v_for_dot对齐
        """
        v_shifted = velocity[:, 1:, :]
        v_norm = v_shifted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        F_active = v_shifted / v_norm
        if target_len is not None and F_active.size(1) > target_len:
            F_active = F_active[:, :target_len, :]
        return F_active

    def estimate_active_force_method_D(self, velocity: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """方案D: 用速度本身作为F_active（非保守驱动力的最直接估计）

        F_active ∝ ḣ (未归一化)
        物理含义: v7.0中P(t) = (m·ḧ + γ·ḣ)·ḣ >> 0
        如果系统被一个与速度成正比的力驱动(F ∝ v)，则P = F·v ∝ ‖v‖² > 0
        这正是"过阻尼驱动系统"的特征: F_active = α·ḣ
        扣除后: F_c = m·ḧ + γ·ḣ - α·ḣ = m·ḧ + (γ-α)·ḣ

        注意: velocity的shape是[B, T-1, D]，其中v[t] = h[t+1]-h[t]
        在compute_corrected_power中，v_for_dot = vel[:, 1:, :] (从v[1]开始)
        所以F_active需要从vel[:, 1:, :]截取，以与v_for_dot对齐
        """
        F_active = velocity[:, 1:, :].clone()
        if target_len is not None and F_active.size(1) > target_len:
            F_active = F_active[:, :target_len, :]
        return F_active

    def estimate_active_force_method_C(
        self,
        hidden: torch.Tensor,
        potential: nn.Module,
        target_len: int = None,
    ) -> torch.Tensor:
        """方案C: 用势函数力场 -∇V(h) 作为保守力部分"""
        h_req = hidden.detach().requires_grad_(True)
        with torch.enable_grad():
            V = potential(h_req)
            grad_V = torch.autograd.grad(V.sum(), h_req, create_graph=False)[0]
        F_active = -grad_V
        if target_len is not None and F_active.size(1) > target_len:
            F_active = F_active[:, :target_len, :]
        return F_active

    def compute_corrected_power(
        self,
        hidden: torch.Tensor,
        F_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算修正后的约束力做功功率

        P_c(t) = (m·ḧ + γ·ḣ - F_active) · ḣ
        同时计算 P_active = F_active · ḣ
        """
        velocity, acceleration = self.compute_derivatives(hidden)

        F_total = self.m * acceleration + self.gamma * velocity[:, 1:, :]

        min_t = min(F_total.size(1), F_active.size(1))
        F_total = F_total[:, :min_t, :]
        F_active_aligned = F_active[:, :min_t, :]
        v_for_dot = velocity[:, 1:, :][:, :min_t, :]

        F_constraint = F_total - F_active_aligned

        P_c = torch.sum(F_constraint * v_for_dot, dim=-1)
        P_active = torch.sum(F_active_aligned * v_for_dot, dim=-1)

        return P_c, velocity, F_constraint, P_active

    def compute_active_force_alignment(
        self,
        velocity: torch.Tensor,
        F_active: torch.Tensor,
    ) -> torch.Tensor:
        """计算F_active与速度方向的对齐度

        如果F_active是驱动力，应与速度方向正相关
        """
        v_trimmed = velocity[:, 1:, :]
        if v_trimmed.size(1) != F_active.size(1):
            min_t = min(v_trimmed.size(1), F_active.size(1))
            v_trimmed = v_trimmed[:, :min_t, :]
            F_active = F_active[:, :min_t, :]
        alignment = F.cosine_similarity(v_trimmed, F_active, dim=-1)
        return alignment

    def compute_energy_decomposition(
        self,
        pos_P_raw: np.ndarray,
        neg_P_raw: np.ndarray,
        pos_P_active: np.ndarray,
        neg_P_active: np.ndarray,
        pos_P_c: np.ndarray,
        neg_P_c: np.ndarray,
    ) -> Dict:
        """能量分解: 总做功 = 约束力做功 + 主动力做功"""
        return {
            "P_total_mean_pos": float(np.mean(pos_P_raw)),
            "P_total_mean_neg": float(np.mean(neg_P_raw)),
            "P_active_mean_pos": float(np.mean(pos_P_active)),
            "P_active_mean_neg": float(np.mean(neg_P_active)),
            "P_constraint_mean_pos": float(np.mean(pos_P_c)),
            "P_constraint_mean_neg": float(np.mean(neg_P_c)),
            "active_fraction_pos": float(np.mean(pos_P_active) / (abs(np.mean(pos_P_raw)) + 1e-10)),
            "active_fraction_neg": float(np.mean(neg_P_active) / (abs(np.mean(neg_P_raw)) + 1e-10)),
        }

    def full_analysis(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
        method: str = "D",
        model: Optional[nn.Module] = None,
        potential: Optional[nn.Module] = None,
        alpha: float = 1.0,
    ) -> Dict:
        """完整v8.0验证: 扣除主动力后的约束力做功分析

        Args:
            method: F_active估计方法
                "A" - 交叉熵梯度 (需要model)
                "B" - 速度方向归一化 (保守)
                "C" - 势函数力场 (需要potential)
                "D" - 速度本身 F_active=α·ḣ (最直接，含α搜索)
            alpha: 方案D的驱动力系数
        """
        pos_P_raw_all = []
        neg_P_raw_all = []
        pos_P_c_all = []
        neg_P_c_all = []
        pos_P_active_all = []
        neg_P_active_all = []
        pos_alignment_all = []
        neg_alignment_all = []

        with torch.no_grad():
            for h in pos_stories_hidden:
                if h.size(0) < 4:
                    continue
                if h.dim() == 2:
                    h = h.unsqueeze(0)

                velocity, acceleration = self.compute_derivatives(h)
                target_len = acceleration.size(1)

                P_raw, vel, acc = self.compute_raw_power(h)

                if method == "B":
                    F_active = self.estimate_active_force_method_B(vel, target_len=target_len)
                elif method == "C" and potential is not None:
                    F_active = self.estimate_active_force_method_C(h, potential, target_len=target_len)
                elif method == "D":
                    F_active = self.estimate_active_force_method_D(vel, target_len=target_len) * alpha
                else:
                    F_active = self.estimate_active_force_method_D(vel, target_len=target_len) * alpha

                P_c, vel_out, F_c, P_active = self.compute_corrected_power(h, F_active)

                alignment = self.compute_active_force_alignment(vel, F_active)

                pos_P_raw_all.extend(P_raw.flatten().tolist())
                pos_P_c_all.extend(P_c.flatten().tolist())
                pos_P_active_all.extend(P_active.flatten().tolist())
                pos_alignment_all.extend(alignment.flatten().tolist())

            for h in neg_stories_hidden:
                if h.size(0) < 4:
                    continue
                if h.dim() == 2:
                    h = h.unsqueeze(0)

                velocity, acceleration = self.compute_derivatives(h)
                target_len = acceleration.size(1)

                P_raw, vel, acc = self.compute_raw_power(h)

                if method == "B":
                    F_active = self.estimate_active_force_method_B(vel, target_len=target_len)
                elif method == "C" and potential is not None:
                    F_active = self.estimate_active_force_method_C(h, potential, target_len=target_len)
                elif method == "D":
                    F_active = self.estimate_active_force_method_D(vel, target_len=target_len) * alpha
                else:
                    F_active = self.estimate_active_force_method_D(vel, target_len=target_len) * alpha

                P_c, vel_out, F_c, P_active = self.compute_corrected_power(h, F_active)

                alignment = self.compute_active_force_alignment(vel, F_active)

                neg_P_raw_all.extend(P_raw.flatten().tolist())
                neg_P_c_all.extend(P_c.flatten().tolist())
                neg_P_active_all.extend(P_active.flatten().tolist())
                neg_alignment_all.extend(alignment.flatten().tolist())

        results = {}

        if pos_P_c_all and neg_P_c_all:
            pos_c = np.array(pos_P_c_all)
            neg_c = np.array(neg_P_c_all)
            pos_raw = np.array(pos_P_raw_all)
            neg_raw = np.array(neg_P_raw_all)
            pos_active = np.array(pos_P_active_all)
            neg_active = np.array(neg_P_active_all)
            pos_align = np.array(pos_alignment_all)
            neg_align = np.array(neg_alignment_all)

            pos_c_mean = np.mean(pos_c)
            neg_c_mean = np.mean(neg_c)
            pos_c_skew = float(stats.skew(pos_c))
            neg_c_skew = float(stats.skew(neg_c))

            t_pos_zero, p_pos_zero = stats.ttest_1samp(pos_c, 0)
            t_diff, p_diff = stats.ttest_ind(pos_c, neg_c, equal_var=False)
            ks_stat, ks_p = stats.ks_2samp(pos_c, neg_c)

            pooled_std = np.sqrt((np.std(pos_c)**2 + np.std(neg_c)**2) / 2)
            cohens_d = (pos_c_mean - neg_c_mean) / (pooled_std + 1e-10)

            pos_near_zero = abs(pos_c_mean) < abs(np.mean(pos_raw)) * 0.1 or (abs(pos_c_mean) < 1.0 and p_pos_zero > 0.01)
            neg_far_from_zero = abs(neg_c_mean) > abs(pos_c_mean) * 2 and p_diff < 0.05
            pos_closer_than_neg = abs(pos_c_mean) < abs(neg_c_mean) and p_diff < 0.05

            criteria = {
                "pos_near_zero_after_correction": pos_near_zero,
                "neg_far_from_zero": neg_far_from_zero,
                "pos_closer_to_zero_than_neg": pos_closer_than_neg,
            }
            passed = sum(1 for v in criteria.values() if v)

            t_align, p_align = stats.ttest_ind(pos_align, neg_align, equal_var=False)

            improvement = abs(np.mean(pos_raw)) / (abs(pos_c_mean) + 1e-10)

            results["corrected_constraint_power"] = {
                "pos_Pc_mean": float(pos_c_mean),
                "neg_Pc_mean": float(neg_c_mean),
                "pos_Pc_std": float(np.std(pos_c)),
                "neg_Pc_std": float(np.std(neg_c)),
                "pos_Pc_skew": pos_c_skew,
                "neg_Pc_skew": neg_c_skew,
                "pos_vs_zero_p": float(p_pos_zero),
                "pos_vs_neg_p": float(p_diff),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "cohens_d": float(cohens_d),
                "criteria": criteria,
                "passed_count": passed,
                "total_criteria": 3,
                "verdict": "STRONG_SUPPORT" if passed >= 2 else ("WEAK_SUPPORT" if passed >= 1 else "OPPOSE"),
            }

            results["energy_decomposition"] = {
                "pos_P_raw_mean": float(np.mean(pos_raw)),
                "neg_P_raw_mean": float(np.mean(neg_raw)),
                "pos_P_active_mean": float(np.mean(pos_active)),
                "neg_P_active_mean": float(np.mean(neg_active)),
                "pos_P_constraint_mean": float(pos_c_mean),
                "neg_P_constraint_mean": float(neg_c_mean),
                "active_fraction_pos": float(np.mean(pos_active) / (abs(np.mean(pos_raw)) + 1e-10)),
                "active_fraction_neg": float(np.mean(neg_active) / (abs(np.mean(neg_raw)) + 1e-10)),
                "correction_improvement": float(improvement),
            }

            results["active_force_alignment"] = {
                "pos_alignment_mean": float(np.mean(pos_align)),
                "neg_alignment_mean": float(np.mean(neg_align)),
                "pos_vs_neg_p": float(p_align),
                "verdict": "SUPPORT" if np.mean(pos_align) > np.mean(neg_align) and p_align < 0.05 else "OPPOSE",
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
            "framework": "v8.0 Lagrangian with Active Force",
            "method": method,
            "mass": self.m,
            "friction": self.gamma,
        }

        return results