import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class HamiltonianVerifier:
    """哈密顿量守恒验证器

    核心预测：如果存在因果势函数 V(h)，则哈密顿量
    H = ½‖ḣ‖² + V(h) 在合法轨迹上应近似守恒。

    这是牛顿版本的关键可证伪假设：
    - 若 H 在合法轨迹上守恒 → 势函数描述正确
    - 若 H 不守恒 → 需要更复杂的动力学（可能升级为爱因斯坦版本）
    """

    def __init__(self, potential: nn.Module):
        self.potential = potential

    def compute_hamiltonian(
        self, hidden: torch.Tensor
    ) -> torch.Tensor:
        """计算轨迹上每个时间步的哈密顿量

        H_t = ½‖v_t‖² + V(h_t)
        v_t = h_{t+1} - h_t (离散速度)

        Args:
            hidden: [B, T, D]

        Returns:
            H: [B, T-1] 每步的哈密顿量
        """
        with torch.no_grad():
            V = self.potential(hidden)
            velocity = hidden[:, 1:, :] - hidden[:, :-1, :]
            kinetic = 0.5 * (velocity ** 2).sum(dim=-1)
            H = kinetic + V[:, :-1]
        return H

    def verify_conservation(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
    ) -> Dict:
        """验证哈密顿量守恒假设

        合法轨迹: H 应近似守恒 (低方差)
        非法轨迹: H 不守恒 (高方差)
        """
        pos_H_vars = []
        neg_H_vars = []

        for h in pos_stories_hidden:
            if h.size(1) < 4:
                continue
            H = self.compute_hamiltonian(h.unsqueeze(0)).squeeze(0)
            H_var = H.var(dim=0).mean().item()
            pos_H_vars.append(H_var)

        for h in neg_stories_hidden:
            if h.size(1) < 4:
                continue
            H = self.compute_hamiltonian(h.unsqueeze(0)).squeeze(0)
            H_var = H.var(dim=0).mean().item()
            neg_H_vars.append(H_var)

        results = {}

        if pos_H_vars and neg_H_vars:
            pos_mean = np.mean(pos_H_vars)
            neg_mean = np.mean(neg_H_vars)
            t_stat, p_val = stats.ttest_ind(pos_H_vars, neg_H_vars)
            more_conserved = pos_mean < neg_mean

            results["hamiltonian_conservation"] = {
                "H_var_pos_mean": float(pos_mean),
                "H_var_neg_mean": float(neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_more_conserved": more_conserved,
                "verdict": "SUPPORT" if more_conserved and p_val < 0.05 else "OPPOSE",
            }

        return results

    def verify_force_alignment(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
    ) -> Dict:
        """验证力场对齐假设（批量版本）"""
        pos_alignments = []
        neg_alignments = []

        with torch.no_grad():
            for h in pos_stories_hidden:
                if h.size(0) < 3:
                    continue
                h_t = h[:-1]
                h_t1 = h[1:]
                force = self.potential.compute_force(h_t)
                delta_h = h_t1 - h_t
                if force.size(0) > 0 and delta_h.size(0) > 0:
                    align = F.cosine_similarity(delta_h, force, dim=-1)
                    pos_alignments.extend(align.tolist())

            for h in neg_stories_hidden:
                if h.size(0) < 3:
                    continue
                h_t = h[:-1]
                h_t1 = h[1:]
                force = self.potential.compute_force(h_t)
                delta_h = h_t1 - h_t
                if force.size(0) > 0 and delta_h.size(0) > 0:
                    align = F.cosine_similarity(delta_h, force, dim=-1)
                    neg_alignments.extend(align.tolist())

        results = {}

        if pos_alignments and neg_alignments:
            pos_mean = np.mean(pos_alignments)
            neg_mean = np.mean(neg_alignments)
            t_stat, p_val = stats.ttest_ind(pos_alignments, neg_alignments)
            more_aligned = pos_mean > neg_mean

            results["force_alignment"] = {
                "alignment_pos_mean": float(pos_mean),
                "alignment_neg_mean": float(neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_more_aligned": more_aligned,
                "verdict": "SUPPORT" if more_aligned and p_val < 0.05 else "OPPOSE",
            }

        return results

    def verify_potential_barrier(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
    ) -> Dict:
        """验证势垒分离假设

        合法转移: ΔV < 0 (势能下降)
        非法转移: ΔV > 0 (势能上升)
        """
        pos_delta_V = []
        neg_delta_V = []

        with torch.no_grad():
            for h in pos_stories_hidden:
                if h.size(1) < 3:
                    continue
                V = self.potential(h)
                for t in range(V.size(0) - 1):
                    pos_delta_V.append((V[t+1] - V[t]).item())

            for h in neg_stories_hidden:
                if h.size(1) < 3:
                    continue
                V = self.potential(h)
                for t in range(V.size(0) - 1):
                    neg_delta_V.append((V[t+1] - V[t]).item())

        results = {}

        if pos_delta_V and neg_delta_V:
            pos_mean = np.mean(pos_delta_V)
            neg_mean = np.mean(neg_delta_V)
            t_stat, p_val = stats.ttest_ind(pos_delta_V, neg_delta_V)
            downhill = pos_mean < neg_mean

            results["potential_barrier"] = {
                "delta_V_pos_mean": float(pos_mean),
                "delta_V_neg_mean": float(neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_downhill": downhill,
                "verdict": "SUPPORT" if downhill and p_val < 0.05 else "OPPOSE",
            }

        return results

    def full_verification(
        self,
        pos_stories_hidden: List[torch.Tensor],
        neg_stories_hidden: List[torch.Tensor],
    ) -> Dict:
        """完整牛顿版本验证：三个独立假设"""
        results = {}
        results.update(self.verify_conservation(pos_stories_hidden, neg_stories_hidden))
        results.update(self.verify_force_alignment(pos_stories_hidden, neg_stories_hidden))
        results.update(self.verify_potential_barrier(pos_stories_hidden, neg_stories_hidden))

        support_count = sum(
            1 for v in results.values() if v.get("verdict") == "SUPPORT"
        )
        total_count = len(results)

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
            "framework": "Newton (flat space + potential)",
            "upgrade_condition": "If all 3 hypotheses pass, consider Einstein upgrade",
        }

        return results