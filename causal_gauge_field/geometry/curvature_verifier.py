import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .metric_tensor import CausalMetricTensor
from .gauge_connection import GaugeConnection


class CurvatureVerifier(nn.Module):
    """
    曲率验证器：验证 Tr(F²)≈0 对合法路径，Tr(F²)显著非零对非法路径。
    
    核心验证：
    1. 因果曲率与生成质量负相关
    2. 合法转移的d_G² < 非法转移的d_G²
    3. 规范不变性：深层因果结构在群G下守恒
    """
    def __init__(self, d_model: int, num_causal_types: int = 3):
        super().__init__()
        self.d_model = d_model
        self.metric = CausalMetricTensor(d_model, num_causal_types)
        self.gauge = GaugeConnection(d_model, lie_dim=num_causal_types)
        self.legal_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def compute_causal_curvature_proxy(
        self, h_t: torch.Tensor, h_t1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        G = self.metric(h_t)
        d_g_sq = self.metric.compute_geodesic_distance_sq(h_t, h_t1, G)
        F_norm_sq = self.gauge.compute_field_strength_norm_sq(h_t, h_t1)
        combined = h_t + self.gauge.scale * self.gauge.compute_field_strength(h_t, h_t1)
        legal_prob = self.legal_classifier(torch.cat([h_t, h_t1], dim=-1)).squeeze(-1)
        F_tilde = 1.0 - legal_prob
        return {
            "d_g_squared": d_g_sq,
            "F_norm_squared": F_norm_sq,
            "F_tilde": F_tilde,
            "legal_probability": legal_prob,
            "type_contributions": self.metric.get_type_contributions(h_t),
        }

    def verify_geodesic_property(
        self,
        h_t: torch.Tensor,
        h_t1_pos: torch.Tensor,
        h_t1_neg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        G = self.metric(h_t)
        d_pos = self.metric.compute_geodesic_distance_sq(h_t, h_t1_pos, G)
        d_neg = self.metric.compute_geodesic_distance_sq(h_t, h_t1_neg, G)
        F_pos = self.gauge.compute_field_strength_norm_sq(h_t, h_t1_pos)
        F_neg = self.gauge.compute_field_strength_norm_sq(h_t, h_t1_neg)
        margin = d_neg - d_pos
        return {
            "d_pos": d_pos,
            "d_neg": d_neg,
            "F_pos": F_pos,
            "F_neg": F_neg,
            "margin": margin,
            "geodesic_satisfied": (margin > 0).float(),
        }

    def verify_gauge_invariance(
        self,
        h: torch.Tensor,
        g_transform: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if g_transform is None:
            g_transform = torch.randn_like(h) * 0.01
        h_transformed = h + g_transform
        G_original = self.metric(h)
        G_transformed = self.metric(h_transformed)
        tr_G_orig = torch.diagonal(G_original, dim1=-2, dim2=-1).sum(dim=-1)
        tr_G_trans = torch.diagonal(G_transformed, dim1=-2, dim2=-1).sum(dim=-1)
        invariance_error = (tr_G_orig - tr_G_trans).abs()
        return {
            "invariance_error": invariance_error,
            "gauge_invariant": (invariance_error < 0.1 * tr_G_orig.abs()).float(),
        }

    def compute_trajectory_curvature(
        self,
        hidden_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, d = hidden_trajectory.shape
        if seq_len < 3:
            return {
                "mean_curvature": torch.zeros(batch_size, device=hidden_trajectory.device),
                "max_curvature": torch.zeros(batch_size, device=hidden_trajectory.device),
                "wilson_loop": torch.zeros(batch_size, device=hidden_trajectory.device),
            }
        deltas = hidden_trajectory[:, 1:, :] - hidden_trajectory[:, :-1, :]
        curvatures = []
        for t in range(1, deltas.size(1)):
            diff = deltas[:, t, :] - deltas[:, t - 1, :]
            norm_product = deltas[:, t, :].norm(dim=-1) * deltas[:, t - 1, :].norm(dim=-1)
            kappa = torch.where(
                norm_product > 1e-8,
                diff.norm(dim=-1) / norm_product,
                torch.zeros(batch_size, device=hidden_trajectory.device),
            )
            curvatures.append(kappa)
        kappa_stack = torch.stack(curvatures, dim=1)
        wilson = self.gauge.compute_wilson_loop(hidden_trajectory)
        return {
            "mean_curvature": kappa_stack.mean(dim=1),
            "max_curvature": kappa_stack.max(dim=1)[0],
            "wilson_loop": wilson,
        }

    def full_diagnosis(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        is_legal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        curvature_proxy = self.compute_causal_curvature_proxy(h_t, h_t1)
        result = {**curvature_proxy}
        if is_legal is not None:
            legal_mask = (is_legal == 1)
            if legal_mask.any() and (~legal_mask).any():
                h_t_pos = h_t[legal_mask]
                h_t1_pos = h_t1[legal_mask]
                h_t_neg = h_t[~legal_mask]
                h_t1_neg = h_t1[~legal_mask]
                min_len = min(h_t_pos.size(0), h_t_neg.size(0))
                if min_len > 0:
                    geodesic = self.verify_geodesic_property(
                        h_t_pos[:min_len],
                        h_t1_pos[:min_len],
                        h_t1_neg[:min_len],
                    )
                    result["geodesic_verification"] = geodesic
        return result