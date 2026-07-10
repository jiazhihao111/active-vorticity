import torch

from ..core.thermodynamics import ThermodynamicEngine


class AffineProjector:
    """Project hidden states onto the affine constraint manifold during inference.

    When P_c/P_raw exceeds threshold, this projector forces the hidden state
    back onto the constraint manifold by removing the constraint force component
    perpendicular to velocity.

    The constraint manifold is defined by: P_c = F_c . v ≈ 0
    """

    def __init__(
        self,
        engine: ThermodynamicEngine,
        project_threshold: float = 0.05,
        strength: float = 1.0,
    ):
        self.engine = engine
        self.project_threshold = project_threshold
        self.strength = strength

    def project(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
    ) -> torch.Tensor:
        """Project h_curr onto the constraint manifold if P_c/P_raw exceeds threshold.

        Args:
            h_curr: Current hidden state [B, D] or [D]
            h_prev: Previous hidden state [B, D] or [D]
            h_prev2: Two-steps-ago hidden state [B, D] or [D]

        Returns:
            Projected h_curr (unchanged if below threshold)
        """
        squeeze = False
        if h_curr.dim() == 1:
            squeeze = True
            h_curr = h_curr.unsqueeze(0)
            h_prev = h_prev.unsqueeze(0)
            h_prev2 = h_prev2.unsqueeze(0)

        P_raw, _, P_c = self.engine.compute_work_and_power(h_curr, h_prev, h_prev2)
        ratio = self.engine.compute_constraint_ratio(P_c, P_raw)

        needs_projection = ratio > self.project_threshold

        if not needs_projection.any():
            result = h_curr.squeeze(0) if squeeze else h_curr
            return result

        v_t = h_curr - h_prev
        a_t = h_curr - 2 * h_prev + h_prev2
        F_c = self.engine.mass * a_t + (self.engine.gamma - self.engine.alpha_star) * v_t

        v_norm_sq = torch.sum(v_t * v_t, dim=-1, keepdim=True) + 1e-8
        fc_proj_v = (torch.sum(F_c * v_t, dim=-1, keepdim=True) / v_norm_sq) * v_t
        fc_perp = F_c - fc_proj_v

        correction = self.strength * fc_perp
        h_projected = h_curr - correction

        mask = needs_projection.float()
        h_result = h_curr * (1 - mask) + h_projected * mask

        result = h_result.squeeze(0) if squeeze else h_result
        return result