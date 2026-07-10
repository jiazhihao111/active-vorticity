import torch
from typing import Tuple, Optional, Dict
from .kinematics import KinematicsExtractor


class ThermodynamicEngine:
    """Non-equilibrium thermodynamics computation engine for LLM hidden states.

    Based on GUIT-TRT v9.2 motion equation:
        m*a + gamma*v = alpha*v + F_c + xi

    Core identity:
        P_raw = F_res . v = (m*a + gamma*v) . v
        P_active = alpha * ||v||^2
        P_c = P_raw - P_active  (constraint force power, ~0 for valid trajectories)

    Validated on:
        - NPNW-MVE (32-dim): alpha* ~ 0.33
        - MiniCPM5-1B (1536-dim): alpha* ~ 1.46
        - Qwen2.5-7B (3584-dim): alpha* ~ 1.41
    """

    def __init__(
        self,
        alpha_star: float = 1.46,
        gamma: float = 0.01,
        mass: float = 1.0,
        velocity_method: str = "raw_diff",
    ):
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.mass = mass
        self._kinematics = KinematicsExtractor(method=velocity_method)

    def compute_work_and_power(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute single-step dynamical work and power.

        Args:
            h_curr: Current hidden state [B, D] (time step t)
            h_prev: Previous hidden state [B, D] (time step t-1)
            h_prev2: Two-steps-ago hidden state [B, D] (time step t-2)

        Returns:
            (P_raw, P_active, P_c) each of shape [B, 1]
        """
        v_t = h_curr - h_prev
        a_t = h_curr - 2 * h_prev + h_prev2

        F_res = self.mass * a_t + self.gamma * v_t

        P_raw = torch.sum(F_res * v_t, dim=-1, keepdim=True)
        P_active = self.alpha_star * torch.sum(v_t * v_t, dim=-1, keepdim=True)
        P_c = P_raw - P_active

        return P_raw, P_active, P_c

    def compute_constraint_ratio(
        self,
        P_c: torch.Tensor,
        P_raw: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute constraint force work ratio |P_c| / |P_raw|.

        This is the core hallucination detection signal.
        Validated gradient: pos (0.13%) < scr (2.25%) < rnd (6.21%)

        WARNING: This computes ratio per-element. For trajectory-level
        P_c/P_raw, use compute_per_token_ratio() which correctly handles
        the batch-vs-per-token methodology pitfall. Batch averaging
        produces P_c/P_raw ~ 0.84 (artifact); per-token averaging
        gives the correct value ~ 0.12. See Section 4.9 of the paper.
        """
        return torch.abs(P_c) / (torch.abs(P_raw) + eps)

    def compute_per_token_ratio(
        self,
        hidden_states: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict:
        """Compute P_c/P_raw using the CORRECT per-token method.

        This is the only correct way to compute trajectory-level P_c/P_raw.
        The batch method (ratio of averages) produces ~0.84 artifact because
        positive and negative P_c partially cancel, making |<P_c>| / |<P_raw>|
        artificially large.

        Per-token method: ratio_t = |P_c(t)| / |P_raw(t)| for each t,
        then mean(ratio_t). This gives the correct value ~0.12.

        Args:
            hidden_states: [T, D] trajectory

        Returns:
            Dict with per_token_ratio (correct), batch_ratio (artifact),
            and per-step ratios
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected 2D [T, D], got {hidden_states.dim()}D")
        T, D = hidden_states.shape
        if T < 4:
            return {"error": "Need T >= 4"}

        vel = hidden_states[1:] - hidden_states[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        acc = acc[:min_t]
        v_for = v_for[:min_t]

        F_res = self.mass * acc + self.gamma * v_for
        P_raw = (F_res * v_for).sum(dim=-1)
        P_active = self.alpha_star * (v_for * v_for).sum(dim=-1)
        P_c = P_raw - P_active

        per_token_ratios = torch.abs(P_c) / (torch.abs(P_raw) + eps)
        per_token_mean = float(per_token_ratios.mean().item())

        batch_ratio = float(P_c.abs().mean().item() / (P_raw.abs().mean().item() + eps))

        return {
            "per_token_ratio": per_token_mean,
            "batch_ratio_artifact": batch_ratio,
            "per_step_ratios": per_token_ratios,
            "P_c_mean": float(P_c.mean().item()),
            "P_raw_mean": float(P_raw.abs().mean().item()),
            "method_warning": "batch_ratio_artifact is INCORRECT (inflated by sign cancellation); use per_token_ratio",
        }

    def compute_fc_vel_cosine(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute cosine similarity between F_c and velocity.

        For valid trajectories, F_c . v ~ 0 (ideal constraint).
        Validated: cos ~ 0.015 for MiniCPM5-1B pos trajectories.
        """
        v_t = h_curr - h_prev
        a_t = h_curr - 2 * h_prev + h_prev2
        F_c = self.mass * a_t + (self.gamma - self.alpha_star) * v_t

        fc_dot_v = torch.sum(F_c * v_t, dim=-1, keepdim=True)
        fc_norm = torch.norm(F_c, dim=-1, keepdim=True)
        v_norm = torch.norm(v_t, dim=-1, keepdim=True)

        return fc_dot_v / (fc_norm * v_norm + eps)

    def compute_trajectory_metrics(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute full trajectory metrics from hidden state sequence.

        Args:
            hidden_states: [T, D] hidden state sequence

        Returns:
            Dictionary of trajectory-level metrics
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected 2D tensor [T, D], got {hidden_states.dim()}D")

        T, D = hidden_states.shape
        if T < 4:
            return {"error": "Trajectory too short (need T >= 4)"}

        vel = hidden_states[1:] - hidden_states[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))

        F_res = acc[:min_t] + self.gamma * v_for[:min_t]
        F_c = acc[:min_t] + (self.gamma - self.alpha_star) * v_for[:min_t]

        P_raw = (F_res * v_for[:min_t]).sum(dim=-1)
        P_active = (v_for[:min_t] * v_for[:min_t]).sum(dim=-1)
        P_c = (F_c * v_for[:min_t]).sum(dim=-1)

        pc_raw_ratio = float(P_c.abs().mean() / (P_raw.abs().mean() + 1e-8))
        if min_t > 0:
            fc_dot_v = (F_c * v_for[:min_t]).sum(dim=-1)
            fc_norms = F_c.norm(dim=-1) * v_for[:min_t].norm(dim=-1) + 1e-8
            fc_vel_cos = float((fc_dot_v / fc_norms).mean())
        else:
            fc_vel_cos = 0.0

        vel_norm_mean = float(vel.norm(dim=-1).mean())
        p_active_ratio = float(P_active.abs().mean() / (P_raw.abs().mean() + 1e-8))

        return {
            "P_c_mean": float(P_c.mean()),
            "P_c_abs_mean": float(P_c.abs().mean()),
            "P_raw_mean": float(P_raw.abs().mean()),
            "P_active_ratio": p_active_ratio,
            "P_c_P_raw_ratio": pc_raw_ratio,
            "fc_vel_cosine": fc_vel_cos,
            "vel_norm_mean": vel_norm_mean,
            "alpha_star": self.alpha_star,
            "trajectory_length": T,
            "hidden_dim": D,
        }

    def calibrate_alpha_star(
        self,
        hidden_states_list: list[torch.Tensor],
        gamma: Optional[float] = None,
    ) -> float:
        """Auto-calibrate alpha* from a set of valid (pos) trajectories.

        alpha* = <P_raw> / <P_active> = <(m*a + gamma*v) . v> / <v . v>

        Args:
            hidden_states_list: List of [T, D] tensors from valid trajectories
            gamma: Override gamma for calibration (uses self.gamma if None)

        Returns:
            Calibrated alpha* value
        """
        if gamma is None:
            gamma = self.gamma

        alpha_estimates = []
        for h in hidden_states_list:
            if h.dim() != 2 or h.size(0) < 4:
                continue
            vel = h[1:] - h[:-1]
            acc = vel[1:] - vel[:-1]
            v_for = vel[1:]
            min_t = min(acc.size(0), v_for.size(0))

            F_res = self.mass * acc[:min_t] + gamma * v_for[:min_t]
            P_raw = (F_res * v_for[:min_t]).sum(dim=-1)
            P_active = (v_for[:min_t] * v_for[:min_t]).sum(dim=-1)

            if P_active.abs().mean() > 1e-10:
                alpha_estimates.append(
                    float(P_raw.mean().item() / P_active.mean().item())
                )

        if not alpha_estimates:
            return self.alpha_star

        import numpy as np
        self.alpha_star = float(np.mean(alpha_estimates))
        return self.alpha_star