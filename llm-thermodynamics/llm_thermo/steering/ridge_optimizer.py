import torch
from typing import Tuple, Dict, Optional, List
from collections import deque

from ..core.thermodynamics import ThermodynamicEngine


class RidgeOptimizer:
    """Auto ridge extraction and thermodynamic optimization engine.

    Based on GUIT-TRT v9.2: affine-nonholonomic hybrid constraint principle.
    Three-phase pipeline:
    1. Auto Ridge Finding: SVD on calibration hidden states to find affine skeleton
    2. Auto Alpha Calibration: fit alpha* from consecutive trajectory steps
    3. Phase Transition Monitoring: real-time P_c/P_raw with adaptive fallback

    Key findings from Phase 1 v9 experiments:
    - Decode SVD basis (r~63) is far more precise than prefill basis (r~318)
    - SVD basis is prompt-dependent; cross-prompt basis completely fails
    - Replacing h_t with h_rec causes greedy decoding loops (monitor mode preferred)
    - r=128 achieves 100% top-1 match, KL=0.14 on same-prompt online verification

    Usage:
        optimizer = RidgeOptimizer(variance_threshold=0.95)
        ridge_info = optimizer.auto_find_ridge(hidden_states)  # [N, D]
        alpha_info = optimizer.auto_calibrate_alpha([h_t2, h_t1, h_t])
        h_recon, metrics = optimizer.step_decode(h_curr)
    """

    def __init__(
        self,
        variance_threshold: float = 0.95,
        gamma: float = 0.01,
        pc_ratio_threshold: float = 0.05,
        fallback_window: int = 3,
        max_ridge_dim: Optional[int] = None,
    ):
        self.var_threshold = variance_threshold
        self.gamma = gamma
        self.pc_thresh = pc_ratio_threshold
        self.fallback_window = fallback_window
        self.max_ridge_dim = max_ridge_dim

        self.ridge_basis: Optional[torch.Tensor] = None
        self.ridge_mean: Optional[torch.Tensor] = None
        self.r: int = 0
        self.ambient_dim: int = 0
        self.alpha_star: float = 1.0

        self.h_history: deque = deque(maxlen=3)
        self.alert_counter = 0
        self.is_calibrated = False
        self._step_count = 0
        self._fallback_count = 0
        self._total_count = 0

    @torch.no_grad()
    def auto_find_ridge(self, hidden_states: torch.Tensor) -> Dict:
        """Phase 1: Auto find affine subspace ridge via SVD.

        IMPORTANT: Use decode-phase hidden states for calibration, not prefill.
        Prefill basis (r~318) is too broad; decode basis (r~63) is precise.

        Args:
            hidden_states: [N, D] calibration hidden states (prefer decode-phase)

        Returns:
            Dict with ridge_dim, ambient_dim, compression_ratio, explained_variance
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        N, D = hidden_states.shape
        self.ambient_dim = D

        self.ridge_mean = hidden_states.mean(dim=0)
        h_centered = hidden_states - self.ridge_mean

        _, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)

        explained_var = (S ** 2) / (S ** 2).sum()
        cum_var = torch.cumsum(explained_var, dim=0)

        self.r = int((cum_var < self.var_threshold).sum().item()) + 1
        self.r = min(self.r, D)
        if self.max_ridge_dim is not None:
            self.r = min(self.r, self.max_ridge_dim)

        self.ridge_basis = Vh[: self.r]

        self.h_history.clear()
        self.alert_counter = 0
        self._step_count = 0
        self._fallback_count = 0
        self._total_count = 0

        return {
            "ridge_dim": self.r,
            "ambient_dim": D,
            "compression_ratio": 1.0 - self.r / D,
            "explained_variance": cum_var[self.r - 1].item(),
        }

    @torch.no_grad()
    def auto_calibrate_alpha(self, hidden_states_seq: List[torch.Tensor]) -> Dict:
        """Phase 2: Auto calibrate active driving force coefficient alpha*.

        alpha* = <P_raw> / <P_active_base> from consecutive trajectory steps.

        Args:
            hidden_states_seq: At least 3 consecutive hidden states [D] or [1, D]

        Returns:
            Dict with alpha_star
        """
        if len(hidden_states_seq) < 3:
            raise ValueError("Need at least 3 steps to calibrate alpha*")

        h_t2 = hidden_states_seq[-3].detach().float().squeeze()
        h_t1 = hidden_states_seq[-2].detach().float().squeeze()
        h_t = hidden_states_seq[-1].detach().float().squeeze()

        v_t = h_t - h_t1
        a_t = v_t - (h_t1 - h_t2)

        F_res = a_t + self.gamma * v_t
        P_raw = torch.sum(F_res * v_t).item()
        P_active_base = torch.sum(v_t * v_t).item()

        self.alpha_star = P_raw / (P_active_base + 1e-8)
        self.is_calibrated = True

        return {"alpha_star": self.alpha_star}

    @torch.no_grad()
    def step_decode(self, h_curr: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Phase 3: Low-dim projection + reconstruction + phase transition monitoring.

        IMPORTANT: This returns (h_recon, metrics) but does NOT modify the forward
        pass. Use in monitor mode to compare logits, or store only low-dim
        coordinates for compression. Replacing h_t with h_rec in the forward pass
        causes greedy decoding repetition loops (v9 experiment).

        Args:
            h_curr: [1, D] or [D] current decode step hidden state

        Returns:
            (h_reconstructed, metrics_dict)
        """
        if self.ridge_basis is None:
            raise RuntimeError("Ridge not found. Call auto_find_ridge first.")

        h_flat = h_curr.detach().float().squeeze()

        h_low = (h_flat - self.ridge_mean) @ self.ridge_basis.T
        h_recon = h_low @ self.ridge_basis + self.ridge_mean

        self.h_history.append(h_recon)
        self._step_count += 1
        self._total_count += 1

        pc_ratio = 0.0
        status = "WARMUP"
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                h_flat.unsqueeze(0), h_recon.unsqueeze(0), dim=-1
            ).item()
        )
        mse = float(((h_flat - h_recon) ** 2).mean().item())

        if len(self.h_history) == 3 and self.is_calibrated:
            h_t = self.h_history[2]
            h_t1 = self.h_history[1]
            h_t2 = self.h_history[0]

            v_t = h_t - h_t1
            a_t = v_t - (h_t1 - h_t2)

            F_res = a_t + self.gamma * v_t
            P_raw = torch.sum(F_res * v_t).item()
            P_active = self.alpha_star * torch.sum(v_t * v_t).item()
            P_c = P_raw - P_active

            pc_ratio = abs(P_c) / (abs(P_raw) + 1e-8)

            if pc_ratio > self.pc_thresh:
                self.alert_counter += 1
            else:
                self.alert_counter = max(0, self.alert_counter - 1)

            if self.alert_counter >= self.fallback_window:
                status = "PHASE_TRANSITION"
                self._fallback_count += 1
            else:
                status = "IN_AFFINE_MANIFOLD"

        h_out = h_recon.unsqueeze(0) if h_curr.dim() > 1 else h_recon

        return h_out, {
            "pc_ratio": pc_ratio,
            "status": status,
            "alpha_star": self.alpha_star,
            "cosine_sim": cos_sim,
            "mse": mse,
            "step": self._step_count,
            "low_dim_coords": h_low,
        }

    def get_low_dim_coords(self, h_curr: torch.Tensor) -> torch.Tensor:
        """Project hidden state to low-dim coordinates only (no reconstruction).

        For storage compression: store r coordinates instead of D values.
        Reconstruction can be done later with reconstruct_from_coords().

        Args:
            h_curr: [D] or [1, D] hidden state

        Returns:
            [r] low-dimensional coordinates
        """
        if self.ridge_basis is None:
            raise RuntimeError("Ridge not found. Call auto_find_ridge first.")
        h_flat = h_curr.detach().float().squeeze()
        return (h_flat - self.ridge_mean) @ self.ridge_basis.T

    def reconstruct_from_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Reconstruct hidden state from low-dim coordinates.

        Args:
            coords: [r] low-dimensional coordinates

        Returns:
            [D] reconstructed hidden state
        """
        if self.ridge_basis is None:
            raise RuntimeError("Ridge not found. Call auto_find_ridge first.")
        return coords @ self.ridge_basis + self.ridge_mean

    @property
    def fallback_rate(self) -> float:
        """Ratio of fallback triggers to total decode steps."""
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count

    def should_reset_ridge(self, threshold: float = 0.3) -> bool:
        """Check if ridge should be reset due to high fallback rate (OOD adaptation).

        Args:
            threshold: Fallback rate threshold for triggering reset

        Returns:
            True if ridge should be reset
        """
        return self._total_count >= 10 and self.fallback_rate > threshold

    def reset_ridge(self):
        """Online manifold adaptation: reset ridge when encountering severe OOD."""
        self.ridge_basis = None
        self.ridge_mean = None
        self.r = 0
        self.h_history.clear()
        self.alert_counter = 0
        self.is_calibrated = False
        self._step_count = 0
        self._fallback_count = 0
        self._total_count = 0