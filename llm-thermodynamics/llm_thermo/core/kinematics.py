import torch
from typing import Optional


class KinematicsExtractor:
    """Extract velocity and acceleration from hidden state trajectories.

    Supports three velocity extraction methods:
    - raw_diff: Forward difference (no smoothing, most faithful to discrete dynamics)
    - central_diff: Central difference (higher accuracy, no smoothing)
    - sg: Savitzky-Golay filter (smooths high-frequency jumps, may suppress causal signals)

    WARNING: SG filtering can artificially suppress effective rank.
    See v14b experiment: sg(w=11) reduces rank to 55-78% of raw_diff.
    For faithful dynamics analysis, prefer raw_diff or central_diff.
    """

    def __init__(self, method: str = "raw_diff", sg_window: int = 11, sg_polyorder: int = 3):
        if method not in ("raw_diff", "central_diff", "sg"):
            raise ValueError(f"Unknown method: {method}. Use 'raw_diff', 'central_diff', or 'sg'.")
        if method == "sg" and sg_window % 2 == 0:
            raise ValueError(f"SG window must be odd, got {sg_window}")
        if method == "sg" and sg_polyorder >= sg_window:
            raise ValueError(f"SG polyorder ({sg_polyorder}) must be < window ({sg_window})")
        self.method = method
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder

    def extract_velocity(self, h: torch.Tensor) -> torch.Tensor:
        """Extract velocity from hidden state sequence.

        Args:
            h: Hidden states [T, D] or [B, T, D]

        Returns:
            Velocity tensor, same shape as input (last position padded)
        """
        if h.dim() == 2:
            return self._extract_velocity_single(h)
        elif h.dim() == 3:
            results = []
            for i in range(h.size(0)):
                results.append(self._extract_velocity_single(h[i]))
            return torch.stack(results)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {h.dim()}D")

    def _extract_velocity_single(self, h: torch.Tensor) -> torch.Tensor:
        T, D = h.shape
        vel = torch.zeros_like(h)

        if self.method == "raw_diff":
            vel[:-1] = h[1:] - h[:-1]
            vel[-1] = vel[-2]

        elif self.method == "central_diff":
            vel[1:-1] = (h[2:] - h[:-2]) / 2.0
            vel[0] = h[1] - h[0]
            vel[-1] = h[-1] - h[-2]

        elif self.method == "sg":
            try:
                from scipy.signal import savgol_filter
            except ImportError:
                raise ImportError("scipy is required for SG filtering: pip install scipy")
            h_np = h.numpy()
            vel_np = torch.zeros_like(h).numpy()
            window = min(self.sg_window, T)
            if window % 2 == 0:
                window -= 1
            if window < 5:
                vel[:-1] = h[1:] - h[:-1]
                vel[-1] = vel[-2]
                return vel
            polyorder = min(self.sg_polyorder, window - 1)
            for d in range(D):
                vel_np[:, d] = savgol_filter(h_np[:, d], window, polyorder, deriv=1, delta=1.0)
            vel = torch.from_numpy(vel_np)

        return vel

    def extract_acceleration(self, h: torch.Tensor) -> torch.Tensor:
        """Extract acceleration from hidden state sequence.

        Uses second-order finite difference: a_t = h_{t+2} - 2*h_{t+1} + h_t
        """
        if h.dim() == 2:
            vel = self._extract_velocity_single(h)
            acc = torch.zeros_like(h)
            if vel.size(0) >= 3:
                acc[:-2] = vel[1:-1] - vel[:-2]
                acc[-2:] = acc[-3]
            return acc
        elif h.dim() == 3:
            results = []
            for i in range(h.size(0)):
                results.append(self.extract_acceleration(h[i]))
            return torch.stack(results)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {h.dim()}D")

    @staticmethod
    def compute_effective_rank(vel: torch.Tensor, threshold: float = 0.95) -> int:
        """Compute effective rank of velocity matrix using cumulative variance ratio.

        Args:
            vel: Velocity matrix [T, D]
            threshold: Cumulative variance ratio threshold (default 0.95)

        Returns:
            Effective rank (number of singular values needed to explain threshold% variance)
        """
        if vel.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {vel.dim()}D")
        U, S, Vt = torch.linalg.svd(vel, full_matrices=False)
        S2 = S ** 2
        total = S2.sum()
        if total < 1e-10:
            return 0
        cum = torch.cumsum(S2, dim=0) / total
        rank = int(torch.searchsorted(cum, threshold)) + 1
        return min(rank, vel.size(1))