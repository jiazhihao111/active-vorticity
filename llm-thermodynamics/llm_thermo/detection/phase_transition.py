from collections import deque
from enum import Enum
from typing import Optional, Dict

import numpy as np
import torch

from ..core.thermodynamics import ThermodynamicEngine


class AlertLevel(str, Enum):
    WARMUP = "WARMUP"
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class HallucinationDetector:
    """Zero-shot real-time hallucination detector based on dynamical phase transition.

    Monitors P_c/P_raw ratio via sliding window. When the constraint force
    work ratio exceeds threshold for consecutive steps, the model is likely
    generating content that violates the causal manifold.

    Validated gradient: pos (0.13%) < scr (2.25%) < rnd (6.21%)

    IMPORTANT: P_c/P_raw must be computed per-token (not batch-averaged).
    Batch averaging produces ~0.84 artifact; per-token gives correct ~0.12.
    """

    def __init__(
        self,
        engine: ThermodynamicEngine,
        window_size: int = 5,
        threshold_ratio: float = 0.05,
        consecutive_hits: int = 3,
        enable_vel_norm: bool = False,
        vel_norm_threshold: Optional[float] = None,
    ):
        self.engine = engine
        self.window_size = window_size
        self.threshold = threshold_ratio
        self.consecutive_hits = consecutive_hits
        self.enable_vel_norm = enable_vel_norm
        self.vel_norm_threshold = vel_norm_threshold
        self.h_history: deque = deque(maxlen=3)
        self.ratio_history: deque = deque(maxlen=window_size)
        self.vel_norm_history: deque = deque(maxlen=window_size)
        self.alert_counter = 0
        self._step_count = 0

    def reset(self):
        self.h_history.clear()
        self.ratio_history.clear()
        self.vel_norm_history.clear()
        self.alert_counter = 0
        self._step_count = 0

    def step(self, h_t: torch.Tensor) -> dict:
        """Call once per generated token.

        Args:
            h_t: Current hidden state [B, D] or [D]

        Returns:
            Detection result dict with keys:
            - is_hallucinating: bool
            - smooth_ratio: float (window-averaged P_c/P_raw)
            - raw_ratio: float (current step P_c/P_raw)
            - alert_level: AlertLevel
            - step: int
            - vel_norm: float (velocity norm, if enabled)
            - fc_vel_cosine: float (F_c-v cosine similarity)
        """
        self._step_count += 1

        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)

        self.h_history.append(h_t)

        if len(self.h_history) < 3:
            return {
                "is_hallucinating": False,
                "smooth_ratio": 0.0,
                "raw_ratio": 0.0,
                "alert_level": AlertLevel.WARMUP,
                "step": self._step_count,
                "vel_norm": 0.0,
                "fc_vel_cosine": 0.0,
            }

        h_prev2, h_prev, h_curr = self.h_history[0], self.h_history[1], self.h_history[2]
        P_raw, _, P_c = self.engine.compute_work_and_power(h_curr, h_prev, h_prev2)
        ratio = self.engine.compute_constraint_ratio(P_c, P_raw).item()

        v_t = h_curr - h_prev
        vel_norm = float(v_t.norm(dim=-1).mean().item())

        fc_vel_cos = self.engine.compute_fc_vel_cosine(h_curr, h_prev, h_prev2).item()

        self.ratio_history.append(ratio)
        self.vel_norm_history.append(vel_norm)
        smooth_ratio = float(np.mean(self.ratio_history))

        alert_triggers = 0
        if smooth_ratio > self.threshold:
            alert_triggers += 1

        if self.enable_vel_norm and self.vel_norm_threshold is not None:
            smooth_vel = float(np.mean(self.vel_norm_history))
            if smooth_vel < self.vel_norm_threshold:
                alert_triggers += 1

        if alert_triggers > 0:
            self.alert_counter += 1
        else:
            self.alert_counter = max(0, self.alert_counter - 1)

        is_hallucinating = self.alert_counter >= self.consecutive_hits

        if is_hallucinating:
            alert_level = AlertLevel.CRITICAL
        elif self.alert_counter > 0:
            alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.NORMAL

        return {
            "is_hallucinating": is_hallucinating,
            "smooth_ratio": smooth_ratio,
            "raw_ratio": ratio,
            "alert_level": alert_level,
            "step": self._step_count,
            "vel_norm": vel_norm,
            "fc_vel_cosine": fc_vel_cos,
        }