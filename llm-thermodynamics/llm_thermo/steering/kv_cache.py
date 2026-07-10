from typing import Optional

import numpy as np
import torch

from ..core.thermodynamics import ThermodynamicEngine


class DynamicKVCacheEvictor:
    """KV Cache compression based on causal backtracking and sub-Riemannian contribution.

    Core innovation: combines sub-Riemannian normal acceleration (physical causal
    turning strength) with attention-weighted causal backtracking (information
    dependency path). This identifies "load-bearing wall" tokens that may not
    produce dramatic turns themselves but are highly depended upon by subsequent
    critical turns.

    WARNING: Only suitable for strong-causality tasks (logical reasoning,
    factual QA, code generation). May misidentify "inspiration nodes" in
    creative writing tasks.
    """

    def __init__(
        self,
        engine: ThermodynamicEngine,
        max_capacity: int = 1024,
        n_sink: int = 4,
        n_recent: int = 10,
    ):
        self.engine = engine
        self.max_capacity = max_capacity
        self.n_sink = n_sink
        self.n_recent = n_recent
        self.causal_scores: list[float] = []

    def reset(self):
        self.causal_scores.clear()

    def update_scores(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ):
        """Compute causal turning strength for the latest token and backpropagate to history.

        Args:
            h_curr: Current hidden state [B, D] or [D]
            h_prev: Previous hidden state [B, D] or [D]
            h_prev2: Two-steps-ago hidden state [B, D] or [D]
            attn_weights: Attention weights from current layer
                [1, num_heads, 1, seq_len] for causal backtracking.
                If None, falls back to instantaneous score only.
        """
        if h_curr.dim() == 1:
            h_curr = h_curr.unsqueeze(0)
            h_prev = h_prev.unsqueeze(0)
            h_prev2 = h_prev2.unsqueeze(0)

        v_t = h_curr - h_prev
        a_t = h_curr - 2 * h_prev + h_prev2

        v_norm_sq = torch.sum(v_t * v_t, dim=-1, keepdim=True) + 1e-8
        a_parallel = (torch.sum(a_t * v_t, dim=-1, keepdim=True) / v_norm_sq) * v_t
        a_perpendicular = a_t - a_parallel

        current_causal_impact = float(torch.norm(a_perpendicular, dim=-1).mean().item())
        self.causal_scores.append(current_causal_impact)

        if attn_weights is not None and len(self.causal_scores) > 1:
            attn_to_history = attn_weights[0, :, 0, :-1].mean(dim=0).cpu().numpy()
            attn_sum = attn_to_history.sum()
            if attn_sum > 1e-8:
                attn_to_history = attn_to_history / attn_sum
            for i in range(len(self.causal_scores) - 1):
                self.causal_scores[i] += float(attn_to_history[i]) * current_causal_impact

    def get_eviction_mask(self) -> list[bool]:
        """Return eviction mask. True = evict this position.

        Protects sink tokens (first n_sink) and recent tokens (last n_recent).
        Evicts lowest-causal-score tokens from the middle region.
        """
        n_total = len(self.causal_scores)
        if n_total <= self.max_capacity:
            return [False] * n_total

        n_to_evict = n_total - self.max_capacity

        evictable_start = self.n_sink
        evictable_end = max(evictable_start, n_total - self.n_recent)

        if evictable_end <= evictable_start:
            return [False] * n_total

        evictable_scores = np.array(self.causal_scores[evictable_start:evictable_end])
        sorted_indices = np.argsort(evictable_scores)
        evict_local_indices = set(sorted_indices[:n_to_evict].tolist())
        evict_global_indices = {idx + evictable_start for idx in evict_local_indices}

        return [i in evict_global_indices for i in range(n_total)]

    def get_eviction_positions(self) -> list[int]:
        """Return list of positions to evict."""
        mask = self.get_eviction_mask()
        return [i for i, evict in enumerate(mask) if evict]
