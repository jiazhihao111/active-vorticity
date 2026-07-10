"""DynamicKVCacheEvictor — 基于因果贡献度的动力学 KV Cache 压缩 (CausalKV)。

核心创新: 早期启发式仅用当前 token 瞬时法向加速度, 无法衡量历史 token
对长程因果链的支撑。引入因果回溯 (Causal Backtracking): 利用注意力权重
将当前 token 的"因果转折强度"反向分配给历史 token —— 被关键转折高度
依赖的隐性承重墙 token 免于淘汰。

对比基线 (benchmark 中实现): H2O (注意力分数), Random。
"""

from typing import List, Optional
import torch
import numpy as np


class DynamicKVCacheEvictor:
    def __init__(self, max_capacity: int = 1024, n_sink: int = 4, n_recent: int = 10):
        self.max_capacity = max_capacity
        self.n_sink = n_sink
        self.n_recent = n_recent
        self.causal_scores: List[float] = []

    def update_scores(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        h_prev2: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ):
        """计算最新 token 的因果转折强度 (法向加速度), 并回溯更新历史分数。

        attn_weights: [1, heads, 1, seq_len] 当前层对历史 token 的注意力
        """
        v = h_curr - h_prev
        a = h_curr - 2 * h_prev + h_prev2

        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True) + 1e-8
        a_parallel = (torch.sum(a * v, dim=-1, keepdim=True) / v_norm_sq) * v
        a_perp = a - a_parallel

        impact = float(torch.norm(a_perp, dim=-1).item())
        self.causal_scores.append(impact)

        if attn_weights is not None and len(self.causal_scores) > 1:
            aw = attn_weights[0, :, 0, :-1].mean(dim=0).cpu().numpy()
            s = aw.sum()
            if s > 1e-8:
                aw = aw / s
            for i in range(len(self.causal_scores) - 1):
                self.causal_scores[i] += aw[i] * impact

    def get_eviction_mask(self) -> List[bool]:
        seq = len(self.causal_scores)
        if seq <= self.max_capacity:
            return [False] * seq
        evictable = self.causal_scores[self.n_sink: -self.n_recent]
        n_evict = seq - self.max_capacity
        order = np.argsort(evictable)
        evict_local = set(order[:n_evict].tolist())
        return [i in evict_local for i in range(seq)]
