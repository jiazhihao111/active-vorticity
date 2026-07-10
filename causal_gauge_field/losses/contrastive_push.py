"""
InfoNCE 对比推离损失 — exp10 核心替换方案
==========================================

替换 exp7-9 中失效的铰链损失 (Cohen's d=0.007)，采用温度缩放的
InfoNCE 风格对比损失，实现 H-push 的动力学推离。

理论基础 (§6.4, GUIT-TRT 融合理论):
    合法叙事轨迹应沿流形 F* 附近运动（几何必然点），非法叙事
    应被推离该区域。InfoNCE 通过批量负例提供更丰富的推离信号，
    避免铰链损失的"边际坍塌"。

公式:
    L_push = -log[ exp(sim(h_pos, h_anchor) / τ) /
                   Σ_{j∈batch} exp(sim(h_j, h_anchor) / τ) ]

    其中 sim(·,·) 采用度量距离 d_G² = Δh^T G Δh 的负值，
    或直接在基空间中使用余弦相似度。

对比旧铰链损失的改进:
    - 旧: L_hinge = relu(margin - ‖W_neg - I‖)
      → 仅惩罚"不够扭曲"的负例，无正例牵引力
    - 新: L_InfoNCE = 显式拉近正例 + 批量推离所有负例
      → 温度 τ 控制推离强度，避免梯度消失

用法:
    from causal_gauge_field.losses.contrastive_push import InfoNCEContrastivePush
    push_loss = InfoNCEContrastivePush(temperature=0.1)
    loss = push_loss(h_anchor, h_pos, h_neg_pool, G_composed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InfoNCEContrastivePush(nn.Module):
    """InfoNCE 对比推离损失 — 替代铰链损失。

    核心设计:
    1. 正例 (合法叙事) 被拉向流形 F* → 通过 anchor-positive 相似度最大化
    2. 负例 (非法/破缺叙事) 被推离 F* → 通过 anchor-negative 相似度最小化
    3. 温度 τ 控制推离强度：低 τ → 强推离，高 τ → 软推离

    参数:
        temperature: InfoNCE 温度 (默认 0.1，对应强推离)
        distance_metric: "cosine" | "euclidean" | "geodesic"
        normalize: 是否在计算相似度前做 L2 归一化
    """

    def __init__(
        self,
        temperature: float = 0.1,
        distance_metric: str = "cosine",
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric
        self.normalize = normalize
        # 可学习温度 (如果不需要固定)
        self.log_tau = nn.Parameter(torch.tensor(float(temperature)).log())

    def _pairwise_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        G_composed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 pairwise 相似度矩阵。

        Args:
            a: [N, D] 或 [B, N, D] anchor 向量
            b: [M, D] 或 [B, M, D] candidate 向量
            G_composed: 可选度量张量

        Returns:
            sim: [N, M] 或 [B, N, M] 相似度矩阵
        """
        if self.normalize:
            a = F.normalize(a, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)

        if G_composed is not None:
            # 度量距离: d² = Δh^T G Δh, sim = -d²
            if a.dim() == 2:
                # [N, D] x [N, D, D] x [M, D] -> [N, M]
                diff = a.unsqueeze(1) - b.unsqueeze(0)  # [N, M, D]
                d2 = torch.einsum('nmd,ndk,nmk->nm', diff, G_composed, diff)
            else:
                # [B, N, D] x [B, N, D, D] x [B, M, D] -> [B, N, M]
                diff = a.unsqueeze(2) - b.unsqueeze(1)  # [B, N, M, D]
                d2 = torch.einsum('bnmd,bndk,bnmk->bnm', diff, G_composed, diff)
            sim = -d2  # 距离越小 → 相似度越高
        elif self.distance_metric == "cosine":
            # 余弦相似度直接作为 sim
            sim = torch.matmul(a, b.transpose(-2, -1))  # [N, M] 或 [B, N, M]
        elif self.distance_metric == "euclidean":
            # 负欧氏距离平方作为 sim
            if a.dim() == 2:
                d2 = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0) ** 2
            else:
                # batched: [B, N, D] vs [B, M, D]
                d2 = torch.cdist(a.reshape(-1, a.size(-1)),
                                 b.reshape(-1, b.size(-1))) ** 2
                d2 = d2.reshape(a.size(0), a.size(1), b.size(1))
            sim = -d2
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        return sim

    def forward(
        self,
        h_anchor: torch.Tensor,
        h_pos: torch.Tensor,
        h_neg_pool: torch.Tensor,
        G_composed: Optional[torch.Tensor] = None,
        use_learnable_tau: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """InfoNCE 对比推离前向。

        Args:
            h_anchor: [N, D] 锚点隐状态 (如每段首 token 的 hidden)
            h_pos:    [N, D] 正例隐状态 (合法叙事对应位置)
            h_neg_pool: [N, K, D] 负例池 (K 个非法叙事对应位置)
            G_composed: 可选 [N, D, D] 度量张量
            use_learnable_tau: 是否使用可学习温度

        Returns:
            loss: 标量 InfoNCE 损失
            info: 包含 pos_sim, neg_sim_mean, tau 等诊断信息
        """
        N = h_anchor.size(0)
        tau = self.log_tau.exp().clamp(min=1e-4) if use_learnable_tau else self.temperature

        # 正例相似度: [N, 1]
        pos_sim = self._pairwise_similarity(
            h_anchor.unsqueeze(1), h_pos.unsqueeze(1), G_composed
        ).squeeze(-1).squeeze(-1)  # [N]

        # 批量对比: anchor vs [pos + all_negatives]
        # 将 h_pos 和 h_neg_pool 拼接为对比池
        # h_neg_pool: [N, K, D] → 展平为 [N*K, D] 用于 pairwise
        K = h_neg_pool.size(1)
        h_candidates = torch.cat([
            h_pos.unsqueeze(1),      # [N, 1, D]
            h_neg_pool,              # [N, K, D]
        ], dim=1)                     # [N, 1+K, D]

        # 计算所有 similarity: [N, 1+K]
        all_sim = self._pairwise_similarity(
            h_anchor.unsqueeze(1),  # [N, 1, D]
            h_candidates,           # [N, 1+K, D]
            G_composed,
        ).squeeze(1)                # [N, 1+K]

        # InfoNCE: -log(exp(sim_pos/τ) / Σ_j exp(sim_j/τ))
        logits = all_sim / tau  # [N, 1+K]
        labels = torch.zeros(N, dtype=torch.long, device=h_anchor.device)  # 正例在位置 0
        loss = F.cross_entropy(logits, labels)

        # 诊断信息
        neg_sim = all_sim[:, 1:]  # [N, K]
        with torch.no_grad():
            info = {
                "pos_sim_mean": float(pos_sim.mean().item()),
                "neg_sim_mean": float(neg_sim.mean().item()),
                "sim_margin": float((pos_sim.mean() - neg_sim.mean()).item()),
                "temperature": float(tau.item()),
                "n_negatives": K,
            }

        return loss, info

    def compute_push_strength(
        self,
        h_anchor: torch.Tensor,
        h_neg_pool: torch.Tensor,
        G_composed: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """评估推离强度（不计算梯度）。

        返回推离效果诊断：
        - push_ratio: 负例相似度 / 正例相似度（<1 表示有效推离）
        - min_margin: 最小正负间距
        - effective_push: 有效推离的负例比例 (sim_neg < sim_pos 的比例)
        """
        with torch.no_grad():
            all_sim = self._pairwise_similarity(
                h_anchor.unsqueeze(1),
                h_neg_pool,
                G_composed,
            ).squeeze(1)  # [N, K]

            pos_sim = self._pairwise_similarity(
                h_anchor.unsqueeze(1),
                h_anchor.unsqueeze(1),  # self-similarity as reference
                G_composed,
            ).squeeze(-1).squeeze(-1)  # [N]

            # 每个 anchor 的推离效果
            push_ratios = (all_sim / (pos_sim.unsqueeze(-1) + 1e-8)).mean(dim=-1)  # [N]

            return {
                "push_ratio_mean": float(push_ratios.mean().item()),
                "frac_effective_push": float((all_sim < pos_sim.unsqueeze(-1)).float().mean().item()),
                "min_sim_margin": float((pos_sim.unsqueeze(-1) - all_sim).min().item()),
            }


def build_negative_pool(
    h_sequence: torch.Tensor,
    method: str = "temporal_contrast",
    num_negatives: int = 8,
    window_size: int = 4,
) -> torch.Tensor:
    """从隐状态序列构造负例池。

    支持三种构造策略：
    1. "temporal_contrast": 取不同时间窗口的隐状态作为负例
       → 对应"叙事时间线上的因果断裂"
    2. "divergent": 对正例序列施加小扰动生成发散负例
       → 对应"合法→非法的推离方向"
    3. "shuffle": 随机打乱批次内样本顺序
       → 对应"跨故事因果不连续性"

    Args:
        h_sequence: [N, T, D] 隐状态序列
        method: 负例构造方法
        num_negatives: 负例数量
        window_size: 时间窗口大小

    Returns:
        neg_pool: [N, K, D] 负例池
    """
    N, T, D = h_sequence.shape
    K = min(num_negatives, T - window_size)

    if method == "temporal_contrast":
        # 从不同时间位置采样作为负例
        # 远离当前窗口的位置 → 更强的因果断裂信号
        anchor_center = T // 2
        # 在序列两端采样
        neg_indices_left = torch.arange(0, anchor_center - window_size,
                                         step=max(1, (anchor_center - window_size) // K))
        neg_indices_right = torch.arange(anchor_center + window_size, T - window_size,
                                          step=max(1, (T - anchor_center - 2 * window_size) // K))
        all_indices = torch.cat([neg_indices_left[:K//2], neg_indices_right[:K - K//2]])
        all_indices = all_indices[:K]

        neg_pool = h_sequence[:, all_indices, :]  # [N, K, D]

    elif method == "divergent":
        # 对正例施加高斯噪声生成发散负例
        noise_scale = 0.1
        anchor_h = h_sequence[:, T//2:T//2+window_size, :].mean(dim=1)  # [N, D]
        neg_pool = anchor_h.unsqueeze(1).expand(-1, K, -1)  # [N, K, D]
        noise = torch.randn_like(neg_pool) * noise_scale
        # 逐渐增大的噪声 → 模拟"推离梯度"
        noise = noise * torch.linspace(0.5, 2.0, K, device=h_sequence.device).view(1, K, 1)
        neg_pool = neg_pool + noise

    elif method == "shuffle":
        # 打乱批次顺序
        perm = torch.randperm(N, device=h_sequence.device)
        shuffled = h_sequence[perm]  # [N, T, D]
        anchor_h = h_sequence[:, T//2, :]  # [N, D]
        neg_h = shuffled[:, T//2:T//2+K, :]  # [N, K, D]
        neg_pool = neg_h

    else:
        raise ValueError(f"Unknown negative pool method: {method}")

    return neg_pool
