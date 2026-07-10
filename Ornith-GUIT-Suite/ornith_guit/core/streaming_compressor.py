"""StreamingAffineCompressor — 流式仿射子空间压缩器 (战役一)。

核心痛点 (Phase 1 验证): SVD 基是 prompt-dependent 的, 且 Prefill
提取的基无法用于 Decode。隐空间是"时变呼吸子黎曼流形", 静态基的
重构误差 (KL) 随 Decode 步数放大。

破局: 放弃静态 SVD, 采用流式增量正交迭代 —— 每个新 token 的残差
能量超阈值时, 旋转基底 V_r 追踪当前因果脊线 ("向日葵"追踪)。

工程收益: 4096 维 → 25~32 维, 层间激活显存暴降 99%, 且规避 KL 爆炸。
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class StreamingAffineCompressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ridge_dim: int = 32,
        drift_threshold: float = 0.05,
        warmup_steps: int = 8,
    ):
        super().__init__()
        self.D = hidden_dim
        self.r = ridge_dim
        self.drift_thresh = drift_threshold
        self.warmup = warmup_steps

        Q, _ = torch.linalg.qr(torch.randn(self.D, self.r))
        self.register_buffer("basis", Q)          # [D, r]
        self.register_buffer("is_initialized", torch.tensor(False))

        self._step = 0

    @torch.no_grad()
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """h: [B, N, D] 或 [B, 1, D] (decode)。

        Returns: (h_recon, info)
        """
        B, N, D = h.shape
        h_flat = h.reshape(-1, D).float()

        coords = h_flat @ self.basis                 # [B*N, r]
        h_recon = coords @ self.basis.T              # [B*N, D]

        residual = h_flat - h_recon
        rel_err = (residual.norm(dim=-1) /
                   (h_flat.norm(dim=-1) + 1e-8)).mean().item()

        self._step += 1
        if (not self.is_initialized) or (rel_err > self.drift_thresh
                                         and self._step > self.warmup):
            self._update_basis(h_flat)

        info = {
            "relative_error": rel_err,
            "basis_updated": bool(not self.is_initialized) or rel_err > self.drift_thresh,
            "step": self._step,
        }
        return h_recon.reshape(B, N, D), info

    @torch.no_grad()
    def _update_basis(self, h_new: torch.Tensor):
        """增量正交迭代: 用新数据的"新概念方向"微调基底。"""
        h_recon = (h_new @ self.basis) @ self.basis.T
        novelty = h_new - h_recon                      # [M, D]

        combined = torch.cat([self.basis.T, novelty], dim=0)  # [r+M, D]
        _, S, Vh = torch.linalg.svd(combined, full_matrices=False)
        self.basis.copy_(Vh[: self.r].T)
        self.is_initialized.fill_(True)

    @torch.no_grad()
    def compress_store(self, h: torch.Tensor) -> torch.Tensor:
        """仅存储 r 维坐标 (用于 KV/激活压缩落盘)。"""
        h_flat = h.reshape(-1, self.D).float()
        return h_flat @ self.basis

    @torch.no_grad()
    def restore(self, coords: torch.Tensor) -> torch.Tensor:
        return coords @ self.basis.T + 0.0
