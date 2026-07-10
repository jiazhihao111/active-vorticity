import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CausalMemoryBank(nn.Module):
    """因果记忆银行。

    输入: 已投影到底因基底 B 的隐含态 (B, T, base_dim) (C-03)。
    输出: 组合度量 G^eff = G^curve + Σ_k α_k ΔG_k (B-04)，及类型权重。

    - B-04: G^eff 必须含 G^curve(由规范场 F 派生) 加性项；当外部未提供时
            退化为 PD 基底 δ·I，仍保证正定。
    - C-05: 每个局部度量 local_metrics[k] 经对称化 + 最小特征值裁剪保证 PD。
    - C-08: 三层分解(type_assignment)是表示分块工作假设(H-decomp)，
            非先验公理；num_causal_types 可配置，聚类结果仅作候选假设报告。
    """

    def __init__(self, base_dim: int, num_kernels: int = 64,
                 num_causal_types: int = 3, pd_eps: float = 1e-3):
        super().__init__()
        self.base_dim = base_dim
        self.num_kernels = num_kernels
        self.num_causal_types = num_causal_types
        self.pd_eps = pd_eps
        self.anchors = nn.Parameter(torch.randn(num_kernels, base_dim) * 0.02)
        self.local_metrics = nn.ParameterList([
            nn.Parameter(torch.eye(base_dim) * 0.1)
            for _ in range(num_kernels)
        ])
        self.type_assignment = nn.Parameter(
            torch.randn(num_kernels, num_causal_types) * 0.01
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # B-04: G^curve 基座项（由规范场 F 派生增强；此处为 PD 基底 δ·I）
        self.curvature_scale = nn.Parameter(torch.tensor(0.1))
        # C-08: 三层分解为待涌现的工作假设（H-decomp），非先验公理
        self.decomposition_hypothesis = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        G_curve: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (B, T, base_dim) 已投影
        batch_size, seq_len, _ = hidden_states.shape
        h_last = hidden_states[:, -1, :]
        sim = torch.matmul(h_last, self.anchors.t())
        alpha = F.softmax(sim / self.temperature.clamp(min=0.1), dim=-1)
        base_dim = self.base_dim
        G_composed = torch.zeros(batch_size, base_dim, base_dim, device=hidden_states.device)
        for k in range(self.num_kernels):
            G_k = self._make_pd(self.local_metrics[k])            # C-05 正定保证
            G_composed += alpha[:, k].unsqueeze(-1).unsqueeze(-1) * G_k.unsqueeze(0)
        # B-04: G^eff = G^curve + Σ_k α_k ΔG_k
        if G_curve is None:
            G_curve = (self.curvature_scale
                       * torch.eye(base_dim, device=hidden_states.device)
                       .unsqueeze(0).expand(batch_size, -1, -1))
        G_composed = G_composed + G_curve
        type_weights = F.softmax(self.type_assignment, dim=-1)
        return G_composed, type_weights

    def _make_pd(self, M: torch.Tensor) -> torch.Tensor:
        # C-05: 对称化 + 最小特征值裁剪，保证正定
        M = 0.5 * (M + M.t())
        eig = torch.linalg.eigvalsh(M)
        min_eig = eig.min()
        if min_eig < self.pd_eps:
            M = M + (self.pd_eps - min_eig) * torch.eye(M.size(0), device=M.device)
        return M

    def get_effective_kernels(self, hidden_states: torch.Tensor, threshold: float = 0.01) -> int:
        with torch.no_grad():
            h_last = hidden_states[:, -1, :]
            sim = torch.matmul(h_last, self.anchors.t())
            alpha = F.softmax(sim / self.temperature.clamp(min=0.1), dim=-1)
            k_eff = (alpha > threshold).sum(dim=-1).float().mean().item()
        return int(k_eff)

    def get_type_clustering(self) -> Dict[str, List[int]]:
        # C-08: 聚类仅作为三层分解工作假设(H-decomp)的候选报告，非已验证公理
        with torch.no_grad():
            type_weights = F.softmax(self.type_assignment, dim=-1)
            type_names = [f"type_{i}" for i in range(self.num_causal_types)]
            clustering = {name: [] for name in type_names}
            dominant = type_weights.argmax(dim=-1)
            for k in range(self.num_kernels):
                clustering[type_names[dominant[k].item()]].append(k)
        return clustering
