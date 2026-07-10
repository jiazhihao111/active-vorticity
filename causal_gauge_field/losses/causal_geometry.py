import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CausalGeometryLoss(nn.Module):
    """因果几何损失。

    约定(C-03): 输入 h_* 已是模型投影到底因基底 B (base_dim 维) 的隐含态，
    距离一律在投影空间计算；当提供 G_composed(=G^eff) 时，采用度量距离
    d_G^2 = Δh^T G Δh，其中 G^eff = G^curve + Σ α_k ΔG_k (B-04)。

    损失形式(B-11): 带 σ_t^2 归一化的对比 log-loss
        L_gauge = -log[ e^{-d_+^2/σ_t^2} / (e^{-d_+^2/σ_t^2} + Σ_j e^{-d_-^2/σ_t^2}) ]
    σ_t^2 为可学习归一化温度（下溢修复：指数项天然有界）。
    """

    def __init__(self, margin: float = 1.0, distance_metric: str = "cosine",
                 sigma2_init: float = 0.5):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.sigma2 = nn.Parameter(torch.tensor(float(sigma2_init)))

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_pos: torch.Tensor,
        h_t1_neg: torch.Tensor,
        G_composed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # h_t, h_t1_pos: (N, db) | h_t1_neg: (N, K, db) | G_composed: (N, db, db)
        if G_composed is not None:
            d2_pos = self._geodesic_sq(h_t, h_t1_pos, G_composed)          # (N,)
            db = h_t.size(-1)
            G_exp = G_composed.unsqueeze(1).expand(-1, h_t1_neg.size(1), db, db).contiguous()
            diff_neg = (h_t.unsqueeze(1) - h_t1_neg)                       # (N, K, db)
            d2_neg = torch.einsum('nka,nkab,nkb->nk', diff_neg, G_exp, diff_neg)  # (N, K)
        else:
            d2_pos = self._pair_sq(h_t, h_t1_pos)                          # (N,)
            d2_neg = self._pair_sq(h_t.unsqueeze(1), h_t1_neg)             # (N, K)

        # B-11: 带 σ_t^2 归一化的对比 log-loss
        s = self.sigma2.clamp(min=1e-4)
        pos_term = torch.exp(-d2_pos / s)
        neg_term = torch.exp(-d2_neg / s)
        denom = pos_term + neg_term.sum(dim=-1) + 1e-12
        loss = -torch.log((pos_term / denom).clamp(min=1e-12))
        return loss.mean()

    def _geodesic_sq(self, a: torch.Tensor, b: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        diff = a - b
        return torch.einsum('ni,nij,nj->n', diff, G, diff)

    def _pair_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            a_n = F.normalize(a, p=2, dim=-1)
            b_n = F.normalize(b, p=2, dim=-1)
            d = 1.0 - (a_n * b_n).sum(dim=-1)
            return d ** 2
        elif self.distance_metric == "euclidean":
            return ((a - b) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def compute_geodesic_proxy(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        G_composed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # C-03: 输入已投影；提供 G 时用度量距离，否则退化为欧氏/余弦平方
        if G_composed is not None:
            return self._geodesic_sq(h_t, h_t1, G_composed)
        return self._pair_sq(h_t, h_t1)


def loop_back_contrastive_loss(
    flatness: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """回环 holonomy 的对比训练信号 (实验7, §10.7 任务2).

    与旧 CausalGeometryLoss 的根本区别: 旧损失对所有轨迹【统一压缩】
    (把正/负一起压平, exp4 的 SUPPORT 是此循环假象); 这里按【标签】分化:

      - 正例(闭环叙事): 最小化其回环 holonomy 平坦度 → 拉向平坦 (flat→0);
      - 负例(破缺叙事): 惩罚其被压平 → 推离平坦 (flat 需 > margin, 即『扭曲』).

    这是 C-11「叙事闭环 ⇔ 规范场平坦」应被检验的【公平】训练:
    用标签(闭环/破缺)而非几何量本身作监督, 看几何能否分化两类.
    若训练后 正例平坦 << 负例平坦 → 机制可编码闭环, C-11 在公平设计下成立;
    若仍全局坍缩(两者都平) → 几何根本无法承载该区分, C-11 应退役.

    flatness: (B,) 来自 GaugeField.loop_back_holonomy_flatness
    labels:   (B,) 1=正例(闭环) 0=负例(破缺)
    margin:   负例应被推到的扭曲目标 (‖W-I‖_F > margin)
    """
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    loss = torch.tensor(0.0, device=flatness.device)
    info: Dict[str, float] = {"pos_flat": 0.0, "neg_flat": 0.0, "n_pos": 0, "n_neg": 0}
    if pos_mask.any():
        pos_f = flatness[pos_mask]
        loss = loss + pos_f.mean()                         # 拉平正例
        info["pos_flat"] = float(pos_f.mean().item())
        info["n_pos"] = int(pos_mask.sum().item())
    if neg_mask.any():
        neg_f = flatness[neg_mask]
        # 推离平坦: 当负例比 margin 更平时施加惩罚 (relu(margin - flat))
        loss = loss + torch.relu(
            torch.tensor(margin, device=flatness.device) - neg_f
        ).mean()                                          # 推扭曲负例
        info["neg_flat"] = float(neg_f.mean().item())
        info["n_neg"] = int(neg_mask.sum().item())
    return loss, info


class CombinedLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.causal_loss = CausalGeometryLoss(
            margin=config["training"].get("margin", 1.0),
            distance_metric=config["training"].get("distance_metric", "cosine"),
            sigma2_init=config["training"].get("sigma2_init", 0.5),
        )
        self.lambda_train = config["training"]["lambda_default"]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        h_t1_pos: Optional[torch.Tensor] = None,
        h_t1_neg: Optional[torch.Tensor] = None,
        G_composed: Optional[torch.Tensor] = None,
        lambda_value: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        lm_loss = self.lm_loss(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[:, 1:].contiguous().view(-1),
        )
        result = {"lm_loss": lm_loss}
        if (h_t is not None and h_t1_pos is not None
                and h_t1_neg is not None and G_composed is not None):
            c_loss = self.causal_loss(h_t, h_t1_pos, h_t1_neg, G_composed=G_composed)
            lam = lambda_value if lambda_value is not None else self.lambda_train
            result["causal_loss"] = c_loss
            result["total_loss"] = lm_loss + lam * c_loss
        else:
            result["total_loss"] = lm_loss
        return result
