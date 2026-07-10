"""OrnithAutoCalibrator — 代码因果脊线提取与热力学标定器。

理论锚点: Ornith 在 v5 报告中展现极低有效秩 r(0.95)=25。
本组件从 SWE-bench 代码流形 / 提供的隐状态中, 提取专属的
"代码因果脊线 (Code Causal Ridge)" 并标定对抗 RMSNorm 收缩的 alpha*。

设计: 默认提供张量级 API (hidden-states), 便于在仿真器/真实模型上统一测试;
      同时保留可选的 model+tokenizer 封装接口 (真实 Ornith 部署时使用)。
"""

from typing import List, Dict, Optional
import torch
import numpy as np


class OrnithAutoCalibrator:
    def __init__(
        self,
        hidden_dim: int = 4096,
        target_r: int = 32,
        variance_threshold: float = 0.95,
        gamma: float = 0.01,
    ):
        self.D = hidden_dim
        self.target_r = target_r
        self.var_threshold = variance_threshold
        self.gamma = gamma

        self.code_ridge_basis: Optional[torch.Tensor] = None  # [r, D]
        self.code_ridge_mean: Optional[torch.Tensor] = None   # [D]
        self.alpha_star: float = 1.41
        self.r: int = 0
        self.explained_variance: float = 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(
        self, hidden_states_list: List[torch.Tensor], calibrate_alpha: bool = True
    ) -> Dict:
        """从隐状态序列列表标定代码因果脊线与 alpha*。

        Args:
            hidden_states_list: 每条为 [T_i, D] 的代码/文本隐状态序列
                                (建议用 decode 阶段隐状态, 见 GUIT 铁律)
        Returns:
            report dict
        """
        if not hidden_states_list:
            raise ValueError("hidden_states_list 不能为空")

        mats = []
        for h in hidden_states_list:
            h = h.detach().float()
            if h.dim() == 3:
                h = h.squeeze(0)
            if h.dim() != 2:
                raise ValueError(f"期望 [T, D], 得到 {tuple(h.shape)}")
            mats.append(h)
        H = torch.cat(mats, dim=0)  # [Total, D]
        self.D = H.shape[1]

        # 1. 仿射中心 (位置空间完整约束)
        self.code_ridge_mean = H.mean(dim=0)
        Hc = H - self.code_ridge_mean

        # 2. SVD 提取切空间 (速度空间非完整分布)
        _, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
        explained = (S ** 2) / (S ** 2).sum()
        cum = torch.cumsum(explained, dim=0)

        r = int((cum < self.var_threshold).sum().item()) + 1
        r = min(r, self.target_r, self.D)
        self.r = r
        self.code_ridge_basis = Vh[:r]
        self.explained_variance = float(cum[r - 1].item())

        # 3. alpha* 标定 (用最后一条序列的末 3 步)
        if calibrate_alpha:
            last = mats[-1]
            if last.shape[0] >= 3:
                from ..physics import calibrate_alpha_star
                self.alpha_star = calibrate_alpha_star(
                    [last], gamma=self.gamma
                )

        return {
            "ambient_dim (D)": self.D,
            "ridge_dim (r)": self.r,
            "compression_ratio": 1.0 - self.r / self.D,
            "alpha_star": round(self.alpha_star, 4),
            "explained_variance": self.explained_variance,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_from_swebench(
        self,
        model,
        tokenizer,
        code_snippets: List[str],
        max_length: int = 1024,
    ) -> Dict:
        """真实 Ornith 部署封装: 用 SWE-bench 代码段提取脊线。

        仅在拥有真实模型时调用; 测试与仿真请使用 calibrate()。
        """
        model.eval()
        states = []
        for code in code_snippets:
            prompt = f"<system>You are an expert Python developer.</system>\n<code>\n{code}\n</code>"
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_length).to(model.device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            states.append(out.hidden_states[-1].squeeze(0).float())
        return self.calibrate(states)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_online(
        self, hidden_states_list: List[torch.Tensor], momentum: float = 0.7
    ) -> Dict:
        """元层进化: 用 PI-LOOP 收敛的优质轨迹在线更新脊线基底。

        将新优质数据投影出"新概念方向"(residual), 与现有基底拼接后重做
        SVD, 并以 momentum 对新旧基底做子空间融合 (避免灾难性遗忘)。
        首次调用 (尚无基底) 时等价于 calibrate()。

        Args:
            hidden_states_list: 每条为 [T, D] 的收敛优质轨迹
            momentum: 旧基底保留权重 (0~1, 越大越稳定)
        Returns:
            report dict (含漂移量 drift)
        """
        if self.code_ridge_basis is None:
            return self.calibrate(hidden_states_list)

        mats = []
        for h in hidden_states_list:
            h = h.detach().float()
            if h.dim() == 3:
                h = h.squeeze(0)
            if h.dim() == 2 and h.shape[0] >= 1:
                mats.append(h)
        if not mats:
            return {"updated": False, "reason": "no valid trajectory"}
        H = torch.cat(mats, dim=0)                       # [M, D]

        old_basis = self.code_ridge_basis.clone()        # [r, D]
        old_mean = self.code_ridge_mean.clone()

        # 均值 EMA 更新
        new_mean = H.mean(dim=0)
        self.code_ridge_mean = momentum * old_mean + (1 - momentum) * new_mean

        Hc = H - self.code_ridge_mean
        recon = (Hc @ old_basis.T) @ old_basis           # [M, D]
        novelty = Hc - recon                             # 新概念方向

        combined = torch.cat([old_basis, novelty], dim=0)  # [r+M, D]
        _, _S, Vh = torch.linalg.svd(combined, full_matrices=False)
        cand = Vh[: self.r]                               # [r, D]

        # 子空间 momentum 融合 + 重正交
        fused = momentum * old_basis + (1 - momentum) * cand
        Q, _ = torch.linalg.qr(fused.T)                  # [D, r]
        self.code_ridge_basis = Q[:, : self.r].T

        # 漂移量: 新旧基底主子空间夹角 (1 - 平均对齐)
        align = float(torch.abs(self.code_ridge_basis @ old_basis.T).diagonal()
                      .mean().item())
        drift = 1.0 - align
        return {
            "updated": True,
            "ridge_dim (r)": self.r,
            "subspace_drift": drift,
            "alpha_star": round(self.alpha_star, 4),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def project(self, h: torch.Tensor) -> torch.Tensor:
        """投影到 r 维脊线坐标 [..., r]。"""
        if self.code_ridge_basis is None:
            raise RuntimeError("未标定脊线, 请先调用 calibrate()")
        f = h.detach().float()
        if f.dim() == 1:
            return (f - self.code_ridge_mean) @ self.code_ridge_basis.T
        return (f - self.code_ridge_mean) @ self.code_ridge_basis.T

    @torch.no_grad()
    def reconstruct(self, coords: torch.Tensor) -> torch.Tensor:
        """从 r 维坐标重构回 D 维。"""
        if self.code_ridge_basis is None:
            raise RuntimeError("未标定脊线, 请先调用 calibrate()")
        return coords @ self.code_ridge_basis + self.code_ridge_mean

    def compression_bytes_per_token(self, dtype_bytes: int = 2) -> Dict:
        """对比未压缩 D 维 vs 脊线 r 维的每 token 存储 (bytes)。"""
        return {
            "raw_bytes": self.D * dtype_bytes,
            "ridge_bytes": self.r * dtype_bytes,
            "save_ratio": 1.0 - self.r / self.D,
        }
