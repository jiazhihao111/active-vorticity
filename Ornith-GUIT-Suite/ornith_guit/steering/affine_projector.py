"""仿射硬约束投影器 (论文 §5.5 问题二: 推理时硬约束对齐)。

原理: 合法因果逻辑被严格限制在仿射子空间内
    C_i(h) = n_i^T h + b_i = 0   (R^2=1.0)
在推理时显式施加"拉格朗日乘子力", 将隐状态每步正交投影回安全的
因果脊线子空间 (子黎曼流形的水平分布 Δ_h), 杜绝模型逃逸出安全流形。

工程实现: 给定正交脊线基 basis [r, D], 投影
    h_proj = center + basis @ (basis^T @ (h - center))
即丢弃补空间 (V_perp) 的 off-ridge 分量。对缺陷/幻觉轨迹, off-ridge
注入被清除 → P_c/P_raw 大幅下降。

测试场景: 用 OrnithLatentSimulator 的脊线基 Vr 作为 basis, 对含缺陷块
的轨迹投影前后比较 mean P_c/P_raw。
"""

from typing import Dict, Optional
import numpy as np
import torch

from ..physics import ThermoPhysics


class AffineConstraintProjector:
    """将隐状态正交投影回仿射因果脊线子空间。"""

    def __init__(
        self,
        ridge_basis: np.ndarray,
        center: Optional[np.ndarray] = None,
    ):
        self.basis = np.asarray(ridge_basis, float)   # [r, D] 正交
        self.center = None if center is None else np.asarray(center, float)

    def project(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, float)
        x = h - self.center if self.center is not None else h
        coord = self.basis @ x                 # [r]
        h_proj = self.basis.T @ coord           # [D]
        if self.center is not None:
            h_proj = h_proj + self.center
        return h_proj

    def project_trajectory(self, H: torch.Tensor) -> np.ndarray:
        H = np.asarray(H, float)
        return np.array([self.project(h) for h in H])

    def pc_before_after(
        self,
        H: torch.Tensor,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        region: Optional[tuple] = None,
    ) -> Dict:
        """投影前后 mean P_c/P_raw 对比 (逐 token)。

        region: 可选 (lo, hi) 步区间, 仅在该区间比较。
        用于隔离 off-ridge 缺陷/幻觉块 —— 合法脊线旋转本身因 a⊥v 使 P_c/P_raw
        偏高 (详见模块头部说明), 全轨迹均值无参考意义; 投影对 off-ridge 注入
        块才是干净的降噪 (论文 §5.5 问题二)。
        """
        eng = ThermoPhysics(alpha_star=alpha_star, gamma=gamma)
        H = np.asarray(H, float)
        center = H.mean(axis=0) if self.center is None else self.center
        H_proj = self.project_trajectory(H)

        def mean_pc(X: np.ndarray, region) -> float:
            vals = []
            T = X.shape[0]
            lo, hi = (0, T) if region is None else region
            for t in range(max(2, lo), min(hi, T)):
                vals.append(eng.pc_ratio(
                    torch.tensor(X[t]), torch.tensor(X[t - 1]),
                    torch.tensor(X[t - 2]))[0])
            return float(np.mean(vals)) if vals else float("nan")

        raw = mean_pc(H, region)
        proj = mean_pc(H_proj, region)
        return {
            "pc_raw": raw,
            "pc_projected": proj,
            "reduction_ratio": float(1.0 - proj / (raw + 1e-12)),
            "region": list(region) if region is not None else None,
            "alpha_star": alpha_star,
        }
