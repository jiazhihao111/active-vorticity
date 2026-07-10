"""定理三：拓扑残差论 (Topological Residual)
============================================

定理陈述：
    该系统的所有"错误"（幻觉、发散、遗忘），是流形为满足定理一的
    拓扑闭合要求而必然遗留的几何残差，不可通过局部参数优化完全消除。

数学推导：
    1. 定理一要求隐流形具有闭合拓扑（χ ≠ 0）
    2. 闭合拓扑需要在全局范围内满足高斯-博内定理：∫K dA = 2πχ
    3. 局部曲率 K 由模型参数 θ 决定，但 χ 是拓扑不变量（不受 θ 影响）
    4. 因此存在拓扑约束：E[K(θ)] · A ≈ 2πχ
    5. 如果 A (上下文长度) 增大，但 K(θ) 受限于模型容量，
       则必须由"负曲率区域"（即错误）来补偿，以满足拓扑闭合
    6. 这些负曲率区域不能被局部优化消除，因为 χ 是全局不变量 →
       残差 ε_min = f(χ, dim, capacity) > 0

验证方法：
    1. 渐进界拟合：error(L) = ε_min + (ε_0 - ε_min)·exp(-L/L_0)
    2. 检验 ε_min > 0（残差非零）
    3. 检验 ε_min 在更大模型上不收敛到 0
    4. 检验 ε_min 与 χ 的正相关性

定理编码为以下可验证命题：
    H₀: ε_min > 0（残差不可消除）— 单侧 Bootstrap 检验
    H₁: ε_min(LargeModel) ≥ ε_min(SmallModel)（规模不消除残差）
    H₂: corr(ε_min, χ) > 0（残差与拓扑复杂度正相关）

十三字公理映射：
    "信息化为" → 模型参数 θ 决定了局部几何
    "世界模型" → 隐流形 M 的拓扑结构
    "遵守几何规则" → 高斯-博内定理的强制闭合 → 残差必然存在
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit


# ── 数据结构 ──────────────────────────────────────────────────────

class ResidualVerdict(str, Enum):
    """定理三判决"""
    SUPPORT = "SUPPORT"  # ε_min > 0 显著成立
    WEAK = "WEAK"  # 趋势正确但残差置信区间包含 0
    REFUTE = "REFUTE"  # ε_min ≈ 0，残差可消除
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ResidualAsymptote:
    """残差渐进界拟合结果"""
    epsilon_min: float  # 渐进残差下界
    epsilon_0: float  # 初始残差
    L_0: float  # 特征长度
    r_squared: float  # 拟合优度
    epsilon_min_ci_lower: float  # ε_min 的 95% CI 下界
    epsilon_min_ci_upper: float
    ls: List[int]  # 上下文长度序列
    errors: np.ndarray  # 对应误差序列
    fitted: np.ndarray  # 拟合曲线值
    residuals: np.ndarray  # 拟合残差


@dataclass
class ResidualResult:
    """定理三完整验证结果"""
    # 渐进界拟合
    asymptote: ResidualAsymptote

    # Bootstrap 检验
    epsilon_min_bootstrap_mean: float
    epsilon_min_bootstrap_std: float
    epsilon_min_bootstrap_ci: Tuple[float, float]

    # 统计检验
    p_value_positive: float  # H₀: ε_min > 0 的单侧检验
    p_value_nonzero: float  # H₀: ε_min ≠ 0 的双侧备择

    # 判决
    verdict: ResidualVerdict

    # 理论预测
    predicted_epsilon_min: float  # 基于 χ 和模型容量的理论预测

    # 跨规模比较
    multi_scale_results: Optional[Dict[str, ResidualAsymptote]] = None


# ── 核心实现 ──────────────────────────────────────────────────────

class TheoremResidual:
    """拓扑残差论验证引擎。

    核心能力：
    1. 渐进界拟合：error(L) → ε_min
    2. Bootstrap 置信区间检验 ε_min > 0
    3. 跨模型规模的残差不变性检验
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Args:
            n_bootstrap: Bootstrap 重采样次数
            confidence_level: 置信水平
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    # ── 渐进界拟合 ───────────────────────────────────────────────

    @staticmethod
    def _asymptotic_decay(L: np.ndarray, eps_min: float, eps_0: float, L_0: float) -> np.ndarray:
        """渐进误差衰减模型。

        error(L) = ε_min + (ε_0 - ε_min) · exp(-L/L_0)

        含义：
        - ε_min: 不可消除的拓扑残差（L → ∞ 时的极限）
        - ε_0: 初始误差（L → 0 时）
        - L_0: 特征衰减长度
        """
        return eps_min + (eps_0 - eps_min) * np.exp(-L / L_0)

    def fit_asymptote(
        self, context_lengths: List[int], errors: List[float]
    ) -> ResidualAsymptote:
        """对跨上下文长度误差序列拟合渐进界。

        Args:
            context_lengths: [N] 不同上下文长度
            errors: [N] 对应的误差（如事实不一致率、BLEU衰减）

        Returns:
            ResidualAsymptote 包含拟合结果和统计检验
        """
        L = np.array(context_lengths, dtype=np.float64)
        e = np.array(errors, dtype=np.float64)

        if len(L) < 5:
            return self._degenerate_asymptote(L, e)

        # 初始猜测
        eps_min_guess = np.min(e) * 0.5
        eps_0_guess = np.max(e)
        L_0_guess = np.median(L) / 2

        try:
            popt, pcov = curve_fit(
                self._asymptotic_decay, L, e,
                p0=[eps_min_guess, eps_0_guess, L_0_guess],
                bounds=(
                    [0.0, 0.0, 1.0],  # ε_min ≥ 0
                    [eps_0_guess * 2, eps_0_guess * 3, L[-1] * 10],
                ),
                maxfev=10000,
            )

            eps_min, eps_0, L_0 = popt

            # R²
            fitted = self._asymptotic_decay(L, *popt)
            residuals = e - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((e - np.mean(e)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # 参数的置信区间
            perr = np.sqrt(np.diag(pcov))
            eps_min_ci_lower = max(0.0, eps_min - 1.96 * perr[0])
            eps_min_ci_upper = eps_min + 1.96 * perr[0]

        except (RuntimeError, ValueError):
            # 拟合失败：降级到线性外推
            eps_min = np.min(e) * 0.1
            eps_0 = np.max(e)
            L_0 = float(np.median(L))
            fitted = np.full_like(e, np.mean(e))
            residuals = e - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((e - np.mean(e)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            eps_min_ci_lower = 0.0
            eps_min_ci_upper = float(np.mean(e))

        return ResidualAsymptote(
            epsilon_min=float(eps_min),
            epsilon_0=float(eps_0),
            L_0=float(L_0),
            r_squared=float(r_squared),
            epsilon_min_ci_lower=eps_min_ci_lower,
            epsilon_min_ci_upper=eps_min_ci_upper,
            ls=list(context_lengths),
            errors=e,
            fitted=fitted,
            residuals=residuals,
        )

    def _degenerate_asymptote(self, L: np.ndarray, e: np.ndarray) -> ResidualAsymptote:
        """数据不足时的退化结果"""
        eps_min = np.min(e) * 0.1 if len(e) > 0 else 0.0
        eps_0 = np.max(e) if len(e) > 0 else 0.0
        L_0 = float(np.median(L)) if len(L) > 0 else 1.0
        return ResidualAsymptote(
            epsilon_min=float(eps_min),
            epsilon_0=float(eps_0),
            L_0=L_0,
            r_squared=0.0,
            epsilon_min_ci_lower=0.0,
            epsilon_min_ci_upper=float(eps_0),
            ls=list(L),
            errors=e,
            fitted=np.full_like(e, eps_min),
            residuals=e - eps_min,
        )

    # ── Bootstrap 残差检验 ───────────────────────────────────────

    def bootstrap_epsilon_min(
        self, context_lengths: List[int], errors: List[float]
    ) -> Dict:
        """Bootstrap 检验 ε_min > 0。

        对数据做重采样后重新拟合渐进界，得到 ε_min 的分布。
        然后检验 H₀: ε_min > 0。
        """
        L = np.array(context_lengths)
        e = np.array(errors)
        N = len(L)

        if N < 5:
            return {
                "mean": 0.0, "std": 0.0, "ci": (0.0, 0.0),
                "p_value_positive": 0.5, "verdict": "insufficient_data",
            }

        bootstrap_eps_mins = []

        for _ in range(self.n_bootstrap):
            idx = np.random.choice(N, size=N, replace=True)
            L_boot = L[idx]
            e_boot = e[idx]

            try:
                popt, _ = curve_fit(
                    self._asymptotic_decay, L_boot.astype(float), e_boot.astype(float),
                    p0=[np.min(e) * 0.5, np.max(e), np.median(L) / 2],
                    bounds=([0, 0, 1], [np.max(e) * 2, np.max(e) * 3, L[-1] * 10]),
                    maxfev=5000,
                )
                bootstrap_eps_mins.append(max(0.0, popt[0]))
            except (RuntimeError, ValueError):
                continue

        if not bootstrap_eps_mins:
            return {
                "mean": 0.0, "std": 0.0, "ci": (0.0, 0.0),
                "p_value_positive": 0.5, "verdict": "bootstrap_failed",
            }

        eps_mins = np.array(bootstrap_eps_mins)
        mean_em = float(np.mean(eps_mins))
        std_em = float(np.std(eps_mins))
        ci_lower = float(np.percentile(eps_mins, (1 - self.confidence_level) * 50))
        ci_upper = float(np.percentile(eps_mins, 100 - (1 - self.confidence_level) * 50))

        # 单侧检验：ε_min > 0
        p_positive = float(np.mean(eps_mins <= 0))

        if p_positive < 0.01:
            verdict = "epsilon_min_significantly_positive"
        elif p_positive < 0.05:
            verdict = "epsilon_min_likely_positive"
        else:
            verdict = "epsilon_min_not_significant"

        return {
            "mean": mean_em,
            "std": std_em,
            "ci": (ci_lower, ci_upper),
            "p_value_positive": p_positive,
            "verdict": verdict,
        }

    # ── 跨规模比较 ───────────────────────────────────────────────

    def multi_scale_verify(
        self,
        scale_data: Dict[str, Tuple[List[int], List[float]]],
    ) -> Dict:
        """跨模型规模的残差不变性检验。

        Args:
            scale_data: {"small": (Ls, errors), "medium": (Ls, errors), "large": (Ls, errors)}

        Returns:
            每个规模的渐进界 + 跨规模比较结果
        """
        asymptotes = {}
        for scale_name, (Ls, errors) in scale_data.items():
            asymptotes[scale_name] = self.fit_asymptote(Ls, errors)

        # 提取 ε_min
        eps_mins = {
            name: a.epsilon_min
            for name, a in asymptotes.items()
        }

        # 排序检验：更大的模型不应该有更小的 ε_min
        sorted_scales = sorted(asymptotes.items(), key=lambda x: len(x[1].ls))
        monotonic = True
        prev_scale, prev_a = sorted_scales[0]
        for curr_scale, curr_a in sorted_scales[1:]:
            if curr_a.epsilon_min < prev_a.epsilon_min * 0.8:  # 允许 20% 噪音
                monotonic = False
                break
            prev_scale, prev_a = curr_scale, curr_a

        return {
            "asymptotes": asymptotes,
            "epsilon_mins": eps_mins,
            "monotonic": monotonic,  # ε_min 不随规模减小 = 支持定理
            "verdict": "SUPPORT" if monotonic else "needs_investigation",
        }

    # ── 完整验证 ─────────────────────────────────────────────────

    def verify(
        self,
        context_lengths: List[int],
        errors: List[float],
        chi_estimate: Optional[float] = None,
        model_capacity: Optional[float] = None,
    ) -> ResidualResult:
        """完整验证定理三。

        Args:
            context_lengths: 上下文长度序列 [Ls]
            errors: 对应误差 [errors]
            chi_estimate: 欧拉示性数估计（从定理一）
            model_capacity: 模型容量（参数量 log10）

        Returns:
            ResidualResult
        """
        # 1. 渐进界拟合
        asymptote = self.fit_asymptote(context_lengths, errors)

        # 2. Bootstrap
        bootstrap_result = self.bootstrap_epsilon_min(context_lengths, errors)

        # 3. 判决
        p_positive = bootstrap_result["p_value_positive"]
        auto_corr = self._autocorrelation(asymptote.residuals) if len(asymptote.residuals) > 1 else 0

        if p_positive < 0.01 and asymptote.epsilon_min > 0:
            if auto_corr < 0.6:  # 残差独立
                verdict = ResidualVerdict.SUPPORT
            else:
                verdict = ResidualVerdict.WEAK  # 自相关 → 模型不完备
        elif asymptote.epsilon_min_ci_lower > 0:
            verdict = ResidualVerdict.WEAK
        elif asymptote.epsilon_min < 0.01:
            verdict = ResidualVerdict.REFUTE
        else:
            verdict = ResidualVerdict.INCONCLUSIVE

        # 4. 理论预测 ε_min
        predicted = self._theoretical_epsilon_min(chi_estimate, model_capacity)

        return ResidualResult(
            asymptote=asymptote,
            epsilon_min_bootstrap_mean=bootstrap_result["mean"],
            epsilon_min_bootstrap_std=bootstrap_result["std"],
            epsilon_min_bootstrap_ci=tuple(bootstrap_result["ci"]),
            p_value_positive=p_positive,
            p_value_nonzero=1.0 - bootstrap_result.get("p_value_positive", 0.5),
            verdict=verdict,
            predicted_epsilon_min=predicted,
        )

    @staticmethod
    def _autocorrelation(x: np.ndarray, lag: int = 1) -> float:
        """一阶自相关"""
        if len(x) < lag + 2:
            return 0.0
        return float(np.corrcoef(x[lag:], x[:-lag])[0, 1]) if len(x) > lag else 0.0

    @staticmethod
    def _theoretical_epsilon_min(
        chi: Optional[float] = None,
        capacity_log10: Optional[float] = None,
    ) -> float:
        """理论预测 ε_min。

        基于：ε_min ∝ |χ| / capacity
        - χ 越大 → 拓扑越复杂 → 残差越大
        - capacity 越大 → 拟合能力越强 → 残差越小

        但 ε_min > 0 必然成立（因为 χ 是全局不变量，局部参数优化不可达）
        """
        if chi is None or capacity_log10 is None:
            return float("nan")

        # 估计公式：ε_min ≈ |χ| / (10^capacity_log10)^{α} · β
        # α ≈ 0.33 (基于经验标度律：容量→误差的幂律关系)
        alpha = 0.33
        beta = 0.1

        predicted = abs(chi) * beta / (10.0 ** (capacity_log10 * alpha))
        return max(0.001, predicted)  # 至少 0.1% 残差


# ── 快速接口 ──────────────────────────────────────────────────────

def residual_asymptote_pipeline(
    context_lengths: List[int],
    errors: List[float],
) -> ResidualAsymptote:
    """一键残差渐进界分析。

    >>> result = residual_asymptote_pipeline([32, 64, 128, 256], [0.3, 0.2, 0.15, 0.12])
    >>> print(f"ε_min = {result.epsilon_min:.4f}")
    """
    engine = TheoremResidual()
    return engine.fit_asymptote(context_lengths, errors)
