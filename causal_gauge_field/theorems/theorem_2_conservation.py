"""定理二：几何守恒论 (Geometric Conservation)
==============================================

定理陈述：
    合法生成路径是语义流形上保持 τ/κ 共形不变的测地线或类测地平行移动。

数学推导：
    1. 语义流形 (M, g) 上，FS 标架 {T, N, B} 构成局部正交基
    2. 平行移动：沿路径 γ，T 的协变导数 ∇_T T = κN = 0（测地线）
       或 ∇_T T = κN ≠ 0 但 κ 保持有界（类测地）
    3. 共形不变：τ/κ = const 意味着在共形变换 g → λ(x)·g 下，
       tanΘ = τ/κ 不变（因为 τ 和 κ 在共形变换下同比缩放）
    4. 合法路径上，τ/κ 的变异系数 CV 应显著低于随机路径

验证方法（三种变体实验）：
    1. 原本 vs 打乱句子：同一故事的句子随机排列后，τ/κ 应显著变化
    2. 原本 vs 打乱段落：段落重排后，局部 τ/κ 断崖
    3. 平行移动检验：在滑动窗口内，τ/κ 的局部 CV < 全局 CV

定理编码为以下可验证命题：
    H₀: CV_original < CV_shuffled  （单侧检验）
    H₁: 滑动窗口 CV < 全局 CV  （局部-整体比较）
    H₂: 平行移动残差 |Δ(τ/κ)| < ε  （局部守恒界的余量检验）

十三字公理映射：
    "信息化为" → FS 标架提取 (κ, τ, tanΘ)
    "世界模型" → 生成路径 = 流形上的曲线
    "遵守几何规则" → τ/κ = const（共形不变量）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats


# ── 数据结构 ──────────────────────────────────────────────────────

class ConformalVerdict(str, Enum):
    """定理二判决"""
    SUPPORT = "SUPPORT"  # τ/κ 共形不变显著成立
    WEAK = "WEAK"  # 趋势正确但未达显著性
    REFUTE = "REFUTE"  # τ/κ 不满足共形不变
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ConservationResult:
    """定理二完整验证结果"""
    # 原始数据的 tanΘ 统计
    tan_original: np.ndarray
    cv_original: float
    mean_original: float
    std_original: float

    # 打乱数据的 tanΘ 统计
    tan_shuffled: np.ndarray
    cv_shuffled: float
    mean_shuffled: float
    std_shuffled: float

    # 滑动窗口统计
    window_cv_mean: float
    window_cv_std: float

    # 平行移动残差
    parallel_transport_residual: float  # 相邻窗口 tanΘ 差的 RMS
    pt_residual_threshold: float  # 残差上界

    # 统计检验
    t_statistic: float
    p_value: float
    cohens_d: float
    verdict: ConformalVerdict

    # 变体实验
    sentence_level: Optional[Dict] = None  # 句子打乱结果
    paragraph_level: Optional[Dict] = None  # 段落打乱结果


# ── 核心实现 ──────────────────────────────────────────────────────

class TheoremConservation:
    """几何守恒论验证引擎。

    核心能力：
    1. 计算 τ/κ 的共形不变性度量
    2. 平行移动残留分析
    3. 句子/段落打乱对比实验
    """

    def __init__(
        self,
        window_size: int = 8,
        stride: int = 4,
        cv_threshold: float = 0.3,
        significance_level: float = 0.01,
    ):
        """
        Args:
            window_size: 滑动窗口大小
            stride: 窗口步长
            cv_threshold: 共形不变最大允许 CV
            significance_level: 统计显著性水平
        """
        self.window_size = window_size
        self.stride = stride
        self.cv_threshold = cv_threshold
        self.significance_level = significance_level

    # ── 共形不变检验 ─────────────────────────────────────────────

    def compute_conformal_invariance(
        self,
        tan_theta: np.ndarray,
    ) -> Dict[str, float]:
        """计算 τ/κ（= tanΘ）的共形不变量性质。

        Args:
            tan_theta: [T] 或 [B, T] 的 tanΘ 序列

        Returns:
            dict with mean, std, cv, skewness, kurtosis, gini
        """
        if tan_theta.ndim > 1:
            tan_theta = tan_theta.flatten()

        # 去除 NaN 和 Inf
        valid = np.isfinite(tan_theta)
        t = tan_theta[valid]

        if len(t) < 4:
            return {"mean": float("nan"), "std": float("nan"), "cv": float("nan"),
                    "n_valid": len(t), "verdict": "insufficient_data"}

        mean_t = float(np.mean(t))
        std_t = float(np.std(t, ddof=1))
        cv = std_t / mean_t if mean_t > 0 else float("inf")

        # 偏度（对称性）
        skew = float(scipy_stats.skew(t)) if len(t) > 8 else 0.0

        # 峰度（尾部分布）
        kurt = float(scipy_stats.kurtosis(t)) if len(t) > 15 else 0.0

        # Gini 系数（异质性）
        gini = self._gini(t)

        # 判决
        if cv < self.cv_threshold:
            verdict = "conformally_invariant"
        elif cv < self.cv_threshold * 2:
            verdict = "marginally_invariant"
        else:
            verdict = "not_invariant"

        return {
            "mean": mean_t,
            "std": std_t,
            "cv": cv,
            "skewness": skew,
            "kurtosis": kurt,
            "gini": gini,
            "n_valid": int(valid.sum()),
            "verdict": verdict,
        }

    @staticmethod
    def _gini(x: np.ndarray) -> float:
        """Gini 系数"""
        n = len(x)
        if n < 2:
            return 0.0
        sorted_x = np.sort(x)
        index = np.arange(1, n + 1)
        gini = (2 * (index * sorted_x).sum()) / (n * sorted_x.sum()) - (n + 1) / n
        return float(gini)

    # ── 滑动窗口平行移动检验 ─────────────────────────────────────

    def sliding_window_test(self, tan_theta: np.ndarray) -> Dict:
        """滑动窗口检验 τ/κ 的局域守恒。

        原理：如果 τ/κ 在测地平行移动下共形不变，
        那么滑动窗口内的 CV 应远小于全局 CV。

        Returns:
            dict with local_cv_stats, global_cv, cv_ratio, pass_fraction
        """
        if tan_theta.ndim > 1:
            tan_theta = tan_theta.flatten()
        valid = np.isfinite(tan_theta)
        t = tan_theta[valid]
        T = len(t)

        if T < self.window_size:
            return {"verdict": "insufficient_data", "pass_fraction": 0.0}

        global_cv = self.compute_conformal_invariance(t)["cv"]

        window_cvs = []
        residuals = []  # 相邻窗口 tanΘ 差

        prev_mean = None
        for start in range(0, T - self.window_size + 1, self.stride):
            window = t[start:start + self.window_size]
            mean_w = np.mean(window)
            std_w = np.std(window, ddof=1)
            cv_w = std_w / mean_w if mean_w > 0 else float("inf")
            window_cvs.append(cv_w)

            if prev_mean is not None:
                residuals.append(abs(mean_w - prev_mean))
            prev_mean = mean_w

        window_cvs = np.array(window_cvs)
        residuals = np.array(residuals) if residuals else np.array([0.0])

        # 局部 CV 的均值应远低于全局 CV
        local_mean_cv = float(np.mean(window_cvs))
        local_std_cv = float(np.std(window_cvs))
        cv_ratio = local_mean_cv / global_cv if global_cv > 0 else 1.0

        # 通过率：局部 CV < 全局 CV 的窗口比例
        pass_frac = float(np.mean(window_cvs < global_cv)) if global_cv > 0 else 0.0

        # 平行移动残差
        pt_residual = float(np.sqrt(np.mean(residuals ** 2)))
        pt_threshold = float(np.std(t)) * 0.3  # 30% 全局标准差作为上界

        if cv_ratio < 0.5 and pass_frac > 0.7:
            verdict = "strong_local_conservation"
        elif cv_ratio < 1.0 and pass_frac > 0.5:
            verdict = "weak_local_conservation"
        else:
            verdict = "no_local_conservation"

        return {
            "verdict": verdict,
            "local_cv_mean": local_mean_cv,
            "local_cv_std": local_std_cv,
            "global_cv": global_cv,
            "cv_ratio": cv_ratio,
            "pass_fraction": pass_frac,
            "parallel_transport_residual": pt_residual,
            "pt_residual_threshold": pt_threshold,
            "n_windows": len(window_cvs),
            "window_size": self.window_size,
            "stride": self.stride,
        }

    # ── 变体实验：打乱对比 ───────────────────────────────────────

    def shuffle_test(
        self,
        tan_original: np.ndarray,
        shuffle_mode: str = "sentence",
        n_shuffles: int = 10,
    ) -> Dict:
        """打乱变体实验。

        对 tanΘ 序列做块状打乱，模拟句子/段落重排。
        检验打乱后 CV 是否显著升高。

        Args:
            tan_original: 原始 tanΘ 序列
            shuffle_mode: "sentence" (小块打乱) 或 "paragraph" (大块打乱)
            n_shuffles: 打乱次数
        """
        if tan_original.ndim > 1:
            tan_original = tan_original.flatten()
        valid = np.isfinite(tan_original)
        t = tan_original[valid]
        T = len(t)

        if T < 10:
            return {"verdict": "insufficient_data"}

        # 块大小
        if shuffle_mode == "sentence":
            block_size = max(3, T // 10)  # ~句子长度
        else:
            block_size = max(8, T // 4)  # ~段落长度

        original_cv = self.compute_conformal_invariance(t)["cv"]

        shuffled_cvs = []
        for _ in range(n_shuffles):
            # 块状打乱
            n_blocks = T // block_size
            blocks = np.array_split(t, n_blocks)
            np.random.shuffle(blocks)
            shuffled = np.concatenate(blocks)
            cv_s = self.compute_conformal_invariance(shuffled)["cv"]
            shuffled_cvs.append(cv_s)

        shuffled_cvs = np.array(shuffled_cvs)
        mean_shuffled_cv = float(np.mean(shuffled_cvs))
        std_shuffled_cv = float(np.std(shuffled_cvs))

        # 统计检验
        if n_shuffles > 1:
            # 单样本 t 检验：打乱后的 CV 均值 > 原始 CV
            t_stat, p_val = scipy_stats.ttest_1samp(shuffled_cvs, original_cv)
            p_val = float(p_val / 2)  # 单侧
        else:
            t_stat = float("nan")
            p_val = float("nan")

        cv_increase = mean_shuffled_cv - original_cv
        relative_increase = cv_increase / original_cv if original_cv > 0 else float("inf")

        if p_val < 0.05 and relative_increase > 0.2:
            verdict = "conservation_broken"  # 打乱破坏了守恒
        elif relative_increase > 0.1:
            verdict = "conservation_weakened"
        else:
            verdict = "no_effect"

        return {
            "verdict": verdict,
            "shuffle_mode": shuffle_mode,
            "block_size": block_size,
            "original_cv": original_cv,
            "shuffled_cv_mean": mean_shuffled_cv,
            "shuffled_cv_std": std_shuffled_cv,
            "cv_increase": cv_increase,
            "relative_increase": relative_increase,
            "t_statistic": t_stat,
            "p_value": p_val,
            "n_shuffles": n_shuffles,
        }

    # ── 完整验证 ─────────────────────────────────────────────────

    def verify(
        self,
        tan_theta_original: np.ndarray,
        tan_theta_shuffled: Optional[np.ndarray] = None,
        run_shuffle_test: bool = True,
    ) -> ConservationResult:
        """完整验证定理二。

        Args:
            tan_theta_original: 原始正例的 tanΘ 序列
            tan_theta_shuffled: 打乱版本的 tanΘ 序列（可选，若为 None 则自动生成）
            run_shuffle_test: 是否运行打乱实验

        Returns:
            ConservationResult
        """
        # 原始数据共形不变统计
        orig_stats = self.compute_conformal_invariance(tan_theta_original)
        tan_orig = tan_theta_original[np.isfinite(tan_theta_original)]

        # 滑动窗口平行移动检验
        window_result = self.sliding_window_test(tan_theta_original)

        # 打乱实验
        sentence_test = None
        paragraph_test = None

        if run_shuffle_test:
            sentence_test = self.shuffle_test(tan_theta_original, "sentence")
            paragraph_test = self.shuffle_test(tan_theta_original, "paragraph")

        # 如果有提供打乱数据，做直接对比
        if tan_theta_shuffled is not None:
            tan_shuf = tan_theta_shuffled[np.isfinite(tan_theta_shuffled)]
            shuf_stats = self.compute_conformal_invariance(tan_theta_shuffled)
            shuffled_cv = shuf_stats["cv"]
            shuffled_mean = shuf_stats["mean"]
            shuffled_std = shuf_stats["std"]

            # 统计检验
            t_stat, p_val = scipy_stats.ttest_ind(
                np.abs(np.diff(tan_orig)),
                np.abs(np.diff(tan_shuf)),
                equal_var=False,
            )
            p_val = float(p_val / 2)  # 单侧

            cohens_d = (shuffled_cv - orig_stats["cv"]) / (
                np.sqrt((orig_stats["std"] ** 2 + shuffled_std ** 2) / 2) + 1e-10
            )
        else:
            shuffled_cv = float("nan")
            shuffled_mean = float("nan")
            shuffled_std = float("nan")
            t_stat = float("nan")
            p_val = float("nan")
            cohens_d = float("nan")
            tan_shuf = np.array([])

        # 综合判决
        if window_result["verdict"].startswith("strong") and (
            sentence_test and sentence_test["verdict"] == "conservation_broken"
        ):
            verdict = ConformalVerdict.SUPPORT
        elif window_result["verdict"].startswith("weak") or (
            sentence_test and sentence_test.get("verdict") == "conservation_weakened"
        ):
            verdict = ConformalVerdict.WEAK
        elif window_result["verdict"] == "no_local_conservation":
            verdict = ConformalVerdict.REFUTE
        else:
            verdict = ConformalVerdict.INCONCLUSIVE

        return ConservationResult(
            tan_original=tan_orig,
            cv_original=orig_stats["cv"],
            mean_original=orig_stats["mean"],
            std_original=orig_stats["std"],
            tan_shuffled=tan_shuf,
            cv_shuffled=shuffled_cv,
            mean_shuffled=shuffled_mean,
            std_shuffled=shuffled_std,
            window_cv_mean=window_result["local_cv_mean"],
            window_cv_std=window_result["local_cv_std"],
            parallel_transport_residual=window_result["parallel_transport_residual"],
            pt_residual_threshold=window_result["pt_residual_threshold"],
            t_statistic=float(t_stat) if not np.isnan(float(t_stat)) else float("nan"),
            p_value=p_val,
            cohens_d=float(cohens_d) if cohens_d is not None else float("nan"),
            verdict=verdict,
            sentence_level=sentence_test,
            paragraph_level=paragraph_test,
        )


# ── 快速接口 ──────────────────────────────────────────────────────

def conformal_invariance_pipeline(
    kappa: torch.Tensor,
    tau: torch.Tensor,
    window_size: int = 8,
) -> ConservationResult:
    """一键 τ/κ 共形不变验证。

    Args:
        kappa: [T] 或 [B, T] 曲率
        tau: [T] 或 [B, T] 挠率
        window_size: 滑动窗口大小

    Returns:
        ConservationResult
    """
    k_np = kappa.detach().cpu().numpy() if isinstance(kappa, torch.Tensor) else np.array(kappa)
    t_np = tau.detach().cpu().numpy() if isinstance(tau, torch.Tensor) else np.array(tau)

    tan_theta = t_np / (k_np + 1e-10)

    engine = TheoremConservation(window_size=window_size)
    return engine.verify(tan_theta)
