"""定理验证器 — 统一框架
========================

将三条定理的验证整合为统一流水线，以十三字公理为归一化入口。

流水线：
    输入：模型隐状态 + FS 几何量 + 跨上下文长度误差
      ↓
    定理一：持久同调 → χ → 本体论判决
      ↓
    定理二：τ/κ 共形不变 → 守恒论判决
      ↓
    定理三：残差渐进界 → 残差论判决（使用定理一的 χ）
      ↓
    综合报告：三条定理的整体置信度 + 公理验证状态

十三字公理映射：
    "信息化为世界模型"  → 定理一（χ ≠ 0）
    "世界模型遵守"       → 定理二（τ/κ const）
    "几何规则"           → 定理三（残差不可避免）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .theorem_1_ontology import TheoremOntology, OntologyResult, OntologyVerdict
from .theorem_2_conservation import TheoremConservation, ConservationResult, ConformalVerdict
from .theorem_3_residual import TheoremResidual, ResidualResult, ResidualVerdict


# ── 数据结构 ──────────────────────────────────────────────────────

class TheoremStatus(str, Enum):
    """单个定理的验证状态"""
    PASS = "PASS"  # 定理被实验支持
    WEAK = "WEAK"  # 趋势正确但不显著
    FAIL = "FAIL"  # 定理未被支持
    PENDING = "PENDING"  # 未执行
    ERROR = "ERROR"  # 执行出错


@dataclass
class TheoremResult:
    """单个定理的验证结果"""
    theorem_id: int
    theorem_name: str
    status: TheoremStatus
    verdict: Any  # OntologyVerdict / ConformalVerdict / ResidualVerdict
    confidence: float  # [0, 1] 置信度
    details: Dict = field(default_factory=dict)
    error_message: str = ""


@dataclass
class VerificationReport:
    """完整验证报告"""
    axiom: str = "信息化为世界模型，世界模型遵守几何规则"

    # 三条定理结果
    theorem_1: Optional[TheoremResult] = None
    theorem_2: Optional[TheoremResult] = None
    theorem_3: Optional[TheoremResult] = None

    # 综合指标
    overall_support: float = 0.0  # [0, 1] 三条定理综合支持度
    passed_count: int = 0
    total_count: int = 3
    axiom_verified: bool = False

    # 理论自洽性
    self_consistency_score: float = 0.0  # [0,1] 三条定理结果是否自洽

    # 诊断
    diagnostics: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"━━━ GUIT-TRT 定理验证报告 ━━━", "",
                 f"公理：{self.axiom}", "",
                 f"定理一（信息本体论）：{self._t1_status()}",
                 f"定理二（几何守恒论）：{self._t2_status()}",
                 f"定理三（拓扑残差论）：{self._t3_status()}",
                 "",
                 f"综合支持度：{self.overall_support:.2%}",
                 f"通过数：{self.passed_count}/{self.total_count}",
                 f"公理验证：{'✓ 通过' if self.axiom_verified else '✗ 未通过'}",
                 f"自洽性：{self.self_consistency_score:.2%}",
        ]
        return "\n".join(lines)

    def _t1_status(self) -> str:
        t = self.theorem_1
        if t is None:
            return "⏳ 未执行"
        return f"{self._emoji(t.status)} {t.verdict.value if t.verdict else 'N/A'} (置信度 {t.confidence:.2%})"

    def _t2_status(self) -> str:
        t = self.theorem_2
        if t is None:
            return "⏳ 未执行"
        return f"{self._emoji(t.status)} {t.verdict.value if t.verdict else 'N/A'} (置信度 {t.confidence:.2%})"

    def _t3_status(self) -> str:
        t = self.theorem_3
        if t is None:
            return "⏳ 未执行"
        return f"{self._emoji(t.status)} {t.verdict.value if t.verdict else 'N/A'} (置信度 {t.confidence:.2%})"

    @staticmethod
    def _emoji(status: TheoremStatus) -> str:
        return {
            TheoremStatus.PASS: "✅",
            TheoremStatus.WEAK: "⚠️",
            TheoremStatus.FAIL: "❌",
            TheoremStatus.PENDING: "⏳",
            TheoremStatus.ERROR: "💥",
        }.get(status, "❓")

    def to_dict(self) -> Dict:
        return {
            "axiom": self.axiom,
            "theorem_1": self._t_to_dict(self.theorem_1),
            "theorem_2": self._t_to_dict(self.theorem_2),
            "theorem_3": self._t_to_dict(self.theorem_3),
            "overall_support": self.overall_support,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "axiom_verified": self.axiom_verified,
            "self_consistency_score": self.self_consistency_score,
        }

    @staticmethod
    def _t_to_dict(t: Optional[TheoremResult]) -> Optional[Dict]:
        if t is None:
            return None
        return {
            "id": t.theorem_id,
            "name": t.theorem_name,
            "status": t.status.value,
            "verdict": t.verdict.value if hasattr(t.verdict, 'value') else str(t.verdict),
            "confidence": t.confidence,
            "details": {k: str(v) for k, v in t.details.items()},
            "error": t.error_message,
        }


# ── 核心实现 ──────────────────────────────────────────────────────

class TheoremVerifier:
    """三条定理的统一验证引擎。

    使用方式：
        verifier = TheoremVerifier(fs_analyzer)
        report = verifier.verify_all(
            hidden_pos=pos_hidden,
            hidden_neg=neg_hidden,
            tan_theta_original=tan_orig,
            context_lengths=[32, 64, 128, 256],
            errors=[0.3, 0.2, 0.15, 0.12],
        )
        print(report.summary())
    """

    def __init__(
        self,
        fs_analyzer=None,
        base_dim: int = 128,
    ):
        """
        Args:
            fs_analyzer: FrenetSerretAnalyzer 实例（计算 κ, τ, tanΘ）
            base_dim: 基础维度（用于定理一持久同调）
        """
        self.fs_analyzer = fs_analyzer
        self.base_dim = base_dim

        # 初始化三条定理引擎
        self.onto = TheoremOntology()
        self.conserv = TheoremConservation()
        self.resid = TheoremResidual()

        # 结果缓存
        self._onto_result: Optional[OntologyResult] = None
        self._conserv_result: Optional[ConservationResult] = None
        self._resid_result: Optional[ResidualResult] = None

    # ── 完整流水线 ───────────────────────────────────────────────

    def verify_all(
        self,
        hidden_pos: Optional[List[torch.Tensor]] = None,
        hidden_neg: Optional[List[torch.Tensor]] = None,
        tan_theta_original: Optional[np.ndarray] = None,
        tan_theta_shuffled: Optional[np.ndarray] = None,
        context_lengths: Optional[List[int]] = None,
        errors: Optional[List[float]] = None,
        model_capacity: Optional[float] = None,
    ) -> VerificationReport:
        """运行完整的三定理验证流水线。

        所有参数均为可选——未提供的定理将标记为 PENDING。

        Args:
            hidden_pos: 正例隐状态列表 [用于定理一]
            hidden_neg: 负例隐状态列表 [用于定理一]
            tan_theta_original: 原始 tanΘ 序列 [用于定理二]
            tan_theta_shuffled: 打乱 tanΘ 序列 [用于定理二]
            context_lengths: 上下文长度序列 [用于定理三]
            errors: 对应误差序列 [用于定理三]
            model_capacity: 模型容量的 log10 [用于定理三理论预测]

        Returns:
            VerificationReport
        """
        report = VerificationReport()

        # ── 定理一 ──
        if hidden_pos is not None and hidden_neg is not None:
            try:
                self._onto_result = self.onto.verify(hidden_pos, hidden_neg)
                t1_confidence = self._compute_confidence_onto(self._onto_result)
                report.theorem_1 = TheoremResult(
                    theorem_id=1,
                    theorem_name="信息本体论 (χ ≠ 0)",
                    status=self._map_onto_status(self._onto_result.verdict),
                    verdict=self._onto_result.verdict,
                    confidence=t1_confidence,
                    details={
                        "chi_pos_mean": self._onto_result.mean_chi_pos,
                        "chi_neg_mean": self._onto_result.mean_chi_neg,
                        "cohens_d": self._onto_result.cohens_d,
                        "p_value": self._onto_result.p_value,
                    },
                )
            except Exception as e:
                report.theorem_1 = TheoremResult(
                    theorem_id=1, theorem_name="信息本体论 (χ ≠ 0)",
                    status=TheoremStatus.ERROR, verdict="ERROR",
                    confidence=0.0, error_message=str(e),
                )
        else:
            report.theorem_1 = TheoremResult(
                theorem_id=1, theorem_name="信息本体论 (χ ≠ 0)",
                status=TheoremStatus.PENDING, verdict="PENDING", confidence=0.0,
            )

        # ── 定理二 ──
        if tan_theta_original is not None:
            try:
                self._conserv_result = self.conserv.verify(
                    tan_theta_original,
                    tan_theta_shuffled=tan_theta_shuffled,
                )
                t2_confidence = self._compute_confidence_conserv(self._conserv_result)
                report.theorem_2 = TheoremResult(
                    theorem_id=2,
                    theorem_name="几何守恒论 (τ/κ = const)",
                    status=self._map_conserv_status(self._conserv_result.verdict),
                    verdict=self._conserv_result.verdict,
                    confidence=t2_confidence,
                    details={
                        "cv_original": self._conserv_result.cv_original,
                        "cv_shuffled": self._conserv_result.cv_shuffled,
                        "pt_residual": self._conserv_result.parallel_transport_residual,
                        "window_cv_ratio": (
                            self._conserv_result.window_cv_mean /
                            max(self._conserv_result.cv_original, 1e-10)
                        ),
                    },
                )
            except Exception as e:
                report.theorem_2 = TheoremResult(
                    theorem_id=2, theorem_name="几何守恒论 (τ/κ = const)",
                    status=TheoremStatus.ERROR, verdict="ERROR",
                    confidence=0.0, error_message=str(e),
                )
        else:
            report.theorem_2 = TheoremResult(
                theorem_id=2, theorem_name="几何守恒论 (τ/κ = const)",
                status=TheoremStatus.PENDING, verdict="PENDING", confidence=0.0,
            )

        # ── 定理三 ──
        if context_lengths is not None and errors is not None and len(context_lengths) >= 3:
            try:
                # 从定理一获取 χ
                chi_estimate = None
                if self._onto_result is not None:
                    chi_estimate = self._onto_result.mean_chi_pos

                self._resid_result = self.resid.verify(
                    context_lengths, errors,
                    chi_estimate=chi_estimate,
                    model_capacity=model_capacity,
                )
                t3_confidence = self._compute_confidence_resid(self._resid_result)
                report.theorem_3 = TheoremResult(
                    theorem_id=3,
                    theorem_name="拓扑残差论 (ε_min > 0)",
                    status=self._map_resid_status(self._resid_result.verdict),
                    verdict=self._resid_result.verdict,
                    confidence=t3_confidence,
                    details={
                        "epsilon_min": self._resid_result.asymptote.epsilon_min,
                        "r_squared": self._resid_result.asymptote.r_squared,
                        "p_positive": self._resid_result.p_value_positive,
                        "predicted_eps_min": self._resid_result.predicted_epsilon_min,
                    },
                )
            except Exception as e:
                report.theorem_3 = TheoremResult(
                    theorem_id=3, theorem_name="拓扑残差论 (ε_min > 0)",
                    status=TheoremStatus.ERROR, verdict="ERROR",
                    confidence=0.0, error_message=str(e),
                )
        else:
            report.theorem_3 = TheoremResult(
                theorem_id=3, theorem_name="拓扑残差论 (ε_min > 0)",
                status=TheoremStatus.PENDING, verdict="PENDING", confidence=0.0,
            )

        # ── 综合指标 ──
        report = self._compute_overall(report)

        return report

    # ── 综合指标计算 ─────────────────────────────────────────────

    def _compute_overall(self, report: VerificationReport) -> VerificationReport:
        """计算综合指标"""
        confidences = []
        passed = 0
        total = 0

        for t in [report.theorem_1, report.theorem_2, report.theorem_3]:
            if t is None:
                continue
            total += 1
            if t.status == TheoremStatus.PASS:
                passed += 1
                confidences.append(t.confidence)
            elif t.status == TheoremStatus.WEAK:
                confidences.append(t.confidence * 0.5)
            elif t.status == TheoremStatus.FAIL:
                confidences.append(0.0)

        report.passed_count = passed
        report.total_count = total

        if confidences:
            report.overall_support = float(np.mean(confidences))
        else:
            report.overall_support = 0.0

        # 公理验证：至少两条定理 PASS + 整体支持度 > 0.5
        report.axiom_verified = (
            passed >= 2 and report.overall_support > 0.5
        )

        # 自洽性：三条定理的结果是否一致
        report.self_consistency_score = self._compute_consistency(report)

        # 诊断
        report.diagnostics = {
            "theorem_1_missing": report.theorem_1 is None or report.theorem_1.status == TheoremStatus.PENDING,
            "theorem_2_missing": report.theorem_2 is None or report.theorem_2.status == TheoremStatus.PENDING,
            "theorem_3_missing": report.theorem_3 is None or report.theorem_3.status == TheoremStatus.PENDING,
            "inconsistency_warning": report.self_consistency_score < 0.3,
        }

        return report

    def _compute_consistency(self, report: VerificationReport) -> float:
        """计算三条定理之间的自洽性。

        定理一和二应该同向（χ ≠ 0 意味着 τ/κ 有意义），
        定理三应该是前两条的结果（χ → 残差）。
        """
        scores = []

        t1_pass = report.theorem_1 is not None and report.theorem_1.status == TheoremStatus.PASS
        t2_pass = report.theorem_2 is not None and report.theorem_2.status == TheoremStatus.PASS
        t3_pass = report.theorem_3 is not None and report.theorem_3.status == TheoremStatus.PASS

        # T1 和 T2 应该同向
        if t1_pass is not None and t2_pass is not None:
            scores.append(1.0 if t1_pass == t2_pass else 0.0)

        # T3 应该 ≥ T1, T2 的最小值（残差不比信息更不确定）
        if t3_pass is not None:
            if not t1_pass and not t2_pass:
                scores.append(0.5)
            else:
                scores.append(1.0)

        return float(np.mean(scores)) if scores else 0.0

    # ── 信心计算 ─────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence_onto(result: OntologyResult) -> float:
        if result.verdict == OntologyVerdict.SUPPORT:
            base = 0.9
        elif result.verdict == OntologyVerdict.WEAK:
            base = 0.6
        elif result.verdict == OntologyVerdict.REFUTE:
            base = 0.1
        else:
            return 0.3

        # Cohen's d 惩罚
        d = abs(result.cohens_d) if not np.isnan(result.cohens_d) else 0
        d_factor = min(d / 0.8, 1.0)  # d ≥ 0.8 = 大效应

        # p 值惩罚
        p = result.p_value if not np.isnan(result.p_value) else 0.5
        p_factor = max(0.0, 1.0 - p / 0.05)

        return base * 0.5 + 0.5 * (d_factor * 0.6 + p_factor * 0.4)

    @staticmethod
    def _compute_confidence_conserv(result: ConservationResult) -> float:
        if result.verdict == ConformalVerdict.SUPPORT:
            base = 0.9
        elif result.verdict == ConformalVerdict.WEAK:
            base = 0.6
        elif result.verdict == ConformalVerdict.REFUTE:
            base = 0.1
        else:
            return 0.3

        # CV 比率
        cv = result.cv_original
        cv_factor = max(0.0, 1.0 - cv / 0.5)

        # 平行移动残差
        pt = result.parallel_transport_residual
        pt_thresh = result.pt_residual_threshold
        pt_factor = max(0.0, 1.0 - pt / (pt_thresh + 1e-10)) if pt_thresh > 0 else 0.5

        # 效应量
        d = abs(result.cohens_d) if not np.isnan(result.cohens_d) else 0
        d_factor = min(d / 0.5, 1.0)

        return base * 0.4 + 0.6 * (cv_factor * 0.3 + pt_factor * 0.3 + d_factor * 0.4)

    @staticmethod
    def _compute_confidence_resid(result: ResidualResult) -> float:
        if result.verdict == ResidualVerdict.SUPPORT:
            base = 0.9
        elif result.verdict == ResidualVerdict.WEAK:
            base = 0.6
        elif result.verdict == ResidualVerdict.REFUTE:
            base = 0.1
        else:
            return 0.3

        # 拟合优度
        r2 = max(0.0, result.asymptote.r_squared)
        r2_factor = min(r2 / 0.7, 1.0)

        # Bootstrap
        p_pos = result.p_value_positive
        p_factor = max(0.0, 1.0 - p_pos / 0.05)

        # 效应量（ε_min 的显著程度）
        eps_min = result.asymptote.epsilon_min
        eps_range = result.asymptote.epsilon_0 - eps_min
        eps_factor = min(eps_min / (eps_range + 1e-10) * 2, 1.0) if eps_range > 0 else 0.5

        return base * 0.4 + 0.6 * (r2_factor * 0.3 + p_factor * 0.4 + eps_factor * 0.3)

    # ── 状态映射 ─────────────────────────────────────────────────

    @staticmethod
    def _map_onto_status(v: OntologyVerdict) -> TheoremStatus:
        mapping = {
            OntologyVerdict.SUPPORT: TheoremStatus.PASS,
            OntologyVerdict.WEAK: TheoremStatus.WEAK,
            OntologyVerdict.REFUTE: TheoremStatus.FAIL,
            OntologyVerdict.INCONCLUSIVE: TheoremStatus.PENDING,
        }
        return mapping.get(v, TheoremStatus.PENDING)

    @staticmethod
    def _map_conserv_status(v: ConformalVerdict) -> TheoremStatus:
        mapping = {
            ConformalVerdict.SUPPORT: TheoremStatus.PASS,
            ConformalVerdict.WEAK: TheoremStatus.WEAK,
            ConformalVerdict.REFUTE: TheoremStatus.FAIL,
            ConformalVerdict.INCONCLUSIVE: TheoremStatus.PENDING,
        }
        return mapping.get(v, TheoremStatus.PENDING)

    @staticmethod
    def _map_resid_status(v: ResidualVerdict) -> TheoremStatus:
        mapping = {
            ResidualVerdict.SUPPORT: TheoremStatus.PASS,
            ResidualVerdict.WEAK: TheoremStatus.WEAK,
            ResidualVerdict.REFUTE: TheoremStatus.FAIL,
            ResidualVerdict.INCONCLUSIVE: TheoremStatus.PENDING,
        }
        return mapping.get(v, TheoremStatus.PENDING)

    # ── 增量验证 ─────────────────────────────────────────────────

    def verify_theorem_1(
        self, hidden_pos: List[torch.Tensor], hidden_neg: List[torch.Tensor]
    ) -> TheoremResult:
        """单独验证定理一"""
        return self.verify_all(hidden_pos=hidden_pos, hidden_neg=hidden_neg).theorem_1

    def verify_theorem_2(
        self, tan_theta_original: np.ndarray, tan_theta_shuffled: Optional[np.ndarray] = None
    ) -> TheoremResult:
        """单独验证定理二"""
        return self.verify_all(
            tan_theta_original=tan_theta_original,
            tan_theta_shuffled=tan_theta_shuffled,
        ).theorem_2

    def verify_theorem_3(
        self, context_lengths: List[int], errors: List[float],
        chi_estimate: Optional[float] = None,
    ) -> TheoremResult:
        """单独验证定理三"""
        return self.verify_all(
            context_lengths=context_lengths, errors=errors,
        ).theorem_3


# ── 快速接口 ──────────────────────────────────────────────────────

def axiom_verification_pipeline(
    hidden_pos: Optional[List[torch.Tensor]] = None,
    hidden_neg: Optional[List[torch.Tensor]] = None,
    tan_theta_original: Optional[np.ndarray] = None,
    context_lengths: Optional[List[int]] = None,
    errors: Optional[List[float]] = None,
) -> VerificationReport:
    """一键公理验证流水线。

    十三字公理："信息化为世界模型，世界模型遵守几何规则"
    → 定理一(χ≠0) → 定理二(τ/κ=const) → 定理三(ε_min>0)

    >>> report = axiom_verification_pipeline(
    ...     hidden_pos=pos_hidden_list,
    ...     hidden_neg=neg_hidden_list,
    ...     tan_theta_original=tan_theta_array,
    ...     context_lengths=[32, 64, 128, 256],
    ...     errors=[0.3, 0.2, 0.15, 0.12],
    ... )
    >>> print(report.summary())
    """
    verifier = TheoremVerifier()
    return verifier.verify_all(
        hidden_pos=hidden_pos,
        hidden_neg=hidden_neg,
        tan_theta_original=tan_theta_original,
        context_lengths=context_lengths,
        errors=errors,
    )
