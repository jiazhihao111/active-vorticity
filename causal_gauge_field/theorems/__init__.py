"""GUIT-TRT 三大定理模块
=====================

公理："信息化为世界模型，世界模型遵守几何规则"

三条核心定理：
    定理一（信息本体论）：连贯叙事系统的隐空间 χ ≠ 0
    定理二（几何守恒论）：合法路径上 τ/κ 共形不变
    定理三（拓扑残差论）：错误是拓扑闭合的必然几何残差，不可完全消除

使用方式：
    from causal_gauge_field.theorems import (
        TheoremOntology, TheoremConservation, TheoremResidual,
        TheoremVerifier, TheoremResult,
    )
"""

from .theorem_1_ontology import (
    TheoremOntology,
    PersistenceResult,
    BettiNumbers,
    OntologyVerdict,
)

from .theorem_2_conservation import (
    TheoremConservation,
    ConservationResult,
    ConformalVerdict,
)

from .theorem_3_residual import (
    TheoremResidual,
    ResidualResult,
    ResidualAsymptote,
    ResidualVerdict,
)

from .theorem_verifier import (
    TheoremVerifier,
    TheoremResult,
    VerificationReport,
    TheoremStatus,
)

__all__ = [
    # 定理一
    "TheoremOntology",
    "PersistenceResult",
    "BettiNumbers",
    "OntologyVerdict",
    # 定理二
    "TheoremConservation",
    "ConservationResult",
    "ConformalVerdict",
    # 定理三
    "TheoremResidual",
    "ResidualResult",
    "ResidualAsymptote",
    "ResidualVerdict",
    # 验证器
    "TheoremVerifier",
    "TheoremResult",
    "VerificationReport",
    "TheoremStatus",
]
