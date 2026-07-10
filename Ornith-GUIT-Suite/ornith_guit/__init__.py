"""Ornith-1.0-9B 专属 GUIT 工程化套件 (Ornith-GUIT-Suite).

将 GUIT-TRT (大语言模型隐空间非平衡态热力学) 理论工程化，
为 Ornith-1.0-9B (Qwen3_5 架构, 4096 维, 混合注意力) 的
Agentic-Coding / Self-Scaffolding 场景提供专属组件。

核心理论锚点 (来自 2026.710.0.30 论文实证):
  - 仿射-非完整混合约束原理: R^2=1.0, 异常曲率 K_sub≈0.01
  - 活性驱动力系数 alpha* ≈ 1.41 (Ornith bf16 校准值)
  - 有效秩 r(0.95) = 25 (压缩率 99.4%)
  - 幻觉相变指标 P_c/P_raw ≈ 0.09-0.15 (逐 token 计算)
  - 4-bit 量化使 r 膨胀 50-64%，但 P_c/P_raw 保持鲁棒

边界约束 (GUIT 铁律):
  - 本套件仅适用于推理阶段冻结权重的 LLM
  - 脊线提取 (SVD) 必须在 bf16/fp32 下进行；P_c/P_raw 检测可在量化下使用
  - P_c/P_raw 必须使用逐 token 计算，禁止批量平均 (假象≈0.84)
"""

from .physics import ThermoPhysics, calibrate_alpha_star
from .core.ornith_calibrator import OrnithAutoCalibrator
from .core.streaming_compressor import StreamingAffineCompressor
from .detection.ornith_guard import OrnithGuard, PhaseTransitionException
from .detection.detector import HallucinationDetector, batch_vs_token_pc
from .steering.causal_kv import DynamicKVCacheEvictor
from .steering.affine_projector import AffineConstraintProjector
from .thermo import (
    NESSDiagnostics, ness_metrics, abnormal_curvature,
    VorticityAnalyzer, analyze_vorticity,
)
from .loop import (
    PhysicsInformedLoop,
    GenerationBackend,
    SimulatedCodingBackend,
    GenerationResult,
)

__all__ = [
    "ThermoPhysics", "calibrate_alpha_star",
    "OrnithAutoCalibrator", "StreamingAffineCompressor",
    "OrnithGuard", "PhaseTransitionException",
    "HallucinationDetector", "batch_vs_token_pc",
    "DynamicKVCacheEvictor", "AffineConstraintProjector",
    "NESSDiagnostics", "ness_metrics", "abnormal_curvature",
    "VorticityAnalyzer", "analyze_vorticity",
    "PhysicsInformedLoop", "GenerationBackend",
    "SimulatedCodingBackend", "GenerationResult",
]

__version__ = "1.0.0"
__target_model__ = "Ornith-1.0-9B"
__theory__ = "GUIT-TRT v9.2 (Non-Equilibrium Active Vorticity)"
