"""PI-LOOP — 物理感知自优化循环 (Physics-Informed Self-Optimizing Loop)。

将 GUIT 微观物理熔断 (OrnithGuard) 嵌入 LOOP 宏观语义反思, 构成双轨制闭环:
  - 微观物理轨: Decode 过程中实时监控 P_c/P_raw, 逃逸因果脊线立即熔断
  - 宏观语义轨: 融合物理熔断反馈 + 沙盒执行反馈 → 高优先级定向修正
  - 元层进化: 用收敛的优质轨迹在线更新 OrnithAutoCalibrator 脊线基底

设计遵循 GUIT 铁律 (边界约束): 生成后端抽象为 GenerationBackend 协议,
既可对接真实 Ornith (HFGenerationBackend), 也可用 SimulatedCodingBackend
在无 GPU 环境下验证"事中熔断 vs 事后反思"的相对优劣。
"""

from .pi_loop import (
    PhysicsInformedLoop,
    GenerationBackend,
    SimulatedCodingBackend,
    GenerationResult,
)

__all__ = [
    "PhysicsInformedLoop",
    "GenerationBackend",
    "SimulatedCodingBackend",
    "GenerationResult",
]
