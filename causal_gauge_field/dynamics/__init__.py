"""动力学模块 — 几何约束场的演化动力学监控与制动张量调控

step1: 几何动力学监控器 (geometric_dynamics_monitor.py)
    — 基于演化循环理论的三态分类，观察训练过程中几何约束场的动力学演化

step2: FS制动张量 (fs_brake_tensor.py)
    — 四子空间分解对FS标架施加分层动态约束权重
"""

from causal_gauge_field.dynamics.geometric_dynamics_monitor import (
    GeometricDynamicsMonitor,
    DynamicsState,
    EvolutionPhase,
    CycleRecord,
)
from causal_gauge_field.dynamics.fs_brake_tensor import (
    FSBrakeTensor,
    FSSubspace,
    FSBrakeConfig,
    FSBrakeSnapshot,
)

__all__ = [
    "GeometricDynamicsMonitor",
    "DynamicsState",
    "EvolutionPhase",
    "CycleRecord",
    "FSBrakeTensor",
    "FSSubspace",
    "FSBrakeConfig",
    "FSBrakeSnapshot",
]
