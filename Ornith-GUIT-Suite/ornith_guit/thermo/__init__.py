"""GUIT-TRT 热力学分析子包 (活性涡流论文 §4 实证诊断)。

实现论文提出的微观物理诊断量, 全部可在 OrnithLatentSimulator 代理下
无 GPU 验证 (绝对数值以真实 Ornith 实测为准, 见 GUIT 边界约束声明):

  - NESS (非平衡态定态) 诊断: 熵产生率 σ>0, 概率流 J>0,
    一阶矩速度 ≈0 (宏观细致平衡未破缺), 速度 CV<0.5
  - 异常曲率 K_sub ≈ ||a_perp||/||a_parallel|| (合法轨迹极度平坦 ≈0.01)
  - 有效温度 T_eff 梯度 (pos > rnd, 但 vel_norm 在量化下可能反转)
  - 活性涡流: 速度场雅可比反对称分解 + 随机矩阵 (RMT) Wigner 检验
"""

from .ness import (
    NESSDiagnostics,
    ness_metrics,
    abnormal_curvature,
)
from .vorticity import (
    VorticityAnalyzer,
    estimate_velocity_jacobian,
    decompose_jacobian,
    vorticity_ratio,
    rmt_wigner_test,
    analyze_vorticity,
)

__all__ = [
    "NESSDiagnostics", "ness_metrics", "abnormal_curvature",
    "VorticityAnalyzer", "estimate_velocity_jacobian", "decompose_jacobian",
    "vorticity_ratio", "rmt_wigner_test", "analyze_vorticity",
]
