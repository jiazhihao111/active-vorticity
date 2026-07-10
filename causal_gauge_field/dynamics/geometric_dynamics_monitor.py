"""几何动力学监控器 — 将EvolutionCycleEngine状态分类逻辑嵌入训练循环
================================================================

来源：叙事大纲生成器的 evolution_cycle_engine.py
改造：将 ψ=叙事自由能 → ψ_geo=几何约束自由能
      三态分类应用于训练epoch，观察约束场的动力学演化

核心公式：
    ψ_geo = w₁·CV(κ) + w₂·CV(τ) + w₃·CV(tanΘ) + w₄·(1.0 - 正交性) + w₅·(1.0/条件数)
    
三态分类（继承自EvolutionCycleEngine）：
    ψ_geo < -0.2: STEADY — 几何约束稳定收敛，可继续训练
    ψ_geo ∈ [-0.2, 0.3]: PRESSURIZED — 几何量波动加大，需关注
    ψ_geo > 0.3: BREAKING — 约束场破缺，建议干预

演化循环（per epoch）：
    1. OBSERVE  → 提取FS几何量，计算ψ_geo
    2. DETECT   → 趋势检测：连续N历元的变化方向
    3. PRESSURIZE → 判定加压需求（过于稳定→需要更多约束难度）
    4. BREAK    → 检测破缺窗口期（ψ超阈值）
    5. RECOMBINE→ 生成约束权重重组建议
    6. SOLIDIFY → 固化有效的约束配置
    
brake_tensor 集成：
    在BREAKING状态下，调用FSBrakeTensor降低幅值制动γ_a
    （允许隐空间探索），在RECOVERING状态下恢复。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ── 演化状态枚举 ─────────────────────────────────────────────────

class EvolutionPhase(str, Enum):
    """演化周期七阶段"""
    OBSERVE = "observe"        # 1. 稳态观察：计算ψ_geo
    DETECT = "detect"          # 2. 趋势检测：分析ψ_geo变化方向
    PRESSURIZE = "pressurize"  # 3. 加压决策：过于稳→提高约束难度
    BREAK = "break"            # 4. 破缺触发：ψ_geo超阈值
    RECOMBINE = "recombine"    # 4.5 重组探索：生成约束权重调整方案
    REACT = "react"            # 5. 反应评估：降低加压水平
    SOLIDIFY = "solidify"      # 5.5 缝合固化：固有效约束配置
    UPDATE = "update"          # 6. 基线更新


class DynamicsState(str, Enum):
    """系统动力学状态"""
    STEADY = "steady"           # 稳态：几何约束收敛
    PRESSURIZED = "pressurized" # 加压中：波动加大
    BREAKING = "breaking"       # 破缺中：约束场退相干
    RECOVERING = "recovering"   # 恢复中：破缺后重建


# ── 数据结构 ────────────────────────────────────────────────────


@dataclass
class GeometricSnapshot:
    """单个epoch的几何快照"""
    epoch: int
    # FS几何量统计
    kappa_mean: float = float('nan')
    kappa_cv: float = float('nan')
    tau_mean: float = float('nan')
    tau_cv: float = float('nan')
    tan_theta_mean: float = float('nan')
    tan_theta_cv: float = float('nan')
    # 标架质量
    frame_orthogonality: float = float('nan')  # T·N, N·B, T·B 的平均正交度
    tn_dot: float = float('nan')
    nb_dot: float = float('nan')
    tb_dot: float = float('nan')
    # MVE 健康度
    mve_condition_number: float = float('nan')
    mve_volume: float = float('nan')  # det(SVD奇异值乘积)
    # 训练信号
    train_loss: float = float('nan')
    rf_loss: float = float('nan')
    closure_contrastive_loss: float = float('nan')
    gradient_norm: float = float('nan')
    # 计算得到的ψ_geo
    psi_geo: float = float('nan')
    # 状态分类
    state: str = "unknown"
    # 时间戳
    timestamp: float = 0.0


@dataclass
class CycleRecord:
    """单次演化循环记录"""
    cycle_id: int
    epoch: int
    phase_progression: List[EvolutionPhase] = field(default_factory=list)
    psi_before: float = float('nan')
    psi_after: float = float('nan')
    initial_state: DynamicsState = DynamicsState.STEADY
    final_state: DynamicsState = DynamicsState.STEADY
    break_triggered: bool = False
    pressure_applied: bool = False
    # 制动张量调整建议
    gamma_adjustments: Dict[str, float] = field(default_factory=dict)
    # 建议
    recommendations: List[str] = field(default_factory=list)
    # 快照
    snapshot: Optional[GeometricSnapshot] = None


# ── 主类 ────────────────────────────────────────────────────────


class GeometricDynamicsMonitor:
    """几何动力学监控器 — 训练过程的约束场演化追踪

    用法:
        monitor = GeometricDynamicsMonitor(
            fs_analyzer, base_dim=128,
            steady_threshold=-0.2, break_threshold=0.3,
        )

        # 每个epoch结束后调用
        for epoch in range(max_epochs):
            train(...)
            snapshot = monitor.record_epoch(
                epoch, hidden_states, train_loss=train_loss,
                rf_loss=rf_loss, gradient_norm=grad_norm,
            )
            cycle = monitor.run_cycle(epoch)
            if cycle.break_triggered:
                # 调整约束权重
                ...

        # 训练结束后
        report = monitor.get_dynamics_report()
    """

    # 默认阈值（与EvolutionCycleEngine一致）
    STEADY_THRESHOLD = -0.2
    BREAK_THRESHOLD = 0.3

    # ψ_geo 权重（可调）
    DEFAULT_WEIGHTS = {
        'kappa_cv': 0.30,       # 曲率变异系数 — 语义弯曲结构稳定性
        'tau_cv': 0.25,         # 挠率变异系数 — 扭转结构稳定性
        'tan_theta_cv': 0.20,   # 螺旋升角CV — 整体几何一致性
        'frame_orthogonality': 0.15,  # FS标架正交性（1-正交度）
        'mve_condition': 0.10,  # MVE条件数倒数 — 表示空间质量
    }

    def __init__(
        self,
        fs_analyzer=None,  # FrenetSerretAnalyzer 实例
        base_dim: int = 128,
        steady_threshold: float = -0.2,
        break_threshold: float = 0.3,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.fs_analyzer = fs_analyzer
        self.base_dim = base_dim
        self.steady_threshold = steady_threshold
        self.break_threshold = break_threshold
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

        # 状态机
        self._current_state = DynamicsState.STEADY
        self._pressure_level = 0.0  # [0, 1]

        # 历史记录
        self._snapshots: List[GeometricSnapshot] = []
        self._cycles: List[CycleRecord] = []
        self._cycle_count = 0
        self._consecutive_breaks = 0
        self._consecutive_steady = 0

        # 制动张量引用（可选集成）
        self._brake_tensor = None

    def attach_brake_tensor(self, brake_tensor) -> None:
        """绑定FS制动张量，启用自动权重调节"""
        self._brake_tensor = brake_tensor

    # ── 主接口：每epoch调用 ──────────────────────────────────

    def record_epoch(
        self,
        epoch: int,
        hidden_states: Optional[torch.Tensor] = None,
        train_loss: float = float('nan'),
        rf_loss: float = float('nan'),
        closure_contrastive_loss: float = float('nan'),
        gradient_norm: float = float('nan'),
    ) -> GeometricSnapshot:
        """记录单个epoch的几何快照，计算ψ_geo并分类状态。

        Args:
            epoch: 当前历元编号
            hidden_states: [B, T, D] 一批验证用的隐状态（可选）
            train_loss / rf_loss / closure_contrastive_loss / gradient_norm: 训练指标

        Returns:
            GeometricSnapshot 包含所有几何量和状态分类
        """
        snap = GeometricSnapshot(
            epoch=epoch,
            train_loss=train_loss,
            rf_loss=rf_loss,
            closure_contrastive_loss=closure_contrastive_loss,
            gradient_norm=gradient_norm,
            timestamp=time.time(),
        )

        # 1. 提取FS几何量
        if hidden_states is not None and self.fs_analyzer is not None:
            try:
                self._compute_fs_geometry(snap, hidden_states)
            except Exception:
                pass  # 几何提取失败时保留nan

        # 2. 计算ψ_geo
        snap.psi_geo = self._compute_psi_geo(snap)

        # 3. 三态分类
        snap.state = self._classify_state(snap.psi_geo).value

        self._snapshots.append(snap)

        # 更新连续计数
        if snap.state == DynamicsState.BREAKING.value:
            self._consecutive_breaks += 1
            self._consecutive_steady = 0
        elif snap.state == DynamicsState.STEADY.value:
            self._consecutive_steady += 1
            self._consecutive_breaks = 0
        else:
            self._consecutive_breaks = max(0, self._consecutive_breaks - 1)
            self._consecutive_steady = max(0, self._consecutive_steady - 1)

        return snap

    def run_cycle(self, epoch: int) -> CycleRecord:
        """基于最近一次快照运行完整七步演化循环。

        Args:
            epoch: 当前历元编号

        Returns:
            CycleRecord 包含完整的演化周期记录
        """
        self._cycle_count += 1
        progression: List[EvolutionPhase] = []

        if not self._snapshots:
            return CycleRecord(cycle_id=self._cycle_count, epoch=epoch)

        snap = self._snapshots[-1]
        psi = snap.psi_geo
        state = DynamicsState(snap.state) if snap.state != "unknown" else self._current_state

        record = CycleRecord(
            cycle_id=self._cycle_count,
            epoch=epoch,
            psi_before=psi,
            initial_state=state,
            snapshot=snap,
        )

        # Step 1: 稳态观察
        progression.append(EvolutionPhase.OBSERVE)

        # Step 2: 趋势检测
        progression.append(EvolutionPhase.DETECT)
        trend = self._detect_trend()

        # Step 3: 加压决策
        progression.append(EvolutionPhase.PRESSURIZE)
        if (state == DynamicsState.STEADY
                and self._pressure_level < 0.1
                and self._consecutive_steady >= 2):
            record.pressure_applied = True
            self._pressure_level = min(1.0, self._pressure_level + 0.2)
            self._current_state = DynamicsState.PRESSURIZED
            record.recommendations.append(
                f"系统几何结构过于稳定(ψ={psi:.3f})，连续{self._consecutive_steady}历元无变化"
            )

        # Step 4: 破缺触发
        progression.append(EvolutionPhase.BREAK)
        if state == DynamicsState.BREAKING or self._pressure_level > 0.7:
            record.break_triggered = True
            self._current_state = DynamicsState.BREAKING

            # 确定破缺类型
            break_type = self._diagnose_break_type(snap)
            record.recommendations.append(
                f"几何约束场破缺(ψ={psi:.3f})：{break_type}"
            )

            # Step 4.5: 重组探索
            progression.append(EvolutionPhase.RECOMBINE)
            record.gamma_adjustments = self._propose_gamma_adjustments(snap, break_type)

            if record.gamma_adjustments:
                items = ", ".join(
                    f"γ_{k}={v:+.2f}"
                    for k, v in record.gamma_adjustments.items()
                )
                record.recommendations.append(f"约束场重组建议：{items}")

        # Step 5: 反应评估
        progression.append(EvolutionPhase.REACT)
        if record.break_triggered:
            self._pressure_level = max(0.0, self._pressure_level - 0.4)
            self._current_state = DynamicsState.RECOVERING

            # Step 5.5: 缝合固化
            progression.append(EvolutionPhase.SOLIDIFY)
            if record.gamma_adjustments:
                solid = self._evaluate_solidification(record)
                if solid:
                    record.recommendations.append(f"固化成功：{solid}")

        # Step 6: 基线更新
        progression.append(EvolutionPhase.UPDATE)
        record.psi_after = psi  # 简化：不重算
        record.final_state = self._current_state
        record.phase_progression = progression

        self._cycles.append(record)
        return record

    # ── ψ_geo 计算 ──────────────────────────────────────────

    def _compute_fs_geometry(
        self,
        snap: GeometricSnapshot,
        hidden: torch.Tensor,
    ) -> None:
        """从隐状态提取FS几何量并填入快照。

        使用局部SVD投影到3D再计算FS标架，
        而后在批次上聚合统计量。
        """
        B, T, D = hidden.shape
        if T < 4:
            return

        # 使用fs_analyzer的批量分析
        try:
            from causal_gauge_field.utils.frenet_serret import compute_batch_spiral_helix_statistics
            stats = compute_batch_spiral_helix_statistics(self.fs_analyzer, hidden)
        except Exception:
            return

        if 'error' in stats:
            return

        # 基础几何量
        snap.tan_theta_mean = stats.get('mean_tan_theta', float('nan'))
        snap.tan_theta_cv = stats.get('cv_tan_theta', float('nan'))

        # 从单个batch中提取更精细的几何量
        try:
            result = self.fs_analyzer.analyze(hidden, compute_chern=False)
            valid = result.valid_mask  # [B, T-3]

            kappa_all = result.kappa[valid].detach().cpu().numpy()
            tau_all = result.tau[valid].detach().cpu().numpy()

            if len(kappa_all) > 0:
                snap.kappa_mean = float(np.mean(kappa_all))
                snap.kappa_cv = float(np.std(kappa_all) / max(np.mean(kappa_all), 1e-8))
                snap.tau_mean = float(np.mean(tau_all))
                snap.tau_cv = float(np.std(tau_all) / max(np.mean(tau_all), 1e-8))

            # FS标架正交性
            if result.T is not None and result.N is not None and result.B is not None:
                # 在3D投影空间中，T=[B, T-3, 3]
                T_valid = result.T[0]  # [T-3, 3] 取第一个batch
                N_valid = result.N[0]
                B_valid = result.B[0]

                # 计算点积（正交性指标：0 = 完美正交）
                tn = torch.abs((T_valid * N_valid).sum(dim=-1)).mean().item()
                nb = torch.abs((N_valid * B_valid).sum(dim=-1)).mean().item()
                tb = torch.abs((T_valid * B_valid).sum(dim=-1)).mean().item()

                snap.tn_dot = float(tn)
                snap.nb_dot = float(nb)
                snap.tb_dot = float(tb)
                snap.frame_orthogonality = float(1.0 - (tn + nb + tb) / 3.0)
        except Exception:
            pass

        # MVE健康度（通过SVD近似）
        try:
            # 将hidden展平为[B*T, D]做SVD
            flat = hidden.reshape(-1, D)
            U, S, V = torch.linalg.svd(flat.float(), full_matrices=False)
            snap.mve_condition_number = float(
                (S.max() / max(S.min(), 1e-8)).item()
            )
            snap.mve_volume = float(torch.prod(S).item())
        except Exception:
            pass

    def _compute_psi_geo(self, snap: GeometricSnapshot) -> float:
        """计算几何约束自由能 ψ_geo。

        ψ_geo = Σ w_i · component_i
        正值 → 几何破缺/不稳定；负值 → 几何收敛/有序
        """
        w = self.weights
        components = []

        # 1. 曲率CV：高CV = 语义弯曲不稳定
        if not np.isnan(snap.kappa_cv):
            # 将原始CV映射到[-0.5, 0.5]区间
            # CV < 0.2 = 很好 → 负贡献；CV > 0.5 = 差 → 正贡献
            kappa_score = (snap.kappa_cv - 0.3) * 2.0
            components.append(('kappa_cv', kappa_score))

        # 2. 挠率CV
        if not np.isnan(snap.tau_cv):
            tau_score = (snap.tau_cv - 0.3) * 2.0
            components.append(('tau_cv', tau_score))

        # 3. 螺旋升角CV
        if not np.isnan(snap.tan_theta_cv):
            theta_score = (snap.tan_theta_cv - 0.3) * 2.0
            components.append(('tan_theta_cv', theta_score))

        # 4. FS标架正交性：越接近1越好
        if not np.isnan(snap.frame_orthogonality):
            # 正交度 → 反向映射（低正交度 = 高风险）
            ortho_score = 1.0 - 2.0 * snap.frame_orthogonality
            components.append(('frame_orthogonality', ortho_score))

        # 5. MVE条件数倒数
        if not np.isnan(snap.mve_condition_number):
            # 条件数越小越好，映射到[-0.3, 0.3]
            cond_score = min(max(np.log10(max(snap.mve_condition_number, 1.0)) - 1.0, -0.3), 0.3)
            components.append(('mve_condition', cond_score))

        if not components:
            return float('nan')

        # 加权求和
        psi = sum(w.get(name, 0.1) * score for name, score in components)

        return float(psi)

    # ── 状态分类 ────────────────────────────────────────────

    def _classify_state(self, psi: float) -> DynamicsState:
        """根据ψ_geo分类系统状态（继承自EvolutionCycleEngine）"""
        if np.isnan(psi):
            return self._current_state

        # 破缺/恢复中 → 强制过渡逻辑
        if self._current_state in (DynamicsState.BREAKING, DynamicsState.RECOVERING):
            if psi < self.STEADY_THRESHOLD:
                return DynamicsState.STEADY
            else:
                return DynamicsState.RECOVERING

        if psi < self.STEADY_THRESHOLD:
            return DynamicsState.STEADY
        elif psi < self.BREAK_THRESHOLD:
            return DynamicsState.PRESSURIZED
        else:
            return DynamicsState.BREAKING

    # ── 趋势检测 ────────────────────────────────────────────

    def _detect_trend(self, window: int = 3) -> str:
        """检测ψ_geo的近期趋势"""
        if len(self._snapshots) < window:
            return "insufficient_data"

        recent = self._snapshots[-window:]
        psis = [s.psi_geo for s in recent if not np.isnan(s.psi_geo)]

        if len(psis) < 2:
            return "insufficient_data"

        # 简单线性趋势
        x = np.arange(len(psis))
        y = np.array(psis)
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.02:
            return "stable"
        elif slope > 0:
            return "deteriorating" if slope > 0.05 else "slowly_deteriorating"
        else:
            return "improving" if slope < -0.05 else "slowly_improving"

    # ── 破缺诊断 ────────────────────────────────────────────

    def _diagnose_break_type(self, snap: GeometricSnapshot) -> str:
        """诊断破缺类型（对应FS标架四种退化模式）"""
        parts = []

        if not np.isnan(snap.kappa_cv) and snap.kappa_cv > 0.5:
            parts.append("曲率退化(语义弯曲异常)")
        if not np.isnan(snap.tau_cv) and snap.tau_cv > 0.5:
            parts.append("挠率异常(语义扭转失序)")
        if not np.isnan(snap.frame_orthogonality) and snap.frame_orthogonality < 0.5:
            parts.append("标架退相干(T-N-B正交性丧失)")
        if not np.isnan(snap.mve_condition_number) and snap.mve_condition_number > 100:
            parts.append("MVE条件崩溃(表示空间病态)")

        if not parts:
            # 综合破缺
            if not np.isnan(snap.tan_theta_cv) and snap.tan_theta_cv > 0.5:
                parts.append("螺旋结构崩解(τ/κ比失稳)")

        return " + ".join(parts) if parts else "综合破缺(ψ超标)"

    # ── 约束场重组 ──────────────────────────────────────────

    def _propose_gamma_adjustments(
        self,
        snap: GeometricSnapshot,
        break_type: str,
    ) -> Dict[str, float]:
        """基于破缺类型生成四维制动权重调整建议。

        Returns:
            dict: {'gamma_T': Δ, 'gamma_N': Δ, 'gamma_B': Δ, 'gamma_tau': Δ}
            正值=加强约束，负值=放松约束
        """
        adjustments: Dict[str, float] = {}

        # 曲率退化 → 加强法向约束（限制弯曲强度）
        if '曲率' in break_type and not np.isnan(snap.kappa_cv):
            severity = min(snap.kappa_cv / 0.5, 2.0)
            adjustments['gamma_N'] = +0.15 * severity

        # 挠率异常 → 加强副法向约束（限制扭转自由度）
        if '挠率' in break_type and not np.isnan(snap.tau_cv):
            severity = min(snap.tau_cv / 0.5, 2.0)
            adjustments['gamma_B'] = +0.15 * severity

        # 标架退相干 → 加强方向约束（语义方向一致性）
        if '退相干' in break_type:
            ortho_loss = 1.0 - max(snap.frame_orthogonality, 0.0)
            adjustments['gamma_T'] = +0.20 * ortho_loss

        # MVE崩溃 → 加强结构惯性和幅值约束
        if '条件崩溃' in break_type:
            adjustments['gamma_tau'] = +0.10
            adjustments['gamma_N'] = adjustments.get('gamma_N', 0.0) + 0.05

        # 综合破缺 → 全维度收紧
        if '综合破缺' in break_type:
            adjustments.update({
                'gamma_T': +0.10,
                'gamma_N': +0.10,
                'gamma_B': +0.10,
                'gamma_tau': +0.10,
            })

        return adjustments

    def _evaluate_solidification(self, record: CycleRecord) -> Optional[str]:
        """评估重组方案的有效性"""
        if len(self._snapshots) < 2:
            return None

        # 检查重组后的ψ是否降低
        prev = self._snapshots[-2]
        curr = self._snapshots[-1]

        if np.isnan(prev.psi_geo) or np.isnan(curr.psi_geo):
            return None

        delta_psi = curr.psi_geo - prev.psi_geo

        if delta_psi < -0.05:
            return f"ψ改善({delta_psi:+.3f})，约束重组有效"
        elif delta_psi > 0.05:
            return f"ψ恶化({delta_psi:+.3f})，约束重组需回滚"
        else:
            return f"ψ持平({delta_psi:+.3f})，进入观察期"

    # ── 报告生成 ────────────────────────────────────────────

    def get_dynamics_report(self) -> Dict[str, Any]:
        """生成完整的动力学演化报告"""
        if not self._snapshots:
            return {"error": "no_data"}

        # 基本统计
        psis = [s.psi_geo for s in self._snapshots if not np.isnan(s.psi_geo)]
        states = [s.state for s in self._snapshots]

        n_breaks = states.count(DynamicsState.BREAKING.value)
        n_steady = states.count(DynamicsState.STEADY.value)
        n_pressurized = states.count(DynamicsState.PRESSURIZED.value)

        # 破缺事件详情
        break_events = []
        for cycle in self._cycles:
            if cycle.break_triggered:
                break_events.append({
                    "epoch": cycle.epoch,
                    "psi": cycle.psi_before,
                    "type": (
                        cycle.recommendations[1]
                        if len(cycle.recommendations) > 1
                        else "unknown"
                    ),
                    "gamma_adjustments": cycle.gamma_adjustments,
                })

        # 趋势
        trend = self._detect_trend(window=min(5, len(self._snapshots)))

        # 几何量演化曲线
        evolution = {
            "epochs": [s.epoch for s in self._snapshots],
            "psi_geo": [s.psi_geo for s in self._snapshots],
            "kappa_mean": [s.kappa_mean for s in self._snapshots],
            "tau_mean": [s.tau_mean for s in self._snapshots],
            "tan_theta_cv": [s.tan_theta_cv for s in self._snapshots],
            "frame_orthogonality": [s.frame_orthogonality for s in self._snapshots],
            "mve_condition_number": [s.mve_condition_number for s in self._snapshots],
            "states": states,
        }

        # 动力学阶段判定
        if len(psis) >= 2:
            psi_mean = float(np.mean(psis))
            psi_std = float(np.std(psis))
            psi_final = psis[-1]
            psi_trend = "converging" if trend.startswith("improving") else (
                "diverging" if trend.startswith("deteriorating") else "oscillating"
            )
        else:
            psi_mean = float(psis[0]) if psis else float('nan')
            psi_std = float('nan')
            psi_final = float(psis[0]) if psis else float('nan')
            psi_trend = "unknown"

        return {
            "total_epochs": len(self._snapshots),
            "total_cycles": len(self._cycles),
            "psi_statistics": {
                "mean": psi_mean,
                "std": psi_std,
                "final": psi_final,
                "trend": psi_trend,
            },
            "state_distribution": {
                "steady": n_steady,
                "pressurized": n_pressurized,
                "breaking": n_breaks,
                "break_rate": n_breaks / max(len(states), 1),
            },
            "break_events": break_events,
            "evolution_curve": _to_native(evolution),
            "current_state": self._current_state.value,
            "pressure_level": self._pressure_level,
            "dynamics_verdict": self._dynamics_verdict(psi_final, psi_trend, n_breaks),
        }

    def _dynamics_verdict(
        self,
        psi_final: float,
        psi_trend: str,
        n_breaks: int,
    ) -> str:
        """总体动力学判决"""
        if np.isnan(psi_final):
            return "动力学数据不足"

        if psi_trend == "converging" and psi_final < self.STEADY_THRESHOLD:
            return "几何约束场稳定收敛 — 动力学健康"
        elif psi_trend == "converging" and psi_final < self.BREAK_THRESHOLD:
            return "几何约束场趋于收敛 — 但仍需监控"
        elif psi_trend == "diverging":
            return "几何约束场发散 — 动力学恶化，需检讨约束设计"
        elif n_breaks > 2:
            return "频繁破缺 — 约束场不稳定，建议降低约束强度或检查几何本体论"
        else:
            return "几何约束场在边缘振荡 — 动力学处于临界状态"

    # ── 属性 ────────────────────────────────────────────────

    @property
    def current_state(self) -> DynamicsState:
        return self._current_state

    @property
    def pressure_level(self) -> float:
        return self._pressure_level

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def snapshots(self) -> List[GeometricSnapshot]:
        return list(self._snapshots)

    @property
    def cycles(self) -> List[CycleRecord]:
        return list(self._cycles)


# ── 工具函数 ────────────────────────────────────────────────────


def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch is not None and torch.is_tensor(obj):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.tolist()
    return obj
