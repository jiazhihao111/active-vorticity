"""FS制动张量 — 四子空间分解对Frenet-Serret标架的分层动态约束
===================================================================

来源：CDNE项目的 brake_tensor.py 四子空间正交投影分解
改造：将制动张量 Γ_μν 映射到 FS标架的四维几何量

映射关系：
    CDNE制动张量分解           FS标架约束场映射
    ─────────────────────      ──────────────────
    S_d  (方向性制动)     →    γ_T  — 切向量一致性约束
    S_c  (刚性约束)       →    γ_N  — 法向量弯曲强度约束
    S_a  (幅值性制动)     →    γ_B  — 副法向量扭转自由度约束
    S_s  (结构惯性)       →    γ_τ  — 挠率变化率惯性约束

核心机制（继承自CDNE制动张量）：
    1. 四子空间正交投影分解：Γ = γ_T·P^(T) + γ_N·P^(N) + γ_B·P^(B) + γ_τ·P^(τ)
    2. 破缺窗口期变换：γ_B → ε_B (允许扭转探索)，γ_τ → γ_τ' (部分释放结构惯性)
    3. 自由能预算：γ更新量由剩余预算tanh(B_remaining)调制
    4. 紧急制动：退相干时强保守模式
    5. 距离延迟信号：约束变化不立即全局传播

与GeometricDynamicsMonitor集成：
    monitor检测到BREAKING状态 → brake_tensor.open_breaking_window()
    monitor检测到RECOVERING → brake_tensor.close_breaking_window()
    monitor的gamma_adjustments → brake_tensor.adjust_gamma_by_deltas()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class FSSubspace(str, Enum):
    """FS标架四维子空间"""
    TANGENT = "tangent"        # γ_T: 切向量 — 语义方向一致性
    NORMAL = "normal"          # γ_N: 法向量 — 弯曲强度约束
    BINORMAL = "binormal"      # γ_B: 副法向量 — 扭转自由度约束
    TORSION_RATE = "torsion_rate"  # γ_τ: 挠率变化率 — 结构惯性


@dataclass
class FSBrakeConfig:
    """FS制动张量配置"""
    # 四维约束权重基准值
    gamma_T: float = 0.5    # 切向量约束（方向一致性）— 中等
    gamma_N: float = 1.0    # 法向量约束（弯曲强度）— 较强的束
    gamma_B: float = 0.3    # 副法向量约束（扭转自由度）— 较宽松
    gamma_tau: float = 0.8  # 挠率变化率约束（结构惯性）— 较强

    # 破缺窗口期参数
    epsilon_B: float = 0.01       # 窗口期γ_B降至ε_B（大幅放宽扭转）
    gamma_tau_prime_ratio: float = 0.3  # 窗口期γ_τ降至0.3倍（部分释放惯性）

    # 紧急制动：γ_T和γ_N放大倍数
    emergency_gamma_T_mult: float = 3.0
    emergency_gamma_N_mult: float = 5.0

    # 破缺窗口
    default_window_duration: int = 2  # 默认窗口持续epoch数
    window_gradual_transition: bool = True  # 渐变过渡
    window_transition_epochs: int = 1  # 过渡期epoch数

    # 自由能预算 (E56机制)
    budget_initial: float = 1.0       # 初始预算
    budget_learning_rate: float = 0.1 # 预算消耗学习率
    budget_depleted_mult: float = 5.0 # 预算耗尽时γ放大倍数

    # 停滞检测
    stagnation_threshold: float = 0.01  # 梯度停滞阈值
    min_stagnation_steps: int = 3       # 最小停滞步数触发射击

    # 数值稳定
    gamma_floor: float = 0.001

    # 历史
    history_max_size: int = 100
    history_keep_size: int = 50


@dataclass
class FSBrakeSnapshot:
    """FS制动张量快照"""
    gamma_T: float
    gamma_N: float
    gamma_B: float
    gamma_tau: float
    is_window_active: bool
    effective_gamma_B: float      # 窗口期生效的γ_B
    effective_gamma_tau: float    # 窗口期生效的γ_τ
    spectral_condition: float     # 谱条件数 max(γ)/min(γ)
    budget_remaining: float       # 剩余自由能预算

    @property
    def total_brake_strength(self) -> float:
        return self.gamma_T + self.gamma_N + self.effective_gamma_B + self.effective_gamma_tau

    @property
    def exploration_freedom(self) -> float:
        """探索自由度：与γ_B成反比"""
        return 1.0 / max(self.effective_gamma_B, 0.001)

    @property
    def constraint_balance(self) -> Dict[str, float]:
        """四维约束权重在总制动强度中的占比"""
        total = max(self.total_brake_strength, 1e-8)
        return {
            "tangent_pct": self.gamma_T / total,
            "normal_pct": self.gamma_N / total,
            "binormal_pct": self.effective_gamma_B / total,
            "torsion_rate_pct": self.effective_gamma_tau / total,
        }


class FSBrakeTensor:
    """FS制动张量 — 四子空间分解对FS标架的动态约束权重

    使用示例:
        brake = FSBrakeTensor(base_dim=128)
        monitor.attach_brake_tensor(brake)

        # 训练循环中
        for epoch in range(epochs):
            # ... 训练 ...
            snap = monitor.record_epoch(epoch, hidden, loss)
            cycle = monitor.run_cycle(epoch)

            if cycle.break_triggered:
                brake.open_breaking_window(epoch)
                brake.adjust_gamma_by_deltas(**cycle.gamma_adjustments)

            # 获取当前生效的约束权重
            weights = brake.get_constraint_weights()
            # weights = {'gamma_T': ..., 'gamma_N': ..., ...}

            brake.tick_window(epoch)
    """

    def __init__(
        self,
        config: Optional[FSBrakeConfig] = None,
        base_dim: int = 128,
    ):
        self._config = config or FSBrakeConfig()
        self.base_dim = base_dim

        # 窗口状态
        self._window_active: bool = False
        self._window_start_epoch: int = 0
        self._window_remaining: int = 0
        self._window_closing: bool = False

        # 渐变过渡
        self._transition_active: bool = False
        self._transition_direction: str = ""
        self._transition_remaining: int = 0
        self._transition_total: int = 0

        # 窗口队列（嵌套窗口支持）
        self._window_queue: List[Tuple[int, int]] = []

        # 预算（E56机制）
        self._budget_remaining: float = self._config.budget_initial
        self._prev_error_signal: float = 0.0

        # 基线（供紧急制动回退）
        self._baseline_gamma_T: float = self._config.gamma_T
        self._baseline_gamma_N: float = self._config.gamma_N
        self._baseline_gamma_B: float = self._config.gamma_B
        self._baseline_gamma_tau: float = self._config.gamma_tau

        # 上次稳定值
        self._last_stable_gamma_T: float = self._config.gamma_T
        self._last_stable_gamma_N: float = self._config.gamma_N
        self._last_stable_gamma_B: float = self._config.gamma_B
        self._last_stable_gamma_tau: float = self._config.gamma_tau

        # 历史
        self._history: List[FSBrakeSnapshot] = []

        # 停滞检测
        self._stagnation_count: int = 0
        self._structural_defect_signaled: bool = False

    # ── 窗口管理 ────────────────────────────────────────────

    @property
    def is_window_active(self) -> bool:
        return self._window_active and not self._window_closing

    def open_breaking_window(
        self,
        epoch: int,
        duration: Optional[int] = None,
    ) -> None:
        """开启破缺窗口期。

        在窗口期内，γ_B降至ε_B（允许大幅扭转探索），
        γ_τ降至0.3倍（部分释放结构惯性），
        而γ_T和γ_N保持不变（方向一致性和弯曲强度是刚性约束）。
        """
        if duration is None:
            duration = self._config.default_window_duration

        if self._window_active:
            self._window_queue.append((epoch, duration))
            return

        self._window_active = True
        self._window_start_epoch = epoch
        self._window_remaining = duration

        if self._config.window_gradual_transition:
            self._transition_active = True
            self._transition_direction = "opening"
            self._transition_remaining = self._config.window_transition_epochs
            self._transition_total = self._config.window_transition_epochs

    def close_breaking_window(self) -> None:
        """关闭破缺窗口，恢复约束权重"""
        if not self._window_active:
            return

        if self._config.window_gradual_transition and not self._transition_active:
            self._transition_active = True
            self._transition_direction = "closing"
            self._transition_remaining = self._config.window_transition_epochs
            self._transition_total = self._config.window_transition_epochs
            self._window_closing = True
            return

        self._window_active = False
        self._window_remaining = 0
        self._transition_active = False
        self._window_closing = False

        # 检查队列中挂起的窗口
        if self._window_queue:
            next_epoch, next_duration = self._window_queue.pop(0)
            self._window_active = True
            self._window_start_epoch = next_epoch
            self._window_remaining = next_duration

            if self._config.window_gradual_transition:
                self._transition_active = True
                self._transition_direction = "opening"
                self._transition_remaining = self._config.window_transition_epochs
                self._transition_total = self._config.window_transition_epochs

    def tick_window(self, epoch: int) -> bool:
        """每个epoch结束时调用，推进窗口状态。

        Returns:
            True: 窗口仍在进行中
            False: 窗口已关闭
        """
        if self._transition_active:
            self._transition_remaining -= 1
            if self._transition_remaining <= 0:
                self._transition_active = False
                if self._transition_direction == "closing":
                    self._window_active = False
                    self._window_closing = False
                    self._window_remaining = 0
                    # 检查队列
                    if self._window_queue:
                        next_ep, next_dur = self._window_queue.pop(0)
                        self._window_active = True
                        self._window_start_epoch = next_ep
                        self._window_remaining = next_dur
                        if self._config.window_gradual_transition:
                            self._transition_active = True
                            self._transition_direction = "opening"
                            self._transition_remaining = self._config.window_transition_epochs
                            self._transition_total = self._config.window_transition_epochs
                    return False
            return True

        if not self._window_active:
            return False

        self._window_remaining -= 1
        if self._window_remaining <= 0:
            self.close_breaking_window()
            return False
        return True

    # ── 约束权重获取 ────────────────────────────────────────

    def get_gamma(self, subspace: FSSubspace) -> float:
        """获取指定子空间的当前生效γ值"""
        if subspace == FSSubspace.TANGENT:
            return self._config.gamma_T
        if subspace == FSSubspace.NORMAL:
            return self._config.gamma_N
        if subspace == FSSubspace.BINORMAL:
            return self._get_gradual_gamma(
                self._config.gamma_B,
                self._config.epsilon_B,
            )
        if subspace == FSSubspace.TORSION_RATE:
            return self._get_gradual_gamma(
                self._config.gamma_tau,
                self._config.gamma_tau * self._config.gamma_tau_prime_ratio,
            )
        return 0.0

    def _get_gradual_gamma(self, normal_val: float, window_val: float) -> float:
        """渐变过渡期间在正常值和窗口值之间线性插值"""
        if not self._transition_active:
            return window_val if self._window_active else normal_val

        progress = 1.0 - (
            self._transition_remaining / max(self._transition_total, 1)
        )
        if self._transition_direction == "opening":
            return normal_val + (window_val - normal_val) * progress
        return window_val + (normal_val - window_val) * progress

    def get_constraint_weights(self) -> Dict[str, float]:
        """获取四维约束的当前生效权重。

        用于训练时加权各项几何损失。

        Returns:
            {'gamma_T': ..., 'gamma_N': ..., 'gamma_B': ..., 'gamma_tau': ...}
        """
        return {
            'gamma_T': self.get_gamma(FSSubspace.TANGENT),
            'gamma_N': self.get_gamma(FSSubspace.NORMAL),
            'gamma_B': self.get_gamma(FSSubspace.BINORMAL),
            'gamma_tau': self.get_gamma(FSSubspace.TORSION_RATE),
        }

    # ── 约束损失计算 ────────────────────────────────────────

    def compute_fs_constraint_loss(
        self,
        hidden: torch.Tensor,
        fs_analyzer=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算FS标架四维约束损失。

        对四维几何量分别施加权重γ_T/γ_N/γ_B/γ_τ约束。

        Args:
            hidden: [B, T, D] 隐状态
            fs_analyzer: FrenetSerretAnalyzer实例

        Returns:
            total_loss: 标量损失
            components: 各维度损失分量字典
        """
        device = hidden.device
        B, T, D = hidden.shape

        if T < 4 or fs_analyzer is None:
            return torch.tensor(0.0, device=device), {}

        try:
            result = fs_analyzer.analyze(hidden, compute_chern=False)
        except Exception:
            return torch.tensor(0.0, device=device), {}

        valid = result.valid_mask  # [B, T-3]
        if valid.sum() < 4:
            return torch.tensor(0.0, device=device), {}

        w = self.get_constraint_weights()

        losses = {}

        # 1. γ_T: 切向量一致性 — 惩罚相邻T向量的方向突变
        if w['gamma_T'] > 0 and result.T is not None:
            T_seq = result.T  # [B, T-3, 3]
            # 相邻T向量的点积变化（1 = 完全一致，-1 = 反转）
            T_dot = (T_seq[:, :-1] * T_seq[:, 1:]).sum(dim=-1)  # [B, T-4]
            # 惩罚突变（点积偏离1）
            T_loss = (1.0 - T_dot).mean()
            losses['tangent_consistency'] = w['gamma_T'] * T_loss

        # 2. γ_N: 法向量弯曲强度 — 惩罚曲率的剧烈波动
        if w['gamma_N'] > 0:
            kappa = result.kappa[valid]  # [N]
            if kappa.numel() > 0:
                # 曲率的标准差（高方差 = 弯曲强度不一致）
                kappa_std = kappa.std()
                losses['normal_bending'] = w['gamma_N'] * kappa_std

        # 3. γ_B: 副法向量扭转自由度 — 惩罚挠率的过度发散
        if w['gamma_B'] > 0:
            tau = result.tau[valid]  # [N]
            if tau.numel() > 0:
                # 挠率的99分位数（截断异常值）
                tau_q99 = torch.quantile(tau, 0.99)
                losses['binormal_torsion'] = w['gamma_B'] * tau_q99

        # 4. γ_τ: 挠率变化率 — 惩罚挠率的高频波动
        if w['gamma_tau'] > 0:
            tau = result.tau[result.valid_mask.sum(dim=1) > 3]  # 只取有足够有效点的batch
            if tau.numel() > 0:
                tau_batch = result.tau  # [B, T-3]
                # 挠率的一阶差分的std
                tau_diff = (tau_batch[:, 1:] - tau_batch[:, :-1]).abs()
                tau_vol = tau_diff[valid[:, 1:]].mean() if valid[:, 1:].sum() > 0 else torch.tensor(0.0, device=device)
                losses['torsion_rate_inertia'] = w['gamma_tau'] * tau_vol

        if not losses:
            return torch.tensor(0.0, device=device), {}

        total = torch.tensor(0.0, device=device)
        for v in losses.values():
            total = total + v
        component_values = {k: float(v.item()) for k, v in losses.items()}

        return total, component_values

    # ── 权重调节 ────────────────────────────────────────────

    def adjust_gamma_by_deltas(
        self,
        gamma_T: float = 0.0,
        gamma_N: float = 0.0,
        gamma_B: float = 0.0,
        gamma_tau: float = 0.0,
    ) -> None:
        """按增量调整四维γ值（来自GeometricDynamicsMonitor的建议）。

        正值=加强约束，负值=放松约束。
        γ_T和γ_N只允许加强（方向一致性和弯曲强度是刚性约束）。
        """
        self._config.gamma_T = max(0.01, self._config.gamma_T + gamma_T)
        self._config.gamma_N = max(0.1, self._config.gamma_N + gamma_N)
        self._config.gamma_B = max(0.001, self._config.gamma_B + gamma_B)
        self._config.gamma_tau = max(0.01, self._config.gamma_tau + gamma_tau)

    def adjust_gamma_by_error_signal(self, error_signal: float) -> None:
        """E56机制：基于误差信号调节制动强度。

        调节公式：γ_{t+1} = γ_t - η · (dE/dt) · tanh(B_remaining)

        Args:
            error_signal: 误差信号E（如ψ_geo的值）
        """
        eta = self._config.budget_learning_rate
        dE_dt = error_signal - self._prev_error_signal
        self._prev_error_signal = error_signal
        budget_factor = math.tanh(self._budget_remaining)

        # 三段式逻辑：
        # E增大（恶化）→ γ上调（保守防御）
        # E减小（改善但M停滞）→ γ上调（固化新结构）
        # E持续低位 → γ下调（探索新路径）
        if error_signal < 0.2 and abs(dE_dt) < eta * 0.5:
            delta_gamma = -eta * budget_factor * 0.5
        else:
            delta_gamma = eta * abs(dE_dt) * budget_factor

        # 预算耗尽保护
        if self._budget_remaining < 0.05:
            depleted_mult = self._config.budget_depleted_mult
            new_gamma_T = max(self._baseline_gamma_T, self._config.gamma_T) * depleted_mult
            new_gamma_N = max(self._baseline_gamma_N, self._config.gamma_N) * depleted_mult
            new_gamma_B = max(self._baseline_gamma_B, self._config.gamma_B) * depleted_mult
            new_gamma_tau = max(self._baseline_gamma_tau, self._config.gamma_tau) * depleted_mult
        else:
            new_gamma_B = max(
                self._config.gamma_B + delta_gamma,
                self._baseline_gamma_B * 0.5,
            )
            new_gamma_tau = max(
                self._config.gamma_tau + delta_gamma,
                self._baseline_gamma_tau * 0.5,
            )
            # γ_T和γ_N保持刚性
            new_gamma_T = self._config.gamma_T + max(0.0, delta_gamma * 0.2)
            new_gamma_N = self._config.gamma_N + max(0.0, delta_gamma * 0.5)

        # 更新
        self._config.gamma_T = max(0.01, new_gamma_T)
        self._config.gamma_N = max(0.1, new_gamma_N)
        self._config.gamma_B = max(0.001, new_gamma_B)
        self._config.gamma_tau = max(0.01, new_gamma_tau)

        # 记录稳定值
        if dE_dt < 0 and error_signal < 0.3:
            self._last_stable_gamma_T = self._config.gamma_T
            self._last_stable_gamma_N = self._config.gamma_N
            self._last_stable_gamma_B = self._config.gamma_B
            self._last_stable_gamma_tau = self._config.gamma_tau

    def apply_emergency_brake(self) -> None:
        """紧急制动：所有约束急剧收紧，回退到上一个稳定状态。

        用于系统出现结构性缺陷信号时。
        """
        self._config = FSBrakeConfig(
            gamma_T=max(self._last_stable_gamma_T * self._config.emergency_gamma_T_mult,
                        self._config.gamma_T * 3.0),
            gamma_N=max(self._last_stable_gamma_N * self._config.emergency_gamma_N_mult,
                        self._config.gamma_N * 5.0),
            gamma_B=max(self._last_stable_gamma_B * 3.0,
                        self._config.gamma_B * 3.0),
            gamma_tau=max(self._last_stable_gamma_tau * 3.0,
                          self._config.gamma_tau * 3.0),
            epsilon_B=self._config.epsilon_B,
            gamma_tau_prime_ratio=self._config.gamma_tau_prime_ratio,
            window_gradual_transition=self._config.window_gradual_transition,
            window_transition_epochs=self._config.window_transition_epochs,
            budget_initial=self._config.budget_initial,
            budget_learning_rate=self._config.budget_learning_rate,
            budget_depleted_mult=self._config.budget_depleted_mult,
            stagnation_threshold=self._config.stagnation_threshold,
            min_stagnation_steps=self._config.min_stagnation_steps,
            history_max_size=self._config.history_max_size,
            history_keep_size=self._config.history_keep_size,
            default_window_duration=self._config.default_window_duration,
            gamma_floor=self._config.gamma_floor,
        )

    # ── 预算管理 ────────────────────────────────────────────

    def consume_budget(self, amount: float) -> None:
        """消耗自由能预算"""
        self._budget_remaining = max(0.0, self._budget_remaining - amount)

    def get_budget_remaining(self) -> float:
        return self._budget_remaining

    def reset_budget(self, value: Optional[float] = None) -> None:
        """重置预算"""
        self._budget_remaining = (
            value if value is not None else self._config.budget_initial
        )
        self._prev_error_signal = 0.0

    # ── 快照 ────────────────────────────────────────────────

    def snapshot(self) -> FSBrakeSnapshot:
        """生成当前状态快照"""
        snap = FSBrakeSnapshot(
            gamma_T=self._config.gamma_T,
            gamma_N=self._config.gamma_N,
            gamma_B=self._config.gamma_B,
            gamma_tau=self._config.gamma_tau,
            is_window_active=self._window_active,
            effective_gamma_B=self.get_gamma(FSSubspace.BINORMAL),
            effective_gamma_tau=self.get_gamma(FSSubspace.TORSION_RATE),
            spectral_condition=self._compute_spectral_condition(),
            budget_remaining=self._budget_remaining,
        )
        self._history.append(snap)
        if len(self._history) > self._config.history_max_size:
            self._history = self._history[-self._config.history_keep_size:]
        return snap

    def _compute_spectral_condition(self) -> float:
        """计算谱条件数：max(γ)/min(γ)"""
        gammas = [
            self._config.gamma_T,
            self._config.gamma_N,
            self.get_gamma(FSSubspace.BINORMAL),
            self.get_gamma(FSSubspace.TORSION_RATE),
        ]
        min_g = min(gammas)
        max_g = max(gammas)
        if min_g <= 0:
            return float('inf')
        return max_g / min_g

    # ── 诊断 ────────────────────────────────────────────────

    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要（供报告使用）"""
        w = self.get_constraint_weights()
        snap = self.snapshot()
        return {
            "constraint_weights": w,
            "window_active": self._window_active,
            "budget_remaining": round(self._budget_remaining, 4),
            "spectral_condition": round(snap.spectral_condition, 2),
            "exploration_freedom": round(snap.exploration_freedom, 4),
            "constraint_balance": snap.constraint_balance,
            "history_length": len(self._history),
        }

    def should_open_window(self, gradient_norm: float) -> bool:
        """判断是否应开启破缺窗口（梯度停滞）"""
        if gradient_norm < self._config.stagnation_threshold:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        return self._stagnation_count >= self._config.min_stagnation_steps

    # ── 属性 ────────────────────────────────────────────────

    @property
    def budget_remaining(self) -> float:
        return self._budget_remaining

    @property
    def window_active(self) -> bool:
        return self.is_window_active
