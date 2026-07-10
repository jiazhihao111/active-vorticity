"""定理一：信息本体论 (Information Ontology)
==========================================

定理陈述：
    任何能够生成连贯叙事的信息加工系统，其隐状态空间必然具有非平凡拓扑（χ ≠ 0）。

数学推导：
    1. 连贯叙事 ≡ 语义流形上存在闭合结构（故事有头有尾，因果闭环）
    2. 闭合结构 → 流形同胚于 S²（球面）或更高亏格曲面 → χ = 2 - 2g
    3. 破碎叙事 ≡ 流形退化为开线段或树 → χ = 1（线段）或更高（树）
    4. 因此：连贯系统的 χ 必然 ≥ 2（闭合流形），验证 χ ≠ 0

验证方法：
    1. 持久同调：对隐状态距离矩阵做 Vietoris-Rips 过滤
    2. 提取 Betti 数 b₀（连通分量）、b₁（环）、b₂（空洞）
    3. 计算欧拉示性数 χ = b₀ - b₁ + b₂
    4. 比较正例（连贯叙事）vs 负例（破碎叙事）的 χ 分布
    5. 统计检验：正例 χ 显著 > 负例 χ

同调计算（轻量级，纯NumPy/PyTorch实现）：
    - 无需 GUDHI/TDA 等重依赖
    - 使用谱图拉普拉斯的零特征值计数估计 b₀
    - 使用图循环计数（edges - vertices + components）估计 b₁
    - 使用局部曲率的聚集结构估计 b₂

定理编码为以下可验证命题：
    H₀: E[χ_pos] > E[χ_neg]  （单侧检验，α=0.01）
    H₁: χ_pos ~ χ(球面) 分布（χ ≥ 1.5）
    H₂: χ_neg ~ χ(树/线段) 分布（χ ∈ [0, 1.5]）

十三字公理映射：
    "信息化为" → 隐状态流形构建（距离矩阵→持久同调）
    "世界模型" → Betti数捕获的拓扑结构
    "遵守几何规则" → χ 的闭合性约束
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats


# ── 数据结构 ──────────────────────────────────────────────────────

class OntologyVerdict(str, Enum):
    """定理一判决"""
    SUPPORT = "SUPPORT"  # χ ≠ 0 显著成立
    WEAK = "WEAK"  # 趋势正确但未达显著性
    REFUTE = "REFUTE"  # χ ≈ 0，不支持定理
    INCONCLUSIVE = "INCONCLUSIVE"  # 数据不足


@dataclass
class BettiNumbers:
    """持久同调 Betti 数"""
    b0: int  # 连通分量数（H₀ 维数）
    b1: int  # 环数（H₁ 维数）
    b2: int  # 空洞数（H₂ 维数）
    filtration_value: float  # 过滤参数 ε

    @property
    def euler_char(self) -> int:
        """欧拉示性数 χ = b₀ - b₁ + b₂"""
        return self.b0 - self.b1 + self.b2


@dataclass
class PersistenceResult:
    """持久同调完整结果"""
    points: np.ndarray  # [N, D] 输入点
    distance_matrix: np.ndarray  # [N, N] 成对距离
    filtration_epsilons: List[float]  # 过滤阈值序列
    betti_curve: List[BettiNumbers]  # 随 ε 变化的 Betti 数
    persistent_intervals: Dict[str, List[Tuple[float, float]]]  # H₀, H₁, H₂ 的生灭对
    optimal_epsilon: float  # 最大持久对对应的 ε
    chi_at_optimal: int  # 最优 ε 处的 χ


@dataclass
class OntologyResult:
    """定理一验证结果"""
    chi_pos: np.ndarray  # 正例的 χ 值 [N_pos]
    chi_neg: np.ndarray  # 负例的 χ 值 [N_neg]
    mean_chi_pos: float
    mean_chi_neg: float
    std_chi_pos: float
    std_chi_neg: float
    cohens_d: float  # 效应量
    t_statistic: float
    p_value: float
    verdict: OntologyVerdict
    persistence_details: List[PersistenceResult] = field(default_factory=list)


# ── 核心实现 ──────────────────────────────────────────────────────

class TheoremOntology:
    """信息本体论验证引擎。

    核心能力：
    1. 从隐状态构建持久同调
    2. 提取 Betti 数与欧拉示性数
    3. 比较正/负例的 χ 分布，验证 χ ≠ 0
    """

    def __init__(
        self,
        eps_range: Tuple[float, float] = (0.01, 0.95),
        n_filtration_steps: int = 20,
        significance_level: float = 0.01,
    ):
        """
        Args:
            eps_range: 过滤参数 ε 的范围（距离的分位数）
            n_filtration_steps: 过滤步数
            significance_level: 统计显著性水平
        """
        self.eps_range = eps_range
        self.n_filtration_steps = n_filtration_steps
        self.significance_level = significance_level

    # ── 持久同调 ─────────────────────────────────────────────────

    def compute_persistence(
        self, hidden_states: torch.Tensor
    ) -> PersistenceResult:
        """对隐状态序列计算持久同调。

        Args:
            hidden_states: [T, D] 或 [B, T, D] 隐状态

        Returns:
            PersistenceResult 包含所有同调信息
        """
        # 处理 batch 维度
        if hidden_states.dim() == 3:
            # 取 batch 平均
            h = hidden_states.mean(dim=0)  # [T, D]
        else:
            h = hidden_states  # [T, D]

        points = h.detach().cpu().numpy()
        N = points.shape[0]

        if N < 4:
            raise ValueError(f"需要至少 4 个点进行同调分析，当前 N={N}")

        # 1. 计算成对距离矩阵
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, N]

        # 2. 过滤参数序列（基于距离分位数）
        flat_dist = dist[np.triu_indices(N, k=1)]
        eps_min, eps_max = np.percentile(flat_dist, [self.eps_range[0] * 100, self.eps_range[1] * 100])

        epsilons = np.linspace(eps_min, eps_max, self.n_filtration_steps)

        # 3. 计算每个 ε 的 Betti 数
        betti_curve = []
        h0_intervals = []  # H₀ 生灭对
        h1_intervals = []  # H₁ 生灭对

        prev_components = N  # 初始：每个点独立
        prev_cycles = 0

        for eps in epsilons:
            # 邻接矩阵：距离 < ε 则连通
            adj = (dist < eps).astype(int)
            np.fill_diagonal(adj, 0)

            # 计算连通分量数 b₀
            components = self._count_components(adj, N)

            # 计算环数 b₁（一维同调）
            # b₁ = edges - vertices + components (对每个连通分量求和)
            n_edges = adj.sum() // 2
            cycles = n_edges - N + components

            # 追踪生灭对
            if prev_components != components:
                delta = prev_components - components
                for _ in range(delta):
                    h0_intervals.append((0.0, float(eps)))  # H₀ 永远生于 ε=0

            if prev_cycles != cycles and cycles > prev_cycles:
                # 新环出现
                for _ in range(cycles - prev_cycles):
                    h1_intervals.append((float(eps), float(eps_max)))  # H₁ 暂设死亡于最大 ε

            prev_components = components
            prev_cycles = cycles

            # b₂ 近似：通过局部曲率聚集（三个方向的交叠区域）
            b2 = self._estimate_b2(points, eps)

            betti_curve.append(BettiNumbers(
                b0=components, b1=cycles, b2=b2,
                filtration_value=float(eps),
            ))

        # 完成 H₁ 的死亡时间：检测环何时被填充
        # 简化版：检查相邻 ε 的 b₁ 变化
        for i, bn in enumerate(betti_curve):
            if i < len(betti_curve) - 1:
                b1_next = betti_curve[i + 1].b1
                if b1_next < bn.b1:
                    # 有环死亡
                    n_dead = bn.b1 - b1_next
                    for _ in range(n_dead):
                        # 更新最近的未结束的 H₁ 生灭对
                        for j in range(len(h1_intervals) - 1, -1, -1):
                            birth, _death = h1_intervals[j]
                            if _death == eps_max:  # 未结束
                                h1_intervals[j] = (birth, bn.filtration_value)
                                break

        # 4. 选择最优 ε（最大化持久区间的 ε）
        if h1_intervals:
            longest_lifetime = 0
            optimal_eps = epsilons[-1]
            for birth, death in h1_intervals:
                lifetime = death - birth
                if lifetime > longest_lifetime:
                    longest_lifetime = lifetime
                    optimal_eps = (birth + death) / 2
        else:
            optimal_eps = epsilons[len(epsilons) // 2]

        # 对应的 χ
        chi_at_optimal = 0
        for bn in betti_curve:
            if bn.filtration_value >= optimal_eps:
                chi_at_optimal = bn.euler_char
                break

        return PersistenceResult(
            points=points,
            distance_matrix=dist,
            filtration_epsilons=list(epsilons),
            betti_curve=betti_curve,
            persistent_intervals={"H0": h0_intervals, "H1": h1_intervals, "H2": []},
            optimal_epsilon=optimal_eps,
            chi_at_optimal=chi_at_optimal,
        )

    @staticmethod
    def _count_components(adj: np.ndarray, N: int) -> int:
        """通过 BFS 计算连通分量数 == b₀"""
        visited = np.zeros(N, dtype=bool)
        components = 0
        for i in range(N):
            if not visited[i]:
                components += 1
                # BFS
                queue = [i]
                visited[i] = True
                while queue:
                    node = queue.pop()
                    neighbors = np.where(adj[node] > 0)[0]
                    for nb in neighbors:
                        if not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
        return components

    @staticmethod
    def _estimate_b2(points: np.ndarray, eps: float) -> int:
        """估计 H₂ 维数（三维空洞/空腔）。

        方法：检测三个局部方向的交叠区域。
        用曲率张量的非零特征值数近似。
        简化实现：如果数据维度 ≥ 3 且点云在球面分布，则 b₂ ≥ 1。
        """
        D = points.shape[1]
        N = points.shape[0]

        if D < 3 or N < 6:
            return 0

        # 计算局部协方差矩阵的三个特征值
        cov = np.cov(points.T)
        eigenvals = np.sort(np.abs(cov))[::-1]

        # 如果三个方向都有足够方差 + 点云呈现球面分布
        ratio_3rd = eigenvals[2] / (eigenvals[0] + 1e-10) if len(eigenvals) > 2 else 0

        # 从重心到各点的距离方差（检测球面性）
        centroid = points.mean(axis=0)
        radii = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        radius_cv = np.std(radii) / (np.mean(radii) + 1e-10)

        # 判断：三维方差不够 + 球面分布 → b₂ = 1
        if ratio_3rd > 0.05 and radius_cv < 0.5:
            return 1
        return 0

    # ── 定理验证 ─────────────────────────────────────────────────

    def verify(
        self,
        hidden_pos: List[torch.Tensor],
        hidden_neg: List[torch.Tensor],
    ) -> OntologyResult:
        """验证定理一：正例 χ 应显著高于负例。

        Args:
            hidden_pos: 正例（连贯叙事）隐状态序列的列表
            hidden_neg: 负例（破碎叙事）隐状态序列的列表

        Returns:
            OntologyResult 包含判决和统计
        """
        chi_pos_list = []
        chi_neg_list = []
        persistence_details = []

        # 计算所有正例的 χ
        for h in hidden_pos:
            try:
                pr = self.compute_persistence(h)
                chi_pos_list.append(pr.chi_at_optimal)
                persistence_details.append(pr)
            except (ValueError, RuntimeError):
                continue

        # 计算所有负例的 χ
        for h in hidden_neg:
            try:
                pr = self.compute_persistence(h)
                chi_neg_list.append(pr.chi_at_optimal)
                persistence_details.append(pr)
            except (ValueError, RuntimeError):
                continue

        chi_pos = np.array(chi_pos_list)
        chi_neg = np.array(chi_neg_list)

        # 统计检验
        result = self._statistical_test(chi_pos, chi_neg)

        result.persistence_details = persistence_details
        return result

    def verify_single(self, hidden: torch.Tensor) -> Dict:
        """对单个样本验证 χ ≠ 0。

        Returns:
            dict with chi, verdict, betti_curve
        """
        try:
            pr = self.compute_persistence(hidden)
            chi = pr.chi_at_optimal

            if chi >= 2:
                verdict = "χ≥2 → 闭合流形（球面同胚）→ SUPPORT"
            elif chi >= 1:
                verdict = "χ=1 → 线/环拓扑 → WEAK (边界的边界是零，但不闭合)"
            else:
                verdict = "χ≤0 → 非平凡拓扑缺失 → REFUTE"

            return {
                "chi": chi,
                "verdict": verdict,
                "betti_final": pr.betti_curve[-1] if pr.betti_curve else None,
                "optimal_epsilon": pr.optimal_epsilon,
            }
        except (ValueError, RuntimeError) as e:
            return {"chi": float("nan"), "verdict": f"ERROR: {e}", "betti_final": None}

    def _statistical_test(
        self, chi_pos: np.ndarray, chi_neg: np.ndarray
    ) -> OntologyResult:
        """统计检验：正例 χ 是否显著 > 负例 χ"""
        n_pos, n_neg = len(chi_pos), len(chi_neg)

        if n_pos < 3 or n_neg < 3:
            return OntologyResult(
                chi_pos=chi_pos, chi_neg=chi_neg,
                mean_chi_pos=float(np.mean(chi_pos)) if n_pos > 0 else float("nan"),
                mean_chi_neg=float(np.mean(chi_neg)) if n_neg > 0 else float("nan"),
                std_chi_pos=float(np.std(chi_pos)) if n_pos > 1 else 0.0,
                std_chi_neg=float(np.std(chi_neg)) if n_neg > 1 else 0.0,
                cohens_d=float("nan"),
                t_statistic=float("nan"),
                p_value=float("nan"),
                verdict=OntologyVerdict.INCONCLUSIVE,
            )

        mean_pos = float(np.mean(chi_pos))
        mean_neg = float(np.mean(chi_neg))
        std_pos = float(np.std(chi_pos, ddof=1))
        std_neg = float(np.std(chi_neg, ddof=1))

        # Cohen's d
        pooled_std = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2)
        cohens_d = (mean_pos - mean_neg) / (pooled_std + 1e-10)

        # Welch's t-test (单侧：pos > neg)
        t_stat, p_value = scipy_stats.ttest_ind(chi_pos, chi_neg, equal_var=False)
        p_value = float(p_value / 2)  # 单侧

        # 判决
        if p_value < self.significance_level and cohens_d > 0.5:
            verdict = OntologyVerdict.SUPPORT
        elif cohens_d > 0.2 and mean_pos > mean_neg:
            verdict = OntologyVerdict.WEAK
        elif mean_pos <= mean_neg:
            verdict = OntologyVerdict.REFUTE
        else:
            verdict = OntologyVerdict.INCONCLUSIVE

        return OntologyResult(
            chi_pos=chi_pos,
            chi_neg=chi_neg,
            mean_chi_pos=mean_pos,
            mean_chi_neg=mean_neg,
            std_chi_pos=std_pos,
            std_chi_neg=std_neg,
            cohens_d=float(cohens_d),
            t_statistic=float(t_stat),
            p_value=p_value,
            verdict=verdict,
        )

    # ── 诊断接口 ─────────────────────────────────────────────────

    def diagnose_topology(self, hidden: torch.Tensor) -> Dict:
        """诊断隐流形的拓扑类型。

        返回类型判定：
            - "sphere": χ ≥ 2（球面同胚）
            - "torus": χ = 0, b₁ ≥ 2（环面同胚）
            - "disc": χ = 1, b₁ = 0（圆盘/线段同胚）
            - "cycle": χ = 0, b₁ = 1（圆环拓扑）
            - "tree": χ = 1, b₁ = 0 + 高 b₀（树拓扑）
            - "degenerate": χ 未定义（数据太少）
        """
        try:
            pr = self.compute_persistence(hidden)
            chi = pr.chi_at_optimal
            last_betti = pr.betti_curve[-1]

            if abs(chi) >= 2:
                return {"type": "sphere", "chi": chi, "g": (2 - chi) // 2 if chi <= 2 else 0}
            elif abs(chi) <= 0.5 and last_betti.b1 >= 2:
                return {"type": "torus", "chi": chi, "g": 1}
            elif abs(chi - 1) < 0.5 and last_betti.b1 == 0:
                return {"type": "disc", "chi": chi}
            elif abs(chi) < 0.5 and last_betti.b1 == 1:
                return {"type": "cycle", "chi": chi}
            elif abs(chi - 1) < 0.5:
                return {"type": "tree", "chi": chi}
            else:
                return {"type": "degenerate", "chi": chi}
        except (ValueError, RuntimeError):
            return {"type": "error", "chi": float("nan")}


# ── 快速接口 ──────────────────────────────────────────────────────

def persistent_homology_pipeline(
    hidden: torch.Tensor,
    eps_range: Tuple[float, float] = (0.01, 0.95),
) -> PersistenceResult:
    """一键持久同调分析。

    >>> result = persistent_homology_pipeline(hidden_states)
    >>> print(f"χ = {result.chi_at_optimal}")
    """
    engine = TheoremOntology(eps_range=eps_range)
    return engine.compute_persistence(hidden)
