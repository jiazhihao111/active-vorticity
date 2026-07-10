"""
实验10: 几何动力学尺度升级 — 叙事驱动力验证
============================================

叙事锚点:
    "如果我们的几何本体论是对的，那么错误一定出在
    '如何让这个几何在动力学上运作'的细节里——
    让我们升级实验尺度，把这些细节找出来。"

核心改动 (相对于 exp7-9):
    1. FS 标架首次集成: FrenetSerretAnalyzer 对高维隐空间计算
       κ(曲率), τ(挠率), Θ_sem(螺旋升角) — 重测 H-helix
    2. InfoNCE 对比推离: 替换铰链损失 — 重测 H-push
    3. 变长序列 H-phase: 多长度桶检验陈数代理的相变 — 重测 H-phase
    4. 高维 MVE 升级: base_dim 32→128，减少 SVD 投影信息损失

子实验:
    Part A: H-helix 重测 — FS 螺旋升角稳定性 (τ/κ ratio)
    Part B: H-push 重测 — InfoNCE 推离 vs 铰链推离对比
    Part C: H-phase 重测 — 变长序列下的陈数代理相变检测
    Part D: 集成判决 — 三层假设的联合 Bayesian 更新

预期:
    - 32D MVE 中 H-helix 的 R²=0.057 源于投影信息损失
    - 128D 空间中信息保留率从 10%→60%，FS 标架有意义
    - InfoNCE 提供梯度丰富的推离信号，避免铰链的边际坍塌
    - 变长序列中检测到陈数在 ~128 token 处的跳变
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from ..models.transformer import CausalTransformer
from ..models.gauge_field import GaugeField
from ..npnw.story_generator import Story
from ..npnw.tokenizer import NPNWTokenizer
from ..utils.logger import setup_logger
from ..utils.frenet_serret import (
    FrenetSerretAnalyzer,
    compute_batch_spiral_helix_statistics,
)
from ..losses.contrastive_push import (
    InfoNCEContrastivePush,
    build_negative_pool,
)
from ..dynamics.geometric_dynamics_monitor import (
    GeometricDynamicsMonitor,
    DynamicsState,
    EvolutionPhase,
    CycleRecord,
    GeometricSnapshot,
)
from ..dynamics.fs_brake_tensor import (
    FSBrakeTensor,
    FSSubspace,
    FSBrakeConfig,
    FSBrakeSnapshot,
)
from ..theorems.theorem_verifier import (
    TheoremVerifier,
    TheoremResult,
    VerificationReport,
    TheoremStatus,
)


@dataclass
class Exp10Config:
    """exp10 专属配置"""
    # MVE 升级
    base_dim_high: int = 128          # 高维 MVE (原 32D)
    base_dim_low: int = 32            # 低维对照 (复现 exp9)

    # FS 标架
    fs_window_size: int = 4           # 局部投影窗口
    fs_svd_components: int = 3        # SVD 投影维度
    fs_eps: float = 1e-8              # 数值稳定

    # InfoNCE 推离
    infonce_temperature: float = 0.1  # 温度 (低→强推离)
    infonce_num_negatives: int = 16   # 每样本负例数
    infonce_neg_method: str = "temporal_contrast"

    # 变长 H-phase
    phase_seq_lengths: List[int] = field(default_factory=lambda:
        [16, 32, 64, 128, 256, 512])
    phase_stories_per_length: int = 200
    phase_bootstrap_samples: int = 1000

    # 判决
    significance_level: float = 0.05

    # 训练
    train_epochs_high: int = 30       # 高维训练轮数
    train_epochs_low: int = 30        # 低维训练轮数
    batch_size: int = 32


class Experiment10:
    """实验10 主类: 几何动力学尺度升级验证。

    叙事位置: 崩溃IV→涅槃I 的过渡期。
    承认"动力学实现需要重新设计"，以 FS 标架和 InfoNCE
    为工具，升级实验尺度，寻找动力学细节中的真相。
    """

    def __init__(self, config: dict, gauge_field: GaugeField):
        self.config = config
        self.exp10_cfg = Exp10Config()
        self.logger = setup_logger("Experiment10")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gauge_field = gauge_field
        self.fs_analyzer = FrenetSerretAnalyzer(eps=self.exp10_cfg.fs_eps)
        self.infonce_push = InfoNCEContrastivePush(
            temperature=self.exp10_cfg.infonce_temperature,
        ).to(self.device)

        # ── 动力学监控器 & 制动张量 (NEW) ──
        base_dim = config["model"].get("base_dim", 32)
        self.dynamics_monitor = GeometricDynamicsMonitor(
            fs_analyzer=self.fs_analyzer,
            base_dim=base_dim,
        )
        self.brake_tensor = FSBrakeTensor(
            base_dim=base_dim,
        )
        self.dynamics_monitor.attach_brake_tensor(self.brake_tensor)

        # ── 三定理验证器 (NEW — 十三字公理固化) ──
        self.theorem_verifier = TheoremVerifier(
            fs_analyzer=self.fs_analyzer,
            base_dim=base_dim,
        )

        # 结果存储
        self.results: Dict = {}
        self._dynamics_snapshots: List[GeometricSnapshot] = []
        self._dynamics_cycles: List[CycleRecord] = []

    # ─────────────────────────────────────────────
    # Part A: H-helix 重测 — FS 螺旋升角稳定性
    # ─────────────────────────────────────────────

    def run_part_a_helix(
        self,
        model_high: CausalTransformer,
        model_low: Optional[CausalTransformer],
        test_pos: List[Story],
        test_neg: List[Story],
    ) -> Dict:
        """Part A: FS 标架螺旋升角稳定性检验 (H-helix 重测)。

        核心假设:
            H-helix: 合法叙事轨迹的隐空间螺旋升角 τ/κ 比值
                     应在高维空间中收敛到窄带 (CV < 0.3)。
            exp9 结果: 32D 空间中 R²=0.057 (REFUTED)，
                      怀疑是 SVD 投影到 3D 损失 90% 信息导致。

        检验逻辑:
            1. 在高维(128D) MVE 中对正/负例分别计算 FS 标架
            2. 提取 tanΘ = τ/κ 的分布
            3. 检验: CV(tanΘ_pos) < CV(tanΘ_neg) 且 CV(tanΘ_pos) < 0.3
            4. 与低维(32D) 对照对比，量化信息增益
        """
        self.logger.info("=" * 60)
        self.logger.info("Part A: H-helix 重测 — FS 螺旋升角稳定性")
        self.logger.info("=" * 60)

        results = {"high_dim": {}, "low_dim": {}}

        # ── 高维 MVE (128D) ──
        self.logger.info("[高维 128D] 提取隐状态并计算 FS 标架...")
        h_pos_high = self._extract_hidden_batch(model_high, test_pos, max_batch=128)
        h_neg_high = self._extract_hidden_batch(model_high, test_neg, max_batch=128)

        if h_pos_high is not None and h_neg_high is not None:
            stats_pos_high = compute_batch_spiral_helix_statistics(
                self.fs_analyzer, h_pos_high
            )
            stats_neg_high = compute_batch_spiral_helix_statistics(
                self.fs_analyzer, h_neg_high
            )
            results["high_dim"]["pos"] = stats_pos_high
            results["high_dim"]["neg"] = stats_neg_high

            # 判决
            cv_pos = stats_pos_high.get("cv_tan_theta", float("nan"))
            cv_neg = stats_neg_high.get("cv_tan_theta", float("nan"))
            r2 = stats_pos_high.get("r_squared", float("nan"))

            if not np.isnan(cv_pos):
                if cv_pos < 0.3 and cv_pos < cv_neg:
                    verdict = "SUPPORT"
                elif cv_pos > 0.5:
                    verdict = "REFUTED"
                else:
                    verdict = "INCONCLUSIVE"
            else:
                verdict = "INCONCLUSIVE"

            self.logger.info(
                f"  [高维] 正例 CV={cv_pos:.4f}, 负例 CV={cv_neg:.4f}, "
                f"R²={r2:.4f} → {verdict}"
            )
            results["high_dim"]["verdict"] = verdict
        else:
            results["high_dim"]["verdict"] = "INCONCLUSIVE"
            results["high_dim"]["error"] = "insufficient_hidden_states"

        # ── 低维对照 (32D) ──
        if model_low is not None:
            self.logger.info("[低维 32D] 提取隐状态并计算 FS 标架...")
            h_pos_low = self._extract_hidden_batch(model_low, test_pos, max_batch=128)
            h_neg_low = self._extract_hidden_batch(model_low, test_neg, max_batch=128)

            if h_pos_low is not None and h_neg_low is not None:
                stats_pos_low = compute_batch_spiral_helix_statistics(
                    self.fs_analyzer, h_pos_low
                )
                stats_neg_low = compute_batch_spiral_helix_statistics(
                    self.fs_analyzer, h_neg_low
                )
                results["low_dim"]["pos"] = stats_pos_low
                results["low_dim"]["neg"] = stats_neg_low

                cv_pos_low = stats_pos_low.get("cv_tan_theta", float("nan"))
                r2_low = stats_pos_low.get("r_squared", float("nan"))
                self.logger.info(
                    f"  [低维] 正例 CV={cv_pos_low:.4f}, R²={r2_low:.4f}"
                )
                results["low_dim"]["verdict"] = "REFUTED" if not np.isnan(r2_low) and r2_low < 0.1 else "INCONCLUSIVE"

        # ── 维度增益量化 ──
        if results["high_dim"].get("pos") and results["low_dim"].get("pos"):
            r2_high = results["high_dim"]["pos"].get("r_squared", 0)
            r2_low = results["low_dim"]["pos"].get("r_squared", 0)
            gain = (r2_high - r2_low) / max(abs(r2_low), 1e-8) if r2_low != 0 else float("inf")
            results["dimensional_gain"] = {
                "r2_high": r2_high,
                "r2_low": r2_low,
                "relative_gain": gain,
            }
            self.logger.info(f"  维度增益: R² {r2_low:.4f}→{r2_high:.4f} (+{gain:.1%})")

        self.results["part_a"] = results
        return results

    # ─────────────────────────────────────────────
    # Part B: H-push 重测 — InfoNCE vs 铰链推离
    # ─────────────────────────────────────────────

    def run_part_b_push(
        self,
        model_high: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
    ) -> Dict:
        """Part B: InfoNCE 对比推离 vs 铰链推离 (H-push 重测)。

        核心假设:
            H-push: 合法/非法叙事轨迹在几何信号上可被推离分离。
            exp9 结果: 铰链损失 Cohen's d=0.007 (REFUTED)，
                      对纯几何信号推离完全失效。

        检验逻辑:
            1. 提取高维隐状态的锚点-正例-负例三元组
            2. 用 InfoNCE 计算推离损失和推离强度
            3. 对比: InfoNCE 推离效果 vs 铰链推离效果
            4. 检验: push_ratio < 0.8 (负例相似度显著低于正例)
        """
        self.logger.info("=" * 60)
        self.logger.info("Part B: H-push 重测 — InfoNCE vs 铰链推离")
        self.logger.info("=" * 60)

        # 提取隐状态
        h_pos = self._extract_hidden_batch(model_high, test_pos, max_batch=64)
        h_neg = self._extract_hidden_batch(model_high, test_neg, max_batch=64)

        if h_pos is None or h_neg is None:
            return {"verdict": "INCONCLUSIVE", "error": "insufficient_data"}

        # 构造锚点: 使用序列中心位置的隐状态
        N = min(h_pos.size(0), h_neg.size(0))
        T = min(h_pos.size(1), h_neg.size(1))
        anchor_idx = T // 2

        h_anchor = h_pos[:N, anchor_idx, :].to(self.device)  # [N, D]
        h_pos_ref = h_pos[:N, anchor_idx + 1, :].to(self.device)  # [N, D]

        # 构造负例池 (多种方法)
        neg_results = {}
        for method in ["temporal_contrast", "divergent", "shuffle"]:
            self.logger.info(f"  InfoNCE 负例构造方法: {method}")
            neg_pool = build_negative_pool(
                h_neg[:N],
                method=method,
                num_negatives=self.exp10_cfg.infonce_num_negatives,
            ).to(self.device)

            # InfoNCE 损失
            loss, info = self.infonce_push(h_anchor, h_pos_ref, neg_pool)
            push_diag = self.infonce_push.compute_push_strength(
                h_anchor, neg_pool
            )

            neg_results[method] = {
                "loss": float(loss.item()),
                **info,
                **push_diag,
            }

        # ── 铰链对照 ──
        self.logger.info("  铰链推离对照...")
        hinge_results = self._compute_hinge_push(h_anchor, h_pos_ref, h_neg[:N])
        neg_results["hinge_baseline"] = hinge_results

        # ── 判决 ──
        best_method = max(
            neg_results,
            key=lambda m: neg_results[m].get("frac_effective_push", 0)
            if m != "hinge_baseline"
            else neg_results[m].get("hinge_cohens_d", 0),
        )
        best_push = neg_results[best_method].get("frac_effective_push", 0)
        infonce_best = max(
            neg_results[m].get("frac_effective_push", 0)
            for m in ["temporal_contrast", "divergent", "shuffle"]
        )

        if infonce_best > 0.6:
            verdict = "SUPPORT"
        elif infonce_best < 0.3:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"

        self.logger.info(
            f"  InfoNCE 最佳推离率={infonce_best:.3f} (方法={best_method}), "
            f"铰链 Cohen's d={hinge_results.get('hinge_cohens_d', 0):.4f} → {verdict}"
        )

        results = {
            "methods": neg_results,
            "best_method": best_method,
            "verdict": verdict,
            "infonce_vs_hinge_improvement": (
                infonce_best - abs(hinge_results.get("hinge_cohens_d", 0))
            ),
        }
        self.results["part_b"] = results
        return results

    def _compute_hinge_push(
        self,
        h_anchor: torch.Tensor,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
    ) -> Dict:
        """计算铰链推离效果 (exp9 对照基准)。"""
        with torch.no_grad():
            # 正例距离
            pos_dist = torch.norm(h_anchor - h_pos, dim=-1)
            # 负例距离 (取序列中心)
            T = h_neg.size(1)
            neg_h = h_neg[:, T // 2, :]
            neg_dist = torch.norm(h_anchor - neg_h, dim=-1)

            # Cohen's d
            pooled_std = torch.sqrt(
                (pos_dist.var() + neg_dist.var()) / 2
            )
            d = (neg_dist.mean() - pos_dist.mean()) / (pooled_std + 1e-8)

            # 铰链损失: relu(margin - neg_dist)
            margin = 2.0
            hinge = torch.relu(margin - neg_dist).mean()

        return {
            "hinge_cohens_d": float(d.item()),
            "hinge_loss": float(hinge.item()),
            "pos_dist_mean": float(pos_dist.mean().item()),
            "neg_dist_mean": float(neg_dist.mean().item()),
        }

    # ─────────────────────────────────────────────
    # Part C: H-phase 重测 — 变长序列陈数相变
    # ─────────────────────────────────────────────

    def run_part_c_phase(
        self,
        model_high: CausalTransformer,
        story_generator,  # EnhancedClosureGenerator
    ) -> Dict:
        """Part C: 变长序列陈数代理相变检测 (H-phase 重测)。

        核心假设:
            H-phase: 在某个序列长度临界值处，累积陈数代理会出现
                     跳变，对应"叙事复杂度的拓扑相变"。
            exp9 结果: INCONCLUSIVE (数据等长导致无法检测)。

        检验逻辑:
            1. 对每个长度桶 (16~512 token) 生成独立的测试故事
            2. 提取隐状态并计算累积陈数代理
            3. Bootstrap 检验相邻长度桶的陈数差是否显著
            4. 检测是否存在突变点 (相邻桶差 > 2σ)
        """
        self.logger.info("=" * 60)
        self.logger.info("Part C: H-phase 重测 — 变长序列陈数相变检测")
        self.logger.info("=" * 60)

        lengths = self.exp10_cfg.phase_seq_lengths
        n_per_length = self.exp10_cfg.phase_stories_per_length
        n_bootstrap = self.exp10_cfg.phase_bootstrap_samples

        chern_by_length: Dict[int, List[float]] = {}
        fs_stats_by_length: Dict[int, Dict] = {}

        for L in lengths:
            self.logger.info(f"  长度桶 L={L}...")
            # 生成变长故事
            stories_pos, stories_neg = self._generate_variable_length_stories(
                story_generator, L, n_per_length
            )

            # 提取隐状态 (只用正例，保持因果一致性)
            h_batch = self._extract_hidden_batch(
                model_high, stories_pos, max_batch=n_per_length
            )
            if h_batch is None:
                chern_by_length[L] = []
                continue

            # 计算 FS 标架和陈数代理
            try:
                fs_result = self.fs_analyzer.analyze(h_batch, compute_chern=True)
                chern_vals = fs_result.chern_proxy.detach().cpu().numpy().tolist()
                chern_by_length[L] = chern_vals

                # 螺旋升角统计量
                helix_stats = compute_batch_spiral_helix_statistics(
                    self.fs_analyzer, h_batch
                )
                fs_stats_by_length[L] = helix_stats

                self.logger.info(
                    f"    Chern mean={np.mean(chern_vals):.4f}±{np.std(chern_vals):.4f}, "
                    f"CV_tanΘ={helix_stats.get('cv_tan_theta', 'nan')}"
                )
            except Exception as e:
                self.logger.warning(f"    FS 分析失败: {e}")
                chern_by_length[L] = []

        # ── Bootstrap 相邻长度桶差异检验 ──
        transitions = []
        sorted_lengths = sorted(chern_by_length.keys())
        for i in range(len(sorted_lengths) - 1):
            L1, L2 = sorted_lengths[i], sorted_lengths[i + 1]
            vals1 = np.array(chern_by_length[L1])
            vals2 = np.array(chern_by_length[L2])

            if len(vals1) < 10 or len(vals2) < 10:
                transitions.append({
                    "L_from": L1, "L_to": L2,
                    "verdict": "INCONCLUSIVE",
                    "reason": "insufficient_samples",
                })
                continue

            # Bootstrap 差异分布
            diffs = []
            for _ in range(n_bootstrap):
                s1 = np.random.choice(vals1, size=len(vals1), replace=True)
                s2 = np.random.choice(vals2, size=len(vals2), replace=True)
                diffs.append(np.mean(s2) - np.mean(s1))
            diffs = np.array(diffs)

            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            ci_low = np.percentile(diffs, 2.5)
            ci_high = np.percentile(diffs, 97.5)

            # 判决: CI 不包含 0 且效应量 > 0.2
            is_significant = (ci_low > 0 or ci_high < 0)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0
            is_large_effect = abs(cohens_d) > 0.2

            if is_significant and is_large_effect:
                verdict = "PHASE_TRANSITION"
            elif is_significant:
                verdict = "GRADUAL_SHIFT"
            else:
                verdict = "NO_TRANSITION"

            transitions.append({
                "L_from": L1,
                "L_to": L2,
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "cohens_d": float(cohens_d),
                "verdict": verdict,
            })
            self.logger.info(
                f"    {L1}→{L2}: 陈数Δ={mean_diff:.4f}±{std_diff:.4f}, "
                f"d={cohens_d:.2f} → {verdict}"
            )

        # ── 全局判决 ──
        phase_transitions = [t for t in transitions if t["verdict"] == "PHASE_TRANSITION"]
        if phase_transitions:
            global_verdict = "SUPPORT"
            critical_length = phase_transitions[0]["L_from"]
            self.logger.info(f"  检测到相变! 临界长度 ≈ {critical_length}")
        elif any(t["verdict"] == "GRADUAL_SHIFT" for t in transitions):
            global_verdict = "GRADUAL"  # 渐变而非突变
        else:
            global_verdict = "REFUTED"

        results = {
            "chern_by_length": {str(k): v for k, v in chern_by_length.items()},
            "transitions": transitions,
            "fs_stats_by_length": {str(k): v for k, v in fs_stats_by_length.items()},
            "verdict": global_verdict,
        }
        self.results["part_c"] = results
        return results

    def _generate_variable_length_stories(
        self,
        generator,
        target_length: int,
        n_stories: int,
    ):
        """生成指定长度的测试故事。

        通过调整 generator 的 min_steps/max_steps 参数
        来近似控制序列长度。
        """
        # 估计步数: 每步约 2-3 token
        est_steps = max(4, target_length // 3)
        old_min, old_max = generator.min_steps, generator.max_steps
        generator.min_steps = est_steps
        generator.max_steps = est_steps + 2

        pos_stories, neg_stories = [], []
        for i in range(n_stories):
            try:
                pos, neg = generator.generate_pair(i + 100000)
                if pos and neg and len(pos.steps) >= 3:
                    pos_stories.append(pos)
                    neg_stories.append(neg)
            except Exception:
                continue

        # 恢复
        generator.min_steps, generator.max_steps = old_min, old_max
        return pos_stories, neg_stories

    # ─────────────────────────────────────────────
    # Part D: 集成判决 — 三层假设联合更新
    # ─────────────────────────────────────────────

    def run_part_d_integrated_verdict(self) -> Dict:
        """Part D: 三层假设的联合 Bayesian 更新。

        将 Part A (H-helix), Part B (H-push), Part C (H-phase)
        的判决整合为统一理论置信度更新。
        """
        self.logger.info("=" * 60)
        self.logger.info("Part D: 集成判决 — 三层假设联合更新")
        self.logger.info("=" * 60)

        part_a = self.results.get("part_a", {}).get("high_dim", {}).get("verdict", "INCONCLUSIVE")
        part_b = self.results.get("part_b", {}).get("verdict", "INCONCLUSIVE")
        part_c = self.results.get("part_c", {}).get("verdict", "INCONCLUSIVE")

        # 先验: 基于 exp8+exp9 的固化度 0.48
        prior_certainty = 0.48

        # 判决→证据权重映射
        verdict_weight = {
            "SUPPORT": +0.15,
            "GRADUAL": +0.05,
            "INCONCLUSIVE": 0.0,
            "REFUTED": -0.10,
        }

        delta_a = verdict_weight.get(part_a, 0)
        delta_b = verdict_weight.get(part_b, 0)
        delta_c = verdict_weight.get(part_c, 0)

        # Bayesian 更新 (简化: 线性加权)
        # 权重: FS 标架首次使用，权重 0.4；InfoNCE 验证推离动力学，权重 0.35；
        #       H-phase 变长序列，权重 0.25
        total_delta = 0.40 * delta_a + 0.35 * delta_b + 0.25 * delta_c
        posterior_certainty = max(0.05, min(0.85, prior_certainty + total_delta))

        # 叙事位置判定
        if posterior_certainty > 0.60:
            narrative_stage = "涅槃III-灵魂扩容: 几何动力学重设计成功"
        elif posterior_certainty > 0.50:
            narrative_stage = "涅槃II-断舍剥离: 保留刚性+拓扑，中间层仍需迭代"
        elif posterior_certainty > 0.40:
            narrative_stage = "涅槃I-直面混沌: 方向正确但细节待验证"
        else:
            narrative_stage = "崩溃IV: 高维升级未改变结论，需检讨几何本体论"

        # 可固化项识别
        solidifiable = []
        if part_a == "SUPPORT":
            solidifiable.append("FS标架在128D空间有效 (H-helix 升级为铁律)")
        if part_b == "SUPPORT":
            solidifiable.append("InfoNCE推离取代铰链推离 (H-push 从REFUTED复活)")
        if part_c in ("SUPPORT", "GRADUAL"):
            solidifiable.append("变长序列陈数演化规律 (H-phase 从INCONCLUSIVE收敛)")

        integrated = {
            "prior_certainty": prior_certainty,
            "verdicts": {"H-helix": part_a, "H-push": part_b, "H-phase": part_c},
            "deltas": {"H-helix": delta_a, "H-push": delta_b, "H-phase": delta_c},
            "total_delta": total_delta,
            "posterior_certainty": posterior_certainty,
            "narrative_stage": narrative_stage,
            "solidifiable_items": solidifiable,
            "next_experiment": self._recommend_next(posterior_certainty, part_a, part_b, part_c),
        }

        self.logger.info(f"  先验固化度: {prior_certainty:.2f}")
        self.logger.info(f"  判决: A={part_a}({delta_a:+.2f}) B={part_b}({delta_b:+.2f}) C={part_c}({delta_c:+.2f})")
        self.logger.info(f"  后验固化度: {posterior_certainty:.2f} (+{total_delta:+.2f})")
        self.logger.info(f"  叙事位置: {narrative_stage}")
        if solidifiable:
            for item in solidifiable:
                self.logger.info(f"  可固化: {item}")

        self.results["part_d"] = integrated
        return integrated

    def _recommend_next(
        self,
        certainty: float,
        part_a: str,
        part_b: str,
        part_c: str,
    ) -> str:
        """基于当前结果推荐下一步实验。"""
        if certainty > 0.60:
            return "exp11: 真实LLM FS标架验证 (GPT-2-small/LLaMA-7B) + 预注册交叉验证"
        elif part_a == "SUPPORT" and part_b == "SUPPORT":
            return "exp11: 扩大NPNW世界规模 (10x10网格) + 多层嵌套叙事"
        elif part_b == "REFUTED" and part_a != "REFUTED":
            return "exp11: 混合信号推离 (几何+语义) 替代纯几何推离"
        else:
            return "exp11: 检讨几何本体论边界条件 + 缩小理论适用范围"

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────

    def _extract_hidden_batch(
        self,
        model: CausalTransformer,
        stories: List[Story],
        max_batch: int = 128,
    ) -> Optional[torch.Tensor]:
        """批量提取故事隐状态。

        Returns:
            hidden: [B, T, D] 或 None (如果失败)
        """
        model.eval()
        hidden_list = []
        max_len = self.config["model"]["max_seq_len"]

        for story in stories[:max_batch]:
            try:
                steps_data = [
                    {"state": s.state, "action": s.action, "causal_labels": s.causal_labels}
                    for s in story.steps
                ]
                token_ids = self.tokenizer.encode_story(steps_data)
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                if len(token_ids) < 4:
                    continue

                input_ids = torch.tensor(
                    [token_ids], dtype=torch.long
                ).to(self.device)

                with torch.no_grad():
                    _, hidden = model(input_ids)  # [1, T, D]

                hidden_list.append(hidden[0])  # [T, D]
            except Exception:
                continue

        if not hidden_list:
            return None

        # 填充到相同长度
        max_T = min(max(h.size(0) for h in hidden_list), max_len)
        padded = []
        for h in hidden_list:
            T = h.size(0)
            if T < max_T:
                pad = torch.zeros(max_T - T, h.size(1), device=h.device)
                h = torch.cat([h, pad], dim=0)
            else:
                h = h[:max_T]
            padded.append(h)

        return torch.stack(padded, dim=0)  # [B, T, D]

    # ─────────────────────────────────────────────
    # Part D2: 几何动力学演化监控 (NEW — 基于EvolutionCycleEngine)
    # ─────────────────────────────────────────────

    def run_dynamics_monitoring(
        self,
        model_high: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        training_history: Optional[Dict] = None,
    ) -> Dict:
        """运行几何动力学演化监控。

        在训练完成后，用测试数据观察模型隐空间的几何约束场状态。
        模拟多个"伪epoch"来生成动力学演化轨迹。

        Args:
            model_high: 训练好的高维模型
            test_pos: 正例测试故事
            test_neg: 负例测试故事
            training_history: 训练历史（可选，用于注入真实训练指标）

        Returns:
            动力学演化报告
        """
        self.logger.info("=" * 60)
        self.logger.info("Part D2: 几何动力学演化监控 (基于EvolutionCycleEngine)")
        self.logger.info("=" * 60)

        # 提取隐状态用作几何分析
        self.logger.info("  提取测试集隐状态用于几何动力学分析...")
        h_pos = self._extract_hidden_batch(model_high, test_pos, max_batch=32)
        h_neg = self._extract_hidden_batch(model_high, test_neg, max_batch=32)

        if h_pos is None:
            self.logger.warning("  无法提取隐状态，动力学监控跳过")
            return {"verdict": "INCONCLUSIVE", "error": "no_hidden_states"}

        # 构建"伪epoch序列"：在正例和负例上分别采样不同子集
        # 模拟训练过程中约束场的演化
        B_pos = h_pos.size(0)
        n_pseudo_epochs = min(8, B_pos // 4)  # 伪epoch数
        if n_pseudo_epochs < 2:
            n_pseudo_epochs = 2

        self.logger.info(f"  运行 {n_pseudo_epochs} 个伪epoch的动力学监控...")

        train_losses = []
        if training_history and training_history.get("train_loss"):
            train_losses = training_history["train_loss"]

        rf_losses = []
        if training_history and training_history.get("rf_loss"):
            rf_losses = training_history["rf_loss"]

        for ep in range(n_pseudo_epochs):
            # 取正例的一个子集
            idx_start = (ep * 4) % max(1, B_pos - 4)
            idx_end = min(idx_start + 4, B_pos)
            h_subset = h_pos[idx_start:idx_end]  # [subset, T, D]

            # 混合少量负例（模拟训练中的正负样本混合）
            if h_neg is not None and h_neg.size(0) > 0:
                neg_idx = (ep * 2) % h_neg.size(0)
                h_neg_subset = h_neg[neg_idx:neg_idx+2]
                h_subset = torch.cat([h_subset, h_neg_subset], dim=0)

            # 训练指标（如果有训练历史则用真实数据）
            train_loss = train_losses[ep] if ep < len(train_losses) else float('nan')
            rf_loss = rf_losses[ep] if ep < len(rf_losses) else float('nan')

            # 记录epoch
            snap = self.dynamics_monitor.record_epoch(
                epoch=ep,
                hidden_states=h_subset,
                train_loss=train_loss,
                rf_loss=rf_loss,
            )

            # 运行演化循环
            cycle = self.dynamics_monitor.run_cycle(ep)

            state_emoji = {
                "steady": "◯", "pressurized": "◎", "breaking": "◉", "recovering": "◑"
            }
            emoji = state_emoji.get(snap.state, "?")

            self.logger.info(
                f"    Epoch {ep}: ψ_geo={snap.psi_geo:+.3f} "
                f"[{emoji} {snap.state}] "
                f"κ_mean={snap.kappa_mean:.4f} "
                f"τ_mean={snap.tau_mean:.4f} "
                f"正交度={snap.frame_orthogonality:.3f} "
                f"条件数={snap.mve_condition_number:.1f}"
            )

            if cycle.break_triggered:
                self.logger.warning(
                    f"      ⚠ 破缺触发! {cycle.recommendations[0] if cycle.recommendations else ''}"
                )
                if cycle.gamma_adjustments:
                    self.logger.warning(
                        f"      制动调整: {cycle.gamma_adjustments}"
                    )
                # 开启破缺窗口
                self.brake_tensor.open_breaking_window(ep)

            # tick制动张量窗口
            self.brake_tensor.tick_window(ep)

            # 如果有制动调整，应用之
            if cycle.gamma_adjustments:
                self.brake_tensor.adjust_gamma_by_deltas(**cycle.gamma_adjustments)

            # 基于ψ_geo调节制动
            if not np.isnan(snap.psi_geo):
                self.brake_tensor.adjust_gamma_by_error_signal(snap.psi_geo)

            # 消耗预算
            self.brake_tensor.consume_budget(0.05)

            self._dynamics_snapshots.append(snap)
            self._dynamics_cycles.append(cycle)

        # 生成动力学报告
        dynamics_report = self.dynamics_monitor.get_dynamics_report()
        brake_summary = self.brake_tensor.get_state_summary()

        self.logger.info(f"  动力学判决: {dynamics_report.get('dynamics_verdict', 'N/A')}")
        self.logger.info(f"  制动状态: 窗口={brake_summary['window_active']}, "
                        f"预算={brake_summary['budget_remaining']:.3f}, "
                        f"谱条件数={brake_summary['spectral_condition']}")

        results = {
            "dynamics_report": dynamics_report,
            "brake_tensor_summary": brake_summary,
            "n_pseudo_epochs": n_pseudo_epochs,
            "verdict": dynamics_report.get("dynamics_verdict", "INCONCLUSIVE"),
        }

        self.results["part_d2_dynamics"] = results
        return results

    # ─────────────────────────────────────────────
    # Part D3: 三条定理验证 (NEW — 十三字公理固化)
    # ─────────────────────────────────────────────

    def run_theorem_verification(
        self,
        model_high: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        context_lengths: Optional[List[int]] = None,
        context_errors: Optional[List[float]] = None,
    ) -> "VerificationReport":
        """运行三定理统一验证。

        定理一（信息本体论）：正例 vs 负例的 χ 比较
        定理二（几何守恒论）：τ/κ 的共形不变性
        定理三（拓扑残差论）：跨上下文长度的误差渐进界
        """
        self.logger.info("=" * 60)
        self.logger.info("Part D3: 三定理验证（十三字公理固化）")
        self.logger.info("  公理：'信息化为世界模型，世界模型遵守几何规则'")
        self.logger.info("=" * 60)

        # 提取隐状态
        h_pos_all = self._extract_hidden_batch(model_high, test_pos, max_batch=16)
        h_neg_all = self._extract_hidden_batch(model_high, test_neg, max_batch=16)

        if h_pos_all is None:
            self.logger.warning("  无法提取隐状态，定理验证跳过")
            from ..theorems.theorem_verifier import VerificationReport
            return VerificationReport()

        # 拆分为独立样本列表
        B_pos = h_pos_all.size(0)
        h_pos_list = [h_pos_all[b] for b in range(min(B_pos, 12))]
        h_neg_list = []
        if h_neg_all is not None:
            B_neg = h_neg_all.size(0)
            h_neg_list = [h_neg_all[b] for b in range(min(B_neg, 12))]

        # 定理二的 tanΘ 数据
        tan_theta_original = None
        if h_pos_all is not None:
            try:
                result = self.fs_analyzer.analyze(h_pos_all, compute_chern=False)
                tan_theta_original = result.tan_theta.flatten().detach().cpu().numpy()
            except Exception as e:
                self.logger.warning(f"  tanΘ 提取失败: {e}")

        # 模型容量
        model_capacity_log10 = np.log10(
            sum(p.numel() for p in model_high.parameters())
        ) if hasattr(model_high, 'parameters') else None

        # 运行三定理验证
        report = self.theorem_verifier.verify_all(
            hidden_pos=h_pos_list,
            hidden_neg=h_neg_list,
            tan_theta_original=tan_theta_original,
            context_lengths=context_lengths,
            errors=context_errors,
            model_capacity=model_capacity_log10,
        )

        # 日志输出
        t1_v = str(report.theorem_1.verdict.value) if report.theorem_1 and hasattr(report.theorem_1.verdict, 'value') else str(report.theorem_1.verdict) if report.theorem_1 else "PENDING"
        t1_c = f"{report.theorem_1.confidence:.2%}" if report.theorem_1 else "N/A"
        t2_v = str(report.theorem_2.verdict.value) if report.theorem_2 and hasattr(report.theorem_2.verdict, 'value') else str(report.theorem_2.verdict) if report.theorem_2 else "PENDING"
        t2_c = f"{report.theorem_2.confidence:.2%}" if report.theorem_2 else "N/A"
        t3_v = str(report.theorem_3.verdict.value) if report.theorem_3 and hasattr(report.theorem_3.verdict, 'value') else str(report.theorem_3.verdict) if report.theorem_3 else "PENDING"
        t3_c = f"{report.theorem_3.confidence:.2%}" if report.theorem_3 else "N/A"

        self.logger.info(f"  定理一（信息本体论 χ≠0）：{t1_v} (置信度 {t1_c})")
        self.logger.info(f"  定理二（几何守恒论 τ/κ=const）：{t2_v} (置信度 {t2_c})")
        self.logger.info(f"  定理三（拓扑残差论 ε_min>0）：{t3_v} (置信度 {t3_c})")
        self.logger.info(f"  综合支持度：{report.overall_support:.2%}")
        self.logger.info(f"  公理验证：{'✓ 通过' if report.axiom_verified else '✗ 未通过'}")
        self.logger.info(f"  自洽性：{report.self_consistency_score:.2%}")

        self.results["part_d3_theorems"] = report.to_dict()
        return report

    def run_full(
        self,
        model_high: CausalTransformer,
        model_low: Optional[CausalTransformer],
        story_generator,
        test_pos: List[Story],
        test_neg: List[Story],
        training_history: Optional[Dict] = None,
    ) -> Dict:
        """运行完整 exp10 四部分实验 + 动力学监控。

        Args:
            model_high: 高维 MVE (128D) 训练好的模型
            model_low:  低维 MVE (32D) 对照模型 (可选)
            story_generator: EnhancedClosureGenerator
            test_pos: 正例测试故事列表
            test_neg: 负例测试故事列表
            training_history: 训练历史 (可选，用于动力学监控)

        Returns:
            完整的实验结果字典
        """
        start_time = time.time()

        # Part A: H-helix (FS 螺旋升角)
        self.run_part_a_helix(model_high, model_low, test_pos, test_neg)

        # Part B: H-push (InfoNCE 推离)
        self.run_part_b_push(model_high, test_pos, test_neg)

        # Part C: H-phase (变长序列相变)
        self.run_part_c_phase(model_high, story_generator)

        # Part D: 集成判决
        integrated = self.run_part_d_integrated_verdict()

        # Part D2: 几何动力学演化监控 (NEW)
        self.run_dynamics_monitoring(
            model_high, test_pos, test_neg,
            training_history=training_history,
        )

        # Part D3: 三条定理验证 (NEW — 十三字公理固化)
        self.run_theorem_verification(
            model_high, test_pos, test_neg,
        )

        elapsed = time.time() - start_time
        self.logger.info(f"exp10 完成，耗时 {elapsed:.1f}s")

        return {
            **self.results,
            "elapsed_seconds": elapsed,
            "narrative_driver": (
                "如果我们的几何本体论是对的，那么错误一定出在"
                "'如何让这个几何在动力学上运作'的细节里——"
                "让我们升级实验尺度，把这些细节找出来。"
            ),
        }


def create_high_dim_model(
    config: dict,
    base_dim: int = 128,
    max_seq_len: int = 256,
) -> CausalTransformer:
    """创建高维 MVE 模型实例。

    相对于原 32D 配置，修改 base_dim、d_model 和 max_seq_len。
    d_model = base_dim * 4 (保持 4:1 的过完备比例)。
    """
    high_config = dict(config)  # 浅拷贝
    high_config["model"] = dict(config["model"])
    high_config["model"]["base_dim"] = base_dim
    high_config["model"]["d_model"] = base_dim * 4
    high_config["model"]["max_seq_len"] = max_seq_len
    # 按比例调整 n_heads (需整除)
    high_config["model"]["n_heads"] = max(4, base_dim // 32)

    model = CausalTransformer(high_config)
    return model


def create_low_dim_model(
    config: dict,
    max_seq_len: int = 128,
) -> CausalTransformer:
    """创建低维对照模型 (32D, 复现 exp9 配置)。"""
    low_config = dict(config)
    low_config["model"] = dict(config["model"])
    low_config["model"]["max_seq_len"] = max_seq_len
    model = CausalTransformer(low_config)
    return model
