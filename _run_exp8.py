"""exp8: 吸引子定向分化损失公式(A)的三体制受控检验.

实验设计依据:
- 论文: 对称、破缺与约束场_GUIT六步闭环再生版.md §8.3
- 实验设计: 严格验证exp8.md (含P0-2/P0-4修复)
- 审计: 最终理论验证_GUIT六模块闭合审计报告.md

三体制:
- A (Baseline): 仅 LM 损失
- B (旧阈值): LM + 公式(R) — 应复现塌缩
- C (新吸引子): LM + 公式(A) — 核心检验

三阶段训练:
- Phase 1: 冻结模型, 估计 F_flex* 和 m
- Phase 2: 联合训练
- Phase 3: λ 扫描与稳定性分析
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent / "causal_gauge_field"))
sys.path.insert(0, str(Path(__file__).parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.enhanced_generator import EnhancedClosureGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.models.gauge_field import GaugeField
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from torch.utils.data import DataLoader

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy 未安装, 部分统计检验将使用近似方法")


# ═══════════════════════════════════════════════════════════════════
# 模块1: 可学习子空间投影 (H-decomp 最小侵入式实现)
# ═══════════════════════════════════════════════════════════════════

class SubspaceProjection(nn.Module):
    """将隐状态投影到物理子空间和柔性子空间.

    对应 exp8 设计 §4: W_phys, W_flex 两个轻量投影矩阵,
    由损失引导分化. 不强制正交, 以观察是否自发分化.
    """

    def __init__(self, d_model: int, r: int = 16):
        super().__init__()
        self.d_model = d_model
        self.r = r
        # r x d_model 投影矩阵
        self.W_phys = nn.Parameter(torch.randn(r, d_model) * 0.02)
        self.W_flex = nn.Parameter(torch.randn(r, d_model) * 0.02)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: [B, T, d_model] -> h_phys: [B, T, r], h_flex: [B, T, r]"""
        # [B, T, d] @ [d, r] -> [B, T, r]
        h_phys = torch.einsum('btd,rd->btr', h, self.W_phys)
        h_flex = torch.einsum('btd,rd->btr', h, self.W_flex)
        return h_phys, h_flex

    def orthogonality_loss(self) -> torch.Tensor:
        """辅助: 鼓励 W_phys 和 W_flex 近似正交 (不强制)."""
        # W_phys: [r, d], W_flex: [r, d]
        gram = self.W_phys @ self.W_flex.T  # [r, r]
        return (gram ** 2).mean()


# ═══════════════════════════════════════════════════════════════════
# 模块2: 离散曲率估计
# ═══════════════════════════════════════════════════════════════════

def estimate_discrete_curvature(
    h_seq: torch.Tensor,
    projector: SubspaceProjection,
    mode: str = "phys"
) -> torch.Tensor:
    """用二阶差分近似曲率 ||F||².

    h_seq: [B, T, d_model]
    mode: "phys" | "flex"

    返回 [B] 的平均曲率范数的平方.
    """
    B, T, d = h_seq.shape
    if T < 3:
        return torch.zeros(B, device=h_seq.device)

    h_prev = h_seq[:, :-2, :]   # [B, T-2, d]
    h_curr = h_seq[:, 1:-1, :]  # [B, T-2, d]
    h_next = h_seq[:, 2:, :]    # [B, T-2, d]

    # 一阶差分 (向前)
    d1 = h_curr - h_prev    # [B, T-2, d]
    # 二阶差分 (曲率近似)
    d2 = h_next - 2 * h_curr + h_prev  # [B, T-2, d]

    # 投影到子空间
    if mode == "phys":
        d1_proj = torch.einsum('btd,rd->btr', d1, projector.W_phys)
        d2_proj = torch.einsum('btd,rd->btr', d2, projector.W_phys)
    else:
        d1_proj = torch.einsum('btd,rd->btr', d1, projector.W_flex)
        d2_proj = torch.einsum('btd,rd->btr', d2, projector.W_flex)

    # ||F||² ≈ ||d2||² (简化, 不使用 O_{t+1,t} 形式)
    curv_norm2 = (d2_proj ** 2).sum(dim=-1).mean(dim=-1)  # [B]

    return curv_norm2


# ═══════════════════════════════════════════════════════════════════
# 模块3: 公式(A) 损失 — 吸引子定向分化损失
# ═══════════════════════════════════════════════════════════════════

def formula_a_loss(
    phys_curv: torch.Tensor,         # [B_phys]
    flex_pos_curv: torch.Tensor,     # [B_pos]
    flex_neg_curv: torch.Tensor,     # [B_neg]
    F_flex_star: float,              # 柔性吸引子 (标量, 平方曲率均值)
    m: float,                         # 安全边际
    lambda_phys: float = 1.0,
    lambda_flex_pos: float = 0.5,
    lambda_flex_neg: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """公式(A): 吸引子定向分化损失.

    返回: {"total": loss, "phys": L_phys, "flex_pos": L_pos, "flex_neg": L_neg}
    """
    F_star_tensor = torch.tensor(F_flex_star, device=phys_curv.device)
    m_tensor = torch.tensor(m, device=phys_curv.device)

    # 刚性项: 拉向零吸引子
    L_phys = lambda_phys * phys_curv.mean()

    # 正例: 拉向 F_flex*
    L_flex_pos = lambda_flex_pos * ((flex_pos_curv - F_star_tensor) ** 2).mean()

    # 负例: 推离吸引子 (铰链损失)
    if flex_neg_curv.numel() > 0:
        distances = (flex_neg_curv - F_star_tensor).abs()
        L_flex_neg = lambda_flex_neg * torch.clamp(m_tensor - distances, min=0).mean()
    else:
        L_flex_neg = torch.tensor(0.0, device=phys_curv.device)

    total = L_phys + L_flex_pos + L_flex_neg

    return {
        "total": total,
        "phys": L_phys,
        "flex_pos": L_flex_pos,
        "flex_neg": L_flex_neg,
    }


def formula_r_loss(
    phys_curv: torch.Tensor,
    flex_curv: torch.Tensor,
    tau: float = 0.5,
    lambda_phys: float = 1.0,
    lambda_flex: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """公式(R): 阈值惩罚损失 (三次重构版本, 体制B用)."""
    L_phys = lambda_phys * phys_curv.mean()
    L_flex = lambda_flex * torch.clamp(flex_curv - tau, min=0).mean()
    return {"total": L_phys + L_flex, "phys": L_phys, "flex": L_flex}


# ═══════════════════════════════════════════════════════════════════
# 模块4: 灵活标签数据生成
# ═══════════════════════════════════════════════════════════════════

class FlexibleLabelGenerator:
    """从增强闭合生成器中构造 phys/flex_pos/flex_neg 标注的转移对.

    策略 (exp8 §3):
    - phys: 物理 token 对应的隐状态转移 (位置/属性变化)
    - flex_pos: 合法叙事变化 (同义改写, 插入无关细节)
    - flex_neg: 非法跳跃 (因果矛盾, 突兀切换)
    """

    PHYS_TOKENS = {"[POS]", "[ATTR]", "[MOVE]", "[PICKUP]", "[DROP]", "[USE_DOOR]"}
    ACTION_TOKENS = {"go_to", "pick_up", "drop", "open_door", "put_in_bag"}

    def __init__(self, generator: EnhancedClosureGenerator, tokenizer: NPNWTokenizer,
                 config: dict, seed: int = 42):
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config
        self.rng = np.random.RandomState(seed)
        # 缓存正例故事的隐状态用于构造变体
        self._story_cache: List[dict] = []

    def generate(self, num_stories: int = 100) -> Dict[str, list]:
        """生成带灵活标签的数据集.

        Story 对象来自 npnw.story_generator.Story (dataclass),
        每个 Story 有 .steps (List[StoryStep]), 每个 StoryStep 有 .state/.action/.causal_labels.

        返回: {"phys": [...], "flex_pos": [...], "flex_neg": [...],
                "counts": {...}, "train_pos": [...], ...}
        """
        # 使用增强生成器生成基础正负例对
        (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = \
            self.generator.generate_dataset(num_stories=num_stories)

        all_pos = train_pos + val_pos + test_pos  # List[Story]
        all_neg = train_neg + val_neg + test_neg  # List[Story]

        # 构造灵活标签
        phys_transfers = []
        flex_pos_transfers = []
        flex_neg_transfers = []

        for story in all_pos:
            # story 是 Story dataclass, story.steps 是 List[StoryStep]
            steps = story.steps
            if len(steps) < 3:
                continue

            # phys: 检查相邻步骤的物理状态变化 (位置/属性)
            for i in range(len(steps) - 1):
                s_cur = steps[i].state
                s_nxt = steps[i+1].state
                # 检查位置变化
                pos_changed = (s_cur.get("pos_x") != s_nxt.get("pos_x") or
                              s_cur.get("pos_y") != s_nxt.get("pos_y"))
                attr_changed = (s_cur.get("holding") != s_nxt.get("holding") or
                               s_cur.get("stamina") != s_nxt.get("stamina"))
                if pos_changed or attr_changed:
                    end = min(i + 3, len(steps))
                    if end - i >= 3:
                        phys_transfers.append({"idx": i, "steps": steps[i:end],
                                               "type": "phys"})

            # flex_pos: 取中间位置的3步相邻叙事步骤
            mid = max(1, len(steps) // 2)
            if mid + 2 < len(steps):
                flex_pos_transfers.append({"idx": mid, "steps": steps[mid-1:mid+2],
                                           "type": "flex_pos"})

            # flex_neg: 逻辑跳跃 — 取不相邻的3步序列
            if len(steps) >= 5:
                i, j = len(steps)//4, 3*len(steps)//4
                if j - i >= 2:
                    neg_steps = [steps[i], steps[i+1], steps[j]]  # 跳跃
                    flex_neg_transfers.append({"steps": neg_steps, "type": "flex_neg"})

        # 从负例中提取更多 flex_neg (它们天然是因果破裂的)
        for story in all_neg:
            steps = story.steps
            if len(steps) >= 4:
                mid = max(1, len(steps) // 2)
                if mid + 2 < len(steps):
                    flex_neg_transfers.append({"idx": mid, "steps": steps[mid-1:mid+2],
                                           "type": "flex_neg"})

        # 截断到合理数量
        n_phys = min(len(phys_transfers), 300)
        n_flex_pos = min(len(flex_pos_transfers), 300)
        n_flex_neg = min(len(flex_neg_transfers), 300)

        phys_transfers = phys_transfers[:n_phys]
        flex_pos_transfers = flex_pos_transfers[:n_flex_pos]
        flex_neg_transfers = flex_neg_transfers[:n_flex_neg]

        return {
            "phys": phys_transfers,
            "flex_pos": flex_pos_transfers,
            "flex_neg": flex_neg_transfers,
            "train_pos": train_pos, "train_neg": train_neg,
            "val_pos": val_pos, "val_neg": val_neg,
            "test_pos": test_pos, "test_neg": test_neg,
            "counts": {"phys": n_phys, "flex_pos": n_flex_pos, "flex_neg": n_flex_neg},
        }


# ═══════════════════════════════════════════════════════════════════
# 模块5: 评估指标
# ═══════════════════════════════════════════════════════════════════

def compute_evaluation_metrics(
    flex_pos_curvs: np.ndarray,
    flex_neg_curvs: np.ndarray,
    phys_curvs: np.ndarray,
    F_flex_star_initial: float,
    F_flex_star_final: float,
) -> Dict:
    """计算 exp8 设计的全部评估指标.

    返回字典含: snr, cohens_d, wilcoxon_p, dip_p_pos, dip_p_neg,
               auc, gradient_cos_sim, f_star_drift, decoupling_p, verdicts
    """
    metrics = {}

    # --- SNR ---
    mu_pos = np.mean(flex_pos_curvs)
    mu_neg = np.mean(flex_neg_curvs)
    var_pos = np.var(flex_pos_curvs)
    var_neg = np.var(flex_neg_curvs)
    pooled_sd = np.sqrt(0.5 * (var_pos + var_neg))
    metrics["snr"] = abs(mu_pos - mu_neg) / max(pooled_sd, 1e-8)
    metrics["mu_pos"] = float(mu_pos)
    metrics["mu_neg"] = float(mu_neg)

    # --- Cohen's d ---
    n_pos, n_neg = len(flex_pos_curvs), len(flex_neg_curvs)
    pooled_var = ((n_pos-1)*var_pos + (n_neg-1)*var_neg) / (n_pos+n_neg-2) if n_pos+n_neg>2 else 1.0
    metrics["cohens_d"] = abs(mu_pos - mu_neg) / max(np.sqrt(pooled_var), 1e-8)

    # --- Wilcoxon ---
    if HAS_SCIPY and n_pos > 0 and n_neg > 0:
        try:
            min_n = min(n_pos, n_neg)
            s_pos = np.random.choice(flex_pos_curvs, min_n, replace=False) if n_pos > min_n else flex_pos_curvs
            s_neg = np.random.choice(flex_neg_curvs, min_n, replace=False) if n_neg > min_n else flex_neg_curvs
            _, metrics["wilcoxon_p"] = scipy_stats.wilcoxon(s_pos, s_neg)
        except Exception:
            metrics["wilcoxon_p"] = 1.0
    else:
        # 近似: Mann-Whitney U 通过排序模拟
        combined = np.concatenate([flex_pos_curvs, flex_neg_curvs])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        order = np.argsort(combined)
        ranks = np.zeros(len(combined))
        ranks[order] = np.arange(1, len(combined)+1)
        U = ranks[labels==1].sum() - n_pos*(n_pos+1)/2
        z = (U - n_pos*n_neg/2) / np.sqrt(n_pos*n_neg*(n_pos+n_neg+1)/12 + 1e-8)
        metrics["wilcoxon_p"] = float(2 * (1 - _norm_cdf(abs(z))))

    # --- Hartigan's Dip Test (双峰性) ---
    metrics["dip_p_pos"] = _simple_dip_test(flex_pos_curvs) if HAS_SCIPY else -1
    metrics["dip_p_neg"] = _simple_dip_test(flex_neg_curvs) if HAS_SCIPY else -1
    # 对合并分布做 dip test
    combined_all = np.concatenate([flex_pos_curvs, flex_neg_curvs])
    metrics["dip_p_combined"] = _simple_dip_test(combined_all) if HAS_SCIPY else -1

    # --- AUC (合法/非法分类) ---
    metrics["auc"] = _compute_auc(flex_pos_curvs, flex_neg_curvs, F_flex_star_final)

    # --- F* 漂移率 ---
    if F_flex_star_initial > 1e-8:
        metrics["f_star_drift"] = abs(F_flex_star_final - F_flex_star_initial) / F_flex_star_initial
    else:
        metrics["f_star_drift"] = 0.0 if F_flex_star_final < 1e-8 else 1.0
    metrics["F_flex_star_initial"] = float(F_flex_star_initial)
    metrics["F_flex_star_final"] = float(F_flex_star_final)

    # --- 物理曲率 ---
    metrics["phys_curv_mean"] = float(np.mean(phys_curvs)) if len(phys_curvs) > 0 else 0.0

    # --- 判决 ---
    verdicts = _compute_verdicts(metrics)
    metrics.update(verdicts)

    return metrics


def _norm_cdf(x):
    """标准正态 CDF 近似."""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def _simple_dip_test(data: np.ndarray, n_bootstrap: int = 100) -> float:
    """简化的 Hartigan Dip Test.

    返回近似 p 值 (越小越拒绝单峰).
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 10:
        return 1.0

    try:
        from scipy.stats import gaussian_kde
        n = len(data)
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 200)
        density = kde(x_grid)

        # 找谷值/峰值比率
        peaks, valleys = _find_peaks_valleys(density)
        if len(peaks) < 2 or len(valleys) < 1:
            return 1.0  # 无明显双峰

        # dip statistic: 最深谷的相对深度
        max_peak = max(density[p] for p in peaks)
        min_valley = min(density[v] for v in valleys)
        dip = min_valley / max(max_peak, 1e-8)

        # Bootstrap 校准
        dip_bootstrap = []
        for _ in range(n_bootstrap):
            boot = np.random.choice(data, n, replace=True)
            b_kde = gaussian_kde(boot)
            b_x = np.linspace(boot.min(), boot.max(), 200)
            b_dens = b_kde(b_x)
            b_peaks, b_valleys = _find_peaks_valleys(b_dens)
            if len(b_peaks) >= 2 and len(b_valleys) >= 1:
                b_max_p = max(b_dens[p] for p in b_peaks)
                b_min_v = min(b_dens[v] for v in b_valleys)
                dip_bootstrap.append(b_min_v / max(b_max_p, 1e-8))
            else:
                dip_bootstrap.append(0.0)

        p_val = sum(1 for d in dip_bootstrap if d < dip) / max(n_bootstrap, 1)
        return float(p_val)
    except Exception:
        return -1.0


def _find_peaks_valleys(arr):
    """找局部峰值和谷值的索引."""
    peaks, valleys = [], []
    for i in range(1, len(arr)-1):
        if arr[i-1] < arr[i] > arr[i+1]:
            peaks.append(i)
        elif arr[i-1] > arr[i] < arr[i+1]:
            valleys.append(i)
    return peaks, valleys


def _compute_auc(pos_curvs, neg_curvs, F_star):
    """以距离 |curv - F_star| 为分数计算 AUC."""
    pos_scores = np.abs(pos_curvs - F_star)
    neg_scores = np.abs(neg_curvs - F_star)

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # 排序计算
    order = np.argsort(all_scores)
    sorted_labels = all_labels[order]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # ROC
    tpr = np.cumsum(sorted_labels) / n_pos
    fpr = np.cumsum(1 - sorted_labels) / n_neg

    # AUC via trapezoidal
    auc = np.trapz(np.concatenate([[0], tpr, [1]]),
                   np.concatenate([[0], fpr, [1]]))
    return float(max(auc, 1-auc))  # 确保 AUC >= 0.5


def _compute_verdicts(metrics: Dict) -> Dict:
    """根据不可豁免判决规则生成判决."""
    verdicts = {}

    # H-attractor: F* 退化 > 80%
    if metrics["f_star_drift"] > 0.8:
        verdicts["H_attractor"] = "REFUTED"
        verdicts["H_attractor_detail"] = f"F* 退化 {metrics['f_star_drift']:.1%} > 80%"
    elif metrics["f_star_drift"] > 0.5:
        verdicts["H_attractor"] = "UNCERTAIN"
        verdicts["H_attractor_detail"] = f"F* 退化 {metrics['f_star_drift']:.1%} 在灰色地带 (50-80%)"
    else:
        verdicts["H_attractor"] = "SUPPORT"
        verdicts["H_attractor_detail"] = f"F* 稳定 (退化 {metrics['f_star_drift']:.1%})"

    # H-push: Cohen's d < 0.2 且 p > 0.05
    if metrics["cohens_d"] < 0.2 and metrics["wilcoxon_p"] > 0.05:
        verdicts["H_push"] = "REFUTED"
        verdicts["H_push_detail"] = f"Cohen's d={metrics['cohens_d']:.3f} < 0.2, p={metrics['wilcoxon_p']:.3f} > 0.05"
    elif 0.2 <= metrics["cohens_d"] < 0.5:
        verdicts["H_push"] = "UNCERTAIN"
        verdicts["H_push_detail"] = f"Cohen's d={metrics['cohens_d']:.3f} 在灰色地带 (0.2-0.5)"
    else:
        verdicts["H_push"] = "SUPPORT"
        verdicts["H_push_detail"] = f"Cohen's d={metrics['cohens_d']:.3f} >= 0.5"

    # 整体: 三问
    snr_ok = metrics["snr"] > 1.5
    wilcoxon_ok = metrics["wilcoxon_p"] < 0.01
    auc_ok = metrics["auc"] > 0.6
    drift_ok = metrics["f_star_drift"] < 0.8

    passed = sum([snr_ok, wilcoxon_ok, auc_ok, drift_ok])
    verdicts["questions_passed"] = passed
    verdicts["overall"] = "SUPPORT" if passed >= 4 else \
                          "PARTIAL" if passed >= 2 else "REFUTED"

    verdicts["Q1_stable_Fstar"] = "PASS" if drift_ok else "FAIL"
    verdicts["Q2_differentiation"] = "PASS" if (snr_ok and wilcoxon_ok) else "FAIL"
    verdicts["Q3_decoupling"] = "PENDING"  # 需要 λ 扫描数据

    return verdicts


# ═══════════════════════════════════════════════════════════════════
# 模块6: 三阶段训练协议
# ═══════════════════════════════════════════════════════════════════

def _storysteps_to_dicts(steps) -> List[dict]:
    """将 List[StoryStep] 转换为 tokenizer.encode_story 所需的 List[dict].

    每个 dict 包含 "state", "action", "causal_labels" 三个键.
    """
    result = []
    for step in steps:
        if hasattr(step, 'state'):
            result.append({
                "state": step.state,
                "action": step.action,
                "causal_labels": step.causal_labels,
            })
        elif isinstance(step, dict):
            result.append(step)
    return result


def _encode_transfer_steps(steps, tokenizer, max_seq_len: int) -> List[int]:
    """将 transfer 的 steps (List[StoryStep]) 编码为 token IDs."""
    step_dicts = _storysteps_to_dicts(steps)
    if len(step_dicts) < 3:
        return []
    tokens = tokenizer.encode_story(step_dicts)  # 返回 List[int]
    return tokens[:max_seq_len] if tokens else []

class Exp8Trainer:
    """exp8 专属训练器, 支持三阶段训练协议."""

    def __init__(self, config: dict, logger, device: str = "cpu"):
        self.config = config
        self.logger = logger
        # 🔧 P0-设备修复: 自动检测CUDA, 与Trainer保持一致
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda"
        self.device = torch.device(device)
        self.d_model = config["model"]["d_model"]
        # 🔧 P0-维度修复: 模型返回的hidden可能是base_dim而非d_model,
        #     SubspaceProjection必须使用实际hidden维度
        self.base_dim = config["model"].get("base_dim", None)
        self.hidden_dim = self.base_dim if self.base_dim else self.d_model
        self.r = min(16, self.hidden_dim // 2)
        self._nb_encoding_errs = 0  # 统计编码/推理异常次数
        logger.info(f"Exp8Trainer init: d_model={self.d_model}, base_dim={self.base_dim}, "
                    f"hidden_dim={self.hidden_dim}, r={self.r} (🔥 维度修复: 投影矩阵用hidden_dim而非d_model)")

    def train_regime(self, tag: str, regime_type: str, data_dict: Dict,
                     out_dir: Path, **kwargs) -> Dict:
        """训练一个体制并返回全部指标.

        regime_type: "A" | "B" | "C"
        """
        self.logger.info(f"{'='*60}")
        self.logger.info(f"体制 {regime_type} ({tag}): 开始训练")
        self.logger.info(f"{'='*60}")

        # 初始化模型
        causal_model = CausalTransformer(self.config).to(self.device)
        gauge_field = GaugeField(self.hidden_dim)  # 🔧 用实际hidden维度
        projector = SubspaceProjection(self.hidden_dim, r=self.r).to(self.device)

        # 构建数据加载器
        tokenizer = NPNWTokenizer()
        self.config["model"]["vocab_size"] = tokenizer.vocab_size

        train_loader = self._make_loader(data_dict.get("train_pos", []),
                                         data_dict.get("train_neg", []), True, tokenizer)
        val_loader = self._make_loader(data_dict.get("val_pos", []),
                                       data_dict.get("val_neg", []), False, tokenizer)

        # 创建 Trainer
        trainer = Trainer(self.config, causal_model, None, gauge_field)

        # --- Phase 1: 冻结模型下的吸引子估计 (仅体制C) ---
        phase1_results = {}
        if regime_type == "C":
            phase1_results = self._phase1_estimate_attractor(
                causal_model, projector, data_dict, tokenizer
            )
            self.logger.info(f"Phase 1 完成: F_flex*={phase1_results['F_flex_star']:.6f}, "
                           f"m={phase1_results['m']:.6f}, "
                           f"Shapiro-Wilk p={phase1_results.get('shapiro_p', 'N/A')}")

        # --- Phase 2: 联合训练 ---
        if regime_type == "A":
            # 纯 LM
            trainer.train_full(train_loader, val_loader,
                             lambda_value=0.0, closure_contrastive_lambda=0.0)
        elif regime_type == "B":
            # 公式(R): 阈值惩罚
            rf_cfg = self.config.get("rigid_flexible", {})
            trainer.train_full(train_loader, val_loader,
                             layered_rf_lambda=rf_cfg.get("layered_rf_lambda", 1.0),
                             rf_tau=rf_cfg.get("tau_narr", 0.5),
                             rf_lambda_phys=rf_cfg.get("lambda_phys", 1.0),
                             rf_lambda_flex=rf_cfg.get("lambda_flex", 1.0),
                             rf_margin=rf_cfg.get("margin", 0.3))
        else:  # regime_type == "C"
            self._phase2_joint_training(
                causal_model, projector, data_dict, tokenizer,
                train_loader, val_loader, phase1_results, **kwargs
            )

        # --- Phase 2 后的评估 ---
        eval_metrics = self._evaluate_after_training(
            causal_model, projector, data_dict, tokenizer,
            regime_type, phase1_results
        )

        # --- Phase 3: λ 扫描 (仅体制C) ---
        phase3_results = {}
        if regime_type == "C" and kwargs.get("run_phase3", True):
            phase3_results = self._phase3_lambda_scan(
                data_dict, tokenizer, phase1_results, projector, **kwargs
            )

        # 汇总
        return {
            "regime": regime_type,
            "tag": tag,
            "phase1": phase1_results,
            "evaluation": eval_metrics,
            "phase3": phase3_results,
        }

    def _make_loader(self, pos, neg, shuffle, tokenizer):
        ds = StoryDataset(pos + neg, tokenizer, self.config["model"]["max_seq_len"])
        return DataLoader(ds, batch_size=self.config["training"]["batch_size"], shuffle=shuffle)

    def _phase1_estimate_attractor(
        self, model: CausalTransformer, projector: SubspaceProjection,
        data_dict: Dict, tokenizer: NPNWTokenizer
    ) -> Dict:
        """Phase 1: 冻结模型, 遍历 flex_pos 转移估计 F_flex* 和 m."""
        model.eval()
        projector.eval()

        flex_pos_transfers = data_dict.get("flex_pos", [])
        curv_values = []

        max_seq_len = self.config["model"]["max_seq_len"]

        with torch.no_grad():
            for transfer in flex_pos_transfers:
                steps = transfer.get("steps", [])
                if len(steps) < 3:
                    continue

                # Tokenize 并编码
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    if len(tokens) < 3:
                        continue
                    tokens = tokens[:max_seq_len]
                    model_device = next(model.parameters()).device
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                except Exception as _e:
                    if not hasattr(self, '_p1_errs'):
                        self._p1_errs = []
                    if len(self._p1_errs) < 3:
                        self._p1_errs.append(f"{type(_e).__name__}")
                    continue

                # 计算柔性曲率
                curv = estimate_discrete_curvature(hidden, projector, mode="flex")
                curv_values.append(curv.item())

        if len(curv_values) < 5:
            err_info = ""
            if hasattr(self, '_p1_errs') and self._p1_errs:
                err_info = f" 错误类型: {', '.join(set(self._p1_errs))}"
                self._p1_errs = []
            self.logger.warning(f"flex_pos 转移不足 ({len(curv_values)}/300 成功), 使用默认值 F*=0.01, m=0.1.{err_info}")
            return {"F_flex_star": 0.01, "m": 0.1, "sigma": 0.01, "n_samples": len(curv_values)}

        curv_array = np.array(curv_values)
        F_star = float(np.mean(curv_array))
        sigma = float(np.std(curv_array))

        # P0-4 修复: Shapiro-Wilk 正态性检验 + 双 m 估计
        shapiro_p = -1.0
        m_param = F_star + 2 * sigma
        m_perc = float(np.percentile(curv_array, 95)) if len(curv_array) >= 10 else m_param

        if HAS_SCIPY:
            try:
                _, shapiro_p = scipy_stats.shapiro(curv_array)
            except Exception:
                pass

        results = {
            "F_flex_star": F_star,
            "sigma": sigma,
            "m": m_param,  # 默认使用参数法
            "m_percentile": m_perc,
            "shapiro_p": float(shapiro_p) if shapiro_p > 0 else -1.0,
            "n_samples": len(curv_array),
            "skewness": float(_safe_skewness(curv_array)),
            "kurtosis": float(_safe_kurtosis(curv_array)),
        }

        # 若非正态, 标记需交叉验证
        if shapiro_p > 0 and shapiro_p <= 0.05:
            results["normality"] = "REJECTED"
            results["note"] = "分布非正态, 需在报告中交叉验证 m_param 和 m_percentile"
        else:
            results["normality"] = "NORMAL" if shapiro_p > 0.05 else "UNKNOWN"

        self.logger.info(f"F_flex* = {F_star:.6f}, σ = {sigma:.6f}, "
                        f"m_param = {m_param:.6f}, m_perc = {m_perc:.6f}, "
                        f"Shapiro-Wilk p = {shapiro_p}")

        return results

    def _phase2_joint_training(
        self, model, projector, data_dict, tokenizer,
        train_loader, val_loader, phase1_results, **kwargs
    ):
        """Phase 2: 公式(A) 联合训练."""
        # 简化实现: 复用 Trainer 的基础训练, 然后用自定义几何损失做微调
        trainer = Trainer(self.config, model, None, GaugeField(self.hidden_dim))

        # 先做基础 LM 训练
        trainer.train_full(train_loader, val_loader,
                         lambda_value=0.0, closure_contrastive_lambda=0.0,
                         )  # 短训练

        # 然后用公式(A)做几何微调
        self._fine_tune_with_formula_a(
            model, projector, data_dict, tokenizer,
            trainer, phase1_results, **kwargs
        )

    def _fine_tune_with_formula_a(
        self, model, projector, data_dict, tokenizer,
        trainer, phase1_results, n_steps=200, lr=1e-4, **kwargs
    ):
        """用公式(A)微调模型参数和投影矩阵."""
        F_star = phase1_results["F_flex_star"]
        m_val = phase1_results["m"]
        lambda_phys = kwargs.get("lambda_phys", 1.0)
        lambda_flex_pos = kwargs.get("lambda_flex_pos", 0.5)
        lambda_flex_neg = kwargs.get("lambda_flex_neg", 0.5)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(projector.parameters()), lr=lr
        )

        phys_data = data_dict.get("phys", [])
        flex_pos_data = data_dict.get("flex_pos", [])
        flex_neg_data = data_dict.get("flex_neg", [])
        max_seq_len = self.config["model"]["max_seq_len"]

        model.train()
        projector.train()

        batch_size = min(16, len(phys_data), len(flex_pos_data))
        grad_cos_log = []
        phys_curv_log = []
        flex_curv_log = []
        p_errs = fp_errs = fn_errs = 0  # 异常计数
        model_device = next(model.parameters()).device

        for step in range(n_steps):
            # 随机采样
            p_batch = np.random.choice(len(phys_data), min(batch_size, len(phys_data)), replace=False)
            fp_batch = np.random.choice(len(flex_pos_data), min(batch_size, len(flex_pos_data)), replace=False)
            fn_batch = np.random.choice(len(flex_neg_data), min(batch_size, len(flex_neg_data)), replace=False)

            phys_curvs = []
            flex_pos_curvs = []
            flex_neg_curvs = []

            # 物理曲率
            for idx in p_batch:
                t = phys_data[idx]
                steps = t.get("steps", [])
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "phys")
                    phys_curvs.append(c)
                except Exception:
                    p_errs += 1
                    continue

            # 柔性正例曲率
            for idx in fp_batch:
                t = flex_pos_data[idx]
                steps = t.get("steps", [])
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "flex")
                    flex_pos_curvs.append(c)
                except Exception:
                    fp_errs += 1
                    continue

            # 柔性负例曲率
            for idx in fn_batch:
                t = flex_neg_data[idx]
                steps = t.get("steps", [])
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "flex")
                    flex_neg_curvs.append(c)
                except Exception:
                    fn_errs += 1
                    continue

            if not phys_curvs or not flex_pos_curvs:
                continue

            p_curv = torch.stack(phys_curvs)
            fp_curv = torch.stack(flex_pos_curvs)
            fn_curv = torch.stack(flex_neg_curvs) if flex_neg_curvs else torch.tensor([], device=self.device)

            # 计算公式(A)损失
            loss_dict = formula_a_loss(
                p_curv, fp_curv, fn_curv,
                F_star, m_val,
                lambda_phys, lambda_flex_pos, lambda_flex_neg
            )

            # 正交辅助损失
            ortho_loss = 0.01 * projector.orthogonality_loss()
            total_loss = loss_dict["total"] + ortho_loss

            optimizer.zero_grad()
            total_loss.backward()

            # 记录梯度余弦相似度 (预测8)
            if step > n_steps * 0.9 and len(fp_curv) > 0 and len(fn_curv) > 0:
                # 正例梯度: d||F-F*||²/dF = 2(F-F*), 负例梯度: d max(0,m-|F-F*|²)/dF
                # 简化为记录损失对曲率的梯度方向
                fp_grad = 2 * (fp_curv.detach() - F_star)  # 近似
                fn_dist = (fn_curv.detach() - F_star).abs()
                fn_active = fn_dist < m_val
                if fn_active.any():
                    fn_grad = -2 * (fn_curv.detach() - F_star) * fn_active.float()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        fp_grad.mean(dim=0, keepdim=True),
                        fn_grad.mean(dim=0, keepdim=True),
                        dim=0
                    ).item()
                    grad_cos_log.append(cos_sim)

            optimizer.step()

            phys_curv_log.append(float(p_curv.mean()))
            flex_curv_log.append(float(fp_curv.mean()))

            if step % 50 == 0 or step == n_steps - 1:
                self.logger.info(f"  Step {step}/{n_steps}: "
                               f"total={total_loss.item():.6f}, "
                               f"phys={loss_dict['phys'].item():.6f}, "
                               f"flex_pos={loss_dict['flex_pos'].item():.6f}, "
                               f"flex_neg={loss_dict['flex_neg'].item():.6f}")

        # 存储训练记录
        self._train_log = {
            "grad_cos_sim": grad_cos_log,
            "phys_curv_history": phys_curv_log,
            "flex_curv_history": flex_curv_log,
            "grad_cos_sim_mean": float(np.mean(grad_cos_log)) if grad_cos_log else 0.0,
        }

        if p_errs or fp_errs or fn_errs:
            self.logger.info(f"  Phase 2 微调异常统计: phys={p_errs}, flex_pos={fp_errs}, flex_neg={fn_errs}")

    def _evaluate_after_training(
        self, model, projector, data_dict, tokenizer,
        regime_type: str, phase1_results: Dict
    ) -> Dict:
        """Phase 2 后的全面评估."""
        model.eval()
        projector.eval()

        flex_pos_data = data_dict.get("flex_pos", [])
        flex_neg_data = data_dict.get("flex_neg", [])
        phys_data = data_dict.get("phys", [])
        max_seq_len = self.config["model"]["max_seq_len"]

        flex_pos_curvs = []
        flex_neg_curvs = []
        phys_curvs = []
        eval_errs = 0
        model_device = next(model.parameters()).device

        with torch.no_grad():
            # 柔性正例
            for t in flex_pos_data:
                steps = t.get("steps", [])
                if len(steps) < 3:
                    continue
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "flex")
                    flex_pos_curvs.append(c.item())
                except Exception:
                    eval_errs += 1
                    continue

            # 柔性负例
            for t in flex_neg_data:
                steps = t.get("steps", [])
                if len(steps) < 3:
                    continue
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "flex")
                    flex_neg_curvs.append(c.item())
                except Exception:
                    eval_errs += 1
                    continue

            # 物理
            for t in phys_data:
                steps = t.get("steps", [])
                if len(steps) < 3:
                    continue
                try:
                    tokens = _encode_transfer_steps(steps, tokenizer, max_seq_len)
                    input_ids = torch.tensor([tokens], device=model_device)
                    _, hidden = model(input_ids)
                    c = estimate_discrete_curvature(hidden, projector, "phys")
                    phys_curvs.append(c.item())
                except Exception:
                    eval_errs += 1
                    continue

        fp_arr = np.array(flex_pos_curvs) if flex_pos_curvs else np.array([0.0])
        fn_arr = np.array(flex_neg_curvs) if flex_neg_curvs else np.array([0.0])
        pp_arr = np.array(phys_curvs) if phys_curvs else np.array([0.0])

        F_flex_star_initial = phase1_results.get("F_flex_star", 0.01)
        F_flex_star_final = float(np.mean(fp_arr)) if len(fp_arr) > 0 else F_flex_star_initial

        metrics = compute_evaluation_metrics(
            fp_arr, fn_arr, pp_arr,
            F_flex_star_initial, F_flex_star_final
        )

        # 附加训练日志中的梯度信息
        if hasattr(self, '_train_log') and regime_type == "C":
            metrics["grad_cos_sim_mean"] = self._train_log.get("grad_cos_sim_mean", 0.0)
            metrics["grad_cos_sim_log"] = self._train_log.get("grad_cos_sim", [])

        metrics["n_pos_samples"] = len(fp_arr)
        metrics["n_neg_samples"] = len(fn_arr)
        metrics["n_phys_samples"] = len(pp_arr)
        metrics["eval_errors"] = eval_errs

        if eval_errs > 0:
            self.logger.warning(f"  评估阶段 {eval_errs} 个样本编码/推理异常 (已跳过)")

        return metrics

    def _phase3_lambda_scan(
        self, data_dict, tokenizer, phase1_results, projector,
        n_repeats=3, **kwargs
    ) -> Dict:
        """Phase 3: λ 扫描与稳定性分析."""
        self.logger.info("Phase 3: λ 扫描开始...")

        lambda_phys_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        lambda_flex_pos_values = [0.1, 0.5, 1.0, 2.0]
        scan_results = []

        for lp in lambda_phys_values:
            for lfp in lambda_flex_pos_values:
                for _ in range(n_repeats):
                    model = CausalTransformer(self.config).to(self.device)
                    proj = SubspaceProjection(self.hidden_dim, r=self.r).to(self.device)

                    # 快速训练
                    self._fine_tune_with_formula_a(
                        model, proj, data_dict, tokenizer,
                        None, phase1_results, n_steps=100, lr=1e-4,
                        lambda_phys=lp, lambda_flex_pos=lfp,
                        lambda_flex_neg=kwargs.get("lambda_flex_neg", 0.5),
                    )

                    # 评估
                    metrics = self._evaluate_after_training(
                        model, proj, data_dict, tokenizer, "C", phase1_results
                    )
                    scan_results.append({
                        "lambda_phys": lp,
                        "lambda_flex_pos": lfp,
                        "F_flex_star_final": metrics["F_flex_star_final"],
                        "snr": metrics["snr"],
                        "cohens_d": metrics["cohens_d"],
                    })

        # P1-3 修复: 样本量消融
        ablation_results = self._sample_size_ablation(
            data_dict, tokenizer, phase1_results, projector
        )

        return {
            "lambda_scan": scan_results,
            "sample_ablation": ablation_results,
            "n_configs": len(scan_results),
        }

    def _sample_size_ablation(
        self, data_dict, tokenizer, phase1_results, projector,
        sizes=[10, 50, 100, None]
    ) -> List[Dict]:
        """P1-3 修复: 样本量消融 — 测试 F* 对不同 N+ 的稳定性."""
        self.logger.info(" 样本量消融: N+ = [10, 50, 100, 全量]...")
        results = []

        flex_pos_data = data_dict.get("flex_pos", [])
        total_n = len(flex_pos_data)

        for size in sizes:
            n = size if size is not None else total_n
            n = min(n, total_n)
            if n < 10:
                continue

            # 随机子采样
            indices = np.random.choice(total_n, n, replace=False)
            subset_data = {k: v for k, v in data_dict.items()}
            subset_data["flex_pos"] = [flex_pos_data[i] for i in indices]

            # 重新估计 F*
            model = CausalTransformer(self.config).to(self.device)
            proj = SubspaceProjection(self.hidden_dim, r=self.r).to(self.device)
            est = self._phase1_estimate_attractor(model, proj, subset_data, tokenizer)

            results.append({
                "N_pos": n,
                "F_flex_star": est["F_flex_star"],
                "sigma": est["sigma"],
                "m": est["m"],
                "m_percentile": est.get("m_percentile", est["m"]),
            })

        return results


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _safe_skewness(arr):
    try:
        n = len(arr)
        m = np.mean(arr)
        sd = np.std(arr)
        if sd < 1e-8:
            return 0.0
        return float(np.sum((arr - m) ** 3) / (n * sd ** 3))
    except Exception:
        return 0.0


def _safe_kurtosis(arr):
    try:
        n = len(arr)
        m = np.mean(arr)
        sd = np.std(arr)
        if sd < 1e-8:
            return 0.0
        return float(np.sum((arr - m) ** 4) / (n * sd ** 4) - 3)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════

def main():
    config = load_config()

    # ---- 实验配置 (exp8 专用) ----
    config["data"]["num_stories"] = 300
    config["data"]["enh_min_steps"] = 6
    config["data"]["enh_max_steps"] = 9
    config["npnw"]["max_stamina"] = 20
    config["model"]["max_seq_len"] = 160
    config["training"]["max_epochs"] = 8
    config["training"]["patience"] = 4
    config["training"]["batch_size"] = 16
    config["experiment4"]["significance_level"] = 0.05

    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(__file__).parent
    logger = setup_logger("Exp8", str(out_dir / "logs"), "exp8.log")

    logger.info("=" * 70)
    logger.info("exp8: 吸引子定向分化损失 公式(A) 三体制受控检验")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info(f"配置: {config['data']['num_stories']} 故事, "
                f"max_seq_len={config['model']['max_seq_len']}, "
                f"epochs={config['training']['max_epochs']}")
    logger.info(f"scipy 可用: {HAS_SCIPY}")
    logger.info("=" * 70)

    # ---- 数据生成 ----
    logger.info("生成增强闭合数据集 + 灵活标签...")
    generator = EnhancedClosureGenerator(config, seed=seed)
    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    label_gen = FlexibleLabelGenerator(generator, tokenizer, config, seed=seed)
    flex_data = label_gen.generate(num_stories=config["data"]["num_stories"])

    logger.info(f"数据统计: phys={flex_data['counts']['phys']}, "
                f"flex_pos={flex_data['counts']['flex_pos']}, "
                f"flex_neg={flex_data['counts']['flex_neg']}")

    # 构造完整数据集
    full_data = {
        "train_pos": flex_data["train_pos"],
        "train_neg": flex_data["train_neg"],
        "val_pos": flex_data["val_pos"],
        "val_neg": flex_data["val_neg"],
        "test_pos": flex_data["test_pos"],
        "test_neg": flex_data["test_neg"],
        "phys": flex_data["phys"],
        "flex_pos": flex_data["flex_pos"],
        "flex_neg": flex_data["flex_neg"],
    }

    # ---- 训练三体制 ----
    exp8_trainer = Exp8Trainer(config, logger)

    results = {}

    # 体制A: Baseline (仅LM)
    logger.info("\n" + "=" * 70)
    logger.info("体制 A: Baseline (纯LM)")
    logger.info("=" * 70)
    results["A_baseline"] = exp8_trainer.train_regime(
        "A_baseline_LM", "A", full_data, out_dir
    )

    # 体制B: 旧阈值公式(R)
    logger.info("\n" + "=" * 70)
    logger.info("体制 B: 旧阈值惩罚 公式(R)")
    logger.info("=" * 70)
    results["B_threshold"] = exp8_trainer.train_regime(
        "B_threshold_R", "B", full_data, out_dir
    )

    # 体制C: 新吸引子公式(A)
    logger.info("\n" + "=" * 70)
    logger.info("体制 C: 吸引子定向分化 公式(A)")
    logger.info("=" * 70)
    results["C_attractor"] = exp8_trainer.train_regime(
        "C_attractor_A", "C", full_data, out_dir,
        run_phase3=True,
        lambda_phys=1.0,
        lambda_flex_pos=0.5,
        lambda_flex_neg=0.5,
    )

    # ---- 生成报告 ----
    _generate_report(results, flex_data, out_dir, logger)

    logger.info("exp8 完成.")


def _generate_report(results, flex_data, out_dir, logger):
    """生成 JSON 和 Markdown 报告."""

    # JSON 报告
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_stories": 300,
            "max_seq_len": 160,
            "flex_label_counts": flex_data["counts"],
        },
        "regimes": {},
    }

    for key, res in results.items():
        regime_data = {"regime": res["regime"], "tag": res["tag"]}

        if res.get("phase1"):
            p1 = res["phase1"]
            regime_data["phase1"] = {
                "F_flex_star": p1.get("F_flex_star"),
                "m": p1.get("m"),
                "shapiro_p": p1.get("shapiro_p"),
                "normality": p1.get("normality"),
                "n_samples": p1.get("n_samples"),
            }

        if res.get("evaluation"):
            ev = res["evaluation"]
            regime_data["evaluation"] = {
                k: v for k, v in ev.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }

        if res.get("phase3"):
            p3 = res["phase3"]
            regime_data["phase3"] = {
                "n_configs": p3.get("n_configs"),
                "sample_ablation": p3.get("sample_ablation"),
            }

        json_report["regimes"][key] = regime_data

    with open(out_dir / "exp8_report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2, default=str)

    # Markdown 报告
    md = ["# 实验8: 吸引子定向分化损失 公式(A) 三体制受控检验",
          f"\n时间: {json_report['timestamp']}",
          "",
          "## 实验设计",
          f"- 数据: {json_report['config']['num_stories']} 故事, "
          f"max_seq_len={json_report['config']['max_seq_len']}",
          f"- 灵活标签: phys={flex_data['counts']['phys']}, "
          f"flex_pos={flex_data['counts']['flex_pos']}, "
          f"flex_neg={flex_data['counts']['flex_neg']}",
          "- 三体制: A (纯LM) / B (公式R-阈值惩罚) / C (公式A-吸引子分化)",
          "- 三阶段: Phase1(冻结估计F*) / Phase2(联合训练) / Phase3(λ扫描+消融)",
          "",
          "## 结果",
          ""]

    for key, res in results.items():
        md.append(f"### 体制 {res['regime']}: {res['tag']}")
        md.append("")

        if res.get("phase1"):
            p1 = res["phase1"]
            md.append("#### Phase 1: 吸引子估计")
            md.append(f"- F_flex* = {p1.get('F_flex_star', 'N/A'):.6f}" if isinstance(p1.get('F_flex_star'), float) else f"- F_flex* = {p1.get('F_flex_star', 'N/A')}")
            md.append(f"- m = {p1.get('m', 'N/A')}")
            md.append(f"- Shapiro-Wilk p = {p1.get('shapiro_p', 'N/A')}")
            md.append(f"- 正态性: {p1.get('normality', 'N/A')}")
            md.append(f"- 偏度: {p1.get('skewness', 'N/A')}, 峰度: {p1.get('kurtosis', 'N/A')}")
            md.append("")

        if res.get("evaluation"):
            ev = res["evaluation"]
            md.append("#### 评估指标")
            md.append(f"| 指标 | 值 | 阈值 | 判定 |")
            md.append(f"|---|---|---|---|")

            # SNR
            snr_val = ev.get("snr", 0)
            md.append(f"| SNR | {snr_val:.3f} | > 1.5 | {'✅' if snr_val > 1.5 else '❌'} |")

            # Cohen's d
            d_val = ev.get("cohens_d", 0)
            d_verdict = "✅" if d_val >= 0.5 else ("⚠️" if d_val >= 0.2 else "❌")
            md.append(f"| Cohen's d | {d_val:.3f} | ≥ 0.5 | {d_verdict} |")

            # Wilcoxon
            p_val = ev.get("wilcoxon_p", 1)
            md.append(f"| Wilcoxon p | {p_val:.4f} | < 0.01 | {'✅' if p_val < 0.01 else '❌'} |")

            # AUC
            auc_val = ev.get("auc", 0.5)
            md.append(f"| AUC | {auc_val:.3f} | > 0.6 | {'✅' if auc_val > 0.6 else '❌'} |")

            # F* 漂移
            drift_val = ev.get("f_star_drift", 0)
            drift_verdict = "✅" if drift_val < 0.5 else ("⚠️" if drift_val < 0.8 else "❌")
            md.append(f"| F* 漂移率 | {drift_val:.1%} | < 50% | {drift_verdict} |")

            # 曲率均值
            md.append(f"| F_flex* 初始/最终 | {ev.get('F_flex_star_initial', 0):.6f} / {ev.get('F_flex_star_final', 0):.6f} | — | — |")
            md.append(f"| μ_pos / μ_neg | {ev.get('mu_pos', 0):.6f} / {ev.get('mu_neg', 0):.6f} | — | — |")
            md.append(f"| 物理曲率 μ | {ev.get('phys_curv_mean', 0):.6f} | ≈ 0 | — |")
            md.append("")

            # 判决
            md.append("#### 判决性结果")
            md.append(f"| 问题 | 结果 |")
            md.append(f"|---|---|")
            md.append(f"| Q1: F* 稳定? | {ev.get('Q1_stable_Fstar', 'N/A')} |")
            md.append(f"| Q2: 合法/非法分化? | {ev.get('Q2_differentiation', 'N/A')} |")
            md.append(f"| Q3: 训练解耦? | {ev.get('Q3_decoupling', 'N/A')} |")
            md.append(f"| H-attractor | {ev.get('H_attractor', 'N/A')} |")
            md.append(f"| H-push | {ev.get('H_push', 'N/A')} |")
            md.append("")

            # P1-4 修复: 中等漂移的灰色地带处理
            drift_val = ev.get("f_star_drift", 0)
            if 0.2 <= drift_val < 0.8:
                md.append(f"> ⚠️ **P1-4 灰色地带判决**: F* 漂移率 {drift_val:.1%} 处于中等区间 (20-80%), "
                          f"判决为「不确定——需增补实验诊断」。建议扩大样本量或增加训练轮次后重验。")
                md.append("")

        if res.get("phase3") and res["phase3"].get("sample_ablation"):
            md.append("#### P1-3: 样本量消融")
            md.append(f"| N_pos | F_flex* | σ | m_param | m_perc |")
            md.append(f"|---|---|---|---|---|")
            for ab in res["phase3"]["sample_ablation"]:
                md.append(f"| {ab['N_pos']} | {ab['F_flex_star']:.6f} | {ab['sigma']:.6f} | "
                         f"{ab['m']:.6f} | {ab.get('m_percentile', ab['m']):.6f} |")
            md.append("")

    # 交叉体制对比
    md.append("## 体制间对比")
    md.append("")

    c_eval = results.get("C_attractor", {}).get("evaluation", {})
    b_eval = results.get("B_threshold", {}).get("evaluation", {})

    if c_eval and b_eval:
        c_fstar = c_eval.get("F_flex_star_final", 0)
        b_fstar = b_eval.get("F_flex_star_final", 0)
        c_snr = c_eval.get("snr", 0)
        b_snr = b_eval.get("snr", 0)

        md.append(f"| 对比维度 | 体制B (公式R) | 体制C (公式A) | 差异 |")
        md.append(f"|---|---|---|---|")
        md.append(f"| F* 最终值 | {b_fstar:.6f} | {c_fstar:.6f} | "
                 f"{'体制C更高✅' if c_fstar > b_fstar else '体制C未改善❌'} |")
        md.append(f"| SNR | {b_snr:.3f} | {c_snr:.3f} | "
                 f"{'体制C更高✅' if c_snr > b_snr else '体制C未改善❌'} |")
        md.append("")

        # 核心判决: 体制C > 体制B?
        if c_snr > b_snr * 1.5 and c_fstar > b_fstar:
            md.append("> ✅ **核心判决: 体制C优于体制B** — 公式(A)的吸引子锚定修复有效, "
                      "δ_F* > 0 + SNR 提升。")
        elif c_snr <= b_snr and c_fstar <= b_fstar:
            md.append("> ❌ **核心判决: 公式(A)无效** — 四重重构未对损失景观产生结构性改善, "
                      "δ_F* = 0 且 SNR 未提升。")
        else:
            md.append("> ⚠️ **灰色判决: 体制C部分优于体制B** — 需要进一步诊断原因。")

    # P1-1 修复: exp9 留待声明
    md.append("")
    md.append("## P1-1: 后续实验规划")
    md.append("- **exp9 (跨MVE推广)**: 本实验仅在一个微型叙事世界上验证。")
    md.append("  exp9 将在不同的世界规则、不同tokenizer、不同Transformer架构下复现体制C的吸引子分化, ")
    md.append("  检验理论是否具备跨场景的普适性。")

    # 论文预注册信息
    md.append("")
    md.append("## 预注册: exp8 完成后理论状态更新路径")
    md.append("")
    md.append("| 情景 | 三问状态 | 固化度 | 理论状态 | 下一步 |")
    md.append("|---|---|---|---|---|")
    md.append("| S++ (全通过) | Q1✅ Q2✅ Q3✅ | 0.54 → 0.64 | 理论获强力支持, 进入实证积累阶段 | "
              "撰写 exp8 验证报告, 准备 exp9 跨 MVE 推广 |")
    md.append("| S+ (部分通过) | 2/3 通过 | 0.54 → 0.58 | 核心机制成立, 部分局限需标注 | "
              "诊断失败项, 修订理论边界 |")
    md.append("| S (弱分化) | 仅Q1通过 | 0.54 → 0.50 | H-attractor成立, H-push待重新设计 | "
              "回溯负例推离机制, 考虑替代方案 |")
    md.append("| S- (全面失败) | 0/3 通过 | 0.54 → 0.40 | 刚柔二分被证伪, 退化为局部成功 | "
              "H-rigid (物理层F*=0) 作为唯一实证收获保留 |")
    md.append("| S-- (体制C≤B) | δ_F*=0 | 0.54 → 0.35 | 四重重构无效, 需第五次重构 | "
              "归因: 是Formula(A)结构问题还是MVE规模问题? |")
    md.append("")

    with open(out_dir / "exp8_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    logger.info(f"报告已保存: {out_dir / 'exp8_report.md'}")
    logger.info(f"JSON已保存: {out_dir / 'exp8_report.json'}")


if __name__ == "__main__":
    main()
