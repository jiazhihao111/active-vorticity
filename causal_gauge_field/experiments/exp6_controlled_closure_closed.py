import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple

from ..models.transformer import CausalTransformer
from ..models.gauge_field import GaugeField
from ..npnw.story_generator import Story
from ..npnw.tokenizer import NPNWTokenizer
from ..utils.logger import setup_logger
from ..utils.metrics import discrete_curvature


class Experiment6:
    """受控检验(修正版): 闭合 holonomy 平坦度 正例(闭环) vs 负例(破缺).

    实验5 证明 exp4 的 SUPPORT 是训练目标的循环假象——但实验5 仍用【开环】
    Wilson 与 κ_angle, 二者都量的是「整段轨迹的全局压缩」, 从未真正测过「闭合」.
    本实验改用真正闭合的 holonomy (gauge_field.wilson_loop_closed) 作为主判据:

      - 闭合 holonomy W(γ): 沿 h_0..h_T 乘积后, 再补闭合边 h_T -> h_0.
        闭环叙事使 h_T 回到与 h_0 一致的框架 ⇒ W≈I(平坦);
        破缺叙事 h_T 与 h_0 不一致 ⇒ W≠I(不平坦).
      - 主判据: holonomy_flatness = ‖W - I‖_F  (闭环应更小);
                其次 closed_wilson_trace 与 db 的偏差 (闭环应更小).
      - 设计: 负例由正例派生(story_id 相同), 做【配对】 Wilcoxon (破缺-闭环),
              H1: 破缺更不平坦 (flatness_负 > flatness_正, 即差>0).
      - 对照: 同时报告开环 κ_angle 与开环 Wilson Var (实验5 指标), 用于对比
              「全局压缩」与「闭合敏感」两种判据的差异.

    这能区分两种结论:
      (a) 理论错 —— 即便闭合后仍无差异 ⇒ C-11 确证为假, 应退役/降级为未验证隐喻;
      (b) 旧实现假象 —— 闭合后出现 闭环<破缺 的差异 ⇒ 理论可挽救, 需进一步加闭合
          敏感训练信号以稳固结论.
    """

    def __init__(self, config: dict, gauge_field: GaugeField):
        self.config = config
        self.logger = setup_logger("Experiment6")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gauge_field = gauge_field
        self.db = config["model"]["base_dim"]

    def _truncate(self, token_ids, offset=1):
        max_len = self.config["model"]["max_seq_len"] - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def _per_story(self, model: CausalTransformer, story: Story) -> Optional[Dict[str, float]]:
        model.eval()
        steps_data = [{"state": s.state, "action": s.action, "causal_labels": s.causal_labels}
                      for s in story.steps]
        token_ids = self.tokenizer.encode_story(steps_data)
        token_ids = self._truncate(token_ids)
        if len(token_ids) < 4:
            return None
        input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            _, hidden = model(input_ids)                          # (1, T, base_dim)
            hidden = hidden.to(self.device)
            hidden_np = hidden[0, :, :].cpu().numpy()
            self.gauge_field.eval()
            flat = self.gauge_field.holonomy_flatness(hidden).item()
            tr = self.gauge_field.closed_wilson_trace(hidden).item()
        kappa_mean, _ = discrete_curvature(hidden_np)
        segs = np.diff(hidden_np, axis=0)
        step_norm = float(np.mean(np.linalg.norm(segs, axis=1))) if len(segs) else 0.0
        rec = {
            "kappa_angle": kappa_mean,
            "step_norm": step_norm,
            "closed_flatness": float(flat),
            "closed_trace_dev": float(abs(tr - self.db)),
        }
        return rec

    def run(
        self,
        model: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        model_label: str = "causal",
    ) -> Dict:
        self.logger.info(f"=== 实验6(闭合 holonomy 受控): {model_label} 模型 正例vs负例 ===")
        pos_recs = [r for r in (self._per_story(model, s) for s in test_pos) if r]
        neg_recs = [r for r in (self._per_story(model, s) for s in test_neg) if r]
        if not pos_recs or not neg_recs:
            return {"verdict": "INCONCLUSIVE", "reason": "样本不足"}

        pos_f = np.array([r["closed_flatness"] for r in pos_recs])
        neg_f = np.array([r["closed_flatness"] for r in neg_recs])
        pos_t = np.array([r["closed_trace_dev"] for r in pos_recs])
        neg_t = np.array([r["closed_trace_dev"] for r in neg_recs])
        pos_k = np.array([r["kappa_angle"] for r in pos_recs])
        neg_k = np.array([r["kappa_angle"] for r in neg_recs])

        # ---- 配对比较 (负例派生自正例, story_id 相同) ----
        pos_by_id = {s.story_id: r for s, r in zip(test_pos, pos_recs)}
        paired = [(pos_by_id[s.story_id], r) for s, r in zip(test_neg, neg_recs)
                  if s.story_id in pos_by_id]
        paired_info: Dict = {"n_pairs": len(paired)}
        if paired:
            df = np.array([nr["closed_flatness"] - pr["closed_flatness"] for pr, nr in paired])
            dt = np.array([nr["closed_trace_dev"] - pr["closed_trace_dev"] for pr, nr in paired])
            dk = np.array([nr["kappa_angle"] - pr["kappa_angle"] for pr, nr in paired])
            try:
                wf_stat, wf_p = stats.wilcoxon(df, alternative="greater")
            except ValueError:
                wf_stat, wf_p = 0.0, 1.0
            try:
                wt_stat, wt_p = stats.wilcoxon(dt, alternative="greater")
            except ValueError:
                wt_stat, wt_p = 0.0, 1.0
            try:
                wk_stat, wk_p = stats.wilcoxon(dk, alternative="greater")
            except ValueError:
                wk_stat, wk_p = 0.0, 1.0
            paired_info.update({
                "flatness_diff_median": float(np.median(df)),
                "flatness_diff_mean": float(np.mean(df)),
                "frac_neg_flatter": float(np.mean(df > 0)),     # 破缺更不平坦的比例
                "wilcoxon_flatness_p": float(wf_p),
                "trace_dev_diff_median": float(np.median(dt)),
                "wilcoxon_tracedev_p": float(wt_p),
                "kappa_diff_median": float(np.median(dk)),       # 开环指标(对照)
                "wilcoxon_kappa_p": float(wk_p),
            })

        # ---- 判决 (主判据: 配对 闭合 holonomy 平坦度差, 破缺应更不平坦) ----
        sig = self.config["experiment4"]["significance_level"]
        if not paired:
            verdict = "INCONCLUSIVE"
        elif (paired_info.get("flatness_diff_median", 0.0) > 0
              and paired_info.get("wilcoxon_flatness_p", 1.0) < sig):
            verdict = "SUPPORT"
        elif paired_info.get("flatness_diff_median", 0.0) < 0:
            verdict = "OPPOSE"
        else:
            verdict = "INCONCLUSIVE"

        summary = {
            "model_label": model_label,
            "n_pos": len(pos_recs),
            "n_neg": len(neg_recs),
            "pos_closed_flatness_mean": float(np.mean(pos_f)),
            "pos_closed_flatness_std": float(np.std(pos_f)),
            "neg_closed_flatness_mean": float(np.mean(neg_f)),
            "neg_closed_flatness_std": float(np.std(neg_f)),
            "pos_closed_trace_dev_mean": float(np.mean(pos_t)),
            "neg_closed_trace_dev_mean": float(np.mean(neg_t)),
            "pos_kappa_mean": float(np.mean(pos_k)),
            "neg_kappa_mean": float(np.mean(neg_k)),
            "paired": paired_info,
            "verdict": verdict,
        }
        self.logger.info(
            f"  判决={verdict} | 配对平平坦度差中位={paired_info.get('flatness_diff_median', 0):.4f} "
            f"p={paired_info.get('wilcoxon_flatness_p', 1):.4f} | "
            f"闭合平坦度: 正={np.mean(pos_f):.4f} 负={np.mean(neg_f):.4f}"
        )
        return summary
