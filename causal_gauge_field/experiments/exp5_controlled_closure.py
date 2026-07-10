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


class Experiment5:
    """受控检验: 同一模型内 闭环叙事(正例) vs 破缺叙事(负例) 的平坦度。

    与 exp4(比 baseline/causal 两模型) 不同, 本实验直接检验 C-11「叙事闭环 ⇔
    规范场平坦」的 ⇔ 核心: 破缺叙事应比其源闭环叙事更不平坦。

    设计要点:
    - 负例由正例派生(story_id 相同), 故可做强配对检验(破缺 vs 其源闭环),
      控制住底层叙事差异, 干净地隔离「打破一致性」这一变量的效应。
    - 主判据: 转向角曲率 κ_angle (已修复 exp4 的 1/||v|| 假象, 与步长无关)。
    - 辅判据: Wilson 环量方差 Var[W] — 标注为受污染: 破缺叙事注入更大状态跳变,
      会机械放大联络 A 与 W 的方差, 仅作参考, 不计入主判决。
    - 控制变量: 平均步长 ||v|| 分组/配对对比, 识别曲率差是否仅由步长差驱动。
    """

    def __init__(self, config: dict, gauge_field: Optional[GaugeField] = None):
        self.config = config
        self.logger = setup_logger("Experiment5")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gauge_field = gauge_field

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
            _, hidden = model(input_ids)                 # (1, T, base_dim)
            hidden_np = hidden[0, :, :].cpu().numpy()
        kappa_mean, _ = discrete_curvature(hidden_np)
        segs = np.diff(hidden_np, axis=0)
        step_norm = float(np.mean(np.linalg.norm(segs, axis=1))) if len(segs) else 0.0
        rec = {"kappa_angle": kappa_mean, "step_norm": step_norm}
        if self.gauge_field is not None:
            self.gauge_field.eval()
            W = self.gauge_field.wilson_loop(hidden)
            rec["wilson"] = float(W.item())
        return rec

    def run(
        self,
        model: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        model_label: str = "causal",
    ) -> Dict:
        self.logger.info(f"=== 实验5(受控): {model_label} 模型 正例vs负例平坦度 ===")
        pos_recs = [r for r in (self._per_story(model, s) for s in test_pos) if r]
        neg_recs = [r for r in (self._per_story(model, s) for s in test_neg) if r]
        if not pos_recs or not neg_recs:
            return {"verdict": "INCONCLUSIVE", "reason": "样本不足"}

        pos_k = np.array([r["kappa_angle"] for r in pos_recs])
        neg_k = np.array([r["kappa_angle"] for r in neg_recs])
        pos_n = np.array([r["step_norm"] for r in pos_recs])
        neg_n = np.array([r["step_norm"] for r in neg_recs])
        has_w = self.gauge_field is not None and "wilson" in pos_recs[0]
        if has_w:
            pos_w = np.array([r["wilson"] for r in pos_recs])
            neg_w = np.array([r["wilson"] for r in neg_recs])

        # ---- 非配对组间比较 (Mann-Whitney + Welch t, 双尾报告) ----
        if len(pos_k) > 2 and len(neg_k) > 2:
            u_stat, u_p = stats.mannwhitneyu(pos_k, neg_k, alternative="less")
            t_stat, t_p = stats.ttest_ind(pos_k, neg_k, alternative="less", equal_var=False)
            rb = 1.0 - 2.0 * u_stat / (len(pos_k) * len(neg_k))   # 秩双列效应量
        else:
            u_stat, u_p, t_stat, t_p, rb = 0.0, 1.0, 0.0, 1.0, 0.0

        # ---- 配对比较 (负例派生自正例, story_id 相同) ----
        pos_by_id = {s.story_id: r for s, r in zip(test_pos, pos_recs)}
        paired = [(pos_by_id[s.story_id], r) for s, r in zip(test_neg, neg_recs)
                  if s.story_id in pos_by_id]
        paired_info = {"n_pairs": len(paired)}
        if paired:
            dk = np.array([nr["kappa_angle"] - pr["kappa_angle"] for pr, nr in paired])
            dn = np.array([nr["step_norm"] - pr["step_norm"] for pr, nr in paired])
            try:
                w_stat, w_p = stats.wilcoxon(dk, alternative="greater")
            except ValueError:
                w_stat, w_p = 0.0, 1.0
            paired_info.update({
                "kappa_diff_median": float(np.median(dk)),
                "kappa_diff_mean": float(np.mean(dk)),
                "frac_neg_flatter": float(np.mean(dk > 0)),   # 破缺更不平坦的比例
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "stepnorm_diff_median": float(np.median(dn)),
                "frac_stepnorm_larger": float(np.mean(dn > 0)),
            })
            if has_w:
                dw = np.array([nr["wilson"] - pr["wilson"] for pr, nr in paired])
                try:
                    wp = float(stats.wilcoxon(dw, alternative="greater")[1])
                except ValueError:
                    wp = 1.0
                paired_info.update({
                    "wilson_diff_median": float(np.median(dw)),
                    "wilcoxon_wilson_p": wp,
                })

        # ---- 判决 (主判据: 配对 κ_angle 差, 破缺应更不平坦) ----
        sig = self.config["experiment4"]["significance_level"]
        if not paired:
            verdict = "INCONCLUSIVE"
        elif (paired_info.get("kappa_diff_median", 0.0) > 0
              and paired_info.get("wilcoxon_p", 1.0) < sig):
            verdict = "SUPPORT"
        elif paired_info.get("kappa_diff_median", 0.0) < 0:
            verdict = "OPPOSE"
        else:
            verdict = "INCONCLUSIVE"

        summary = {
            "model_label": model_label,
            "n_pos": len(pos_recs),
            "n_neg": len(neg_recs),
            "pos_kappa_mean": float(np.mean(pos_k)),
            "pos_kappa_std": float(np.std(pos_k)),
            "neg_kappa_mean": float(np.mean(neg_k)),
            "neg_kappa_std": float(np.std(neg_k)),
            "pos_stepnorm_mean": float(np.mean(pos_n)),
            "neg_stepnorm_mean": float(np.mean(neg_n)),
            "mannwhitneyU_p": float(u_p),
            "ttest_p": float(t_p),
            "rank_biserial": float(rb),
            "paired": paired_info,
            "verdict": verdict,
            "wilson_available": has_w,
            "wilson_confounded": has_w,   # 破缺叙事注入更大跳变 => Wilson 方差升高属机械效应
        }
        if has_w:
            summary["pos_wilson_var"] = float(np.var(pos_w))
            summary["neg_wilson_var"] = float(np.var(neg_w))
        self.logger.info(
            f"  判决={verdict} | 配对κ差中位={paired_info.get('kappa_diff_median', 0):.4f} "
            f"p={paired_info.get('wilcoxon_p', 1):.4f} | "
            f"κ: 正={np.mean(pos_k):.3f} 负={np.mean(neg_k):.3f}"
        )
        return summary
