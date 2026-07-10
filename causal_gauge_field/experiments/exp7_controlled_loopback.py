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


class Experiment7:
    """受控检验(实验7, §10.7 三项改造之集成):

    主判据 = 差异化『回环点』holonomy 平坦度 (gauge_field.loop_back_holonomy_flatness),
    而非整段均值式 wilson_loop_closed.

      - 正例(出征→返回原点, 真实闭环): 其回环(h_0↔h_T 等)几何应平坦;
      - 负例(出征→发散尾, 破缺): 其最优回环应更不平坦 (被对比信号推离).

    设计: 负例由正例同一条出征路径派生 (story_id 相同) → 严格配对 Wilcoxon,
          H1: 破缺更不平坦 (loop_flat_负 > loop_flat_正, 即差>0).

    对照指标同时报告:
      - 整段闭合 holonomy 平坦度 (实验6 指标, 用于对比『整段均值』vs『回环点』);
      - 检出的回环数量 (loop_count, 诊断: 正例应检到更多真实回环).

    判决 (主判据: 配对 回环 holonomy 平坦度差, 破缺应更不平坦):
      - 中位>0 且 Wilcoxon p<α  → SUPPORT   (C-11 在公平设计下成立)
      - 中位<0                  → OPPOSE
      - 否则                    → INCONCLUSIVE
    """

    def __init__(self, config: dict, gauge_field: GaugeField):
        self.config = config
        self.logger = setup_logger("Experiment7")
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
            self.gauge_field.eval()
            loop_flat, loop_cnt = self.gauge_field.loop_back_holonomy_flatness(hidden)
            full_flat = self.gauge_field.holonomy_flatness(hidden)
        hidden_np = hidden[0, :, :].cpu().numpy()
        kappa_mean, _ = discrete_curvature(hidden_np)
        rec = {
            "loop_flatness": float(loop_flat.item()),
            "loop_count": int(loop_cnt.item()),
            "full_closed_flatness": float(full_flat.item()),
            "kappa_angle": float(kappa_mean),
        }
        return rec

    def run_layered(
        self,
        model: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        model_label: str = "causal_rf",
    ) -> Dict:
        """刚柔分层诊断 (论文 §10.8 / 附录账本 H-rigid·H-flex).

        对每测试故事计算 per-token 转移曲率 (GaugeField.per_transition_connection_norm),
        按刚柔层掩码 (story_generator.step_layer_kind) 分离:
          - 物理层曲率应近 0  → H-rigid 判据
          - 柔性层: 正例曲率应 < τ, 负例(破缺)应 > τ (配对 Wilcoxon) → H-flex 判据

        判决直接回填附录假设账本 H-rigid / H-flex.
        """
        from ..npnw.story_generator import step_layer_kind
        tau = float(self.config.get("rigid_flexible", {}).get("tau_narr", 0.5))
        phys_thr = float(self.config.get("rigid_flexible", {}).get("phys_curv_threshold", 0.2))
        sig = self.config["experiment4"]["significance_level"]

        model.eval()
        self.gauge_field.eval()
        mdev = next(model.parameters()).device          # 与传入模型同设备, 避免 CPU/CUDA 错配
        phys_curvs: List[float] = []
        flex_pos_by_id: Dict = {}
        flex_neg_by_id: Dict = {}
        for story in list(test_pos) + list(test_neg):
            steps_data = [{"state": s.state, "action": s.action, "causal_labels": s.causal_labels}
                          for s in story.steps]
            token_ids, step_ids = self.tokenizer.encode_story(steps_data, with_step_ids=True)
            token_ids = self._truncate(token_ids)
            step_ids = step_ids[:len(token_ids)]          # 与截断后的 token 对齐
            if len(token_ids) < 4:
                continue
            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(mdev)
            with torch.no_grad():
                _, hidden = model(input_ids)
                curv = self.gauge_field.per_transition_connection_norm(hidden)[0]   # (Tm1,)
            Tm1 = curv.size(0)
            tl = torch.full((len(token_ids),), -1, dtype=torch.long)
            for pos, sid in enumerate(step_ids):
                if sid < 0:
                    continue
                tl[pos] = int(step_layer_kind(story.steps[sid]))
            layer_t = tl[1:Tm1 + 1].tolist()                       # 转移 t 用目标 token t+1
            for t in range(Tm1):
                ly = layer_t[t]
                if ly < 0:
                    continue
                c = float(curv[t].item())
                if ly == 0:
                    phys_curvs.append(c)
                elif story.is_positive:
                    flex_pos_by_id.setdefault(story.story_id, []).append(c)
                else:
                    flex_neg_by_id.setdefault(story.story_id, []).append(c)

        phys_mean = float(np.mean(phys_curvs)) if phys_curvs else float("nan")
        flex_pos = np.array([v for vs in flex_pos_by_id.values() for v in vs])
        flex_neg = np.array([v for vs in flex_neg_by_id.values() for v in vs])
        flex_pos_mean = float(np.mean(flex_pos)) if flex_pos.size else float("nan")
        flex_neg_mean = float(np.mean(flex_neg)) if flex_neg.size else float("nan")

        # H-flex: 每故事 柔性负例均值 − 柔性正例均值, 配对 Wilcoxon (alternative greater)
        paired_ids = set(flex_pos_by_id) & set(flex_neg_by_id)
        hflex_info: Dict = {"n_pairs": len(paired_ids)}
        if paired_ids:
            diffs = np.array([
                float(np.mean(flex_neg_by_id[i])) - float(np.mean(flex_pos_by_id[i]))
                for i in paired_ids])
            try:
                wf_stat, wf_p = stats.wilcoxon(diffs, alternative="greater")
            except ValueError:
                wf_stat, wf_p = 0.0, 1.0
            hflex_info.update({
                "flex_diff_median": float(np.median(diffs)),
                "wilcoxon_flex_p": float(wf_p),
            })

        # 判决
        h_rigid = "SUPPORT" if (len(phys_curvs) > 0 and phys_mean < phys_thr) else "INCONCLUSIVE"
        if flex_pos.size == 0 or flex_neg.size == 0:
            h_flex = "INCONCLUSIVE"
        elif (hflex_info.get("flex_diff_median", 0.0) > 0
              and hflex_info.get("wilcoxon_flex_p", 1.0) < sig
              and flex_pos_mean < tau):
            h_flex = "SUPPORT"
        else:
            h_flex = "INCONCLUSIVE"

        summary = {
            "model_label": model_label,
            "tau": tau,
            "phys_curv_threshold": phys_thr,
            "phys_curv_mean": phys_mean,
            "flex_pos_curv_mean": flex_pos_mean,
            "flex_neg_curv_mean": flex_neg_mean,
            "h_rigid_verdict": h_rigid,
            "h_flex_verdict": h_flex,
            "h_flex_paired": hflex_info,
        }
        self.logger.info(
            f"  [分层] phys_curv_mean={phys_mean:.4f}(阈{phys_thr})→{h_rigid} | "
            f"flex 正={flex_pos_mean:.4f} 负={flex_neg_mean:.4f} τ={tau}→{h_flex}"
        )
        return summary

    def run(
        self,
        model: CausalTransformer,
        test_pos: List[Story],
        test_neg: List[Story],
        model_label: str = "causal",
    ) -> Dict:
        self.logger.info(f"=== 实验7(回环 holonomy 受控): {model_label} ===")
        pos_recs = [r for r in (self._per_story(model, s) for s in test_pos) if r]
        neg_recs = [r for r in (self._per_story(model, s) for s in test_neg) if r]
        if not pos_recs or not neg_recs:
            return {"verdict": "INCONCLUSIVE", "reason": "样本不足"}

        pos_lf = np.array([r["loop_flatness"] for r in pos_recs])
        neg_lf = np.array([r["loop_flatness"] for r in neg_recs])
        pos_lc = np.array([r["loop_count"] for r in pos_recs], dtype=float)
        neg_lc = np.array([r["loop_count"] for r in neg_recs], dtype=float)
        pos_ff = np.array([r["full_closed_flatness"] for r in pos_recs])
        neg_ff = np.array([r["full_closed_flatness"] for r in neg_recs])

        # 配对比较 (负例派生自正例, story_id 相同)
        pos_by_id = {s.story_id: r for s, r in zip(test_pos, pos_recs)}
        paired = [(pos_by_id[s.story_id], r) for s, r in zip(test_neg, neg_recs)
                  if s.story_id in pos_by_id]
        paired_info: Dict = {"n_pairs": len(paired)}
        if paired:
            dlf = np.array([nr["loop_flatness"] - pr["loop_flatness"] for pr, nr in paired])
            try:
                wf_stat, wf_p = stats.wilcoxon(dlf, alternative="greater")
            except ValueError:
                wf_stat, wf_p = 0.0, 1.0
            paired_info.update({
                "loop_flatness_diff_median": float(np.median(dlf)),
                "loop_flatness_diff_mean": float(np.mean(dlf)),
                "frac_neg_flatter": float(np.mean(dlf > 0)),
                "wilcoxon_loop_flatness_p": float(wf_p),
            })

        sig = self.config["experiment4"]["significance_level"]
        if not paired:
            verdict = "INCONCLUSIVE"
        elif (paired_info.get("loop_flatness_diff_median", 0.0) > 0
              and paired_info.get("wilcoxon_loop_flatness_p", 1.0) < sig):
            verdict = "SUPPORT"
        elif paired_info.get("loop_flatness_diff_median", 0.0) < 0:
            verdict = "OPPOSE"
        else:
            verdict = "INCONCLUSIVE"

        summary = {
            "model_label": model_label,
            "n_pos": len(pos_recs),
            "n_neg": len(neg_recs),
            "pos_loop_flatness_mean": float(np.mean(pos_lf)),
            "neg_loop_flatness_mean": float(np.mean(neg_lf)),
            "pos_loop_count_mean": float(np.mean(pos_lc)),
            "neg_loop_count_mean": float(np.mean(neg_lc)),
            "pos_full_closed_flatness_mean": float(np.mean(pos_ff)),
            "neg_full_closed_flatness_mean": float(np.mean(neg_ff)),
            "paired": paired_info,
            "verdict": verdict,
        }
        self.logger.info(
            f"  判决={verdict} | 回环平坦度: 正={np.mean(pos_lf):.4f} 负={np.mean(neg_lf):.4f} | "
            f"回环数: 正={np.mean(pos_lc):.2f} 负={np.mean(neg_lc):.2f} | "
            f"配对差中位={paired_info.get('loop_flatness_diff_median', 0):.4f} "
            f"p={paired_info.get('wilcoxon_loop_flatness_p', 1):.4f}"
        )
        return summary
