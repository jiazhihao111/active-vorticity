import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple

from ..models.transformer import CausalTransformer
from ..models.memory_bank import CausalMemoryBank
from ..losses.causal_geometry import CausalGeometryLoss
from ..npnw.story_generator import Story, StoryGenerator
from ..npnw.tokenizer import NPNWTokenizer
from ..experiments.trainer import Trainer, StoryDataset
from ..utils.logger import setup_logger
from ..utils.config import load_config


class Experiment1:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Experiment1")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_cfg = config["experiment1"]
        self.max_seq_len = config["model"]["max_seq_len"]

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def run(
        self,
        baseline_model: CausalTransformer,
        test_stories: List[Story],
    ) -> Dict:
        self.logger.info("=== 实验1: 因果曲率代理量效度校准 ===")
        self.logger.info("方法A: 隐状态转移不一致分数")
        results_a = self._method_a(baseline_model, test_stories)
        self.logger.info("方法B: 理论测地线距离校准")
        results_b = self._method_b(baseline_model, test_stories)
        r_a = results_a["correlation"]
        r_b = results_b["correlation"]
        if r_a > self.exp_cfg["correlation_threshold_support"]:
            verdict_a = "SUPPORT"
        elif r_a < self.exp_cfg["correlation_threshold_weaken"]:
            verdict_a = "WEAKEN"
        elif r_a < -0.2:
            verdict_a = "STRONG_OPPOSE"
        else:
            verdict_a = "INCONCLUSIVE"
        if r_b > 0.5:
            verdict_b = "SUPPORT"
        elif r_b < 0.2:
            verdict_b = "WEAKEN"
        else:
            verdict_b = "INCONCLUSIVE"
        summary = {
            "method_a": results_a,
            "method_b": results_b,
            "verdict_a": verdict_a,
            "verdict_b": verdict_b,
            "overall_verdict": verdict_a if verdict_a == verdict_b else "INCONCLUSIVE",
        }
        self.logger.info(f"方法A判决: {verdict_a}, r={r_a:.4f}")
        self.logger.info(f"方法B判决: {verdict_b}, r={r_b:.4f}")
        self.logger.info(f"总体判决: {summary['overall_verdict']}")
        return summary

    def _method_a(
        self, model: CausalTransformer, stories: List[Story]
    ) -> Dict:
        # 校准: 模型的投影空间转移幅值 vs NPNW 独立因果方向场 (C-07)
        model.eval()
        model_proxy = []
        indep_field = []
        with torch.no_grad():
            for story in stories:
                steps_data = []
                for step in story.steps:
                    steps_data.append({
                        "state": step.state,
                        "action": step.action,
                        "causal_labels": step.causal_labels,
                    })
                token_ids = self.tokenizer.encode_story(steps_data)
                token_ids = self._truncate_tokens(token_ids)
                if len(token_ids) < 3:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)   # (1, T, base_dim) 已投影(C-03)
                for i in range(hidden.size(1) - 1):
                    h_t = hidden[0, i, :].cpu().numpy()
                    h_t1 = hidden[0, i + 1, :].cpu().numpy()
                    model_proxy.append(float(np.linalg.norm(h_t1 - h_t)))
                    indep_field.append(self._independent_causal_field(story, i))
        if not model_proxy:
            return {"correlation": 0.0, "p_value": 1.0, "num_points": 0}
        r, p = stats.pearsonr(np.array(model_proxy), np.array(indep_field))
        return {"correlation": float(r), "p_value": float(p), "num_points": len(model_proxy)}

    def _method_b(
        self, model: CausalTransformer, stories: List[Story]
    ) -> Dict:
        model.eval()
        d_empirical_list = []
        d_theory_list = []
        with torch.no_grad():
            for story in stories:
                if not story.is_positive:
                    continue
                steps_data = []
                for step in story.steps:
                    steps_data.append({
                        "state": step.state,
                        "action": step.action,
                        "causal_labels": step.causal_labels,
                    })
                token_ids = self.tokenizer.encode_story(steps_data)
                token_ids = self._truncate_tokens(token_ids)
                if len(token_ids) < 3:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)
                for i in range(min(hidden.size(1) - 1, len(story.steps) - 1)):
                    h_t = hidden[0, i, :].cpu().numpy()
                    h_t1 = hidden[0, i+1, :].cpu().numpy()
                    d_emp = np.linalg.norm(h_t1 - h_t)   # C-03: 投影空间距离
                    d_empirical_list.append(d_emp)
                    d_theory = self._compute_theoretical_geodesic(story, i)
                    d_theory_list.append(d_theory)
        if not d_empirical_list:
            return {"correlation": 0.0, "p_value": 1.0, "num_points": 0}
        d_emp_arr = np.array(d_empirical_list)
        d_theory_arr = np.array(d_theory_list)
        r, p = stats.pearsonr(d_emp_arr, d_theory_arr)
        return {"correlation": float(r), "p_value": float(p), "num_points": len(d_empirical_list)}

    def _independent_causal_field(self, story: Story, i: int) -> float:
        # C-07: 来自 NPNW 世界模型的独立因果合法性标注，不依赖模型自身表征
        if i >= len(story.steps) - 1:
            return 0.0
        s_now = story.steps[i]
        s_nxt = story.steps[i + 1]
        inc = 0.0
        if not s_now.physical_legal or not s_nxt.physical_legal:
            inc += 1.0
        if not s_now.narrative_legal or not s_nxt.narrative_legal:
            inc += 0.7
        if not s_now.psychological_legal or not s_nxt.psychological_legal:
            inc += 0.5
        return inc

    def _compute_theoretical_geodesic(self, story: Story, step_idx: int) -> float:
        if step_idx >= len(story.steps) - 1:
            return 0.0
        step = story.steps[step_idx]
        next_step = story.steps[step_idx + 1]
        d = 0.0
        if not step.physical_legal:
            d += 2.0
        if not step.narrative_legal:
            d += 1.5
        if not step.psychological_legal:
            d += 1.0
        pos_diff = abs(step.state.get("pos_x", 0) - next_step.state.get("pos_x", 0)) + \
                   abs(step.state.get("pos_y", 0) - next_step.state.get("pos_y", 0))
        if pos_diff > 1:
            d += float(pos_diff)
        stamina_diff = abs(step.state.get("stamina", 0) - next_step.state.get("stamina", 0))
        if stamina_diff > 1:
            d += float(stamina_diff) * 0.5
        return d