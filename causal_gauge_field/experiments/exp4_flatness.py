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


class Experiment4:
    def __init__(self, config: dict, gauge_field: Optional[GaugeField] = None):
        self.config = config
        self.logger = setup_logger("Experiment4")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_cfg = config["experiment4"]
        self.max_seq_len = config["model"]["max_seq_len"]
        self.gauge_field = gauge_field

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def run(
        self,
        baseline_model: CausalTransformer,
        causal_model: CausalTransformer,
        test_stories: List[Story],
    ) -> Dict:
        self.logger.info("=== 实验4: 规范场平坦化 ===")
        baseline_curvatures = self._compute_path_curvatures(baseline_model, test_stories)
        causal_curvatures = self._compute_path_curvatures(causal_model, test_stories)
        baseline_mean_kappa = np.mean([c[0] for c in baseline_curvatures]) if baseline_curvatures else 0.0
        baseline_max_kappa = np.mean([c[1] for c in baseline_curvatures]) if baseline_curvatures else 0.0
        causal_mean_kappa = np.mean([c[0] for c in causal_curvatures]) if causal_curvatures else 0.0
        causal_max_kappa = np.mean([c[1] for c in causal_curvatures]) if causal_curvatures else 0.0
        self.logger.info(f"基线模型: mean_κ={baseline_mean_kappa:.4f}, max_κ={baseline_max_kappa:.4f}")
        self.logger.info(f"因果模型: mean_κ={causal_mean_kappa:.4f}, max_κ={causal_max_kappa:.4f}")

        # B-06/C-11: Wilson 环量 W(γ) 作为几何闭环/平坦化判据。
        # 论文主张「叙事闭环 ⇔ Var[W]→0」，但该等式仅为待实证声明，故 Wilson 仅作辅证：
        #   平坦(因果)模型的 Wilson 方差应更小；其是否支持不改变 κ 主导的核心判决，
        #   仅在 κ 已支持时提供 corroboration，或在 κ 支持而 Wilson 强烈矛盾时降级为 INCONCLUSIVE。
        baseline_wilson_var = 0.0
        causal_wilson_var = 0.0
        wilson_available = self.gauge_field is not None
        if wilson_available:
            baseline_wilson = self._compute_wilson_loops(baseline_model, test_stories)
            causal_wilson = self._compute_wilson_loops(causal_model, test_stories)
            baseline_wilson_var = float(np.var(baseline_wilson)) if len(baseline_wilson) > 1 else 0.0
            causal_wilson_var = float(np.var(causal_wilson)) if len(causal_wilson) > 1 else 0.0
            self.logger.info(f"基线Wilson方差={baseline_wilson_var:.6f}, 因果Wilson方差={causal_wilson_var:.6f}")
        wilson_flattening_support = (
            wilson_available and baseline_wilson_var > 1e-12
            and causal_wilson_var < baseline_wilson_var
        )

        mean_kappas_b = [c[0] for c in baseline_curvatures]
        mean_kappas_c = [c[0] for c in causal_curvatures]
        if len(mean_kappas_b) > 2 and len(mean_kappas_c) > 2:
            t_stat, p_value = stats.ttest_ind(mean_kappas_b, mean_kappas_c, alternative='greater')
        else:
            t_stat, p_value = 0.0, 1.0

        kappa_support = (
            causal_mean_kappa < baseline_mean_kappa
            and causal_max_kappa < baseline_max_kappa
            and p_value < self.exp_cfg["significance_level"]
        )
        if kappa_support:
            # C-11: Wilson 仅作辅证；κ 已支持时，Wilson 若强烈矛盾(因果方差显著更大)则降级
            if wilson_available and baseline_wilson_var > 1e-12 and causal_wilson_var > baseline_wilson_var * 2.0:
                verdict = "INCONCLUSIVE"
            else:
                verdict = "SUPPORT"
        elif abs(causal_mean_kappa - baseline_mean_kappa) < 0.01:
            verdict = "OPPOSE"
        elif causal_mean_kappa > baseline_mean_kappa + 0.05:
            verdict = "STRONG_OPPOSE"
        else:
            verdict = "INCONCLUSIVE"
        summary = {
            "baseline_mean_kappa": float(baseline_mean_kappa),
            "baseline_max_kappa": float(baseline_max_kappa),
            "causal_mean_kappa": float(causal_mean_kappa),
            "causal_max_kappa": float(causal_max_kappa),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "wilson_available": bool(wilson_available),
            "baseline_wilson_var": float(baseline_wilson_var),
            "causal_wilson_var": float(causal_wilson_var),
            "wilson_flattening_support": bool(wilson_flattening_support),
            "verdict": verdict,
        }
        self.logger.info(f"实验4判决: {verdict} (Wilson辅证={'是' if wilson_flattening_support else '否/未启用'})")
        return summary

    def _compute_path_curvatures(
        self, model: CausalTransformer, stories: List[Story]
    ) -> List[Tuple[float, float]]:
        model.eval()
        curvatures = []
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
                if len(token_ids) < 4:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)
                hidden_np = hidden[0, :, :].cpu().numpy()
                mean_k, max_k = discrete_curvature(hidden_np)
                curvatures.append((mean_k, max_k))
        return curvatures

    def _compute_wilson_loops(
        self, model: CausalTransformer, stories: List[Story]
    ) -> List[float]:
        # B-06: 对每个正例叙事的投影隐含轨迹计算 Wilson 环量 W(γ)=Tr P exp(g∮A)。
        # 返回每个故事的 W 值列表，供 run() 统计 Var[W]（C-11 待实证闭环判据）。
        if self.gauge_field is None:
            return []
        self.gauge_field.eval()
        model.eval()
        loops = []
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
                if len(token_ids) < 4:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)            # (1, T, base_dim) 已投影(C-03)
                W = self.gauge_field.wilson_loop(hidden)  # (B,) -> (1,)
                loops.append(float(W.item()))
        return loops