import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from ..models.transformer import CausalTransformer
from ..models.memory_bank import CausalMemoryBank
from ..npnw.story_generator import Story, StoryGenerator
from ..npnw.tokenizer import NPNWTokenizer
from ..experiments.trainer import Trainer, StoryDataset
from ..utils.logger import setup_logger
from ..utils.metrics import (
    physical_legal_rate,
    narrative_closure_rate,
    personality_consistency_rate,
)


class Experiment2:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Experiment2")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_cfg = config["experiment2"]
        self.max_seq_len = config["model"]["max_seq_len"]

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def run(
        self,
        train_pos: List[Story],
        train_neg: List[Story],
        val_pos: List[Story],
        val_neg: List[Story],
        test_pos: List[Story],
        test_neg: List[Story],
    ) -> Dict:
        self.logger.info("=== 实验2: 因果几何损失对长程一致性的影响 ===")
        all_train = train_pos + train_neg
        all_val = val_pos + val_neg
        train_ds = StoryDataset(all_train, self.tokenizer, self.config["model"]["max_seq_len"])
        val_ds = StoryDataset(all_val, self.tokenizer, self.config["model"]["max_seq_len"])
        train_loader = DataLoader(train_ds, batch_size=self.config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config["training"]["batch_size"])
        self.logger.info("训练基线模型 (lambda=0)...")
        baseline_model = CausalTransformer(self.config)
        baseline_trainer = Trainer(self.config, baseline_model)
        baseline_history = baseline_trainer.train_full(train_loader, val_loader, lambda_value=0.0)
        self.logger.info("训练因果正则化模型...")
        lambda_results = {}
        for lam in self.config["training"]["lambda_values"]:
            self.logger.info(f"  lambda={lam}")
            causal_model = CausalTransformer(self.config)
            memory_bank = CausalMemoryBank(self.config["model"]["d_model"])
            causal_trainer = Trainer(self.config, causal_model, memory_bank)
            history = causal_trainer.train_full(train_loader, val_loader, lambda_value=lam)
            lambda_results[lam] = {
                "history": history,
                "model": causal_model,
                "memory_bank": memory_bank,
            }
        self.logger.info("评估各模型在不同长度上的表现...")
        length_categories = self.exp_cfg["story_lengths"]
        baseline_eval = self._evaluate_by_length(baseline_model, test_pos, test_neg, length_categories)
        causal_evals = {}
        for lam, result in lambda_results.items():
            causal_evals[lam] = self._evaluate_by_length(
                result["model"], test_pos, test_neg, length_categories
            )
        pareto_front = self._compute_pareto_front(baseline_eval, causal_evals)
        verdict = self._render_verdict(baseline_eval, causal_evals)
        summary = {
            "baseline_eval": baseline_eval,
            "causal_evals": causal_evals,
            "pareto_front": pareto_front,
            "verdict": verdict,
        }
        self.logger.info(f"实验2判决: {verdict}")
        return summary

    def _evaluate_by_length(
        self,
        model: CausalTransformer,
        pos_stories: List[Story],
        neg_stories: List[Story],
        length_categories: Dict,
    ) -> Dict:
        model.eval()
        results = {}
        for cat_name, (min_len, max_len) in length_categories.items():
            filtered = [s for s in pos_stories if min_len <= len(s.steps) <= max_len]
            if not filtered:
                results[cat_name] = {
                    "physical_legal": 0.0,
                    "narrative_closure": 0.0,
                    "personality_consistency": 0.0,
                    "num_stories": 0,
                }
                continue
            phys_rates = []
            narr_rates = []
            psych_rates = []
            with torch.no_grad():
                for story in filtered[:self.exp_cfg["stories_per_length"]]:
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
                    logits, hidden = model(input_ids)
                    pred_ids = logits.argmax(dim=-1)
                    phys_rates.append(1.0 if story.steps[0].physical_legal else 0.0)
                    narr_rates.append(1.0 if all(s.narrative_legal for s in story.steps) else 0.0)
                    psych_rates.append(1.0 if all(s.psychological_legal for s in story.steps) else 0.0)
            results[cat_name] = {
                "physical_legal": np.mean(phys_rates) if phys_rates else 0.0,
                "narrative_closure": np.mean(narr_rates) if narr_rates else 0.0,
                "personality_consistency": np.mean(psych_rates) if psych_rates else 0.0,
                "num_stories": len(phys_rates),
            }
        return results

    def _compute_pareto_front(
        self, baseline_eval: Dict, causal_evals: Dict
    ) -> List[Dict]:
        points = []
        points.append({"lambda": 0.0, "consistency": self._avg_consistency(baseline_eval), "label": "baseline"})
        for lam, ev in causal_evals.items():
            points.append({"lambda": lam, "consistency": self._avg_consistency(ev), "label": f"lambda={lam}"})
        points.sort(key=lambda p: p["consistency"], reverse=True)
        return points

    def _avg_consistency(self, eval_result: Dict) -> float:
        vals = []
        for cat in eval_result.values():
            if isinstance(cat, dict):
                vals.append(cat.get("physical_legal", 0))
                vals.append(cat.get("narrative_closure", 0))
                vals.append(cat.get("personality_consistency", 0))
        return np.mean(vals) if vals else 0.0

    def _render_verdict(self, baseline_eval: Dict, causal_evals: Dict) -> str:
        baseline_long = baseline_eval.get("long", {})
        best_causal_long = None
        best_lambda = None
        for lam, ev in causal_evals.items():
            long_ev = ev.get("long", {})
            score = self._avg_consistency({"long": long_ev})
            if best_causal_long is None or score > best_causal_long:
                best_causal_long = score
                best_lambda = lam
        baseline_score = self._avg_consistency({"long": baseline_long})
        if best_causal_long is not None and best_causal_long > baseline_score + 0.05:
            return "STRONG_SUPPORT"
        elif best_causal_long is not None and best_causal_long > baseline_score:
            return "WEAK_SUPPORT"
        elif best_causal_long is not None and best_causal_long < baseline_score - 0.1:
            return "STRONG_OPPOSE"
        else:
            return "OPPOSE"