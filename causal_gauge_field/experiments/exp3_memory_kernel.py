import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple

from ..models.transformer import CausalTransformer
from ..models.memory_bank import CausalMemoryBank
from ..npnw.story_generator import Story
from ..npnw.tokenizer import NPNWTokenizer
from ..utils.logger import setup_logger
from ..utils.metrics import frchet_distance


class Experiment3:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Experiment3")
        self.tokenizer = NPNWTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_cfg = config["experiment3"]
        self.max_seq_len = config["model"]["max_seq_len"]

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def run(
        self,
        model: CausalTransformer,
        memory_bank: CausalMemoryBank,
        test_stories: List[Story],
    ) -> Dict:
        self.logger.info("=== 实验3: 记忆核因果特异性 ===")
        model.eval()
        memory_bank.eval()
        all_hidden = []
        all_causal_types = []
        with torch.no_grad():
            for story in test_stories:
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
                h_last = hidden[0, -1, :].cpu().numpy()
                all_hidden.append(h_last)
                if story.violation_type == "physical":
                    all_causal_types.append(0)
                elif story.violation_type == "narrative":
                    all_causal_types.append(1)
                elif story.violation_type == "psychological":
                    all_causal_types.append(2)
                else:
                    all_causal_types.append(-1)
        if len(all_hidden) < 10:
            self.logger.warning("数据不足，无法进行聚类分析")
            return {"verdict": "INCONCLUSIVE", "reason": "insufficient_data"}
        X = np.array(all_hidden)
        causal_types = np.array(all_causal_types)
        self.logger.info("因果引导聚类...")
        causal_clustering = self._causal_guided_clustering(X, causal_types)
        self.logger.info("随机聚类对照...")
        random_clustering = self._random_clustering(X, self.exp_cfg["random_clusters"])
        k_eff_values = self.exp_cfg["k_eff_values"]
        convergence_results = {}
        for k_eff in k_eff_values:
            k_label = str(k_eff) if k_eff > 0 else "all"
            self.logger.info(f"  K_eff={k_label}")
            divergence_causal = self._measure_convergence(
                model, memory_bank, test_stories, k_eff, causal_clustering
            )
            divergence_random = self._measure_convergence(
                model, memory_bank, test_stories, k_eff, random_clustering
            )
            convergence_results[k_label] = {
                "causal_divergence": divergence_causal,
                "random_divergence": divergence_random,
            }
        causal_divs = [v["causal_divergence"] for v in convergence_results.values()]
        random_divs = [v["random_divergence"] for v in convergence_results.values()]
        causal_decreasing = self._check_monotonic_decrease(causal_divs)
        random_decreasing = self._check_monotonic_decrease(random_divs)
        causal_better = np.mean(causal_divs) < np.mean(random_divs)
        # B-12: 显式幂律拟合 Fréchet 方差 ∝ 1/K_eff (排除 K=-1 "all" 档)
        keff_fit = np.array([k for k in k_eff_values if k > 0], dtype=float)
        causal_fit = np.array([convergence_results[str(k)]["causal_divergence"]
                               for k in k_eff_values if k > 0], dtype=float)
        random_fit = np.array([convergence_results[str(k)]["random_divergence"]
                               for k in k_eff_values if k > 0], dtype=float)
        if len(keff_fit) >= 2:
            slope_c = float(np.polyfit(np.log(keff_fit), np.log(causal_fit + 1e-9), 1)[0])
            slope_r = float(np.polyfit(np.log(keff_fit), np.log(random_fit + 1e-9), 1)[0])
        else:
            slope_c, slope_r = 0.0, 0.0
        if causal_decreasing and causal_better:
            verdict = "SUPPORT"
        elif not causal_decreasing and not random_decreasing:
            verdict = "OPPOSE"
        elif causal_decreasing and not causal_better:
            verdict = "PARTIAL_SUPPORT"
        else:
            verdict = "INCONCLUSIVE"
        summary = {
            "convergence_results": convergence_results,
            "causal_decreasing": causal_decreasing,
            "random_decreasing": random_decreasing,
            "causal_better": causal_better,
            "power_law_slope_causal": slope_c,
            "power_law_slope_random": slope_r,
            "verdict": verdict,
        }
        self.logger.info(f"实验3判决: {verdict}")
        return summary

    def _causal_guided_clustering(
        self, X: np.ndarray, causal_types: np.ndarray
    ) -> Dict[int, List[int]]:
        n_clusters = self.exp_cfg["random_clusters"]
        valid_mask = causal_types >= 0
        X_valid = X[valid_mask]
        types_valid = causal_types[valid_mask]
        clustering = {i: [] for i in range(n_clusters)}
        if len(X_valid) < n_clusters:
            for i in range(len(X_valid)):
                clustering[i % n_clusters].append(i)
            return clustering
        type_centers = []
        for t in range(3):
            mask = types_valid == t
            if mask.any():
                type_centers.append(X_valid[mask].mean(axis=0))
            else:
                type_centers.append(np.zeros(X.shape[1]))
        from sklearn.cluster import KMeans
        init_centers = np.array(type_centers)
        if init_centers.shape[0] < n_clusters:
            extra = n_clusters - init_centers.shape[0]
            extra_centers = X_valid[np.random.choice(len(X_valid), extra, replace=False)]
            init_centers = np.vstack([init_centers, extra_centers])
        kmeans = KMeans(n_clusters=n_clusters, init=init_centers[:n_clusters], n_init=1, random_state=42)
        labels = kmeans.fit_predict(X_valid)
        for i, label in enumerate(labels):
            clustering[int(label)].append(i)
        return clustering

    def _random_clustering(self, X: np.ndarray, n_clusters: int) -> Dict[int, List[int]]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        clustering = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(labels):
            clustering[int(label)].append(i)
        return clustering

    def _measure_convergence(
        self,
        model: CausalTransformer,
        memory_bank: CausalMemoryBank,
        stories: List[Story],
        k_eff: int,
        clustering: Dict[int, List[int]],
    ) -> float:
        model.eval()
        trajectories = []
        num_prefixes = min(self.exp_cfg["num_prefixes"], len(stories))
        with torch.no_grad():
            for story in stories[:num_prefixes]:
                steps_data = []
                for step in story.steps[:5]:
                    steps_data.append({
                        "state": step.state,
                        "action": step.action,
                        "causal_labels": step.causal_labels,
                    })
                token_ids = self.tokenizer.encode_story(steps_data)
                token_ids = self._truncate_tokens(token_ids, offset=0)
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                for _ in range(min(self.exp_cfg["stories_per_prefix"], 5)):
                    generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
                    _, hidden = model(generated)
                    traj = hidden[0, :, :].cpu().numpy()
                    trajectories.append(traj)
        if len(trajectories) < 2:
            return float("inf")
        distances = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                min_len = min(len(trajectories[i]), len(trajectories[j]))
                d = frchet_distance(trajectories[i][:min_len], trajectories[j][:min_len])
                distances.append(d)
        return float(np.mean(distances)) if distances else float("inf")

    def _check_monotonic_decrease(self, values: List[float]) -> bool:
        if len(values) < 2:
            return False
        decreases = 0
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                decreases += 1
        return decreases > len(values) * 0.5