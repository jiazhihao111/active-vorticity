import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.geometry.metric_tensor import CausalMetricTensor
from causal_gauge_field.geometry.gauge_connection import GaugeConnection
from causal_gauge_field.geometry.curvature_verifier import CurvatureVerifier
from causal_gauge_field.experiments.trainer import StoryDataset


class GeometricTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("GeometricTrainer")
        self.tokenizer = NPNWTokenizer()
        config["model"]["vocab_size"] = self.tokenizer.vocab_size
        self.max_seq_len = config["model"]["max_seq_len"]
        self.hidden_dim = config["model"].get("base_dim", config["model"]["d_model"])

    def _truncate_tokens(self, token_ids, offset=1):
        max_len = self.max_seq_len - offset
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return token_ids

    def train_baseline(self, train_loader, val_loader, epochs=15):
        model = CausalTransformer(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["training"]["learning_rate"])
        lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.logger.info(f"基线模型参数量: {model.count_parameters()}")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits, _ = model(input_ids, attention_mask)
                loss = lm_loss_fn(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["training"]["grad_clip"])
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"  基线 Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        return model

    def train_geometric(self, train_loader, val_loader, epochs=15, lambda_causal=1.0):
        model = CausalTransformer(self.config).to(self.device)
        metric = CausalMetricTensor(self.hidden_dim).to(self.device)
        gauge = GaugeConnection(self.hidden_dim).to(self.device)
        all_params = list(model.parameters()) + list(metric.parameters()) + list(gauge.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.config["training"]["learning_rate"])
        lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.logger.info(f"几何模型参数量: {sum(p.numel() for p in all_params)}")
        for epoch in range(epochs):
            model.train()
            metric.train()
            gauge.train()
            total_loss = 0
            total_lm = 0
            total_causal = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                logits, hidden = model(input_ids, attention_mask)
                lm_loss = lm_loss_fn(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1),
                )
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)
                causal_loss = torch.tensor(0.0, device=self.device)
                if pos_mask.any() and neg_mask.any():
                    h_t_pos = hidden[pos_mask, :-1, :].reshape(-1, hidden.size(-1))
                    h_t1_pos = hidden[pos_mask, 1:, :].reshape(-1, hidden.size(-1))
                    h_t_neg = hidden[neg_mask, :-1, :].reshape(-1, hidden.size(-1))
                    h_t1_neg = hidden[neg_mask, 1:, :].reshape(-1, hidden.size(-1))
                    min_len = min(h_t_pos.size(0), h_t_neg.size(0))
                    if min_len > 0:
                        idx = torch.randperm(min_len, device=self.device)[:min(min_len, 64)]
                        h_t_p = h_t_pos[idx]
                        h_t1_p = h_t1_pos[idx]
                        h_t1_n = h_t1_neg[idx]
                        G = metric(h_t_p)
                        d_pos = metric.compute_geodesic_distance_sq(h_t_p, h_t1_p, G)
                        d_neg = metric.compute_geodesic_distance_sq(h_t_p, h_t1_n, G)
                        F_pos = gauge.compute_field_strength_norm_sq(h_t_p, h_t1_p)
                        F_neg = gauge.compute_field_strength_norm_sq(h_t_p, h_t1_n)
                        margin = self.config["training"]["margin"]
                        contrastive = F.relu(d_pos - d_neg + margin).mean() + \
                                      F.relu(F_pos - F_neg + margin * 0.5).mean()
                        eye = torch.eye(self.hidden_dim, device=self.device).unsqueeze(0)
                        G_off_diag = G - eye
                        metric_diversity = (G_off_diag ** 2).sum(dim=(-2, -1)).mean()
                        causal_loss = contrastive + 0.01 * metric_diversity
                total_loss_val = lm_loss + lambda_causal * causal_loss
                optimizer.zero_grad()
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(all_params, self.config["training"]["grad_clip"])
                optimizer.step()
                total_loss += total_loss_val.item()
                total_lm += lm_loss.item()
                total_causal += causal_loss.item()
            n = len(train_loader)
            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"  几何 Epoch {epoch+1}/{epochs} "
                    f"Total: {total_loss/n:.4f} LM: {total_lm/n:.4f} Causal: {total_causal/n:.4f}"
                )
        return model, metric, gauge

    def verify_baseline_geometry(self, model, test_stories):
        self.logger.info("=" * 60)
        self.logger.info("基线几何验证: 纯LM模型的隐状态是否自然具有因果几何结构")
        self.logger.info("=" * 60)
        model.eval()
        pos_euclid = []
        neg_euclid = []
        pos_curvatures = []
        neg_curvatures = []
        pos_cosine = []
        neg_cosine = []

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
                if len(token_ids) < 4:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)
                h_all = hidden[0, :, :]
                for i in range(h_all.size(0) - 1):
                    h_t = h_all[i]
                    h_t1 = h_all[i + 1]
                    euclid_dist = (h_t1 - h_t).norm().item()
                    cos_sim = F.cosine_similarity(h_t.unsqueeze(0), h_t1.unsqueeze(0)).item()
                    if story.is_positive:
                        pos_euclid.append(euclid_dist)
                        pos_cosine.append(cos_sim)
                    else:
                        neg_euclid.append(euclid_dist)
                        neg_cosine.append(cos_sim)

                deltas = h_all[1:] - h_all[:-1]
                if deltas.size(0) >= 2:
                    curvatures = []
                    for t in range(1, deltas.size(0)):
                        diff = deltas[t] - deltas[t - 1]
                        norm_prod = deltas[t].norm() * deltas[t - 1].norm()
                        if norm_prod > 1e-8:
                            kappa = diff.norm() / norm_prod
                            curvatures.append(kappa.item())
                    if curvatures:
                        mean_k = np.mean(curvatures)
                        if story.is_positive:
                            pos_curvatures.append(mean_k)
                        else:
                            neg_curvatures.append(mean_k)

        results = {}

        if pos_euclid and neg_euclid:
            d_pos_mean = np.mean(pos_euclid)
            d_neg_mean = np.mean(neg_euclid)
            t_stat, p_val = stats.ttest_ind(pos_euclid, neg_euclid)
            closer = d_pos_mean < d_neg_mean
            results["euclidean_distance"] = {
                "d_pos_mean": float(d_pos_mean),
                "d_neg_mean": float(d_neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_closer": closer,
                "verdict": "SUPPORT" if closer and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  欧几里得距离: d_pos={d_pos_mean:.4f}, d_neg={d_neg_mean:.4f}, "
                f"closer={closer}, p={p_val:.4f}"
            )

        if pos_cosine and neg_cosine:
            c_pos_mean = np.mean(pos_cosine)
            c_neg_mean = np.mean(neg_cosine)
            t_stat, p_val = stats.ttest_ind(pos_cosine, neg_cosine)
            more_aligned = c_pos_mean > c_neg_mean
            results["cosine_similarity"] = {
                "cos_pos_mean": float(c_pos_mean),
                "cos_neg_mean": float(c_neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_more_aligned": more_aligned,
                "verdict": "SUPPORT" if more_aligned and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  余弦相似度: cos_pos={c_pos_mean:.4f}, cos_neg={c_neg_mean:.4f}, "
                f"aligned={more_aligned}, p={p_val:.4f}"
            )

        if pos_curvatures and neg_curvatures:
            k_pos_mean = np.mean(pos_curvatures)
            k_neg_mean = np.mean(neg_curvatures)
            t_stat, p_val = stats.ttest_ind(pos_curvatures, neg_curvatures)
            flatter = k_pos_mean < k_neg_mean
            results["trajectory_curvature"] = {
                "pos_mean_curvature": float(k_pos_mean),
                "neg_mean_curvature": float(k_neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_flatter": flatter,
                "verdict": "SUPPORT" if flatter and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  轨迹曲率: pos_κ={k_pos_mean:.4f}, neg_κ={k_neg_mean:.4f}, "
                f"flatter={flatter}, p={p_val:.4f}"
            )

        support_count = sum(1 for v in results.values() if v.get("verdict") == "SUPPORT")
        total_count = len(results)
        if support_count >= 2:
            overall = "SUPPORT"
        elif support_count >= 1:
            overall = "WEAK_SUPPORT"
        else:
            overall = "OPPOSE"
        results["overall_verdict"] = {
            "support_count": support_count,
            "total_count": total_count,
            "verdict": overall,
            "interpretation": "几何结构自然涌现" if overall != "OPPOSE" else "几何结构未自然涌现",
        }
        self.logger.info(f"  基线几何总判决: {overall} ({support_count}/{total_count})")
        self.logger.info("=" * 60)
        return results

    def verify_axiom(self, model, metric, gauge, test_stories):
        self.logger.info("=" * 60)
        self.logger.info("验证公理: 信息化为世界模型，世界模型遵守几何规则")
        self.logger.info("=" * 60)
        model.eval()
        metric.eval()
        gauge.eval()
        verifier = CurvatureVerifier(self.hidden_dim).to(self.device)
        verifier.metric = metric
        verifier.gauge = gauge
        results = {
            "experiment_1_curvature_correlation": {},
            "experiment_2_geodesic_separation": {},
            "experiment_3_field_strength_discrimination": {},
            "experiment_4_trajectory_flatness": {},
            "experiment_5_gauge_invariance": {},
        }

        pos_d_g_sq_list = []
        neg_d_g_sq_list = []
        pos_F_sq_list = []
        neg_F_sq_list = []
        F_tilde_list = []
        consistency_list = []
        pos_curvatures = []
        neg_curvatures = []

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
                if len(token_ids) < 4:
                    continue
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                _, hidden = model(input_ids)
                h_all = hidden[0, :, :]
                for i in range(h_all.size(0) - 1):
                    h_t = h_all[i:i+1]
                    h_t1 = h_all[i+1:i+2]
                    proxy = verifier.compute_causal_curvature_proxy(h_t, h_t1)
                    F_tilde_list.append(proxy["F_tilde"].item())
                    is_legal = 1 if story.is_positive else 0
                    consistency_list.append(is_legal)
                    if story.is_positive:
                        pos_d_g_sq_list.append(proxy["d_g_squared"].item())
                        pos_F_sq_list.append(proxy["F_norm_squared"].item())
                    else:
                        neg_d_g_sq_list.append(proxy["d_g_squared"].item())
                        neg_F_sq_list.append(proxy["F_norm_squared"].item())

                traj = hidden
                traj_curv = verifier.compute_trajectory_curvature(traj)
                mean_k = traj_curv["mean_curvature"].item()
                if story.is_positive:
                    pos_curvatures.append(mean_k)
                else:
                    neg_curvatures.append(mean_k)

        if F_tilde_list and consistency_list:
            f_arr = np.array(F_tilde_list)
            s_arr = np.array(consistency_list)
            if len(np.unique(s_arr)) > 1:
                r, p = stats.pointbiserialr(s_arr, f_arr)
                results["experiment_1_curvature_correlation"] = {
                    "correlation": float(r), "p_value": float(p), "n_points": len(f_arr),
                    "verdict": "SUPPORT" if r > 0.3 else ("OPPOSE" if r < -0.2 else "INCONCLUSIVE"),
                }
                self.logger.info(f"  实验1(曲率相关性): r={r:.4f}, p={p:.4f}, 判决={results['experiment_1_curvature_correlation']['verdict']}")
            else:
                results["experiment_1_curvature_correlation"] = {"verdict": "INCONCLUSIVE", "reason": "constant_label"}
                self.logger.info("  实验1: 标签恒定，无法计算相关")

        if pos_d_g_sq_list and neg_d_g_sq_list:
            d_pos_mean = np.mean(pos_d_g_sq_list)
            d_neg_mean = np.mean(neg_d_g_sq_list)
            t_stat, p_val = stats.ttest_ind(pos_d_g_sq_list, neg_d_g_sq_list)
            geodesic_satisfied = d_pos_mean < d_neg_mean
            results["experiment_2_geodesic_separation"] = {
                "d_pos_mean": float(d_pos_mean),
                "d_neg_mean": float(d_neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "geodesic_satisfied": geodesic_satisfied,
                "verdict": "SUPPORT" if geodesic_satisfied and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  实验2(测地线分离): d_pos={d_pos_mean:.4f}, d_neg={d_neg_mean:.4f}, "
                f"satisfied={geodesic_satisfied}, p={p_val:.4f}"
            )

        if pos_F_sq_list and neg_F_sq_list:
            F_pos_mean = np.mean(pos_F_sq_list)
            F_neg_mean = np.mean(neg_F_sq_list)
            t_stat, p_val = stats.ttest_ind(pos_F_sq_list, neg_F_sq_list)
            field_discriminates = F_pos_mean < F_neg_mean
            results["experiment_3_field_strength_discrimination"] = {
                "F_pos_mean": float(F_pos_mean),
                "F_neg_mean": float(F_neg_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "field_discriminates": field_discriminates,
                "verdict": "SUPPORT" if field_discriminates and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  实验3(场强区分): F_pos={F_pos_mean:.4f}, F_neg={F_neg_mean:.4f}, "
                f"discriminates={field_discriminates}, p={p_val:.4f}"
            )

        if pos_curvatures and neg_curvatures:
            pos_k_mean = np.mean(pos_curvatures)
            neg_k_mean = np.mean(neg_curvatures)
            t_stat, p_val = stats.ttest_ind(pos_curvatures, neg_curvatures)
            flatter = pos_k_mean < neg_k_mean
            results["experiment_4_trajectory_flatness"] = {
                "pos_mean_curvature": float(pos_k_mean),
                "neg_mean_curvature": float(neg_k_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "legal_flatter": flatter,
                "verdict": "SUPPORT" if flatter and p_val < 0.05 else "OPPOSE",
            }
            self.logger.info(
                f"  实验4(路径平坦化): pos_κ={pos_k_mean:.4f}, neg_κ={neg_k_mean:.4f}, "
                f"flatter={flatter}, p={p_val:.4f}"
            )

        support_count = sum(1 for k in ["experiment_1_curvature_correlation",
                                         "experiment_2_geodesic_separation",
                                         "experiment_3_field_strength_discrimination",
                                         "experiment_4_trajectory_flatness"]
                           if results[k].get("verdict") == "SUPPORT")
        total_count = 4
        if support_count >= 3:
            overall = "STRONG_SUPPORT"
        elif support_count >= 2:
            overall = "PARTIAL_SUPPORT"
        elif support_count >= 1:
            overall = "WEAK_SUPPORT"
        else:
            overall = "OPPOSE"

        results["overall_verdict"] = {
            "support_count": support_count,
            "total_count": total_count,
            "verdict": overall,
            "axiom_half_1_verified": results["experiment_2_geodesic_separation"].get("verdict") == "SUPPORT" or
                                     results["experiment_3_field_strength_discrimination"].get("verdict") == "SUPPORT",
            "axiom_half_2_verified": results["experiment_4_trajectory_flatness"].get("verdict") == "SUPPORT",
        }
        self.logger.info("=" * 60)
        self.logger.info(f"总判决: {overall} ({support_count}/{total_count} 支持)")
        half1 = "信息化为世界模型"
        half2 = "世界模型遵守几何规则"
        v1 = results["overall_verdict"]["axiom_half_1_verified"]
        v2 = results["overall_verdict"]["axiom_half_2_verified"]
        self.logger.info(f"  '{half1}': {'验证通过' if v1 else '未通过'}")
        self.logger.info(f"  '{half2}': {'验证通过' if v2 else '未通过'}")
        self.logger.info("=" * 60)
        return results


def main():
    config = load_config()
    config["data"]["num_stories"] = 800
    config["training"]["max_epochs"] = 20
    config["training"]["patience"] = 8

    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    logger = setup_logger("Main", config["logging"]["log_dir"], "geometric_experiment.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("几何验证实验: 信息化为世界模型，世界模型遵守几何规则")
    logger.info("=" * 60)

    logger.info("[阶段0] 数据生成...")
    gen = StoryGenerator(config, seed=config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )
    logger.info(f"  正例: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    all_train = train_pos + train_neg
    all_val = val_pos + val_neg
    train_ds = StoryDataset(all_train, tokenizer, config["model"]["max_seq_len"])
    val_ds = StoryDataset(all_val, tokenizer, config["model"]["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    trainer = GeometricTrainer(config)

    logger.info("[阶段1] 训练基线模型...")
    baseline_model = trainer.train_baseline(train_loader, val_loader, epochs=20)

    logger.info("[阶段1.5] 基线几何验证（公理核心检验：几何结构是否自然涌现）...")
    test_stories = test_pos[:40] + test_neg[:40]
    baseline_results = trainer.verify_baseline_geometry(baseline_model, test_stories)

    logger.info("[阶段2] 训练几何增强模型...")
    geo_model, metric, gauge = trainer.train_geometric(
        train_loader, val_loader, epochs=20, lambda_causal=5.0
    )

    logger.info("[阶段3] 几何增强模型公理验证...")
    results = trainer.verify_axiom(geo_model, metric, gauge, test_stories)

    output_dir = Path(config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "axiom": "信息化为世界模型，世界模型遵守几何规则",
        "baseline_geometry": baseline_results,
        "geometric_model": results,
    }
    report_path = output_dir / "geometric_axiom_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    md_lines = [
        "# 几何公理验证报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n## 验证公理",
        f"\n> **信息化为世界模型，世界模型遵守几何规则**",
        "\n---\n",
        "## 基线几何验证（核心：几何结构是否自然涌现）\n",
        "| 指标 | 正例均值 | 负例均值 | 方向正确 | p值 | 判决 |",
        "|------|----------|----------|----------|-----|------|",
    ]
    baseline_metric_names = {
        "euclidean_distance": ("欧几里得距离", "d_pos", "d_neg", "legal_closer"),
        "cosine_similarity": ("余弦相似度", "cos_pos_mean", "cos_neg_mean", "legal_more_aligned"),
        "trajectory_curvature": ("轨迹曲率", "pos_mean_curvature", "neg_mean_curvature", "legal_flatter"),
    }
    for key, (name, pos_key, neg_key, dir_key) in baseline_metric_names.items():
        v = baseline_results.get(key, {})
        if v:
            pos_val = v.get(pos_key, 0)
            neg_val = v.get(neg_key, 0)
            p_val = v.get("p_value", 1)
            pos_str = f"{pos_val:.4f}" if isinstance(pos_val, (int, float)) else str(pos_val)
            neg_str = f"{neg_val:.4f}" if isinstance(neg_val, (int, float)) else str(neg_val)
            p_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
            md_lines.append(
                f"| {name} | {pos_str} | {neg_str} | "
                f"{v.get(dir_key, 'N/A')} | {p_str} | {v.get('verdict', 'N/A')} |"
            )
    bl_verdict = baseline_results.get("overall_verdict", {}).get("verdict", "N/A")
    md_lines.append(f"\n**基线总判决**: {bl_verdict}")
    md_lines.append("\n---\n")
    md_lines.append("## 几何增强模型验证\n")
    md_lines.append("| 实验 | 核心指标 | 判决 |")
    md_lines.append("|------|----------|------|")
    exp_names = {
        "experiment_1_curvature_correlation": ("因果曲率相关性", "r"),
        "experiment_2_geodesic_separation": ("测地线距离分离", "d_pos vs d_neg"),
        "experiment_3_field_strength_discrimination": ("场强区分力", "F_pos vs F_neg"),
        "experiment_4_trajectory_flatness": ("路径平坦化", "pos_κ vs neg_κ"),
    }
    for key, (name, metric_name) in exp_names.items():
        v = results.get(key, {})
        verdict = v.get("verdict", "N/A")
        md_lines.append(f"| {name} | {metric_name} | {verdict} |")
    md_lines.append(f"\n**总体判决**: {results.get('overall_verdict', {}).get('verdict', 'N/A')}")
    half1 = results.get("overall_verdict", {}).get("axiom_half_1_verified", False)
    half2 = results.get("overall_verdict", {}).get("axiom_half_2_verified", False)
    md_lines.append(f"\n- '信息化为世界模型': {'✅ 验证通过' if half1 else '❌ 未通过'}")
    md_lines.append(f"- '世界模型遵守几何规则': {'✅ 验证通过' if half2 else '❌ 未通过'}")

    md_path = output_dir / "geometric_axiom_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return results


if __name__ == "__main__":
    main()