import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.newton.causal_potential import CausalPotential
from causal_gauge_field.newton.acceleration import AccelerationLoss, BarrierLoss
from causal_gauge_field.newton.active_force_analyzer import ActiveForceAnalyzer
from causal_gauge_field.experiments.trainer import StoryDataset
from torch.utils.data import DataLoader


class ResidualExtractor:
    """Proxy B: 提取Transformer各层的残差输出"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                hook_pre = module.register_forward_pre_hook(
                    self._make_pre_hook(name)
                )
                hook_post = module.register_forward_hook(
                    self._make_post_hook(name)
                )
                self.hooks.extend([hook_pre, hook_post])

    def _make_pre_hook(self, name):
        def hook(module, input):
            self.layer_inputs[name] = input[0].detach()
        return hook

    def _make_post_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.layer_outputs[name] = output[0].detach()
            else:
                self.layer_outputs[name] = output.detach()
        return hook

    def get_residuals(self):
        """返回各层残差: {layer_name: residual_tensor}"""
        residuals = {}
        for name in self.layer_inputs:
            if name in self.layer_outputs:
                residuals[name] = self.layer_outputs[name] - self.layer_inputs[name]
        return residuals

    def get_total_residual(self):
        """返回所有层残差之和: [B, T, D_model]"""
        residuals = self.get_residuals()
        if not residuals:
            return None
        return sum(residuals.values())

    def clear(self):
        self.layer_inputs.clear()
        self.layer_outputs.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def extract_hidden_with_residuals(model, stories, tokenizer, max_seq_len, device):
    pos_hidden = []
    neg_hidden = []
    pos_residuals_raw = []
    neg_residuals_raw = []
    pos_transformer_out = []
    neg_transformer_out = []

    extractor = ResidualExtractor(model)
    model.eval()

    with torch.no_grad():
        for story in stories:
            steps_data = []
            for step in story.steps:
                steps_data.append({
                    "state": step.state,
                    "action": step.action,
                    "causal_labels": step.causal_labels,
                })
            token_ids = tokenizer.encode_story(steps_data)
            max_len = max_seq_len - 1
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            if len(token_ids) < 4:
                continue

            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(device)

            extractor.clear()
            logits, hidden = model(input_ids)

            h = hidden[0]
            total_res = extractor.get_total_residual()

            if story.is_positive:
                pos_hidden.append(h)
                if total_res is not None:
                    pos_residuals_raw.append(total_res[0])
            else:
                neg_hidden.append(h)
                if total_res is not None:
                    neg_residuals_raw.append(total_res[0])

    extractor.remove_hooks()
    return pos_hidden, neg_hidden, pos_residuals_raw, neg_residuals_raw


def task_3_1_proxy_b_analysis(pos_hidden, neg_hidden, pos_residuals, neg_residuals, model, logger):
    """任务3.1: Proxy B(残差和) vs Proxy D(α·ḣ) 对比"""
    logger.info("=" * 60)
    logger.info("任务3.1: Proxy B(残差和) vs Proxy D(α·ḣ) 对比验证")
    logger.info("=" * 60)

    results = {}

    all_vel_pos = []
    all_vel_neg = []
    all_res_proj_pos = []
    all_res_proj_neg = []

    with torch.no_grad():
        for h, res in zip(pos_hidden, pos_residuals):
            if h.size(0) < 3 or res.size(0) < 3:
                continue
            min_t = min(h.size(0), res.size(0))
            h = h[:min_t]
            res = res[:min_t]

            vel = h[1:] - h[:-1]
            res_proj = model.base_projection(res[:-1])

            all_vel_pos.append(vel)
            all_res_proj_pos.append(res_proj)

        for h, res in zip(neg_hidden, neg_residuals):
            if h.size(0) < 3 or res.size(0) < 3:
                continue
            min_t = min(h.size(0), res.size(0))
            h = h[:min_t]
            res = res[:min_t]

            vel = h[1:] - h[:-1]
            res_proj = model.base_projection(res[:-1])

            all_vel_neg.append(vel)
            all_res_proj_neg.append(res_proj)

    if all_vel_pos and all_res_proj_pos:
        vel_pos_flat = torch.cat([v.flatten() for v in all_vel_pos]).cpu().numpy()
        res_pos_flat = torch.cat([r.flatten() for r in all_res_proj_pos]).cpu().numpy()
        vel_neg_flat = torch.cat([v.flatten() for v in all_vel_neg]).cpu().numpy()
        res_neg_flat = torch.cat([r.flatten() for r in all_res_proj_neg]).cpu().numpy()

        corr_pos, p_corr_pos = stats.pearsonr(vel_pos_flat, res_pos_flat)
        corr_neg, p_corr_neg = stats.pearsonr(vel_neg_flat, res_neg_flat)

        cos_sims_pos = []
        cos_sims_neg = []
        for v, r in zip(all_vel_pos, all_res_proj_pos):
            if v.size(0) > 0 and r.size(0) > 0:
                min_len = min(v.size(0), r.size(0))
                for t in range(min_len):
                    cs = F.cosine_similarity(v[t:t+1].cpu(), r[t:t+1].cpu(), dim=-1).item()
                    cos_sims_pos.append(cs)
        for v, r in zip(all_vel_neg, all_res_proj_neg):
            if v.size(0) > 0 and r.size(0) > 0:
                min_len = min(v.size(0), r.size(0))
                for t in range(min_len):
                    cs = F.cosine_similarity(v[t:t+1].cpu(), r[t:t+1].cpu(), dim=-1).item()
                    cos_sims_neg.append(cs)

        alpha_pos = np.mean(res_pos_flat) / (np.mean(vel_pos_flat) + 1e-10)
        alpha_neg = np.mean(res_neg_flat) / (np.mean(vel_neg_flat) + 1e-10)

        vel_norms_pos = [v.norm(dim=-1).mean().item() for v in all_vel_pos]
        res_norms_pos = [r.norm(dim=-1).mean().item() for r in all_res_proj_pos]
        ratio_pos = np.mean(res_norms_pos) / (np.mean(vel_norms_pos) + 1e-10)

        results = {
            "pearson_corr_pos": float(corr_pos),
            "pearson_corr_neg": float(corr_neg),
            "pearson_p_pos": float(p_corr_pos),
            "pearson_p_neg": float(p_corr_neg),
            "cosine_sim_pos_mean": float(np.mean(cos_sims_pos)),
            "cosine_sim_neg_mean": float(np.mean(cos_sims_neg)),
            "alpha_from_mean_pos": float(alpha_pos),
            "alpha_from_mean_neg": float(alpha_neg),
            "vel_norm_mean_pos": float(np.mean(vel_norms_pos)),
            "res_norm_mean_pos": float(np.mean(res_norms_pos)),
            "norm_ratio_pos": float(ratio_pos),
        }

        logger.info(f"  Pearson相关: pos={corr_pos:.4f}(p={p_corr_pos:.2e}), neg={corr_neg:.4f}(p={p_corr_neg:.2e})")
        logger.info(f"  余弦相似度: pos={np.mean(cos_sims_pos):.4f}, neg={np.mean(cos_sims_neg):.4f}")
        logger.info(f"  α估计(均值比): pos={alpha_pos:.4f}, neg={alpha_neg:.4f}")
        logger.info(f"  范数比(res/vel): pos={ratio_pos:.4f}")

    return results


def task_3_2_alpha_origin(pos_hidden, neg_hidden, model, config, logger):
    """任务3.2: α*≈1.1的物理来源分析"""
    logger.info("=" * 60)
    logger.info("任务3.2: α*≈1.1的物理来源分析")
    logger.info("=" * 60)

    results = {}

    d_model = config["model"]["d_model"]
    base_dim = config["model"].get("base_dim", d_model)
    n_layers = config["model"]["n_layers"]

    W_proj = model.base_projection.weight.detach().cpu()
    b_proj = model.base_projection.bias.detach().cpu()

    W_norm = W_proj.norm().item()
    singular_values = torch.linalg.svdvals(W_proj).numpy()

    results["architecture"] = {
        "d_model": d_model,
        "base_dim": base_dim,
        "n_layers": n_layers,
        "projection_ratio": base_dim / d_model,
    }
    results["projection_matrix"] = {
        "W_frobenius_norm": float(W_norm),
        "W_spectral_norm": float(singular_values[0]),
        "W_condition_number": float(singular_values[0] / (singular_values[-1] + 1e-10)),
        "sv_top5": singular_values[:5].tolist(),
    }

    logger.info(f"  架构: d_model={d_model}, base_dim={base_dim}, n_layers={n_layers}")
    logger.info(f"  投影矩阵: ‖W‖_F={W_norm:.4f}, σ_max={singular_values[0]:.4f}, cond={singular_values[0]/(singular_values[-1]+1e-10):.4f}")

    all_vel_pos = []
    all_vel_neg = []
    for h in pos_hidden:
        if h.size(0) >= 3:
            all_vel_pos.append(h[1:] - h[:-1])
    for h in neg_hidden:
        if h.size(0) >= 3:
            all_vel_neg.append(h[1:] - h[:-1])

    if all_vel_pos:
        vel_pos_cat = torch.cat(all_vel_pos, dim=0).cpu()
        vel_neg_cat = torch.cat(all_vel_neg, dim=0).cpu()

        vel_pos_mean = vel_pos_cat.mean(dim=0)
        vel_neg_mean = vel_neg_cat.mean(dim=0)

        vel_pos_cov = torch.cov(vel_pos_cat.T)
        eigvals_pos = torch.linalg.eigvalsh(vel_pos_cov).numpy()

        results["velocity_stats"] = {
            "pos_mean_norm": float(vel_pos_mean.norm()),
            "neg_mean_norm": float(vel_neg_mean.norm()),
            "pos_cov_trace": float(torch.trace(vel_pos_cov)),
            "pos_cov_eigenvalues_top5": sorted(eigvals_pos, reverse=True)[:5],
        }

        logger.info(f"  速度均值范数: pos={vel_pos_mean.norm():.4f}, neg={vel_neg_mean.norm():.4f}")
        logger.info(f"  速度协方差迹: pos={torch.trace(vel_pos_cov):.4f}")

        alpha_estimates = []
        for gamma in [0.01, 0.1]:
            for h in pos_hidden[:20]:
                if h.size(0) < 4:
                    continue
                if h.dim() == 2:
                    h2 = h.unsqueeze(0)
                else:
                    h2 = h
                vel = h2[:, 1:, :] - h2[:, :-1, :]
                acc = vel[:, 1:, :] - vel[:, :-1, :]
                v_for = vel[:, 1:, :]
                min_t = min(acc.size(1), v_for.size(1))
                F_res = acc[:, :min_t, :] + gamma * v_for[:, :min_t, :]
                P_raw = (F_res * v_for[:, :min_t, :]).sum(dim=-1)
                P_active = (v_for[:, :min_t, :] * v_for[:, :min_t, :]).sum(dim=-1)
                if P_active.abs().mean() > 1e-10:
                    alpha_local = P_raw.mean() / P_active.mean()
                    alpha_estimates.append(float(alpha_local))

        if alpha_estimates:
            results["alpha_local_estimates"] = {
                "mean": float(np.mean(alpha_estimates)),
                "std": float(np.std(alpha_estimates)),
                "min": float(np.min(alpha_estimates)),
                "max": float(np.max(alpha_estimates)),
            }
            logger.info(f"  α局部估计: mean={np.mean(alpha_estimates):.4f}, std={np.std(alpha_estimates):.4f}")

    return results


def task_3_3_negative_damping(pos_hidden, neg_hidden, logger):
    """任务3.3: 负阻尼验证——隐状态范数是否随时间步增长"""
    logger.info("=" * 60)
    logger.info("任务3.3: 负阻尼验证——隐状态范数随时间步演化")
    logger.info("=" * 60)

    results = {}

    pos_norm_trajectories = []
    neg_norm_trajectories = []

    for h in pos_hidden:
        if h.size(0) >= 4:
            norms = h.norm(dim=-1).cpu().numpy()
            pos_norm_trajectories.append(norms)
    for h in neg_hidden:
        if h.size(0) >= 4:
            norms = h.norm(dim=-1).cpu().numpy()
            neg_norm_trajectories.append(norms)

    if pos_norm_trajectories and neg_norm_trajectories:
        min_len = min(
            min(len(t) for t in pos_norm_trajectories),
            min(len(t) for t in neg_norm_trajectories)
        )

        pos_array = np.array([t[:min_len] for t in pos_norm_trajectories if len(t) >= min_len])
        neg_array = np.array([t[:min_len] for t in neg_norm_trajectories if len(t) >= min_len])

        pos_mean_traj = pos_array.mean(axis=0)
        neg_mean_traj = neg_array.mean(axis=0)

        pos_start = pos_mean_traj[:3].mean()
        pos_end = pos_mean_traj[-3:].mean()
        neg_start = neg_mean_traj[:3].mean()
        neg_end = neg_mean_traj[-3:].mean()

        pos_growth_rate = (pos_end - pos_start) / (pos_start + 1e-10)
        neg_growth_rate = (neg_end - neg_start) / (neg_start + 1e-10)

        pos_vel_norms = []
        neg_vel_norms = []
        for h in pos_hidden:
            if h.size(0) >= 3:
                vel = h[1:] - h[:-1]
                pos_vel_norms.extend(vel.norm(dim=-1).tolist())
        for h in neg_hidden:
            if h.size(0) >= 3:
                vel = h[1:] - h[:-1]
                neg_vel_norms.extend(vel.norm(dim=-1).tolist())

        vel_pos_mean = np.mean(pos_vel_norms)
        vel_neg_mean = np.mean(neg_vel_norms)

        pos_accel_norms = []
        neg_accel_norms = []
        for h in pos_hidden:
            if h.size(0) >= 4:
                vel = h[1:] - h[:-1]
                acc = vel[1:] - vel[:-1]
                pos_accel_norms.extend(acc.norm(dim=-1).tolist())
        for h in neg_hidden:
            if h.size(0) >= 4:
                vel = h[1:] - h[:-1]
                acc = vel[1:] - vel[:-1]
                neg_accel_norms.extend(acc.norm(dim=-1).tolist())

        accel_pos_mean = np.mean(pos_accel_norms)
        accel_neg_mean = np.mean(neg_accel_norms)

        results = {
            "hidden_norm_trajectory": {
                "pos_start": float(pos_start),
                "pos_end": float(pos_end),
                "pos_growth_rate": float(pos_growth_rate),
                "neg_start": float(neg_start),
                "neg_end": float(neg_end),
                "neg_growth_rate": float(neg_growth_rate),
                "pos_growing": bool(pos_growth_rate > 0),
                "neg_growing": bool(neg_growth_rate > 0),
            },
            "velocity_norms": {
                "pos_mean": float(vel_pos_mean),
                "neg_mean": float(vel_neg_mean),
                "pos_vs_neg_p": float(stats.ttest_ind(pos_vel_norms, neg_vel_norms, equal_var=False)[1]),
            },
            "acceleration_norms": {
                "pos_mean": float(accel_pos_mean),
                "neg_mean": float(accel_neg_mean),
                "pos_vs_neg_p": float(stats.ttest_ind(pos_accel_norms, neg_accel_norms, equal_var=False)[1]),
            },
            "trajectory_points": {
                "pos_mean": pos_mean_traj.tolist(),
                "neg_mean": neg_mean_traj.tolist(),
            },
        }

        logger.info(f"  隐状态范数: pos {pos_start:.4f}→{pos_end:.4f} (增长率{pos_growth_rate:.4f}), "
                    f"neg {neg_start:.4f}→{neg_end:.4f} (增长率{neg_growth_rate:.4f})")
        logger.info(f"  速度范数: pos={vel_pos_mean:.4f}, neg={vel_neg_mean:.4f}")
        logger.info(f"  加速度范数: pos={accel_pos_mean:.4f}, neg={accel_neg_mean:.4f}")

        is_self_accelerating = pos_growth_rate > 0
        logger.info(f"  自加速判定: {'是' if is_self_accelerating else '否'} (增长率={pos_growth_rate:.4f})")

    return results


def main():
    config = load_config()
    config["data"]["num_stories"] = 800
    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("v8Deep", config["logging"]["log_dir"], "v8_deep_analysis.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")

    logger.info("[阶段0] 数据生成与模型训练...")
    gen = StoryGenerator(config, seed=config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size
    max_seq_len = config["model"]["max_seq_len"]

    all_train = train_pos + train_neg
    train_ds = StoryDataset(all_train, tokenizer, max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(
        StoryDataset(val_pos + val_neg, tokenizer, max_seq_len),
        batch_size=config["training"]["batch_size"],
    )

    from causal_gauge_field.newton_experiment import NewtonTrainer
    trainer = NewtonTrainer(config)
    newton_model, potential = trainer.train_newton(
        train_loader, val_loader, epochs=20, lambda_accel=5.0, lambda_barrier=1.0
    )

    logger.info("[阶段1] 提取隐状态轨迹+残差...")
    test_stories = test_pos[:40] + test_neg[:40]
    pos_hidden, neg_hidden, pos_residuals, neg_residuals = extract_hidden_with_residuals(
        newton_model, test_stories, tokenizer, max_seq_len, device
    )
    logger.info(f"  正例: {len(pos_hidden)} 轨迹, {len(pos_residuals)} 残差")
    logger.info(f"  负例: {len(neg_hidden)} 轨迹, {len(neg_residuals)} 残差")

    logger.info("[阶段2] 任务3.1: Proxy B vs Proxy D...")
    task31_results = task_3_1_proxy_b_analysis(
        pos_hidden, neg_hidden, pos_residuals, neg_residuals, newton_model, logger
    )

    logger.info("[阶段3] 任务3.2: α*物理来源分析...")
    task32_results = task_3_2_alpha_origin(
        pos_hidden, neg_hidden, newton_model, config, logger
    )

    logger.info("[阶段4] 任务3.3: 负阻尼验证...")
    task33_results = task_3_3_negative_damping(pos_hidden, neg_hidden, logger)

    logger.info("[阶段5] v8.0 Proxy D 对照验证...")
    analyzer = ActiveForceAnalyzer(mass=1.0, friction=0.01)
    v8_results = analyzer.full_analysis(pos_hidden, neg_hidden, method="D", alpha=1.1)
    v8_cp = v8_results.get("corrected_constraint_power", {})
    logger.info(f"  v8.0 α=1.1: pos_Pc={v8_cp.get('pos_Pc_mean', 'N/A'):.4f}, "
                f"neg_Pc={v8_cp.get('neg_Pc_mean', 'N/A'):.4f}, "
                f"verdict={v8_results.get('overall_verdict', {}).get('verdict', 'N/A')}")

    output_dir = Path(config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v8.0 Deep Analysis (Tasks 3.1-3.3)",
        "task_3_1_proxy_b": task31_results,
        "task_3_2_alpha_origin": task32_results,
        "task_3_3_negative_damping": task33_results,
        "v8_baseline": v8_results,
    }
    report_path = output_dir / "v8_deep_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    md_lines = [
        "# GUIT-TRT v8.0 深度分析报告 (Tasks 3.1-3.3)",
        f"\n生成时间: {datetime.now().isoformat()}",
        "\n---\n",
        "## 任务3.1: Proxy B(残差和) vs Proxy D(α·ḣ)\n",
    ]
    if task31_results:
        md_lines.append(f"- Pearson相关: pos={task31_results.get('pearson_corr_pos', 'N/A'):.4f}, "
                        f"neg={task31_results.get('pearson_corr_neg', 'N/A'):.4f}")
        md_lines.append(f"- 余弦相似度: pos={task31_results.get('cosine_sim_pos_mean', 'N/A'):.4f}, "
                        f"neg={task31_results.get('cosine_sim_neg_mean', 'N/A'):.4f}")
        md_lines.append(f"- α估计(均值比): pos={task31_results.get('alpha_from_mean_pos', 'N/A'):.4f}, "
                        f"neg={task31_results.get('alpha_from_mean_neg', 'N/A'):.4f}")
        md_lines.append(f"- 范数比(res/vel): {task31_results.get('norm_ratio_pos', 'N/A'):.4f}")
    else:
        md_lines.append("- 数据不足")

    md_lines.append("\n---\n")
    md_lines.append("## 任务3.2: α*≈1.1物理来源\n")
    if task32_results:
        arch = task32_results.get("architecture", {})
        proj = task32_results.get("projection_matrix", {})
        md_lines.append(f"- 架构: d_model={arch.get('d_model')}, base_dim={arch.get('base_dim')}, "
                        f"n_layers={arch.get('n_layers')}")
        md_lines.append(f"- 投影矩阵: ‖W‖_F={proj.get('W_frobenius_norm', 'N/A'):.4f}, "
                        f"σ_max={proj.get('W_spectral_norm', 'N/A'):.4f}")
        alpha_est = task32_results.get("alpha_local_estimates", {})
        if alpha_est:
            md_lines.append(f"- α局部估计: mean={alpha_est.get('mean', 'N/A'):.4f}, "
                            f"std={alpha_est.get('std', 'N/A'):.4f}")

    md_lines.append("\n---\n")
    md_lines.append("## 任务3.3: 负阻尼验证\n")
    if task33_results:
        hn = task33_results.get("hidden_norm_trajectory", {})
        md_lines.append(f"- 隐状态范数: pos {hn.get('pos_start', 'N/A'):.4f}→{hn.get('pos_end', 'N/A'):.4f} "
                        f"(增长率{hn.get('pos_growth_rate', 'N/A'):.4f})")
        md_lines.append(f"- 隐状态范数: neg {hn.get('neg_start', 'N/A'):.4f}→{hn.get('neg_end', 'N/A'):.4f} "
                        f"(增长率{hn.get('neg_growth_rate', 'N/A'):.4f})")
        md_lines.append(f"- 自加速: {'是' if hn.get('pos_growing') else '否'}")
        vn = task33_results.get("velocity_norms", {})
        md_lines.append(f"- 速度范数: pos={vn.get('pos_mean', 'N/A'):.4f}, neg={vn.get('neg_mean', 'N/A'):.4f}")
        an = task33_results.get("acceleration_norms", {})
        md_lines.append(f"- 加速度范数: pos={an.get('pos_mean', 'N/A'):.4f}, neg={an.get('neg_mean', 'N/A'):.4f}")

    md_path = output_dir / "v8_deep_analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return report


if __name__ == "__main__":
    main()