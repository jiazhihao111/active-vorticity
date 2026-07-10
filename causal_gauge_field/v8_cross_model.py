import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from copy import deepcopy

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


def train_model_with_config(config, train_loader, val_loader, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformer(config).to(device)
    potential = CausalPotential(config["model"].get("base_dim", config["model"]["d_model"])).to(device)
    accel_loss_fn = AccelerationLoss(margin=config["training"]["margin"])
    barrier_loss_fn = BarrierLoss(margin=0.5)
    all_params = list(model.parameters()) + list(potential.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config["training"]["learning_rate"])
    lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        model.train()
        potential.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, hidden = model(input_ids, attention_mask)
            lm_loss = lm_loss_fn(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                target_ids[:, 1:].contiguous().view(-1),
            )
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            accel_loss_val = torch.tensor(0.0, device=device)
            barrier_loss_val = torch.tensor(0.0, device=device)
            if pos_mask.any() and neg_mask.any():
                hidden_pos = hidden[pos_mask]
                hidden_neg = hidden[neg_mask]
                if hidden_pos.size(1) >= 3 and hidden_neg.size(1) >= 3:
                    accel_loss_val, _ = accel_loss_fn(hidden_pos, hidden_neg)
            total_loss = lm_loss + 5.0 * accel_loss_val + barrier_loss_val
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, config["training"]["grad_clip"])
            optimizer.step()

    return model, potential


def extract_hidden_states(model, stories, tokenizer, max_seq_len, device):
    pos_hidden = []
    neg_hidden = []
    model.eval()
    with torch.no_grad():
        for story in stories:
            steps_data = [{"state": s.state, "action": s.action, "causal_labels": s.causal_labels} for s in story.steps]
            token_ids = tokenizer.encode_story(steps_data)
            max_len = max_seq_len - 1
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            if len(token_ids) < 4:
                continue
            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(device)
            _, hidden = model(input_ids)
            h = hidden[0].cpu()
            if story.is_positive:
                pos_hidden.append(h)
            else:
                neg_hidden.append(h)
    return pos_hidden, neg_hidden


def compute_alpha_star(pos_hidden, neg_hidden, gamma=0.01):
    """从轨迹数据直接估计α*"""
    alpha_estimates = []
    for h in pos_hidden:
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
            alpha_local = P_raw.mean().item() / P_active.mean().item()
            alpha_estimates.append(alpha_local)

    if not alpha_estimates:
        return None, None
    return float(np.mean(alpha_estimates)), float(np.std(alpha_estimates))


def extract_constraint_subspace(pos_hidden, neg_hidden, alpha_star, gamma=0.01):
    """从F_c向量提取约束子空间

    F_c = m·ḧ + γ·ḣ - α*·ḣ = m·ḧ + (γ-α*)·ḣ
    如果P_c = F_c·ḣ ≈ 0，则F_c ⊥ ḣ
    约束子空间 = F_c张成的子空间
    """
    F_c_all = []
    vel_all = []

    for h in pos_hidden:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h = h.unsqueeze(0)
        vel = h[:, 1:, :] - h[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))
        F_c = acc[:, :min_t, :] + (gamma - alpha_star) * v_for[:, :min_t, :]
        F_c_all.append(F_c.reshape(-1, F_c.size(-1)))
        vel_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))

    if not F_c_all:
        return None

    F_c_cat = torch.cat(F_c_all, dim=0)
    vel_cat = torch.cat(vel_all, dim=0)

    F_c_np = F_c_cat.numpy()
    vel_np = vel_cat.numpy()

    if F_c_np.shape[0] < 2 or F_c_np.shape[1] < 2:
        return None

    cov_fc = np.cov(F_c_np.T)
    eigenvalues_fc = np.sort(np.abs(np.linalg.eigvalsh(cov_fc)))[::-1]
    total_var = eigenvalues_fc.sum()
    if total_var < 1e-10:
        return None
    cumulative = np.cumsum(eigenvalues_fc) / total_var
    effective_rank_fc = int(np.searchsorted(cumulative, 0.95) + 1)

    orthogonality = np.mean(np.sum(F_c_np * vel_np, axis=-1))

    U, S, Vt = np.linalg.svd(F_c_np, full_matrices=False)
    top_components = Vt[:min(5, Vt.shape[0]), :]

    vel_cov = np.cov(vel_np.T)
    eigenvalues_vel = np.sort(np.abs(np.linalg.eigvalsh(vel_cov)))[::-1]
    total_var_vel = eigenvalues_vel.sum()
    if total_var_vel < 1e-10:
        cumulative_vel = np.zeros_like(eigenvalues_vel)
    else:
        cumulative_vel = np.cumsum(eigenvalues_vel) / total_var_vel
    effective_rank_vel = int(np.searchsorted(cumulative_vel, 0.95) + 1) if total_var_vel > 1e-10 else 0

    cos_angles = []
    for i in range(min(3, top_components.shape[0])):
        for j in range(min(3, top_components.shape[0])):
            cos_a = np.abs(np.dot(top_components[i], top_components[j]))
            cos_angles.append(cos_a)

    return {
        "F_c_effective_rank": effective_rank_fc,
        "F_c_eigenvalues_top5": eigenvalues_fc[:5].tolist(),
        "F_c_orthogonality_to_vel": float(orthogonality),
        "F_c_singular_values_top5": S[:5].tolist(),
        "vel_effective_rank": effective_rank_vel,
        "vel_eigenvalues_top5": eigenvalues_vel[:5].tolist(),
        "F_c_norm_mean": float(np.mean(np.linalg.norm(F_c_np, axis=-1))),
        "vel_norm_mean": float(np.mean(np.linalg.norm(vel_np, axis=-1))),
        "F_c_to_vel_norm_ratio": float(np.mean(np.linalg.norm(F_c_np, axis=-1)) / (np.mean(np.linalg.norm(vel_np, axis=-1)) + 1e-10)),
        "constraint_subspace_dim": effective_rank_fc,
        "total_dim": F_c_np.shape[1],
        "compression_ratio": float(effective_rank_fc / F_c_np.shape[1]),
    }


def main():
    base_config = load_config()
    base_config["data"]["num_stories"] = 600
    torch.manual_seed(base_config["project"]["seed"])
    np.random.seed(base_config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("v8Cross", base_config["logging"]["log_dir"], "v8_cross_model.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")

    tokenizer = NPNWTokenizer()
    base_config["model"]["vocab_size"] = tokenizer.vocab_size

    logger.info("[阶段0] 生成共享数据集...")
    gen = StoryGenerator(base_config, seed=base_config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=600, neg_per_positive=base_config["data"]["neg_per_positive"],
    )

    model_configs = [
        {"name": "baseline_2L_128D_32B", "n_layers": 2, "d_model": 128, "base_dim": 32, "d_ff": 512, "n_heads": 4},
        {"name": "deep_4L_128D_32B", "n_layers": 4, "d_model": 128, "base_dim": 32, "d_ff": 512, "n_heads": 4},
        {"name": "shallow_1L_128D_32B", "n_layers": 1, "d_model": 128, "base_dim": 32, "d_ff": 512, "n_heads": 4},
        {"name": "wide_2L_256D_64B", "n_layers": 2, "d_model": 256, "base_dim": 64, "d_ff": 1024, "n_heads": 8},
        {"name": "narrow_2L_64D_16B", "n_layers": 2, "d_model": 64, "base_dim": 16, "d_ff": 256, "n_heads": 2},
    ]

    all_results = {}

    for mc in model_configs:
        name = mc["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"模型: {name} (L={mc['n_layers']}, D={mc['d_model']}, B={mc['base_dim']})")
        logger.info(f"{'='*60}")

        config = deepcopy(base_config)
        config["model"]["n_layers"] = mc["n_layers"]
        config["model"]["d_model"] = mc["d_model"]
        config["model"]["base_dim"] = mc["base_dim"]
        config["model"]["d_ff"] = mc["d_ff"]
        config["model"]["n_heads"] = mc["n_heads"]
        config["model"]["vocab_size"] = tokenizer.vocab_size

        max_seq_len = config["model"]["max_seq_len"]

        all_train = train_pos + train_neg
        train_ds = StoryDataset(all_train, tokenizer, max_seq_len)
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(
            StoryDataset(val_pos + val_neg, tokenizer, max_seq_len),
            batch_size=config["training"]["batch_size"],
        )

        logger.info(f"  训练模型...")
        model, potential = train_model_with_config(config, train_loader, val_loader, epochs=15)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数量: {n_params}")

        test_stories = test_pos[:30] + test_neg[:30]
        pos_hidden, neg_hidden = extract_hidden_states(model, test_stories, tokenizer, max_seq_len, device)
        logger.info(f"  轨迹: pos={len(pos_hidden)}, neg={len(neg_hidden)}")

        if len(pos_hidden) < 5:
            logger.info(f"  轨迹不足，跳过")
            continue

        logger.info(f"  估计α*...")
        alpha_mean, alpha_std = compute_alpha_star(pos_hidden, neg_hidden, gamma=0.01)
        logger.info(f"  α* = {alpha_mean:.4f} ± {alpha_std:.4f}" if alpha_mean else "  α* 估计失败")

        logger.info(f"  v8.0验证 (α=α*)...")
        if alpha_mean:
            analyzer = ActiveForceAnalyzer(mass=1.0, friction=0.01)
            v8_results = analyzer.full_analysis(pos_hidden, neg_hidden, method="D", alpha=alpha_mean)
            v8_cp = v8_results.get("corrected_constraint_power", {})
            v8_ed = v8_results.get("energy_decomposition", {})
            v8_overall = v8_results.get("overall_verdict", {})
            logger.info(f"    pos_Pc={v8_cp.get('pos_Pc_mean', 'N/A'):.4f}, "
                        f"neg_Pc={v8_cp.get('neg_Pc_mean', 'N/A'):.4f}, "
                        f"verdict={v8_overall.get('verdict', 'N/A')}")
        else:
            v8_results = {}

        logger.info(f"  提取约束子空间...")
        fc_results = extract_constraint_subspace(pos_hidden, neg_hidden, alpha_mean or 1.1, gamma=0.01)
        if fc_results:
            logger.info(f"    F_c有效秩: {fc_results['F_c_effective_rank']}/{fc_results['total_dim']} "
                        f"(压缩比={fc_results['compression_ratio']:.3f})")
            logger.info(f"    F_c⊥ḣ: {fc_results['F_c_orthogonality_to_vel']:.4f}")
            logger.info(f"    速度有效秩: {fc_results['vel_effective_rank']}")
            logger.info(f"    F_c/vel范数比: {fc_results['F_c_to_vel_norm_ratio']:.4f}")

        all_results[name] = {
            "config": mc,
            "n_params": n_params,
            "alpha_star_mean": alpha_mean,
            "alpha_star_std": alpha_std,
            "v8_results": v8_results,
            "constraint_subspace": fc_results,
        }

    logger.info(f"\n{'='*60}")
    logger.info("跨模型对比汇总")
    logger.info(f"{'='*60}")
    logger.info(f"{'模型':<30} {'参数量':>8} {'α*':>8} {'α*std':>8} {'pos_Pc':>8} {'neg_Pc':>8} {'判决':>12} {'F_c秩':>6} {'压缩比':>8}")
    logger.info("-" * 110)
    for name, r in all_results.items():
        mc = r["config"]
        v8_cp = r.get("v8_results", {}).get("corrected_constraint_power", {})
        v8_ov = r.get("v8_results", {}).get("overall_verdict", {})
        fc = r.get("constraint_subspace", {})
        logger.info(
            f"{name:<30} {r['n_params']:>8} "
            f"{r.get('alpha_star_mean', 'N/A') if r.get('alpha_star_mean') else 'N/A':>8} "
            f"{r.get('alpha_star_std', 'N/A') if r.get('alpha_star_std') else 'N/A':>8} "
            f"{v8_cp.get('pos_Pc_mean', 'N/A'):>8} "
            f"{v8_cp.get('neg_Pc_mean', 'N/A'):>8} "
            f"{v8_ov.get('verdict', 'N/A'):>12} "
            f"{fc.get('F_c_effective_rank', 'N/A') if fc else 'N/A':>6} "
            f"{fc.get('compression_ratio', 'N/A') if fc else 'N/A':>8}"
        )

    output_dir = Path(base_config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v8.0 Cross-Model + Constraint Extraction",
        "results": all_results,
    }
    report_path = output_dir / "v8_cross_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    md_lines = [
        "# GUIT-TRT v8.0 跨模型验证 + 约束子空间提取报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        "\n---\n",
        "## 跨模型α*对比\n",
        "| 模型 | 层数 | d_model | base_dim | 参数量 | α* | α*std | pos P_c | neg P_c | 判决 |",
        "|------|------|---------|----------|--------|-----|-------|---------|---------|------|",
    ]
    for name, r in all_results.items():
        mc = r["config"]
        v8_cp = r.get("v8_results", {}).get("corrected_constraint_power", {})
        v8_ov = r.get("v8_results", {}).get("overall_verdict", {})
        a_mean = r.get("alpha_star_mean")
        a_std = r.get("alpha_star_std")
        md_lines.append(
            f"| {name} | {mc['n_layers']} | {mc['d_model']} | {mc['base_dim']} | "
            f"{r['n_params']} | {f'{a_mean:.3f}' if a_mean else 'N/A'} | "
            f"{f'{a_std:.3f}' if a_std else 'N/A'} | "
            f"{v8_cp.get('pos_Pc_mean', 'N/A'):.4f} | "
            f"{v8_cp.get('neg_Pc_mean', 'N/A'):.4f} | "
            f"{v8_ov.get('verdict', 'N/A')} |"
        )

    md_lines.append("\n---\n")
    md_lines.append("## 约束子空间提取\n")
    md_lines.append("| 模型 | F_c有效秩 | 总维度 | 压缩比 | F_c⊥ḣ | vel有效秩 | F_c/vel范数比 |")
    md_lines.append("|------|----------|--------|--------|-------|----------|--------------|")
    for name, r in all_results.items():
        fc = r.get("constraint_subspace")
        if fc:
            md_lines.append(
                f"| {name} | {fc['F_c_effective_rank']} | {fc['total_dim']} | "
                f"{fc['compression_ratio']:.3f} | {fc['F_c_orthogonality_to_vel']:.4f} | "
                f"{fc['vel_effective_rank']} | {fc['F_c_to_vel_norm_ratio']:.4f} |"
            )

    md_lines.append("\n---\n")
    md_lines.append("## 关键发现\n")
    alpha_values = [r["alpha_star_mean"] for r in all_results.values() if r.get("alpha_star_mean")]
    if alpha_values:
        md_lines.append(f"- α*范围: [{min(alpha_values):.3f}, {max(alpha_values):.3f}]")
        md_lines.append(f"- α*均值: {np.mean(alpha_values):.3f}")
        md_lines.append(f"- α*与层数的关联: " + ", ".join(
            f"L={all_results[n]['config']['n_layers']}→α*={all_results[n]['alpha_star_mean']:.3f}"
            for n in all_results if all_results[n].get("alpha_star_mean")
        ))

    fc_ranks = [r["constraint_subspace"]["F_c_effective_rank"] for r in all_results.values()
                if r.get("constraint_subspace")]
    if fc_ranks:
        md_lines.append(f"- F_c有效秩范围: [{min(fc_ranks)}, {max(fc_ranks)}]")
        md_lines.append(f"- 约束子空间压缩比: " + ", ".join(
            f"{n}→{all_results[n]['constraint_subspace']['compression_ratio']:.3f}"
            for n in all_results if all_results[n].get("constraint_subspace")
        ))

    md_path = output_dir / "v8_cross_model_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return all_results


if __name__ == "__main__":
    main()