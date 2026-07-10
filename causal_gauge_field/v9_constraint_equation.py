import sys
import json
import torch
import torch.nn as nn
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


def compute_alpha_star(pos_hidden, gamma=0.01):
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
        return None
    return float(np.mean(alpha_estimates))


def extract_constraint_equations(pos_hidden, neg_hidden, alpha_star, gamma=0.01, mass=1.0):
    """提取约束方程的显式形式

    核心思路:
    1. F_c = m·ḧ + (γ-α*)·ḣ, 且 F_c ⊥ ḣ (P_c ≈ 0)
    2. 对F_c做SVD, 提取速度空间的零空间基 (法向量n_i)
    3. 约束方程: C_i(h, ḣ) = n_i^T · ḣ = 0
    4. 检验: n_i是否跨轨迹稳定, 是否可以参数化为h的函数
    """
    F_c_pos_all = []
    vel_pos_all = []
    h_pos_all = []
    F_c_neg_all = []
    vel_neg_all = []

    for h in pos_hidden:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h = h.unsqueeze(0)
        vel = h[:, 1:, :] - h[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))
        F_c = mass * acc[:, :min_t, :] + (gamma - alpha_star) * v_for[:, :min_t, :]
        F_c_pos_all.append(F_c.reshape(-1, F_c.size(-1)))
        vel_pos_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))
        h_mid = h[:, 1:-1, :] if h.size(1) > 2 else h[:, :1, :]
        h_trim = h_mid[:, :min_t, :]
        h_pos_all.append(h_trim.reshape(-1, h_trim.size(-1)))

    for h in neg_hidden:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h = h.unsqueeze(0)
        vel = h[:, 1:, :] - h[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))
        F_c = mass * acc[:, :min_t, :] + (gamma - alpha_star) * v_for[:, :min_t, :]
        F_c_neg_all.append(F_c.reshape(-1, F_c.size(-1)))
        vel_neg_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))

    if not F_c_pos_all:
        return None

    F_c_pos = torch.cat(F_c_pos_all, dim=0).numpy()
    vel_pos = torch.cat(vel_pos_all, dim=0).numpy()
    h_pos = torch.cat(h_pos_all, dim=0).numpy()
    F_c_neg = torch.cat(F_c_neg_all, dim=0).numpy() if F_c_neg_all else None
    vel_neg = torch.cat(vel_neg_all, dim=0).numpy() if vel_neg_all else None

    d = F_c_pos.shape[1]

    # === 第一步: 提取速度空间的零空间基 (约束法向量) ===
    # 对速度矩阵做SVD, 零空间 = 约束法向量
    U_vel, S_vel, Vt_vel = np.linalg.svd(vel_pos, full_matrices=True)

    total_var_vel = (S_vel**2).sum()
    cumulative_vel = np.cumsum(S_vel**2) / total_var_vel if total_var_vel > 0 else np.zeros_like(S_vel)
    r_vel = int(np.searchsorted(cumulative_vel, 0.95) + 1)

    # 零空间基: Vt_vel[r_vel:] 的行向量
    null_space_basis = Vt_vel[r_vel:]
    n_constraints = d - r_vel

    # === 第二步: 验证 n_i^T · ḣ ≈ 0 ===
    constraint_violations_pos = []
    for i in range(n_constraints):
        n_i = null_space_basis[i]
        violations = vel_pos @ n_i
        constraint_violations_pos.append({
            "index": i,
            "mean_violation": float(np.mean(np.abs(violations))),
            "max_violation": float(np.max(np.abs(violations))),
            "std_violation": float(np.std(violations)),
            "rms_violation": float(np.sqrt(np.mean(violations**2))),
        })

    constraint_violations_neg = []
    if vel_neg is not None:
        for i in range(n_constraints):
            n_i = null_space_basis[i]
            violations = vel_neg @ n_i
            constraint_violations_neg.append({
                "index": i,
                "mean_violation": float(np.mean(np.abs(violations))),
                "max_violation": float(np.max(np.abs(violations))),
                "rms_violation": float(np.sqrt(np.mean(violations**2))),
            })

    # === 第三步: 检验F_c是否在零空间基张成的子空间中 ===
    # F_c应该可以被null_space_basis的行向量线性表示
    F_c_proj_residuals = []
    for i in range(min(n_constraints, 10)):
        n_i = null_space_basis[i]
        proj_coeff = F_c_pos @ n_i
        F_c_proj_residuals.append({
            "constraint_idx": i,
            "mean_projection": float(np.mean(proj_coeff)),
            "std_projection": float(np.std(proj_coeff)),
            "norm_projection": float(np.sqrt(np.mean(proj_coeff**2))),
        })

    # === 第四步: 法向量与h的相关性 (参数化检验) ===
    # 检验 n_i^T · ḣ 是否与 h 的某些维度相关
    # 如果约束是完整的(holonomic), n_i 应该是 h 的函数
    # 如果约束是非完整的(nonholonomic), n_i 可能依赖于 ḣ
    h_vel_correlation = []
    for i in range(min(n_constraints, 5)):
        n_i = null_space_basis[i]
        violation_pos = vel_pos @ n_i
        for j in range(min(d, 8)):
            h_j = h_pos[:, j]
            if np.std(h_j) < 1e-10 or np.std(violation_pos) < 1e-10:
                continue
            corr, p_val = stats.pearsonr(h_j, violation_pos)
            h_vel_correlation.append({
                "constraint_idx": i,
                "h_dim": j,
                "correlation": float(corr),
                "p_value": float(p_val),
            })

    # === 第五步: 约束方程的线性参数化尝试 ===
    # 假设 C_i(h, ḣ) = n_i(h)^T · ḣ = 0
    # 线性近似: n_i(h) ≈ A_i · h + b_i
    # 则 C_i = (A_i · h + b_i)^T · ḣ = 0
    # 用最小二乘拟合 A_i, b_i
    linear_param_results = []
    for i in range(min(n_constraints, 5)):
        n_i = null_space_basis[i]
        target = vel_pos @ n_i  # 应该≈0

        # 构造特征: h ⊗ ḣ 的外积展开 (h_j * v_k)
        # 太高维, 改用简化版: n_i^T · ḣ ≈ w^T · h + b
        # 这等价于检验约束违反是否可以用h线性预测
        from numpy.linalg import lstsq
        A = np.column_stack([h_pos, np.ones(h_pos.shape[0])])
        result = lstsq(A, target, rcond=None)
        coeffs = result[0]
        residual = target - A @ coeffs
        rms_residual = float(np.sqrt(np.mean(residual**2)))
        rms_target = float(np.sqrt(np.mean(target**2)))
        r_squared = 1.0 - rms_residual**2 / (rms_target**2 + 1e-10)

        linear_param_results.append({
            "constraint_idx": i,
            "rms_residual": rms_residual,
            "rms_target": rms_target,
            "r_squared": float(r_squared),
            "n_nonzero_coeffs": int(np.sum(np.abs(coeffs) > 0.01 * np.max(np.abs(coeffs)))),
        })

    # === 第六步: 非完整约束判别 ===
    # 非完整约束: C_i(h, ḣ) = n_i^T · ḣ = 0, 但不存在 f_i(h) 使得 ∇f_i = n_i
    # 检验方法: Frobenius可积性条件
    # 如果 n_i = ∇f_i, 则 ∂n_i^k/∂h^j = ∂n_i^j/∂h^k (Schwarz定理)
    # 用有限差分近似检验
    nonholonomic_test = []
    for i in range(min(n_constraints, 3)):
        n_i = null_space_basis[i]
        # 检验: n_i 是否随 h 变化
        # 将h_pos分成若干bin, 检验每个bin中n_i^T·ḣ的均值是否不同
        n_bins = 5
        h_norm = np.linalg.norm(h_pos, axis=1)
        bin_edges = np.percentile(h_norm, np.linspace(0, 100, n_bins + 1))
        bin_violations = []
        for b in range(n_bins):
            mask = (h_norm >= bin_edges[b]) & (h_norm < bin_edges[b + 1])
            if mask.sum() < 5:
                continue
            violations = vel_pos[mask] @ n_i
            bin_violations.append(float(np.mean(np.abs(violations))))

        is_nonholonomic = len(bin_violations) >= 2 and np.std(bin_violations) > 0.1 * np.mean(bin_violations)

        nonholonomic_test.append({
            "constraint_idx": i,
            "bin_violations": bin_violations,
            "is_nonholonomic": is_nonholonomic,
            "variation_coefficient": float(np.std(bin_violations) / (np.mean(bin_violations) + 1e-10)),
        })

    # === 第七步: pos vs neg 约束违反对比 ===
    pos_neg_comparison = []
    if vel_neg is not None:
        for i in range(min(n_constraints, 10)):
            n_i = null_space_basis[i]
            v_pos = vel_pos @ n_i
            v_neg = vel_neg @ n_i
            t_stat, p_val = stats.ttest_ind(np.abs(v_pos), np.abs(v_neg), equal_var=False)
            pos_neg_comparison.append({
                "constraint_idx": i,
                "pos_mean_abs_violation": float(np.mean(np.abs(v_pos))),
                "neg_mean_abs_violation": float(np.mean(np.abs(v_neg))),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "pos_lower": bool(np.mean(np.abs(v_pos)) < np.mean(np.abs(v_neg))),
            })

    return {
        "n_total_dims": d,
        "vel_effective_rank": r_vel,
        "n_constraints": n_constraints,
        "null_space_basis_shape": list(null_space_basis.shape),
        "constraint_violations_pos": constraint_violations_pos,
        "constraint_violations_neg": constraint_violations_neg,
        "F_c_projection_residuals": F_c_proj_residuals,
        "h_vel_correlation_top": sorted(h_vel_correlation, key=lambda x: abs(x["correlation"]), reverse=True)[:10],
        "linear_parameterization": linear_param_results,
        "nonholonomic_test": nonholonomic_test,
        "pos_neg_comparison": pos_neg_comparison,
        "null_space_basis_top3": null_space_basis[:min(3, n_constraints)].tolist(),
        "singular_values_vel": S_vel[:min(20, len(S_vel))].tolist(),
    }


def main():
    base_config = load_config()
    base_config["data"]["num_stories"] = 600
    torch.manual_seed(base_config["project"]["seed"])
    np.random.seed(base_config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("v9ConstraintEq", base_config["logging"]["log_dir"], "v9_constraint_equation.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")

    tokenizer = NPNWTokenizer()
    base_config["model"]["vocab_size"] = tokenizer.vocab_size

    logger.info("[阶段0] 生成共享数据集...")
    gen = StoryGenerator(base_config, seed=base_config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=600, neg_per_positive=base_config["data"]["neg_per_positive"],
    )

    model_configs = [
        {"name": "baseline_2L", "n_layers": 2, "d_model": 128, "base_dim": 32, "d_ff": 512, "n_heads": 4},
        {"name": "narrow_2L", "n_layers": 2, "d_model": 64, "base_dim": 16, "d_ff": 256, "n_heads": 2},
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

        test_stories = test_pos[:30] + test_neg[:30]
        pos_hidden, neg_hidden = extract_hidden_states(model, test_stories, tokenizer, max_seq_len, device)
        logger.info(f"  轨迹: pos={len(pos_hidden)}, neg={len(neg_hidden)}")

        if len(pos_hidden) < 5:
            logger.info(f"  轨迹不足，跳过")
            continue

        alpha_star = compute_alpha_star(pos_hidden, gamma=0.01)
        logger.info(f"  α* = {alpha_star:.4f}" if alpha_star else "  α* 估计失败")

        if not alpha_star:
            continue

        logger.info(f"  提取约束方程...")
        eq_results = extract_constraint_equations(pos_hidden, neg_hidden, alpha_star, gamma=0.01)
        if eq_results:
            logger.info(f"    总维度: {eq_results['n_total_dims']}")
            logger.info(f"    速度有效秩: {eq_results['vel_effective_rank']}")
            logger.info(f"    约束方程数: {eq_results['n_constraints']}")
            logger.info(f"    pos约束违反(RMS): " + ", ".join(
                f"C{i}={v['rms_violation']:.4f}" for i, v in enumerate(eq_results['constraint_violations_pos'][:5])
            ))
            if eq_results['constraint_violations_neg']:
                logger.info(f"    neg约束违反(RMS): " + ", ".join(
                    f"C{i}={v['rms_violation']:.4f}" for i, v in enumerate(eq_results['constraint_violations_neg'][:5])
                ))
            logger.info(f"    线性参数化R²: " + ", ".join(
                f"C{i}={v['r_squared']:.4f}" for i, v in enumerate(eq_results['linear_parameterization'][:5])
            ))
            logger.info(f"    非完整判别: " + ", ".join(
                f"C{i}={'非完整' if v['is_nonholonomic'] else '完整'}(CV={v['variation_coefficient']:.3f})"
                for i, v in enumerate(eq_results['nonholonomic_test'])
            ))
            if eq_results['pos_neg_comparison']:
                n_pos_lower = sum(1 for c in eq_results['pos_neg_comparison'] if c['pos_lower'])
                logger.info(f"    pos违反<neg违反: {n_pos_lower}/{len(eq_results['pos_neg_comparison'])}")

        all_results[name] = {
            "config": mc,
            "alpha_star": alpha_star,
            "constraint_equations": eq_results,
        }

    # === 跨模型法向量稳定性检验 ===
    if len(all_results) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("跨模型法向量稳定性检验")
        logger.info(f"{'='*60}")

        models = list(all_results.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                eq1 = all_results[m1].get("constraint_equations")
                eq2 = all_results[m2].get("constraint_equations")
                if eq1 is None or eq2 is None:
                    continue

                basis1 = np.array(eq1["null_space_basis_top3"])
                basis2 = np.array(eq2["null_space_basis_top3"])

                if basis1.shape[1] != basis2.shape[1]:
                    logger.info(f"  {m1} vs {m2}: 维度不同({basis1.shape[1]} vs {basis2.shape[1]}), 跳过")
                    continue

                min_rows = min(basis1.shape[0], basis2.shape[0])
                similarities = []
                for k in range(min_rows):
                    cos_sim = np.abs(np.dot(basis1[k], basis2[k])) / (
                        np.linalg.norm(basis1[k]) * np.linalg.norm(basis2[k]) + 1e-10
                    )
                    similarities.append(float(cos_sim))

                avg_sim = np.mean(similarities) if similarities else 0
                logger.info(f"  {m1} vs {m2}: 法向量余弦相似度 = {avg_sim:.4f}")
                for k, s in enumerate(similarities):
                    logger.info(f"    n_{k}: cos_sim = {s:.4f}")

    # === 保存报告 ===
    output_dir = Path(base_config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v9.0 Constraint Equation Extraction",
        "results": all_results,
    }
    report_path = output_dir / "v9_constraint_equation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    # === Markdown报告 ===
    md_lines = [
        "# GUIT-TRT v9.0 约束方程显式提取报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        "\n---\n",
    ]

    for name, r in all_results.items():
        eq = r.get("constraint_equations")
        if eq is None:
            continue
        mc = r["config"]
        md_lines.append(f"## 模型: {name} (L={mc['n_layers']}, B={mc['base_dim']})\n")
        md_lines.append(f"- α* = {r['alpha_star']:.4f}")
        md_lines.append(f"- 总维度: {eq['n_total_dims']}")
        md_lines.append(f"- 速度有效秩: {eq['vel_effective_rank']}")
        md_lines.append(f"- 约束方程数: {eq['n_constraints']}")
        md_lines.append(f"- 约束形式: C_i(h, ḣ) = n_i^T · ḣ = 0, i = 1,...,{eq['n_constraints']}\n")

        md_lines.append("### pos约束违反检验\n")
        md_lines.append("| 约束i | RMS违反 | 最大违反 | 均值违反 |")
        md_lines.append("|-------|---------|---------|---------|")
        for v in eq["constraint_violations_pos"][:10]:
            md_lines.append(f"| {v['index']} | {v['rms_violation']:.6f} | {v['max_violation']:.6f} | {v['mean_violation']:.6f} |")

        if eq["constraint_violations_neg"]:
            md_lines.append("\n### neg约束违反检验\n")
            md_lines.append("| 约束i | RMS违反 | 均值违反 |")
            md_lines.append("|-------|---------|---------|")
            for v in eq["constraint_violations_neg"][:10]:
                md_lines.append(f"| {v['index']} | {v['rms_violation']:.6f} | {v['mean_violation']:.6f} |")

        md_lines.append("\n### 线性参数化 (n_i^T·ḣ ≈ w^T·h + b)\n")
        md_lines.append("| 约束i | R² | 非零系数数 | RMS残差 |")
        md_lines.append("|-------|-----|----------|---------|")
        for v in eq["linear_parameterization"]:
            md_lines.append(f"| {v['constraint_idx']} | {v['r_squared']:.4f} | {v['n_nonzero_coeffs']} | {v['rms_residual']:.6f} |")

        md_lines.append("\n### 非完整约束判别\n")
        md_lines.append("| 约束i | 类型 | 变异系数 | bin违反 |")
        md_lines.append("|-------|------|---------|---------|")
        for v in eq["nonholonomic_test"]:
            md_lines.append(f"| {v['constraint_idx']} | {'非完整' if v['is_nonholonomic'] else '完整'} | {v['variation_coefficient']:.3f} | {[f'{x:.4f}' for x in v['bin_violations']]} |")

        if eq["pos_neg_comparison"]:
            md_lines.append("\n### pos vs neg 约束违反对比\n")
            md_lines.append("| 约束i | pos RMS | neg RMS | p值 | pos更低 |")
            md_lines.append("|-------|---------|---------|-----|--------|")
            for v in eq["pos_neg_comparison"]:
                md_lines.append(f"| {v['constraint_idx']} | {v['pos_mean_abs_violation']:.6f} | {v['neg_mean_abs_violation']:.6f} | {v['p_value']:.4f} | {'✅' if v['pos_lower'] else '❌'} |")

        md_lines.append("\n---\n")

    md_path = output_dir / "v9_constraint_equation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return all_results


if __name__ == "__main__":
    main()