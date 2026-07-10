import sys
import json
import torch
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
from causal_gauge_field.newton.hamiltonian import HamiltonianVerifier
from causal_gauge_field.newton.constraint_power import ConstraintPowerAnalyzer
from causal_gauge_field.newton.active_force_analyzer import ActiveForceAnalyzer
from causal_gauge_field.experiments.trainer import StoryDataset
from torch.utils.data import DataLoader


def extract_hidden_states(model, stories, tokenizer, max_seq_len, device):
    pos_hidden = []
    neg_hidden = []
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
            _, hidden = model(input_ids)
            h = hidden[0]
            if story.is_positive:
                pos_hidden.append(h)
            else:
                neg_hidden.append(h)
    return pos_hidden, neg_hidden


def main():
    config = load_config()
    config["data"]["num_stories"] = 800
    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("v8Main", config["logging"]["log_dir"], "v8_active_force_experiment.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("GUIT-TRT v8.0: 带非保守主动力的拉格朗日系统分析")
    logger.info("=" * 60)

    logger.info("[阶段0] 数据生成与模型训练...")
    gen = StoryGenerator(config, seed=config["project"]["seed"])
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size
    hidden_dim = config["model"].get("base_dim", config["model"]["d_model"])
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

    logger.info("  训练牛顿版本模型...")
    newton_model, potential = trainer.train_newton(
        train_loader, val_loader, epochs=20, lambda_accel=5.0, lambda_barrier=1.0
    )

    logger.info("[阶段1] 提取隐状态轨迹...")
    test_stories = test_pos[:40] + test_neg[:40]
    pos_hidden, neg_hidden = extract_hidden_states(
        newton_model, test_stories, tokenizer, max_seq_len, device
    )
    logger.info(f"  正例轨迹: {len(pos_hidden)}, 负例轨迹: {len(neg_hidden)}")

    logger.info("[阶段2] v7.0对照: 原始约束力做功分析...")
    v7_analyzer = ConstraintPowerAnalyzer(mass=1.0, friction=0.01)
    v7_results = v7_analyzer.full_analysis(pos_hidden, neg_hidden)
    v7_cp = v7_results.get("constraint_power", {})
    logger.info(f"  v7.0 P(t): pos_mean={v7_cp.get('pos_mean', 'N/A'):.4f}, "
                f"neg_mean={v7_cp.get('neg_mean', 'N/A'):.4f}, "
                f"verdict={v7_cp.get('verdict', 'N/A')}")

    logger.info("[阶段3] v8.0核心: 扣除主动力后的约束力分析...")
    gamma_values = [0.01]
    alpha_values = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
    methods = ["D"]
    all_results = {}

    for method in methods:
        for gamma in gamma_values:
            for alpha in alpha_values:
                key = f"method_{method}_gamma_{gamma}_alpha_{alpha}"
                analyzer = ActiveForceAnalyzer(mass=1.0, friction=gamma)
                results = analyzer.full_analysis(
                    pos_hidden, neg_hidden,
                    method=method,
                    potential=potential if method == "C" else None,
                    alpha=alpha,
                )
                all_results[key] = results

                cp = results.get("corrected_constraint_power", {})
                ed = results.get("energy_decomposition", {})
                af = results.get("active_force_alignment", {})
                overall = results.get("overall_verdict", {})

                logger.info(f"  方法={method}, γ={gamma}, α={alpha}:")
                logger.info(f"    修正P_c: pos={cp.get('pos_Pc_mean', 'N/A'):.4f}, "
                            f"neg={cp.get('neg_Pc_mean', 'N/A'):.4f}, "
                            f"p={cp.get('pos_vs_neg_p', 'N/A'):.4f}")
                logger.info(f"    能量分解: P_raw={ed.get('pos_P_raw_mean', 'N/A'):.4f}, "
                            f"P_active={ed.get('pos_P_active_mean', 'N/A'):.4f}, "
                            f"P_constraint={ed.get('pos_P_constraint_mean', 'N/A'):.4f}")
                logger.info(f"    总判决: {overall.get('verdict', 'N/A')} "
                            f"({overall.get('support_count', 0)}/{overall.get('total_count', 0)})")

    best_key = min(
        all_results.keys(),
        key=lambda k: abs(all_results[k].get("corrected_constraint_power", {}).get("pos_Pc_mean", 999))
    )
    logger.info(f"\n  最佳配置: {best_key} (修正后pos P_c最接近0)")

    logger.info("[阶段4] 牛顿版本验证（对照）...")
    verifier = HamiltonianVerifier(potential)
    newton_results = verifier.full_verification(pos_hidden, neg_hidden)
    for key in ["hamiltonian_conservation", "force_alignment", "potential_barrier"]:
        if key in newton_results:
            v = newton_results[key]
            logger.info(f"  {key}: verdict={v.get('verdict', 'N/A')}")

    output_dir = Path(config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v8.0 Lagrangian with Active Force",
        "v7_baseline": v7_results,
        "v8_results": all_results,
        "newton_baseline": newton_results,
    }
    report_path = output_dir / "v8_active_force_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    best_results = all_results[best_key]
    cp_best = best_results.get("corrected_constraint_power", {})
    ed_best = best_results.get("energy_decomposition", {})
    af_best = best_results.get("active_force_alignment", {})
    overall_best = best_results.get("overall_verdict", {})

    md_lines = [
        "# GUIT-TRT v8.0 带非保守主动力的拉格朗日系统分析报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n## 框架",
        f"\n> **v8.0: 扣除非保守主动力F_active后，合法轨迹约束力做功P_c(t)≈0**",
        f"\n> 运动方程: m·ḧ + γ·ḣ + ∇V = F_active + ξ(t)",
        f"\n> 修正约束力: F_c = m·ḧ + γ·ḣ - F_active",
        "\n---\n",
        "## v7.0→v8.0 关键修正\n",
        "| 版本 | 假说 | P(t)定义 | v7.0结果 |",
        "|------|------|----------|----------|",
        "| v7.0 | 理想约束，P(t)≈0 | (m·ḧ+γ·ḣ)·ḣ | ❌ P(t)>>0 |",
        "| v8.0 | 扣除主动力，P_c(t)≈0 | (m·ḧ+γ·ḣ-F_active)·ḣ | ? |",
        "\n---\n",
        "## 核心假说验证（最佳配置）\n",
        f"**配置**: {best_key}\n",
        "| 标准 | 结果 | 判决 |",
        "|------|------|------|",
    ]
    criteria = cp_best.get("criteria", {})
    for k, v in criteria.items():
        md_lines.append(f"| {k} | {'✅' if v else '❌'} | {'通过' if v else '未通过'} |")

    md_lines.append(f"\n**修正P_c统计**: pos_mean={cp_best.get('pos_Pc_mean', 'N/A'):.6f}, "
                    f"neg_mean={cp_best.get('neg_Pc_mean', 'N/A'):.6f}, "
                    f"p={cp_best.get('pos_vs_neg_p', 'N/A'):.4f}")
    md_lines.append(f"\n**偏度**: pos={cp_best.get('pos_Pc_skew', 'N/A'):.3f}, "
                    f"neg={cp_best.get('neg_Pc_skew', 'N/A'):.3f}")

    md_lines.append("\n---\n")
    md_lines.append("## 能量分解\n")
    md_lines.append("| 分量 | 正例均值 | 负例均值 | 含义 |")
    md_lines.append("|------|----------|----------|------|")
    md_lines.append(f"| P_raw (v7.0总功率) | {ed_best.get('pos_P_raw_mean', 'N/A'):.4f} | "
                    f"{ed_best.get('neg_P_raw_mean', 'N/A'):.4f} | 原始残余力做功 |")
    md_lines.append(f"| P_active (主动力做功) | {ed_best.get('pos_P_active_mean', 'N/A'):.4f} | "
                    f"{ed_best.get('neg_P_active_mean', 'N/A'):.4f} | 非保守驱动力做功 |")
    md_lines.append(f"| P_constraint (约束力做功) | {ed_best.get('pos_P_constraint_mean', 'N/A'):.4f} | "
                    f"{ed_best.get('neg_P_constraint_mean', 'N/A'):.4f} | 扣除后约束力做功 |")
    md_lines.append(f"\n**主动力占比**: pos={ed_best.get('active_fraction_pos', 'N/A'):.4f}, "
                    f"neg={ed_best.get('active_fraction_neg', 'N/A'):.4f}")
    md_lines.append(f"\n**修正改善倍数**: {ed_best.get('correction_improvement', 'N/A'):.4f}")

    md_lines.append("\n---\n")
    md_lines.append("## 主动力对齐度\n")
    md_lines.append(f"正例: {af_best.get('pos_alignment_mean', 'N/A'):.4f}, "
                    f"负例: {af_best.get('neg_alignment_mean', 'N/A'):.4f}, "
                    f"p={af_best.get('pos_vs_neg_p', 'N/A'):.4f}, "
                    f"verdict={af_best.get('verdict', 'N/A')}")

    md_lines.append("\n---\n")
    md_lines.append(f"**总判决**: {overall_best.get('verdict', 'N/A')} "
                    f"({overall_best.get('support_count', 0)}/{overall_best.get('total_count', 0)})")

    md_path = output_dir / "v8_active_force_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return all_results


if __name__ == "__main__":
    main()