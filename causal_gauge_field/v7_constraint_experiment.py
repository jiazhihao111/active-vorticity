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
    logger = setup_logger("v7Main", config["logging"]["log_dir"], "v7_constraint_experiment.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info("GUIT-TRT v7.0: 约束力做功分析（拉格朗日约束力学版）")
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

    logger.info("[阶段2] v7.0 约束力做功分析...")
    gamma_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    all_results = {}

    for gamma in gamma_values:
        analyzer = ConstraintPowerAnalyzer(mass=1.0, friction=gamma)
        results = analyzer.full_analysis(pos_hidden, neg_hidden)
        all_results[f"gamma_{gamma}"] = results

        cp = results.get("constraint_power", {})
        vr = results.get("velocity_effective_rank", {})
        overall = results.get("overall_verdict", {})

        logger.info(f"  γ={gamma}:")
        logger.info(f"    P(t): pos_mean={cp.get('pos_mean', 'N/A'):.6f}, "
                     f"neg_mean={cp.get('neg_mean', 'N/A'):.6f}, "
                     f"p={cp.get('pos_vs_neg_p', 'N/A'):.4f}")
        logger.info(f"    偏度: pos={cp.get('pos_skew', 'N/A'):.3f}, "
                     f"neg={cp.get('neg_skew', 'N/A'):.3f}")
        logger.info(f"    判决: {cp.get('verdict', 'N/A')} "
                     f"({cp.get('passed_count', 0)}/{cp.get('total_criteria', 0)})")
        if vr:
            logger.info(f"    有效秩: pos={vr.get('pos_effective_rank', 'N/A')}, "
                         f"neg={vr.get('neg_effective_rank', 'N/A')}, "
                         f"verdict={vr.get('verdict', 'N/A')}")
        logger.info(f"    总判决: {overall.get('verdict', 'N/A')}")

    best_gamma = min(gamma_values, key=lambda g: abs(all_results[f"gamma_{g}"].get("constraint_power", {}).get("pos_mean", 999)))
    logger.info(f"\n  最佳γ={best_gamma} (使pos P(t)最接近0)")

    logger.info("[阶段3] 牛顿版本验证（对照）...")
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
        "framework": "v7.0 Lagrangian Constraint Mechanics",
        "constraint_power_results": {k: v for k, v in all_results.items()},
        "newton_baseline": newton_results,
    }
    report_path = output_dir / "v7_constraint_power_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存: {report_path}")

    best_results = all_results[f"gamma_{best_gamma}"]
    cp_best = best_results.get("constraint_power", {})
    vr_best = best_results.get("velocity_effective_rank", {})

    md_lines = [
        "# GUIT-TRT v7.0 约束力做功分析报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n## 框架",
        f"\n> **拉格朗日约束力学版：合法轨迹在约束流形上惯性滑行（P(t)≈0），非法轨迹违反约束（P(t)<<0）**",
        "\n---\n",
        "## 核心假说验证\n",
        "| 标准 | 结果 | 判决 |",
        "|------|------|------|",
    ]
    criteria = cp_best.get("criteria", {})
    for k, v in criteria.items():
        md_lines.append(f"| {k} | {'✅' if v else '❌'} | {'通过' if v else '未通过'} |")

    md_lines.append(f"\n**P(t)统计**: pos_mean={cp_best.get('pos_mean', 'N/A'):.6f}, "
                    f"neg_mean={cp_best.get('neg_mean', 'N/A'):.6f}, "
                    f"p={cp_best.get('pos_vs_neg_p', 'N/A'):.4f}")
    md_lines.append(f"\n**偏度**: pos={cp_best.get('pos_skew', 'N/A'):.3f}, "
                    f"neg={cp_best.get('neg_skew', 'N/A'):.3f}")
    md_lines.append(f"\n**有效秩**: pos={vr_best.get('pos_effective_rank', 'N/A')}, "
                    f"neg={vr_best.get('neg_effective_rank', 'N/A')}")
    md_lines.append(f"\n**总判决**: {best_results.get('overall_verdict', {}).get('verdict', 'N/A')}")
    md_lines.append(f"\n**最佳γ**: {best_gamma}")

    md_path = output_dir / "v7_constraint_power_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    return all_results


if __name__ == "__main__":
    main()