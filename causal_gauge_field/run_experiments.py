import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.models.memory_bank import CausalMemoryBank
from causal_gauge_field.models.gauge_field import GaugeField
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from causal_gauge_field.experiments.exp1_curvature_correlation import Experiment1
from causal_gauge_field.experiments.exp2_causal_loss_impact import Experiment2
from causal_gauge_field.experiments.exp3_memory_kernel import Experiment3
from causal_gauge_field.experiments.exp4_flatness import Experiment4
from causal_gauge_field.experiments.verdict import VerdictMatrix

from torch.utils.data import DataLoader


def main():
    config = load_config()
    config["data"]["num_stories"] = 500
    config["training"]["max_epochs"] = 15
    config["training"]["patience"] = 5
    logger = setup_logger("Main", config["logging"]["log_dir"], config["logging"]["log_file"])
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("因果规范场理论最小可行实验")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    logger.info("[阶段0] 数据生成...")
    generator = StoryGenerator(config, seed=seed)
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = generator.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )
    logger.info(f"  正例: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")
    logger.info(f"  负例: train={len(train_neg)}, val={len(val_neg)}, test={len(test_neg)}")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size
    logger.info(f"  词汇表大小: {tokenizer.vocab_size}")

    logger.info("[基线训练] 训练基线Transformer...")
    baseline_model = CausalTransformer(config)
    logger.info(f"  基线模型参数量: {baseline_model.count_parameters()}")

    all_train = train_pos + train_neg
    all_val = val_pos + val_neg
    train_ds = StoryDataset(all_train, tokenizer, config["model"]["max_seq_len"])
    val_ds = StoryDataset(all_val, tokenizer, config["model"]["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    baseline_trainer = Trainer(config, baseline_model)
    baseline_history = baseline_trainer.train_full(train_loader, val_loader, lambda_value=0.0)

    verdict_matrix = VerdictMatrix()

    logger.info("[实验1] 因果曲率代理量效度校准...")
    exp1 = Experiment1(config)
    exp1_results = exp1.run(baseline_model, test_pos + test_neg)
    verdict_matrix.update(1, exp1_results["overall_verdict"])
    logger.info(f"  实验1判决: {exp1_results['overall_verdict']}")

    logger.info("[实验2] 因果几何损失长程增益...")
    exp2 = Experiment2(config)
    exp2_results = exp2.run(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg)
    verdict_matrix.update(2, exp2_results["verdict"])
    logger.info(f"  实验2判决: {exp2_results['verdict']}")

    best_lambda = 1.0
    if "pareto_front" in exp2_results and exp2_results["pareto_front"]:
        best_entry = exp2_results["pareto_front"][0]
        best_lambda = best_entry.get("lambda", 1.0)
    logger.info(f"  最佳lambda: {best_lambda}")

    logger.info("[因果模型训练] 使用最佳lambda训练因果正则化模型...")
    causal_model = CausalTransformer(config)
    memory_bank = CausalMemoryBank(
        config["model"]["base_dim"],
        num_causal_types=config["model"].get("num_causal_types", 3),
    )
    gauge_field = GaugeField(config["model"]["base_dim"])
    causal_trainer = Trainer(config, causal_model, memory_bank, gauge_field)
    causal_history = causal_trainer.train_full(train_loader, val_loader, lambda_value=best_lambda)

    logger.info("[实验3] 记忆核因果特异性...")
    exp3 = Experiment3(config)
    exp3_results = exp3.run(causal_model, memory_bank, test_pos + test_neg)
    verdict_matrix.update(3, exp3_results["verdict"])
    logger.info(f"  实验3判决: {exp3_results['verdict']}")

    logger.info("[实验4] 规范场平坦化...")
    exp4 = Experiment4(config, gauge_field)
    exp4_results = exp4.run(baseline_model, causal_model, test_pos)
    verdict_matrix.update(4, exp4_results["verdict"])
    logger.info(f"  实验4判决: {exp4_results['verdict']}")

    logger.info("=" * 60)
    logger.info("总判决矩阵")
    final_verdict = verdict_matrix.render()
    for k, v in final_verdict.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    output_dir = Path(config["data"]["output_dir"]) / ".." / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "experiment1": {k: v for k, v in exp1_results.items() if not isinstance(v, torch.Tensor)},
        "experiment2": {k: v for k, v in exp2_results.items() if k != "causal_evals"},
        "experiment3": {k: v for k, v in exp3_results.items() if k != "convergence_results"},
        "experiment4": exp4_results,
        "final_verdict": final_verdict,
    }
    report_path = output_dir / "experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"报告已保存至: {report_path}")

    markdown_report = _generate_markdown_report(final_verdict, exp1_results, exp2_results, exp3_results, exp4_results)
    md_path = output_dir / "experiment_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    logger.info(f"Markdown报告已保存至: {md_path}")

    return final_verdict


def _generate_markdown_report(verdict, exp1, exp2, exp3, exp4):
    lines = [
        "# 因果规范场理论最小可行实验报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        "\n---\n",
        "## 总判决矩阵\n",
        "| 实验 | 判决 |",
        "|------|------|",
        f"| 实验1: 因果曲率相关性 | {verdict.get('exp1_curvature_correlation', 'N/A')} |",
        f"| 实验2: 因果损失长程增益 | {verdict.get('exp2_causal_loss_impact', 'N/A')} |",
        f"| 实验3: 记忆核因果特异性 | {verdict.get('exp3_memory_kernel', 'N/A')} |",
        f"| 实验4: 规范场平坦化 | {verdict.get('exp4_flatness', 'N/A')} |",
        f"\n**总体结论**: {verdict.get('overall_conclusion', 'N/A')}",
        "\n---\n",
        "## 实验1详情\n",
        f"- 方法A判决: {exp1.get('verdict_a', 'N/A')}",
        f"- 方法B判决: {exp1.get('verdict_b', 'N/A')}",
        f"- 方法A相关系数: {exp1.get('method_a', {}).get('correlation', 'N/A')}",
        f"- 方法B相关系数: {exp1.get('method_b', {}).get('correlation', 'N/A')}",
        "\n---\n",
        "## 实验2详情\n",
        f"- 判决: {exp2.get('verdict', 'N/A')}",
        f"- 帕累托前沿: {exp2.get('pareto_front', 'N/A')}",
        "\n---\n",
        "## 实验3详情\n",
        f"- 判决: {exp3.get('verdict', 'N/A')}",
        f"- 因果聚类单调递减: {exp3.get('causal_decreasing', 'N/A')}",
        f"- 因果聚类优于随机: {exp3.get('causal_better', 'N/A')}",
        "\n---\n",
        "## 实验4详情\n",
        f"- 判决: {exp4.get('verdict', 'N/A')}",
        f"- 基线mean_κ: {exp4.get('baseline_mean_kappa', 'N/A')}",
        f"- 因果mean_κ: {exp4.get('causal_mean_kappa', 'N/A')}",
        f"- t统计量: {exp4.get('t_statistic', 'N/A')}",
        f"- p值: {exp4.get('p_value', 'N/A')}",
        f"- Wilson环量可用: {exp4.get('wilson_available', 'N/A')}",
        f"- 基线Wilson方差: {exp4.get('baseline_wilson_var', 'N/A')}",
        f"- 因果Wilson方差: {exp4.get('causal_wilson_var', 'N/A')}",
        f"- Wilson平坦化辅证: {exp4.get('wilson_flattening_support', 'N/A')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()