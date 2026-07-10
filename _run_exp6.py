import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "causal_gauge_field"))
sys.path.insert(0, str(Path(__file__).parent))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.models.gauge_field import GaugeField
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from causal_gauge_field.experiments.exp5_controlled_closure import Experiment5
from causal_gauge_field.experiments.exp6_controlled_closure_closed import Experiment6
from torch.utils.data import DataLoader


def main():
    config = load_config()
    config["data"]["num_stories"] = 500
    config["training"]["max_epochs"] = 15
    config["training"]["patience"] = 5
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(__file__).parent
    logger = setup_logger("Exp6", str(out_dir / "logs"), "exp6.log")

    logger.info("=== 实验6 数据生成 ===")
    generator = StoryGenerator(config, seed=seed)
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = generator.generate_dataset(
        num_stories=config["data"]["num_stories"],
        neg_per_positive=config["data"]["neg_per_positive"],
    )
    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    all_train = train_pos + train_neg
    all_val = val_pos + val_neg
    train_ds = StoryDataset(all_train, tokenizer, config["model"]["max_seq_len"])
    val_ds = StoryDataset(all_val, tokenizer, config["model"]["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    logger.info("=== 因果模型 + 规范场训练 (lambda=1.0) ===")
    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    causal_trainer = Trainer(config, causal_model, None, gauge_field)
    causal_trainer.train_full(train_loader, val_loader, lambda_value=1.0)

    logger.info("=== 实验6: 闭合 holonomy 受控检验 (因果模型) ===")
    exp6 = Experiment6(config, gauge_field)
    exp6_res = exp6.run(causal_model, test_pos, test_neg, model_label="causal")
    for k, v in exp6_res.items():
        logger.info(f"  {k}: {v}")

    logger.info("=== 实验5(开环对照): 同模型同数据, 用开环指标 ===")
    exp5 = Experiment5(config, gauge_field)
    exp5_res = exp5.run(causal_model, test_pos, test_neg, model_label="causal")
    for k, v in exp5_res.items():
        logger.info(f"  {k}: {v}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "exp6_closed_holonomy": {k: v for k, v in exp6_res.items() if not isinstance(v, dict)},
        "exp6_paired": exp6_res.get("paired", {}),
        "exp5_open_loop": {k: v for k, v in exp5_res.items() if not isinstance(v, dict)},
        "exp5_paired": exp5_res.get("paired", {}),
    }
    report_path = out_dir / "exp6_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = [
        "# 实验6: 闭合 holonomy 受控检验 (正例vs负例)",
        f"\n时间: {report['timestamp']}",
        "\n## 设计",
        "- 主判据: **闭合 holonomy 平坦度** ‖W_closed - I‖_F (gauge_field.wilson_loop_closed)",
        "  —— 沿 h_0..h_T 乘积后补闭合边 h_T→h_0; 闭环叙事 h_T≈h_0 ⇒ W≈I(平坦), 破缺≠I.",
        "- 检验: 负例由正例派生(story_id相同), 做配对 Wilcoxon (破缺-闭环), H1: 破缺更不平坦.",
        "- 对照: 同模型同数据再跑实验5开环指标(κ_angle / 开环 Wilson), 对比两种判据.",
        "",
        "## 实验6 (闭合 holonomy, 主判据)",
        f"- 判决: **{exp6_res['verdict']}**",
        f"- 样本: 正例={exp6_res['n_pos']} 负例={exp6_res['n_neg']}",
        f"- 闭合平坦度: 正={exp6_res['pos_closed_flatness_mean']:.4f} "
        f"负={exp6_res['neg_closed_flatness_mean']:.4f}",
        f"- 闭合迹偏差: 正={exp6_res['pos_closed_trace_dev_mean']:.4f} "
        f"负={exp6_res['neg_closed_trace_dev_mean']:.4f}",
        f"- 开环κ: 正={exp6_res['pos_kappa_mean']:.4f} 负={exp6_res['neg_kappa_mean']:.4f}",
        "",
        "### 配对结果 (破缺 - 闭环)",
        f"- 平坦度差中位: {exp6_res['paired'].get('flatness_diff_median', 0):.5f}  "
        f"Wilcoxon p={exp6_res['paired'].get('wilcoxon_flatness_p', 1):.4f}",
        f"- 破缺更不平坦比例: {exp6_res['paired'].get('frac_neg_flatter', 0):.3f}",
        f"- 迹偏差差中位: {exp6_res['paired'].get('trace_dev_diff_median', 0):.5f}  "
        f"p={exp6_res['paired'].get('wilcoxon_tracedev_p', 1):.4f}",
        "",
        "## 实验5 (开环对照, 同模型同数据)",
        f"- 判决: **{exp5_res['verdict']}**",
        f"- 配对κ差中位: {exp5_res['paired'].get('kappa_diff_median', 0):.5f}  "
        f"p={exp5_res['paired'].get('wilcoxon_p', 1):.4f}  "
        f"破缺更不平坦比例={exp5_res['paired'].get('frac_neg_flatter', 0):.3f}",
        "",
        "## 解读",
        "- 若实验6=SUPPORT 而实验5=INCONCLUSIVE: 旧开环指标假象, 闭合 holonomy 真正捕捉到闭合⇔平坦, 理论可挽救.",
        "- 若实验6=INCONCLUSIVE/OPPOSE: 即便真正闭合后, 模型仍未把叙事闭合编码进几何平坦度 ⇒ C-11 确证为假, 应退役为未验证隐喻.",
        "- 若实验6=OPPOSE (破缺反而更平): 模型表征与理论预测相反.",
    ]
    md_path = out_dir / "exp6_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"报告已保存: {md_path}")


if __name__ == "__main__":
    main()
