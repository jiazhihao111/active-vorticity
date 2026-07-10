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
from causal_gauge_field.experiments.exp6_controlled_closure_closed import Experiment6
from torch.utils.data import DataLoader


def train_and_test(closure_lambda: float, tag: str, config, generator, tokenizer,
                    train_loader, val_loader, test_pos, test_neg, logger, out_dir):
    logger.info(f"=== [{tag}] 训练 causal 模型 (closure_lambda={closure_lambda}) ===")
    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    trainer = Trainer(config, causal_model, None, gauge_field)
    trainer.train_full(train_loader, val_loader, lambda_value=1.0, closure_lambda=closure_lambda)

    logger.info(f"=== [{tag}] 实验6 闭合 holonomy 受控检验 ===")
    exp6 = Experiment6(config, gauge_field)
    res = exp6.run(causal_model, test_pos, test_neg, model_label=tag)
    return res


def main():
    config = load_config()
    config["data"]["num_stories"] = 500
    config["training"]["max_epochs"] = 15
    config["training"]["patience"] = 5
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(__file__).parent
    logger = setup_logger("Exp6b", str(out_dir / "logs"), "exp6b.log")

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

    res_no = train_and_test(0.0, "no_closure", config, generator, tokenizer,
                            train_loader, val_loader, test_pos, test_neg, logger, out_dir)
    res_yes = train_and_test(1.0, "with_closure", config, generator, tokenizer,
                             train_loader, val_loader, test_pos, test_neg, logger, out_dir)

    report = {
        "timestamp": datetime.now().isoformat(),
        "no_closure": {k: v for k, v in res_no.items() if not isinstance(v, dict)},
        "no_closure_paired": res_no.get("paired", {}),
        "with_closure": {k: v for k, v in res_yes.items() if not isinstance(v, dict)},
        "with_closure_paired": res_yes.get("paired", {}),
    }
    with open(out_dir / "exp6b_salvage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = [
        "# 实验6b: 闭合敏感训练信号的挽救性检验",
        f"\n时间: {report['timestamp']}",
        "\n## 假说",
        "若原 C-11 失败仅因 CausalGeometryLoss 不奖励『回到起点』, 则加入 closure 损失",
        "(对正例惩罚 ‖h_T-h_0‖) 后, 实验6 闭合 holonomy 应出现 闭环<破缺.",
        "若仍无差异, 说明规范场机制本身无法编码叙事闭合 ⇒ C-11 应退役.",
        "",
        "## 无 closure 信号 (原设定)",
        f"- 判决: **{res_no['verdict']}**  闭合平坦度 正={res_no['pos_closed_flatness_mean']:.4f} 负={res_no['neg_closed_flatness_mean']:.4f}",
        f"- 配对平坦度差中位={res_no['paired'].get('flatness_diff_median',0):.5f} "
        f"p={res_no['paired'].get('wilcoxon_flatness_p',1):.4f} 破缺更不平坦比例={res_no['paired'].get('frac_neg_flatter',0):.3f}",
        "",
        "## 有 closure 信号 (挽救路径)",
        f"- 判决: **{res_yes['verdict']}**  闭合平坦度 正={res_yes['pos_closed_flatness_mean']:.4f} 负={res_yes['neg_closed_flatness_mean']:.4f}",
        f"- 配对平坦度差中位={res_yes['paired'].get('flatness_diff_median',0):.5f} "
        f"p={res_yes['paired'].get('wilcoxon_flatness_p',1):.4f} 破缺更不平坦比例={res_yes['paired'].get('frac_neg_flatter',0):.3f}",
        "",
        "## 结论",
        ("- 加 closure 信号后 出现 闭环<破缺 的显著差异 ⇒ 机制可挽救, 原损失缺闭合激励."
         if res_yes['verdict'] == 'SUPPORT'
         else "- 即便加 closure 信号, 仍无 闭环<破缺 差异 ⇒ 规范场机制无法编码叙事闭合, C-11 退役."),
    ]
    with open(out_dir / "exp6b_salvage_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"报告已保存: {out_dir / 'exp6b_salvage_report.md'}")


if __name__ == "__main__":
    main()
