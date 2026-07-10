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
from causal_gauge_field.npnw.enhanced_generator import EnhancedClosureGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.models.gauge_field import GaugeField
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from causal_gauge_field.experiments.exp7_controlled_loopback import Experiment7
from torch.utils.data import DataLoader


def train_regime(tag, config, generator, tokenizer, train_loader, val_loader,
                 test_pos, test_neg, logger, out_dir,
                 lambda_value=0.0, closure_contrastive_lambda=0.0, closure_margin=2.0):
    logger.info(f"=== [{tag}] 训练 causal 模型 ===")
    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    trainer = Trainer(config, causal_model, None, gauge_field)
    trainer.train_full(
        train_loader, val_loader,
        lambda_value=lambda_value,
        closure_contrastive_lambda=closure_contrastive_lambda,
        closure_margin=closure_margin,
    )
    logger.info(f"=== [{tag}] 实验7 回环 holonomy 受控检验 ===")
    exp7 = Experiment7(config, gauge_field)
    res = exp7.run(causal_model, test_pos, test_neg, model_label=tag)
    return res


def train_regime_layered(tag, config, generator, tokenizer, train_loader, val_loader,
                         test_pos, test_neg, logger, out_dir,
                         layered_rf_lambda=1.0, rf_tau=0.5,
                         rf_lambda_phys=1.0, rf_lambda_flex=1.0, rf_margin=0.3):
    """体制D: 新分层对比 (论文 §10.8 第三体制, 铁律八·九 同时落地).

    用刚柔分层损失训练, 同时跑回环 holonomy 受控检验 与 分层 H-rigid/H-flex 诊断,
    直接判决附录假设账本 H-rigid / H-flex.
    """
    logger.info(f"=== [{tag}] 训练 刚柔分层因果模型 ===")
    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    trainer = Trainer(config, causal_model, None, gauge_field)
    trainer.train_full(
        train_loader, val_loader,
        layered_rf_lambda=layered_rf_lambda, rf_tau=rf_tau,
        rf_lambda_phys=rf_lambda_phys, rf_lambda_flex=rf_lambda_flex,
        rf_margin=rf_margin,
    )
    logger.info(f"=== [{tag}] 实验7 回环 holonomy + 分层诊断 ===")
    exp7 = Experiment7(config, gauge_field)
    res = exp7.run(causal_model, test_pos, test_neg, model_label=tag)
    layered = exp7.run_layered(causal_model, test_pos, test_neg, model_label=tag)
    return res, layered


def main():
    config = load_config()
    # ---- 更大、更长序列 (§10.7 任务3 的『更大』部分) ----
    config["data"]["num_stories"] = 600
    config["data"]["enh_min_steps"] = 6
    config["data"]["enh_max_steps"] = 9
    config["npnw"]["max_stamina"] = 20          # 允许更长出征+返回
    config["model"]["max_seq_len"] = 160        # 容纳更长叙事
    config["training"]["max_epochs"] = 12
    config["training"]["patience"] = 5
    config["experiment4"]["significance_level"] = 0.05
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(__file__).parent
    logger = setup_logger("Exp7", str(out_dir / "logs"), "exp7.log")

    generator = EnhancedClosureGenerator(config, seed=seed)
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = generator.generate_dataset(
        num_stories=config["data"]["num_stories"])
    logger.info(f"数据集规模: train={len(train_pos)}对 val={len(val_pos)}对 test={len(test_pos)}对")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    def make_loader(pos, neg, shuffle):
        ds = StoryDataset(pos + neg, tokenizer, config["model"]["max_seq_len"])
        return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=shuffle)

    train_loader = make_loader(train_pos, train_neg, True)
    val_loader = make_loader(val_pos, val_neg, False)

    results = {}
    # 体制A: baseline (仅 LM)
    results["baseline_lm"] = train_regime(
        "A_baseline_LM", config, generator, tokenizer, train_loader, val_loader,
        test_pos, test_neg, logger, out_dir, lambda_value=0.0, closure_contrastive_lambda=0.0)
    # 体制B: 旧 CausalGeometryLoss (统一压缩, exp4 假象来源)
    results["uniform_compress"] = train_regime(
        "B_uniform_compress", config, generator, tokenizer, train_loader, val_loader,
        test_pos, test_neg, logger, out_dir, lambda_value=1.0, closure_contrastive_lambda=0.0)
    # 体制C: 新 回环 holonomy 对比信号 (正例拉平/负例推离, 公平设计)
    results["loopback_contrastive"] = train_regime(
        "C_loopback_contrastive", config, generator, tokenizer, train_loader, val_loader,
        test_pos, test_neg, logger, out_dir, lambda_value=0.0,
        closure_contrastive_lambda=1.0, closure_margin=2.0)
    # 体制D: 新 刚柔分层对比 (论文 §10.8 第三体制, 铁律八·九 同时落地)
    rf_cfg = config.get("rigid_flexible", {})
    res_d, layered_d = train_regime_layered(
        "D_rigid_flexible", config, generator, tokenizer, train_loader, val_loader,
        test_pos, test_neg, logger, out_dir,
        layered_rf_lambda=rf_cfg.get("layered_rf_lambda", 1.0),
        rf_tau=rf_cfg.get("tau_narr", 0.5),
        rf_lambda_phys=rf_cfg.get("lambda_phys", 1.0),
        rf_lambda_flex=rf_cfg.get("lambda_flex", 1.0),
        rf_margin=rf_cfg.get("margin", 0.3))

    report = {
        "timestamp": datetime.now().isoformat(),
        "regimes": {},
    }
    for k, res in results.items():
        report["regimes"][k] = {
            v: res[v] for v in [
                "verdict", "n_pos", "n_neg",
                "pos_loop_flatness_mean", "neg_loop_flatness_mean",
                "pos_loop_count_mean", "neg_loop_count_mean",
                "pos_full_closed_flatness_mean", "neg_full_closed_flatness_mean",
            ]
        }
        report["regimes"][k]["paired"] = res.get("paired", {})
    # 体制D 刚柔分层: H-rigid / H-flex 判决 (回填附录账本)
    report["regimes"]["D_rigid_flexible"] = {
        "loopback_verdict": res_d.get("verdict"),
        "h_rigid_verdict": layered_d.get("h_rigid_verdict"),
        "h_flex_verdict": layered_d.get("h_flex_verdict"),
        "phys_curv_mean": layered_d.get("phys_curv_mean"),
        "flex_pos_curv_mean": layered_d.get("flex_pos_curv_mean"),
        "flex_neg_curv_mean": layered_d.get("flex_neg_curv_mean"),
        "tau": layered_d.get("tau"),
        "phys_curv_threshold": layered_d.get("phys_curv_threshold"),
        "h_flex_paired": layered_d.get("h_flex_paired", {}),
    }

    with open(out_dir / "exp7_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = ["# 实验7: 差异化回环 holonomy + 对比训练信号 (§10.7 三项集成)",
          f"\n时间: {report['timestamp']}", ""]
    md.append("## 设计")
    md.append("- 语料: EnhancedClosureGenerator (出征→返回原点; 正例返回/负例发散, 严格配对), "
              "更大更长 (max_seq_len=160, stamina=20, 600 对).")
    md.append("- 几何量: 仅在回环点定义 holonomy (loop_back_holonomy_flatness), 非整段均值.")
    md.append("- 训练: 体制A 仅LM / 体制B 旧统一压缩 / 体制C 回环对比(正拉平·负推离).")
    md.append("")
    md.append("## 结果")
    for k, res in results.items():
        p = res.get("paired", {})
        md.append(f"### {k}")
        md.append(f"- 判决: **{res['verdict']}**")
        md.append(f"- 回环平坦度 正={res['pos_loop_flatness_mean']:.4f} "
                  f"负={res['neg_loop_flatness_mean']:.4f}")
        md.append(f"- 检出回环数 正={res['pos_loop_count_mean']:.2f} "
                  f"负={res['neg_loop_count_mean']:.2f}")
        md.append(f"- 整段闭合平坦度 正={res['pos_full_closed_flatness_mean']:.4f} "
                  f"负={res['neg_full_closed_flatness_mean']:.4f}")
        md.append(f"- 配对差中位={p.get('loop_flatness_diff_median', 0):.4f} "
                  f"p={p.get('wilcoxon_loop_flatness_p', 1):.4f} "
                  f"破缺更不平坦比例={p.get('frac_neg_flatter', 0):.3f}")
        md.append("")
    md.append("## 结果 (体制D 刚柔分层)")
    md.append(f"- 回环 holonomy 判决: **{res_d['verdict']}**")
    md.append(f"- H-rigid (物理层曲率应近0): **{layered_d['h_rigid_verdict']}** "
              f"(phys_curv_mean={layered_d['phys_curv_mean']:.4f}, 阈={layered_d['phys_curv_threshold']})")
    md.append(f"- H-flex (柔性层 正<τ<负): **{layered_d['h_flex_verdict']}** "
              f"(正={layered_d['flex_pos_curv_mean']:.4f} 负={layered_d['flex_neg_curv_mean']:.4f} τ={layered_d['tau']})")
    md.append("")

    md.append("## 结论")
    c = results["loopback_contrastive"]
    if c["verdict"] == "SUPPORT":
        md.append("- 体制C 出现 正例回环平坦 << 负例 → C-11 在公平设计下成立: 叙事闭环可编码为几何平坦.")
    else:
        md.append("- 即便用差异化回环量 + 对比训练信号, 仍无 正<负 的显著分化 "
                  f"(体制C 判决={c['verdict']}) → 规范场机制本身无法承载 C-11, 应退役.")
        md.append("- 体制A/B 作为对照, 若亦不支持, 进一步印证.")
    md.append(f"- 体制D 刚柔分层: H-rigid={layered_d['h_rigid_verdict']}, H-flex={layered_d['h_flex_verdict']} "
              f"(回填附录假设账本).")
    with open(out_dir / "exp7_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"报告已保存: {out_dir / 'exp7_report.md'}")


if __name__ == "__main__":
    main()
