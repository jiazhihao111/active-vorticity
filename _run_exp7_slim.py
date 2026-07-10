"""精简版实验7 — 快速基线对比 (体制A/B/C/D, 减少数据+epoch).
运行后在 exp7_slim_report.json 产生快速诊断报告.
"""
import sys, json, torch, numpy as np
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
                 test_pos, test_neg, logger, out_dir, **kw):
    logger.info(f"=== [{tag}] ===")
    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    trainer = Trainer(config, causal_model, None, gauge_field)
    trainer.train_full(train_loader, val_loader, **kw)
    exp7 = Experiment7(config, gauge_field)
    res = exp7.run(causal_model, test_pos, test_neg, model_label=tag)
    layered = kw.get("layered_rf_lambda", 0) > 0
    ld = exp7.run_layered(causal_model, test_pos, test_neg, model_label=tag) if layered else None
    return res, ld


def main():
    config = load_config()
    config["data"]["num_stories"] = 200
    config["data"]["enh_min_steps"], config["data"]["enh_max_steps"] = 6, 9
    config["npnw"]["max_stamina"] = 12
    config["model"]["max_seq_len"] = 128
    config["training"]["max_epochs"], config["training"]["patience"] = 6, 3
    config["experiment4"]["significance_level"] = 0.05
    seed = config["project"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    out_dir = Path(__file__).parent
    logger = setup_logger("Exp7slim", str(out_dir / "logs"), "exp7_slim.log")

    generator = EnhancedClosureGenerator(config, seed=seed)
    (tp, tn), (vp, vn), (test_p, test_n) = generator.generate_dataset(
        num_stories=config["data"]["num_stories"])
    logger.info(f"数据: train={len(tp)}对 val={len(vp)}对 test={len(test_p)}对")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    def mkld(p, n, sh): 
        return DataLoader(StoryDataset(p+n, tokenizer, config["model"]["max_seq_len"]),
                          batch_size=config["training"]["batch_size"], shuffle=sh)
    trl = mkld(tp, tn, True); vl = mkld(vp, vn, False)

    # 体制A: baseline LM only
    ra, _ = train_regime("A_baseline", config, generator, tokenizer, trl, vl,
                         test_p, test_n, logger, out_dir, lambda_value=0, closure_contrastive_lambda=0)
    # 体制B: uniform compress
    rb, _ = train_regime("B_uniform", config, generator, tokenizer, trl, vl,
                         test_p, test_n, logger, out_dir, lambda_value=1, closure_contrastive_lambda=0)
    # 体制C: loopback contrastive
    rc, _ = train_regime("C_loopback", config, generator, tokenizer, trl, vl,
                         test_p, test_n, logger, out_dir, lambda_value=0,
                         closure_contrastive_lambda=1, closure_margin=2)
    # 体制D: rigid-flexible (用 τ=0.5)
    rf_cfg = config.get("rigid_flexible", {})
    rd, ld = train_regime("D_rf", config, generator, tokenizer, trl, vl,
                          test_p, test_n, logger, out_dir,
                          lambda_value=0, closure_contrastive_lambda=0,
                          layered_rf_lambda=rf_cfg.get("layered_rf_lambda", 1.0),
                          rf_tau=rf_cfg.get("tau_narr", 0.5),
                          rf_lambda_phys=rf_cfg.get("lambda_phys", 1.0),
                          rf_lambda_flex=rf_cfg.get("lambda_flex", 1.0),
                          rf_margin=rf_cfg.get("margin", 0.3))

    report = {"timestamp": datetime.now().isoformat(), "regimes": {}}
    for k, r in [("A_baseline", ra), ("B_uniform", rb), ("C_loopback", rc)]:
        report["regimes"][k] = {f: r[f] for f in [
            "verdict", "n_pos", "n_neg",
            "pos_loop_flatness_mean", "neg_loop_flatness_mean",
            "pos_loop_count_mean", "neg_loop_count_mean",
            "pos_full_closed_flatness_mean", "neg_full_closed_flatness_mean",
        ]}
        report["regimes"][k]["paired"] = r.get("paired", {})

    report["regimes"]["D_rf"] = {
        "loopback_verdict": rd["verdict"],
        "n_pos": rd["n_pos"], "n_neg": rd["n_neg"],
        "pos_loop_flatness_mean": rd["pos_loop_flatness_mean"],
        "neg_loop_flatness_mean": rd["neg_loop_flatness_mean"],
        "loopback_paired": rd.get("paired", {}),
        "h_rigid_verdict": ld["h_rigid_verdict"] if ld else None,
        "h_flex_verdict": ld["h_flex_verdict"] if ld else None,
        "phys_curv_mean": ld["phys_curv_mean"] if ld else None,
        "flex_pos_curv_mean": ld["flex_pos_curv_mean"] if ld else None,
        "flex_neg_curv_mean": ld["flex_neg_curv_mean"] if ld else None,
        "tau": ld["tau"] if ld else None,
    }

    with open(out_dir / "exp7_slim_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # Quick markdown
    md = ["# 实验7 精简版: 体制A/B/C/D 快速对比",
          f"\n时间: {report['timestamp']}", ""]
    for k, r in report["regimes"].items():
        md.append(f"## {k}")
        md.append(f"- 判决: **{r.get('verdict', r.get('loopback_verdict', 'N/A'))}**")
        if "pos_loop_flatness_mean" in r:
            md.append(f"- 回环平坦度: 正={r['pos_loop_flatness_mean']:.4f} 负={r['neg_loop_flatness_mean']:.4f}")
        if "h_rigid_verdict" in r:
            md.append(f"- H-rigid: {r['h_rigid_verdict']} (phys={r['phys_curv_mean']:.4f})")
            md.append(f"- H-flex: {r['h_flex_verdict']} (flex+={r['flex_pos_curv_mean']:.4f} flex-={r['flex_neg_curv_mean']:.4f} τ={r['tau']})")
        md.append("")
    with open(out_dir / "exp7_slim_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("精简版exp7完成.")

if __name__ == "__main__":
    main()
