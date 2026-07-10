"""τ 网格搜索标定 — 刚柔分层阈值敏感性分析 (§3.7 铁律八·九).

对 rf_tau (训练损失中的柔性层阈值) 做扫描:
  τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0}

每 τ 值:
  1. 训练体制D 刚柔分层模型 (减少数据量 / 减少 epoch, 加速扫描)
  2. 运行 run_layered 分层诊断
  3. 记录: H-rigid / H-flex 判决, phys_curv_mean, flex_pos/neg_curv_mean,
     rf_loss 收敛值, paired Wilcoxon 统计量

同时测试诊断阈值 τ (即 run_layered 的 tau 参数) 对判决的敏感性:
  固定模型 (max τ_opt), 用不同 tau_diag 重新判决.

产出: tau_grid_report.json
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product

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

# ─── τ 扫描参数 ───
TAU_TRAIN_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
TAU_DIAG_VALUES  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]  # 诊断后扫描


def train_and_diagnose(config, generator, tokenizer,
                       train_loader, val_loader,
                       test_pos, test_neg, logger, seed,
                       rf_tau: float) -> dict:
    """对给定 rf_tau 训练体制D 并返回分层诊断."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    causal_model = CausalTransformer(config)
    gauge_field = GaugeField(config["model"]["base_dim"])
    trainer = Trainer(config, causal_model, None, gauge_field)

    rf_cfg = config.get("rigid_flexible", {})
    history = trainer.train_full(
        train_loader, val_loader,
        layered_rf_lambda=rf_cfg.get("layered_rf_lambda", 1.0),
        rf_tau=rf_tau,
        rf_lambda_phys=rf_cfg.get("lambda_phys", 1.0),
        rf_lambda_flex=rf_cfg.get("lambda_flex", 1.0),
        rf_margin=rf_cfg.get("margin", 0.3),
    )

    exp7 = Experiment7(config, gauge_field)
    layered = exp7.run_layered(causal_model, test_pos, test_neg,
                               model_label=f"tau_{rf_tau}")

    # 后扫描: 固定模型, 用不同 tau_diag 重新判决
    tau_sweep = {}
    for tdiag in TAU_DIAG_VALUES:
        # 临时覆盖 config 中的 tau_narr 做后扫描诊断
        saved_tau = config["rigid_flexible"]["tau_narr"]
        config["rigid_flexible"]["tau_narr"] = tdiag
        sweep_res = exp7.run_layered(causal_model, test_pos, test_neg,
                                     model_label=f"tau_{rf_tau}_diag{tdiag}")
        config["rigid_flexible"]["tau_narr"] = saved_tau
        tau_sweep[str(tdiag)] = {
            "h_rigid": sweep_res["h_rigid_verdict"],
            "h_flex": sweep_res["h_flex_verdict"],
            "phys_curv_mean": sweep_res["phys_curv_mean"],
            "flex_pos_curv_mean": sweep_res["flex_pos_curv_mean"],
            "flex_neg_curv_mean": sweep_res["flex_neg_curv_mean"],
            "h_flex_paired": sweep_res.get("h_flex_paired", {}),
        }

    return {
        "rf_tau": rf_tau,
        "rf_loss_final": history["rf_loss"][-1] if history["rf_loss"] else None,
        "train_loss_final": history["train_loss"][-1],
        "val_loss_final": history["val_loss"][-1],
        "h_rigid_verdict": layered["h_rigid_verdict"],
        "h_flex_verdict": layered["h_flex_verdict"],
        "phys_curv_mean": layered["phys_curv_mean"],
        "flex_pos_curv_mean": layered["flex_pos_curv_mean"],
        "flex_neg_curv_mean": layered["flex_neg_curv_mean"],
        "tau": layered["tau"],
        "h_flex_paired": layered.get("h_flex_paired", {}),
        "tau_diag_sweep": tau_sweep,
    }


def compute_fom(results: list) -> dict:
    """计算品质因数 (Figure of Merit):
    - H-rigid SUPPORT 率 (物理曲率近0的 τ 范围)
    - H-flex SUPPORT 率 (正<τ<负 显著成立)
    - flex_diff_snr: |flex_neg - flex_pos| / flex_pos (信噪比)
    """
    supported_rigid = [r for r in results if r["h_rigid_verdict"] == "SUPPORT"]
    supported_flex  = [r for r in results if r["h_flex_verdict"]  == "SUPPORT"]

    snrs = []
    for r in results:
        fp = r.get("flex_pos_curv_mean", float("nan"))
        fn = r.get("flex_neg_curv_mean", float("nan"))
        if fp and fp > 0 and not np.isnan(fn):
            snrs.append(abs(fn - fp) / fp)

    return {
        "n_total": len(results),
        "n_rigid_support": len(supported_rigid),
        "n_flex_support": len(supported_flex),
        "rigid_support_taus": [r["rf_tau"] for r in supported_rigid],
        "flex_support_taus": [r["rf_tau"] for r in supported_flex],
        "flex_diff_snr_median": float(np.median(snrs)) if snrs else None,
        "flex_diff_snr_mean": float(np.mean(snrs)) if snrs else None,
    }


def main():
    config = load_config()
    # 加速扫描: 更少数据 + 更少 epoch (快速原型, 定性趋势)
    config["data"]["num_stories"] = 300
    config["data"]["enh_min_steps"] = 6
    config["data"]["enh_max_steps"] = 9
    config["npnw"]["max_stamina"] = 12
    config["model"]["max_seq_len"] = 128
    config["training"]["max_epochs"] = 8
    config["training"]["patience"] = 3
    config["experiment4"]["significance_level"] = 0.05

    seed = config["project"]["seed"]
    out_dir = Path(__file__).parent
    logger = setup_logger("TauGrid", str(out_dir / "logs"), "tau_grid.log")

    # 生成一次数据, 所有 τ 共用
    torch.manual_seed(seed + 999)
    np.random.seed(seed + 999)
    generator = EnhancedClosureGenerator(config, seed=seed + 999)
    (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = generator.generate_dataset(
        num_stories=config["data"]["num_stories"])
    logger.info(f"τ网格: train={len(train_pos)}对 val={len(val_pos)}对 test={len(test_pos)}对")

    tokenizer = NPNWTokenizer()
    config["model"]["vocab_size"] = tokenizer.vocab_size

    def make_loader(pos, neg, shuffle):
        ds = StoryDataset(pos + neg, tokenizer, config["model"]["max_seq_len"])
        return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=shuffle)

    train_loader = make_loader(train_pos, train_neg, True)
    val_loader = make_loader(val_pos, val_neg, False)

    results = []
    for i, rf_tau in enumerate(TAU_TRAIN_VALUES):
        logger.info(f"─── τ 网格 [{i+1}/{len(TAU_TRAIN_VALUES)}] τ_train = {rf_tau} ───")
        try:
            rec = train_and_diagnose(
                config, generator, tokenizer,
                train_loader, val_loader,
                test_pos, test_neg, logger, seed + i,
                rf_tau=rf_tau,
            )
            results.append(rec)
            logger.info(
                f"  τ={rf_tau}: H-rigid={rec['h_rigid_verdict']} | "
                f"H-flex={rec['h_flex_verdict']} | "
                f"phys={rec['phys_curv_mean']:.4f} | "
                f"flex+={rec['flex_pos_curv_mean']:.4f} f-={rec['flex_neg_curv_mean']:.4f} | "
                f"rf_loss={rec['rf_loss_final']:.4f}"
            )
        except Exception as e:
            logger.error(f"  τ={rf_tau} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"rf_tau": rf_tau, "error": str(e)})

    fom = compute_fom(results)

    # ─── 最佳 τ 推荐 ───
    # 准则: 同时 H-rigid SUPPORT 且 H-flex SUPPORT, 且 flex_diff_snr 最大
    best = None
    best_score = -1
    for r in results:
        if r.get("error"):
            continue
        rigid_ok = r["h_rigid_verdict"] == "SUPPORT"
        flex_ok = r["h_flex_verdict"] == "SUPPORT"
        fp = r.get("flex_pos_curv_mean", float("nan"))
        fn = r.get("flex_neg_curv_mean", float("nan"))
        snr = abs(fn - fp) / fp if (fp and fp > 0) else 0
        score = (1 if rigid_ok else 0) + (2 if flex_ok else 0) + snr * 0.5
        if score > best_score:
            best_score = score
            best = r

    report = {
        "timestamp": datetime.now().isoformat(),
        "config_tau_values": TAU_TRAIN_VALUES,
        "diag_tau_values": TAU_DIAG_VALUES,
        "figure_of_merit": fom,
        "best_recommendation": {
            "rf_tau": best["rf_tau"] if best else None,
            "h_rigid": best["h_rigid_verdict"] if best else None,
            "h_flex": best["h_flex_verdict"] if best else None,
            "phys_curv_mean": best["phys_curv_mean"] if best else None,
            "flex_pos_curv_mean": best["flex_pos_curv_mean"] if best else None,
            "flex_neg_curv_mean": best["flex_neg_curv_mean"] if best else None,
        } if best else None,
        "results": results,
    }

    with open(out_dir / "tau_grid_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"τ网格报告已保存: {out_dir / 'tau_grid_report.json'}")

    # ─── Markdown 报告 ───
    md = ["# τ 网格搜索标定报告 — 刚柔分层阈值敏感性",
          f"\n时间: {report['timestamp']}",
          f"\nτ 训练值: {TAU_TRAIN_VALUES}",
          f"τ 诊断值: {TAU_DIAG_VALUES}",
          "",
          "## 训练 τ 扫描 (体制D, 300故事 × 8 epoch)",
          "",
          "| τ_train | rf_loss | phys_curv | flex_pos | flex_neg | H-rigid | H-flex | flex_SNR |",
          "|---------|---------|-----------|----------|----------|---------|--------|----------|"]
    for r in results:
        if r.get("error"):
            md.append(f"| {r['rf_tau']} | ERR | - | - | - | - | - | - |")
            continue
        fp = r.get("flex_pos_curv_mean", float("nan"))
        fn = r.get("flex_neg_curv_mean", float("nan"))
        snr = abs(fn - fp) / fp if (fp and fp > 0) else float("nan")
        md.append(
            f"| {r['rf_tau']:.1f} | {r.get('rf_loss_final', 0):.4f} | "
            f"{r['phys_curv_mean']:.4f} | "
            f"{r['flex_pos_curv_mean']:.4f} | {r['flex_neg_curv_mean']:.4f} | "
            f"**{r['h_rigid_verdict']}** | **{r['h_flex_verdict']}** | "
            f"{snr:.3f} |"
        )

    md.append(f"\n## 品质因数 (Figure of Merit)")
    md.append(f"- 总扫描: {fom['n_total']} 个 τ 值")
    md.append(f"- H-rigid SUPPORT: {fom['n_rigid_support']} ({fom['rigid_support_taus']})")
    md.append(f"- H-flex SUPPORT: {fom['n_flex_support']} ({fom['flex_support_taus']})")
    md.append(f"- SNR 中位: {fom.get('flex_diff_snr_median', 'N/A')}")

    if best:
        md.append(f"\n## 最佳推荐")
        md.append(f"- τ_train = **{best['rf_tau']}**")
        md.append(f"- H-rigid: {best['h_rigid_verdict']} (phys_curv={best['phys_curv_mean']:.4f})")
        md.append(f"- H-flex: {best['h_flex_verdict']} "
                  f"(flex+={best['flex_pos_curv_mean']:.4f} / f-={best['flex_neg_curv_mean']:.4f})")

    # 最佳的诊断 τ 后扫描
    if best and "tau_diag_sweep" in best:
        md.append(f"\n### 对最佳模型 (τ_train={best['rf_tau']}) 的诊断阈值后扫描")
        md.append("| τ_diag | H-rigid | H-flex | phys | flex+ | flex- |")
        md.append("|--------|---------|--------|------|-------|-------|")
        for tval, sw in best["tau_diag_sweep"].items():
            fp = sw.get("flex_pos_curv_mean", float("nan"))
            fn = sw.get("flex_neg_curv_mean", float("nan"))
            md.append(
                f"| {float(tval):.1f} | {sw['h_rigid']} | {sw['h_flex']} | "
                f"{sw['phys_curv_mean']:.4f} | {fp:.4f} | {fn:.4f} |"
            )

    with open(out_dir / "tau_grid_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"MD报告已保存: {out_dir / 'tau_grid_report.md'}")


if __name__ == "__main__":
    main()
