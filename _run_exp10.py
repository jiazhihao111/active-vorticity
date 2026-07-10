#!/usr/bin/env python3
"""
实验10: 几何动力学尺度升级
===========================

叙事驱动力:
    "如果我们的几何本体论是对的，那么错误一定出在
    '如何让这个几何在动力学上运作'的细节里——
    让我们升级实验尺度，把这些细节找出来。"

相对于 exp9 的核心升级:
    1. MVE 尺度升级: 32D → 128D (base_dim)
       → SVD 投影信息保留率从 ~10% 提升到 ~60%
    2. FS 标架首次作为实验判据 (而非独立工具)
       → 螺旋升角 Θ_sem = arctan(τ/κ) 在 128D 空间中的稳定性
    3. InfoNCE 对比推离替换铰链推离
       → 梯度丰富的批量对比信号替代边际坍塌的铰链
    4. 变长序列独立生成 (而非 exp9 的按长度分桶复用)
       → 每个长度桶独立生成故事，消除数据同源性混淆

四部分实验:
    Part A: H-helix 重测 — FS 螺旋升角在高维空间中的稳定性
    Part B: H-push 重测 — InfoNCE 对比推离 vs 铰链推离
    Part C: H-phase 重测 — 变长序列陈数相变 bootstrap 检验
    Part D: 集成判决 — 三层假设联合 Bayesian 更新

预期:
    - 128D 空间中 FS 标架有意义 (CV_tanΘ < 0.5)
    - InfoNCE 推离率 > 60% (铰链仅 50% 基准)
    - 检测到 128 token 附近陈数跳变
    - 固化度从 0.48 提升到 0.55-0.65
"""

import os, sys, json, time, logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# 项目路径
PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT))

from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.models.gauge_field import GaugeField
from causal_gauge_field.npnw.enhanced_generator import EnhancedClosureGenerator
from causal_gauge_field.experiments.exp10_geometric_dynamics import (
    Experiment10,
    create_high_dim_model,
    create_low_dim_model,
)
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer

warnings.filterwarnings("ignore")

# ─── 日志 ───
LOG_DIR = PROJ_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = setup_logger("exp10", LOG_DIR / "exp10.log")

OUTPUT_DIR = PROJ_ROOT / "exp10_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    model: CausalTransformer,
    gauge_field: GaugeField,
    config: dict,
    train_pos: List,
    train_neg: List,
    val_pos: List,
    val_neg: List,
    label: str = "model",
    epochs: int = 20,
) -> Tuple[CausalTransformer, Dict]:
    """训练模型并返回最佳 checkpoint 和训练历史。

    同时使用回环对比信号和刚柔分层损失，
    确保训练后的模型学会区分闭环/破缺叙事。
    """
    logger.info(f"[训练] {label}: {epochs} epochs, "
                f"train={len(train_pos)+len(train_neg)}, "
                f"val={len(val_pos)+len(val_neg)}")

    tokenizer = NPNWTokenizer()

    # 混合正负例
    train_stories = list(train_pos) + list(train_neg)
    val_stories = list(val_pos) + list(val_neg)

    train_dataset = StoryDataset(
        train_stories, tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
    )
    val_dataset = StoryDataset(
        val_stories, tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(config["training"]["batch_size"] // 2, 16),
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(config["training"]["batch_size"] // 2, 16),
        shuffle=False,
    )

    trainer = Trainer(
        dict(config),  # 深拷贝避免原地修改
        model,
        gauge_field=gauge_field,
    )

    rf_cfg = config.get("rigid_flexible", {})
    history = trainer.train_full(
        train_loader,
        val_loader,
        lambda_value=0.1,
        closure_contrastive_lambda=0.5,
        closure_margin=2.0,
        layered_rf_lambda=rf_cfg.get("layered_rf_lambda", 1.0),
        rf_tau=rf_cfg.get("tau_narr", 0.5),
        rf_lambda_phys=rf_cfg.get("lambda_phys", 1.0),
        rf_lambda_flex=rf_cfg.get("lambda_flex", 1.0),
        rf_margin=rf_cfg.get("margin", 0.3),
    )

    return model, history


def main():
    logger.info("=" * 70)
    logger.info("exp10: 几何动力学尺度升级 — 叙事驱动力验证")
    logger.info(f"设备: {DEVICE}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("叙事: 如果几何本体论是对的，那么错误在动力学细节里")
    logger.info("=" * 70)

    # ── 1. 加载配置 ──
    config = load_config(PROJ_ROOT / "causal_gauge_field" / "config.yaml")
    base_dim = config["model"]["base_dim"]  # 32 (低维对照用)
    logger.info(f"原始配置: d_model={config['model']['d_model']}, "
                f"base_dim={base_dim}")

    # ── 2. 创建高维(128D)和低维(32D)模型 ──
    logger.info("\n--- 创建模型 ---")

    # 高维模型 (128D MVE, max_seq_len=256)
    model_high = create_high_dim_model(config, base_dim=128, max_seq_len=256).to(DEVICE)
    gauge_high = GaugeField(base_dim=128).to(DEVICE)
    high_params = sum(p.numel() for p in model_high.parameters())
    logger.info(f"高维模型(128D, seq=256): {high_params:,} 参数")

    # 低维对照模型 (32D, max_seq_len=128)
    model_low = create_low_dim_model(config, max_seq_len=128).to(DEVICE)
    gauge_low = GaugeField(base_dim=32).to(DEVICE)
    low_params = sum(p.numel() for p in model_low.parameters())
    logger.info(f"低维模型(32D): {low_params:,} 参数")

    # ── 3. 生成数据 ──
    logger.info("\n--- 生成叙事数据 ---")
    gen_cfg = dict(config)
    gen_cfg["data"] = dict(config["data"])
    gen_cfg["data"]["enh_min_steps"] = 6
    gen_cfg["data"]["enh_max_steps"] = 12

    # 高维模型训练数据 (快速验证: 少量数据)
    generator_high = EnhancedClosureGenerator(gen_cfg, seed=42)
    (train_pos_h, train_neg_h), (val_pos_h, val_neg_h), (test_pos_h, test_neg_h) = \
        generator_high.generate_dataset(num_stories=60)

    # 低维模型训练数据
    gen_cfg["data"]["enh_min_steps"] = 4
    gen_cfg["data"]["enh_max_steps"] = 8
    generator_low = EnhancedClosureGenerator(gen_cfg, seed=99)
    (train_pos_l, train_neg_l), (val_pos_l, val_neg_l), (test_pos_l, test_neg_l) = \
        generator_low.generate_dataset(num_stories=40)

    logger.info(f"高维数据: train={len(train_pos_h)+len(train_neg_h)}, "
                f"test={len(test_pos_h)+len(test_neg_h)}")
    logger.info(f"低维数据: train={len(train_pos_l)+len(train_neg_l)}, "
                f"test={len(test_pos_l)+len(test_neg_l)}")

    # ── 4. 训练模型 ──
    logger.info("\n--- 训练模型 ---")

    # 训练高维模型 (max_seq_len 已在模型创建时设为 256)
    train_cfg_high = dict(config)
    train_cfg_high["model"] = dict(config["model"])
    train_cfg_high["model"]["base_dim"] = 128
    train_cfg_high["model"]["d_model"] = 512
    train_cfg_high["model"]["max_seq_len"] = 256  # 与模型创建一致
    train_cfg_high["model"]["n_heads"] = 4
    train_cfg_high["training"] = dict(config["training"])
    train_cfg_high["training"]["max_epochs"] = 3
    train_cfg_high["training"]["patience"] = 2
    train_cfg_high["training"]["batch_size"] = 16

    model_high, hist_high = train_model(
        model_high, gauge_high, train_cfg_high,
        train_pos_h, train_neg_h, val_pos_h, val_neg_h,
        label="high_128D", epochs=3,
    )

    # 训练低维模型 (max_seq_len=128)
    train_cfg_low = dict(config)
    train_cfg_low["model"] = dict(config["model"])
    train_cfg_low["model"]["max_seq_len"] = 128  # 与模型创建一致
    train_cfg_low["training"] = dict(config["training"])
    train_cfg_low["training"]["max_epochs"] = 3
    train_cfg_low["training"]["patience"] = 2
    train_cfg_low["training"]["batch_size"] = 16

    model_low, hist_low = train_model(
        model_low, gauge_low, train_cfg_low,
        train_pos_l, train_neg_l, val_pos_l, val_neg_l,
        label="low_32D", epochs=3,
    )

    # ── 5. 运行 exp10 实验 ──
    logger.info("\n" + "=" * 70)
    logger.info("运行 exp10 四部分实验")
    logger.info("=" * 70)

    exp10 = Experiment10(config, gauge_high)

    # 使用高维测试数据 + 新的数据生成器 (用于 Part C 变长序列)
    gen_test = EnhancedClosureGenerator(gen_cfg, seed=777)

    results = exp10.run_full(
        model_high=model_high,
        model_low=model_low,
        story_generator=gen_test,
        test_pos=test_pos_h,
        test_neg=test_neg_h,
        training_history=hist_high,  # 传递高维模型训练历史供动力学监控使用
    )

    # ── 6. 保存结果 ──
    logger.info("\n--- 保存结果 ---")

    def _to_native(obj):
        """递归转换为 Python 原生类型"""
        if isinstance(obj, dict):
            return {str(k): _to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.tolist()
        return obj

    clean_results = _to_native(results)

    # JSON 报告
    json_path = OUTPUT_DIR / "exp10_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 报告: {json_path}")

    # Markdown 报告
    md_path = OUTPUT_DIR / "exp10_report.md"
    _write_md_report(clean_results, md_path)
    logger.info(f"Markdown 报告: {md_path}")

    # ── 7. 总结 ──
    integrated = results.get("part_d", {})
    logger.info("\n" + "=" * 70)
    logger.info("exp10 完成！")
    logger.info(f"  先验固化度: {integrated.get('prior_certainty', 0):.2f}")
    logger.info(f"  后验固化度: {integrated.get('posterior_certainty', 0):.2f}")
    logger.info(f"  叙事位置: {integrated.get('narrative_stage', 'N/A')}")
    logger.info(f"  耗时: {results.get('elapsed_seconds', 0):.1f}s")
    logger.info("=" * 70)

    return clean_results


def _write_md_report(results: Dict, path: Path):
    """生成 exp10 Markdown 报告"""
    lines = [
        "# 实验10: 几何动力学尺度升级报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 叙事驱动力",
        "",
        "> 如果我们的几何本体论是对的，那么错误一定出在"
        "'如何让这个几何在动力学上运作'的细节里——"
        "让我们升级实验尺度，把这些细节找出来。",
        "",
        "---",
        "",
        "## 实验设计总览",
        "",
        "| 维度 | exp9 (基线) | exp10 (升级) |",
        "|---|---|---|",
        "| MVE 维度 | 32D | **128D** |",
        "| FS 标架集成 | 独立工具 (未作判据) | **核心判据** (H-helix) |",
        "| 推离机制 | 铰链损失 (d=0.007) | **InfoNCE 对比推离** |",
        "| H-phase 数据 | 同源按长度分桶 | **独立变长生成** |",
        "| 判决框架 | 独立二元判决 | **联合 Bayesian 更新** |",
        "",
        "---",
        "",
    ]

    # Part A
    part_a = results.get("part_a", {})
    lines.extend([
        "## Part A: H-helix — FS 螺旋升角稳定性",
        "",
        "### 高维 (128D)",
        "",
    ])
    a_high = part_a.get("high_dim", {})
    a_pos = a_high.get("pos", {})
    a_neg = a_high.get("neg", {})
    lines.extend([
        f"| 指标 | 正例 (闭环) | 负例 (破缺) |",
        f"|---|---|---|",
        f"| n_valid | {a_pos.get('n_total', 'N/A')} | {a_neg.get('n_total', 'N/A')} |",
        f"| mean(tanΘ) | {a_pos.get('mean_tan_theta', 0):.4f} | {a_neg.get('mean_tan_theta', 0):.4f} |",
        f"| CV(tanΘ) | {a_pos.get('cv_tan_theta', 0):.4f} | {a_neg.get('cv_tan_theta', 0):.4f} |",
        f"| R²(τ~κ) | {a_pos.get('r_squared', 0):.4f} | {a_neg.get('r_squared', 0):.4f} |",
        f"| Pearson r | {a_pos.get('pearson_r_kappa_tau', 0):.4f} | {a_neg.get('pearson_r_kappa_tau', 0):.4f} |",
        "",
        f"**判决**: {a_high.get('verdict', 'N/A')}",
        "",
    ])

    # 维度增益
    dim_gain = part_a.get("dimensional_gain", {})
    if dim_gain:
        lines.extend([
            "### 维度增益",
            "",
            f"| 指标 | 32D (exp9) | 128D (exp10) | 增益 |",
            f"|---|---|---|---|",
            f"| R²(τ~κ) | {dim_gain.get('r2_low', 0):.4f} | {dim_gain.get('r2_high', 0):.4f} | {dim_gain.get('relative_gain', 0):.2%} |",
            "",
        ])

    # Part B
    part_b = results.get("part_b", {})
    lines.extend([
        "---",
        "",
        "## Part B: H-push — InfoNCE 对比推离",
        "",
        "| 方法 | 推离率 | 推离比 | Loss |",
        "|---|---|---|---|",
    ])
    methods = part_b.get("methods", {})
    for method, mdata in methods.items():
        if method == "hinge_baseline":
            lines.append(
                f"| 铰链 (exp9基线) | {mdata.get('hinge_cohens_d', 0):.4f} (Cohen's d) | — | {mdata.get('hinge_loss', 0):.4f} |"
            )
        else:
            lines.append(
                f"| InfoNCE-{method} | {mdata.get('frac_effective_push', 0):.3f} | {mdata.get('push_ratio_mean', 0):.3f} | {mdata.get('loss', 0):.4f} |"
            )
    lines.extend([
        "",
        f"**最佳方法**: {part_b.get('best_method', 'N/A')}",
        f"**判决**: {part_b.get('verdict', 'N/A')}",
        f"**vs 铰链提升**: {part_b.get('infonce_vs_hinge_improvement', 0):.4f}",
        "",
    ])

    # Part C
    part_c = results.get("part_c", {})
    lines.extend([
        "---",
        "",
        "## Part C: H-phase — 变长序列陈数相变",
        "",
        "| 长度 L₁ → L₂ | 陈数Δ | Bootstrap 95%CI | Cohen's d | 判决 |",
        "|---|---|---|---|---|",
    ])
    for t in part_c.get("transitions", []):
        lines.append(
            f"| {t.get('L_from', 0)} → {t.get('L_to', 0)} | "
            f"{t.get('mean_diff', 0):.6f} | "
            f"[{t.get('ci_low', 0):.6f}, {t.get('ci_high', 0):.6f}] | "
            f"{t.get('cohens_d', 0):.3f} | "
            f"{t.get('verdict', 'N/A')} |"
        )
    lines.extend([
        "",
        f"**全局判决**: {part_c.get('verdict', 'N/A')}",
        "",
    ])

    # Part D
    part_d = results.get("part_d", {})
    lines.extend([
        "---",
        "",
        "## Part D: 集成判决",
        "",
        "### 三层假设联合 Bayesian 更新",
        "",
        f"| 参数 | 值 |",
        f"|---|---|",
        f"| 先验固化度 | {part_d.get('prior_certainty', 0):.2f} |",
    ])
    verdicts = part_d.get("verdicts", {})
    deltas = part_d.get("deltas", {})
    for hyp in ["H-helix", "H-push", "H-phase"]:
        v = verdicts.get(hyp, "N/A")
        d = deltas.get(hyp, 0)
        lines.append(f"| {hyp} | {v} ({d:+.2f}) |")
    lines.extend([
        f"| 总Δ | {part_d.get('total_delta', 0):+.2f} |",
        f"| **后验固化度** | **{part_d.get('posterior_certainty', 0):.2f}** |",
        "",
        f"### 叙事位置判定",
        "",
        f"**{part_d.get('narrative_stage', 'N/A')}**",
        "",
    ])

    solidifiable = part_d.get("solidifiable_items", [])
    if solidifiable:
        lines.append("### 可固化项")
        for item in solidifiable:
            lines.append(f"- ✅ {item}")
        lines.append("")

    # ── Part D2: 动力学演化监控 (NEW) ──
    part_d2 = results.get("part_d2_dynamics", {})
    if part_d2:
        d_report = part_d2.get("dynamics_report", {})
        brake_summary = part_d2.get("brake_tensor_summary", {})

        lines.extend([
            "---",
            "",
            "## Part D2: 几何动力学演化监控（基于EvolutionCycleEngine）",
            "",
            f"### 动力学判决",
            "",
            f"**{part_d2.get('verdict', 'N/A')}**",
            "",
            f"### ψ_geo 统计量",
            "",
        ])
        psi_stats = d_report.get("psi_statistics", {})
        lines.extend([
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| ψ_mean | {psi_stats.get('mean', 'N/A')} |",
            f"| ψ_std | {psi_stats.get('std', 'N/A')} |",
            f"| ψ_final | {psi_stats.get('final', 'N/A')} |",
            f"| 趋势 | {psi_stats.get('trend', 'N/A')} |",
            "",
        ])

        # 状态分布
        state_dist = d_report.get("state_distribution", {})
        lines.extend([
            f"### 状态分布",
            f"",
            f"| 状态 | 计次 |",
            f"|---|---|",
            f"| ◯ STEADY | {state_dist.get('steady', 0)} |",
            f"| ◎ PRESSURIZED | {state_dist.get('pressurized', 0)} |",
            f"| ◉ BREAKING | {state_dist.get('breaking', 0)} |",
            f"| 破缺率 | {state_dist.get('break_rate', 0):.1%} |",
            "",
        ])

        # 破缺事件
        break_events = d_report.get("break_events", [])
        if break_events:
            lines.extend([
                "### 破缺事件",
                "",
                "| Epoch | ψ | 类型 | γ调整 |",
                "|---|---|---|---|",
            ])
            for be in break_events:
                gamma_str = ", ".join(
                    f"{k}={v:+.2f}"
                    for k, v in be.get("gamma_adjustments", {}).items()
                ) if be.get("gamma_adjustments") else "无"
                lines.append(
                    f"| {be.get('epoch', '?')} | "
                    f"{be.get('psi', 0):+.3f} | "
                    f"{be.get('type', '?')[:30]} | "
                    f"{gamma_str} |"
                )
            lines.append("")

        # 制动张量状态
        lines.extend([
            "### FS制动张量状态",
            "",
            f"| 参数 | 值 |",
            f"|---|---|",
            f"| 破缺窗口 | {'开启' if brake_summary.get('window_active') else '关闭'} |",
            f"| 自由能预算 | {brake_summary.get('budget_remaining', 0):.3f} |",
            f"| 谱条件数 | {brake_summary.get('spectral_condition', 'N/A')} |",
            f"| 探索自由度 | {brake_summary.get('exploration_freedom', 'N/A')} |",
            "",
        ])

        # 约束权重分配
        w = brake_summary.get("constraint_weights", {})
        balance = brake_summary.get("constraint_balance", {})
        if w:
            lines.extend([
                "### 四维约束权重",
                "",
                "| 维度 | γ值 | 占比 | 含义 |",
                "|---|---|---|---|",
                f"| γ_T (切向量) | {w.get('gamma_T', 'N/A')} | {balance.get('tangent_pct', 0):.1%} | 语义方向一致性 |",
                f"| γ_N (法向量) | {w.get('gamma_N', 'N/A')} | {balance.get('normal_pct', 0):.1%} | 弯曲强度约束 |",
                f"| γ_B (副法向量) | {w.get('gamma_B', 'N/A')} | {balance.get('binormal_pct', 0):.1%} | 扭转自由度约束 |",
                f"| γ_τ (挠率变化率) | {w.get('gamma_tau', 'N/A')} | {balance.get('torsion_rate_pct', 0):.1%} | 结构惯性约束 |",
                "",
            ])

        # 几何量演化表
        evo = d_report.get("evolution_curve", {})
        if evo and evo.get("epochs"):
            epochs_list = evo["epochs"]
            psis = evo.get("psi_geo", [])
            kappas = evo.get("kappa_mean", [])
            taus = evo.get("tau_mean", [])
            orthos = evo.get("frame_orthogonality", [])
            states = evo.get("states", [])

            lines.extend([
                "### 几何量演化轨迹",
                "",
                "| Epoch | ψ_geo | κ_mean | τ_mean | 正交度 | 状态 |",
                "|---|---|---|---|---|---|",
            ])
            for i, ep in enumerate(epochs_list):
                psi_v = f"{psis[i]:+.3f}" if i < len(psis) and psis[i] is not None else "N/A"
                k_v = f"{kappas[i]:.4f}" if i < len(kappas) and kappas[i] is not None and kappas[i] == kappas[i] else "N/A"
                t_v = f"{taus[i]:.4f}" if i < len(taus) and taus[i] is not None and taus[i] == taus[i] else "N/A"
                o_v = f"{orthos[i]:.3f}" if i < len(orthos) and orthos[i] is not None and orthos[i] == orthos[i] else "N/A"
                s_v = states[i] if i < len(states) else "?"
                lines.append(f"| {ep} | {psi_v} | {k_v} | {t_v} | {o_v} | {s_v} |")
            lines.append("")

    # ── Part D3: 三条定理验证 (NEW — 十三字公理固化) ──
    part_d3 = results.get("part_d3_theorems", {})
    if part_d3:
        lines.extend([
            "---",
            "",
            "## Part D3: 三定理验证（十三字公理固化）",
            "",
            f"> **公理**：{part_d3.get('axiom', '信息化为世界模型，世界模型遵守几何规则')}",
            "",
            f"### 综合指标",
            "",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 综合支持度 | {part_d3.get('overall_support', 0):.2%} |",
            f"| 通过数 | {part_d3.get('passed_count', 0)}/{part_d3.get('total_count', 3)} |",
            f"| 公理验证 | {'✓ 通过' if part_d3.get('axiom_verified') else '✗ 未通过'} |",
            f"| 自洽性 | {part_d3.get('self_consistency_score', 0):.2%} |",
            "",
        ])

        # 定理一
        t1 = part_d3.get("theorem_1", {})
        if t1:
            t1_d = t1.get("details", {})
            lines.extend([
                "### 定理一：信息本体论 (χ ≠ 0)",
                "",
                f"| 参数 | 值 |",
                f"|---|---|",
                f"| 判决 | {t1.get('verdict', 'N/A')} |",
                f"| 置信度 | {t1.get('confidence', 0):.2%} |",
                f"| χ_pos_mean | {t1_d.get('chi_pos_mean', 'N/A')} |",
                f"| χ_neg_mean | {t1_d.get('chi_neg_mean', 'N/A')} |",
                f"| Cohen's d | {t1_d.get('cohens_d', 'N/A')} |",
                f"| p-value | {t1_d.get('p_value', 'N/A')} |",
                "",
                f'**解读**："信息化为" → 隐流形 → 持久同调 → χ≠0',
                "",
            ])

        # 定理二
        t2 = part_d3.get("theorem_2", {})
        if t2:
            t2_d = t2.get("details", {})
            lines.extend([
                "### 定理二：几何守恒论 (τ/κ = const)",
                "",
                f"| 参数 | 值 |",
                f"|---|---|",
                f"| 判决 | {t2.get('verdict', 'N/A')} |",
                f"| 置信度 | {t2.get('confidence', 0):.2%} |",
                f"| CV_original | {t2_d.get('cv_original', 'N/A')} |",
                f"| CV_shuffled | {t2_d.get('cv_shuffled', 'N/A')} |",
                f"| 平行移动残差 | {t2_d.get('pt_residual', 'N/A')} |",
                f"| 局部/全局CV比 | {t2_d.get('window_cv_ratio', 'N/A')} |",
                "",
                f'**解读**："世界模型遵守" → τ/κ共形不变',
                "",
            ])

        # 定理三
        t3 = part_d3.get("theorem_3", {})
        if t3:
            t3_d = t3.get("details", {})
            lines.extend([
                "### 定理三：拓扑残差论 (ε_min > 0)",
                "",
                f"| 参数 | 值 |",
                f"|---|---|",
                f"| 判决 | {t3.get('verdict', 'N/A')} |",
                f"| 置信度 | {t3.get('confidence', 0):.2%} |",
                f"| ε_min (渐进残差) | {t3_d.get('epsilon_min', 'N/A')} |",
                f"| R² (拟合优度) | {t3_d.get('r_squared', 'N/A')} |",
                f"| p(ε_min > 0) | {t3_d.get('p_positive', 'N/A')} |",
                f"| 理论预测 | {t3_d.get('predicted_eps_min', 'N/A')} |",
                "",
                f'**解读**："几何规则" → 高斯-博内 → 残差不可消',
                "",
            ])

        # 定理间逻辑链
        lines.extend([
            "### 三条定理的逻辑链",
            "",
            "```",
            "  定理一 (χ ≠ 0)          定理二 (τ/κ = const)      定理三 (ε_min > 0)",
            "  ─────────────          ──────────────────        ────────────────",
            "  隐流形闭合拓扑     →   合法路径测地线/平行移动  →   拓扑闭合的几何残差",
            "  结构化即有拓扑         τ/κ 共形不变               残差不可局部消除",
            "```",
            "",
            f"十三字公理验证：**{'✓ 成立' if part_d3.get('axiom_verified') else '✗ 待更多实验'}**",
            "",
        ])

    # ── 原有后续内容 ──
    lines.extend([
        "### 下一步建议",
        "",
        f"{part_d.get('next_experiment', 'N/A')}",
        "",
        "---",
        "",
        "## exp9→exp10 认知进化追踪",
        "",
        "| 假设 | exp9 结果 | exp10 结果 | 方向 |",
        "|---|---|---|---|",
        f"| H-rigid | SUPPORT | (已固化，不再重测) | ➡️ 沉淀 |",
        f"| H-helix | REFUTED (R²=0.057) | {verdicts.get('H-helix', 'N/A')} | {'⬆️' if verdicts.get('H-helix') == 'SUPPORT' else '➡️'} |",
        f"| H-push | REFUTED (d=0.007) | {verdicts.get('H-push', 'N/A')} | {'⬆️' if verdicts.get('H-push') == 'SUPPORT' else '➡️'} |",
        f"| H-phase | INCONCLUSIVE | {verdicts.get('H-phase', 'N/A')} | {'⬆️' if verdicts.get('H-phase') in ('SUPPORT', 'GRADUAL') else '➡️'} |",
        f"| H-topo-inhibit | SUPPORT (τ×35.64) | (已固化，不再重测) | ➡️ 沉淀 |",
        f"| **几何动力学** | 未监控 | {part_d2.get('verdict', 'N/A')} | 🆕 新增 |",
        f"| **三定理验证** | 未实现 | {part_d3.get('axiom_verified', 'N/A') if part_d3 else 'N/A'} | 🆕 新增 |",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
