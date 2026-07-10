import json
from pathlib import Path

REPORT_PATH = Path(__file__).parent / "v20_cross_model_comparison_report.json"

comparison = {
    "experiment": "v20 Ornith-1.0-9B Cross-Model Thermodynamic Comparison",
    "date": "2026-07-10",
    "key_findings": [
        "1. P_c/P_raw≈0 (仿射约束) 在Ornith-1.0-9B上也成立 (0.09-0.15)",
        "2. 4bit量化显著增加r(0.95): Qwen2.5 32→40, Ornith 25→37",
        "3. Ornith bf16的r(0.95)=25远小于Qwen2.5的32 (尽管d更大)",
        "4. Ornith的隐状态轨迹在更低维子空间中运动 (压缩率99.4% vs 99.1%)",
        "5. 批量P_c计算方法有bug: 必须用逐token方法 (v17方式)",
        "6. Ornith的混合注意力(linear+full)不改变P_c≈0的结论",
    ],
    "models": {
        "Qwen2.5-7B-Instruct": {
            "params": "7B",
            "hidden_size": 3584,
            "num_layers": 28,
            "kv_heads": 4,
            "gqa": True,
            "attention_type": "full_attention (all layers)",
            "bf16_r095_layer0": 32,
            "bf16_r095_layer14": 33,
            "bf16_r095_layer27": 34,
            "4bit_r095_layer0": 40,
            "4bit_r095_layer14": 42,
            "4bit_r095_layer27": 44,
            "Pc_Praw_bf16": "0.08-0.12",
            "alpha_star": 1.41,
        },
        "Ornith-1.0-9B": {
            "params": "9B",
            "hidden_size": 4096,
            "num_layers": 32,
            "kv_heads": 4,
            "gqa": True,
            "attention_type": "mixed (24 linear_attention + 8 full_attention)",
            "bf16_r095_layer0": 23,
            "bf16_r095_layer15": 25,
            "bf16_r095_layer31": 25,
            "4bit_r095_layer0": 37,
            "4bit_r095_layer15": 43,
            "4bit_r095_layer31": 43,
            "Pc_Praw_4bit": "0.09-0.15",
            "alpha_star": 1.41,
            "note": "Self-evolving model claimed by DeepReinforce AI",
        },
    },
    "quantization_effect_on_r095": {
        "description": "4bit quantization increases effective rank by ~25-60%",
        "Qwen2.5-7B": {"bf16": 33, "4bit": 42, "increase": "27%"},
        "Ornith-1.0-9B": {"bf16": 25, "4bit": 41, "increase": "64%"},
    },
    "compression_potential": {
        "Qwen2.5-7B_bf16": {"r095": 33, "d": 3584, "compression": "99.1%"},
        "Ornith-1.0-9B_bf16": {"r095": 25, "d": 4096, "compression": "99.4%"},
    },
    "methodological_lessons": [
        "CRITICAL: P_c must be computed per-token (v17 method), NOT batch-averaged",
        "Batch P_c/P_raw≈0.84 is an artifact of averaging ratios with mixed signs",
        "Per-token P_c/P_raw≈0.09-0.15 correctly confirms P_c≈0 (affine constraint)",
        "4bit quantization inflates r(0.95) by 25-64%, making it unreliable for ridge analysis",
        "bf16 results should be treated as ground truth for thermodynamic analysis",
    ],
    "next_steps": [
        "Ornith bf16 RidgeOptimizer test (CPU offload, slow but accurate)",
        "Ornith online affine compression test (same-prompt SVD, r=128 verification)",
        "Update llm-thermodynamics library with per-token P_c computation fix",
        "Update verification report v5 with Ornith data",
    ],
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
print(f"Cross-model comparison saved to {REPORT_PATH}")