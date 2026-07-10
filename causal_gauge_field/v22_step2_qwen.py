"""
v22_step2_qwen.py — Qwen2.5-7B 实验 (待办1)
5 prompts, max_new_tokens=15 (CPU offload slow)
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "llm-thermodynamics"))

from llm_thermo import (
    ThermodynamicEngine,
    KinematicsExtractor,
    NESSEvaluator,
    SubRiemannianAnalyzer,
)

REPORT_DIR = Path(__file__).parent

POS_5 = [
    "A scientist walked into the laboratory and discovered that the chemical reaction had produced",
    "The king ordered his army to march north because the enemy had captured",
    "After years of research, the team finally found the cure for the disease by",
    "The detective examined the crime scene and noticed that the broken window indicated",
    "The engineer designed the bridge to withstand earthquakes by using reinforced",
]

SCR_5 = [
    "laboratory the walked scientist A into and discovered that the chemical had reaction produced",
    "army his ordered The king march to north because enemy the captured had",
    "research of years After team the finally found the cure the for disease by",
    "crime the examined detective The scene and noticed that broken the window indicated",
    "engineer The designed bridge the to withstand by earthquakes using reinforced",
]

RND_5 = [
    "purple elephant singing quantum banana telescope philosophical motorcycle dictionary",
    "seven forty algorithm whispering cylinder harmonious blanket satellite vocabulary",
    "triangular philosophy dancing refrigerator seventeen orchestral pavement gravity",
    "hexagonal nostalgia climbing umbrella forty-two symphonic cardboard longitude",
    "cylindrical democracy juggling thermometer seventeen philosophical horizon alphabet",
]


def generate_trajectory(model, tokenizer, prompt, max_new_tokens=15, collect_prefill=False):
    prefill_states = []
    decode_states = []

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            if h.shape[1] > 1:
                for t in range(h.shape[1]):
                    prefill_states.append(h[0, t, :].detach().float().cpu())
            elif h.shape[1] == 1:
                decode_states.append(h[0, 0, :].detach().float().cpu())

    target_layer = model.model.layers[-1]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    handle.remove()

    result = {"decode": None, "prefill": None}
    if len(decode_states) >= 10:
        result["decode"] = torch.stack(decode_states)
    if collect_prefill and len(prefill_states) >= 5:
        result["prefill"] = torch.stack(prefill_states)

    return result


def run_ab(model, tokenizer, alpha_star, prompts_by_type, collect_prefill=False):
    engine = ThermodynamicEngine(alpha_star=alpha_star, gamma=0.01)
    ness_eval = NESSEvaluator(alpha_star=alpha_star)
    sub_riem = SubRiemannianAnalyzer(alpha_star=alpha_star)

    results = {}

    for text_type, prompts in prompts_by_type.items():
        print(f"    {text_type}: {len(prompts)} prompts...", end=" ", flush=True)
        trajectories_decode = []
        trajectories_prefill = []
        valid = 0

        for p in prompts:
            traj = generate_trajectory(model, tokenizer, p, max_new_tokens=15, collect_prefill=collect_prefill)
            if traj["decode"] is not None:
                trajectories_decode.append(traj["decode"])
                valid += 1
            if traj.get("prefill") is not None:
                trajectories_prefill.append(traj["prefill"])

        print(f"{valid} valid")

        pc_ratios = []
        batch_ratios = []
        vel_norms = []
        t_effs = []
        k_subs = []
        r_095s = []
        ness_results = []

        for traj in trajectories_decode:
            if traj.size(0) < 10:
                continue

            r = engine.compute_per_token_ratio(traj)
            if "error" not in r:
                pc_ratios.append(r["per_token_ratio"])
                batch_ratios.append(r["batch_ratio_artifact"])

            vel = traj[1:] - traj[:-1]
            vel_norms.append(float(vel.norm(dim=-1).mean().item()))
            r_095s.append(KinematicsExtractor.compute_effective_rank(vel, 0.95))

            t_eff = ness_eval.effective_temperature(traj)
            t_effs.append(t_eff)

            if traj.size(0) >= 4:
                k_result = sub_riem.analyze_trajectory_curvature(traj)
                if "error" not in k_result:
                    k_subs.append(k_result["K_sub_mean"])

            ness_r = ness_eval.evaluate(traj)
            if "error" not in ness_r:
                ness_results.append(ness_r)

        type_result = {
            "n_trajectories": len(pc_ratios),
            "pc_ratio_per_token_mean": float(np.mean(pc_ratios)) if pc_ratios else None,
            "pc_ratio_per_token_std": float(np.std(pc_ratios)) if pc_ratios else None,
            "batch_ratio_artifact_mean": float(np.mean(batch_ratios)) if batch_ratios else None,
            "inflation_factor": float(np.mean(batch_ratios) / (np.mean(pc_ratios) + 1e-8)) if pc_ratios and batch_ratios else None,
            "vel_norm_mean": float(np.mean(vel_norms)) if vel_norms else None,
            "T_eff_mean": float(np.mean(t_effs)) if t_effs else None,
            "K_sub_mean": float(np.mean(k_subs)) if k_subs else None,
            "r_095_mean": float(np.mean(r_095s)) if r_095s else None,
            "sigma_mean": float(np.mean([r["entropy_production_sigma"] for r in ness_results])) if ness_results else None,
            "J_FP_mean": float(np.mean([r["J_FP_norm_per_dim"] for r in ness_results])) if ness_results else None,
            "CV_mean": float(np.mean([r["coefficient_of_variation"] for r in ness_results])) if ness_results else None,
            "ness_verdict": ness_results[0]["ness_verdict"] if ness_results else None,
        }

        if collect_prefill and trajectories_prefill:
            prefill_r095 = [KinematicsExtractor.compute_effective_rank(t[1:] - t[:-1], 0.95) for t in trajectories_prefill if t.size(0) >= 5]
            type_result["prefill_r_095_mean"] = float(np.mean(prefill_r095)) if prefill_r095 else None
            type_result["decode_r_095_mean"] = type_result["r_095_mean"]

        results[text_type] = type_result

    return results


def main():
    print("=" * 70)
    print("v22 Step 2: Qwen2.5-7B 实验 (5 prompts, max_new_tokens=15)")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
    print("  Loading Qwen2.5-7B (bf16, CPU offload)...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    prompts = {"pos": POS_5, "scr": SCR_5, "rnd": RND_5}

    print("\n[待办1] Qwen2.5-7B bf16 (5 prompts)...")
    results = run_ab(model, tokenizer, 1.41, prompts, collect_prefill=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "library_version": "0.3.0",
        "qwen25_7b_bf16_5prompts": results,
    }

    report_path = REPORT_DIR / "v22_step2_qwen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("Qwen2.5-7B 实验报告")
    print("=" * 70)
    data = results
    print(f"{'类型':>5s} | {'P_c/P_raw':>10s} | {'vel_norm':>10s} | {'T_eff':>8s} | {'r(0.95)':>8s} | {'σ':>10s} | {'NESS':>6s} | N")
    print("-" * 80)
    for t in ["pos", "scr", "rnd"]:
        if t in data:
            d = data[t]
            pc = d.get("pc_ratio_per_token_mean")
            vn = d.get("vel_norm_mean")
            te = d.get("T_eff_mean")
            r9 = d.get("r_095_mean")
            sig = d.get("sigma_mean")
            nv = d.get("ness_verdict", "?")
            n = d.get("n_trajectories", 0)
            print(f"{t:>5s} | {pc:10.4f} | {vn:10.2f} | {te:8.2f} | {r9:8.1f} | {sig:10.1f} | {nv:>6s} | {n}")

    if "pos" in data and data["pos"].get("prefill_r_095_mean"):
        print(f"\n  Prefill vs Decode r(0.95):")
        for t in ["pos", "scr", "rnd"]:
            if t in data and data[t].get("prefill_r_095_mean"):
                pf = data[t]["prefill_r_095_mean"]
                dc = data[t].get("decode_r_095_mean") or data[t].get("r_095_mean")
                print(f"    {t}: prefill={pf:.1f}, decode={dc:.1f}, ratio={pf/(dc+1e-8):.1f}x")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()