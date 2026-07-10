"""
v22_step1_minicpm.py — MiniCPM5-1B 完整实验 (待办1+2+3+5)
20 prompts, prefill/decode, affine compression
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
    RidgeOptimizer,
)

REPORT_DIR = Path(__file__).parent

POS_PROMPTS_20 = [
    "A scientist walked into the laboratory and discovered that the chemical reaction had produced",
    "The king ordered his army to march north because the enemy had captured",
    "After years of research, the team finally found the cure for the disease by",
    "The detective examined the crime scene and noticed that the broken window indicated",
    "The engineer designed the bridge to withstand earthquakes by using reinforced",
    "When the sun set behind the mountains, the villagers knew that winter was",
    "The captain ordered the ship to change course after the lookout spotted",
    "The farmer planted the seeds in early spring because the soil was",
    "The architect redesigned the building after realizing that the foundation was",
    "The doctor prescribed a new medication after the test results showed",
    "The teacher explained the mathematical proof step by step until the students",
    "The general retreated his forces when the scouts reported that the enemy",
    "The chemist adjusted the pH level of the solution and observed that",
    "The programmer debugged the code by tracing the error back to",
    "The historian analyzed the ancient text and concluded that the civilization",
    "The biologist discovered a new species in the rainforest that had evolved",
    "The economist predicted the market crash based on the indicators that",
    "The pilot diverted the flight after the weather radar detected",
    "The philosopher argued that consciousness emerges when the brain processes",
    "The musician composed the symphony by combining the melody with",
]

SCR_PROMPTS_20 = [
    "laboratory the walked scientist A into and discovered that the chemical had reaction produced",
    "army his ordered The king march to north because enemy the captured had",
    "research of years After team the finally found the cure the for disease by",
    "crime the examined detective The scene and noticed that broken the window indicated",
    "engineer The designed bridge the to withstand by earthquakes using reinforced",
    "sun the set behind When mountains the villagers the knew that winter was",
    "captain The ordered ship the to course change after lookout the spotted",
    "farmer The planted seeds the in spring early because the soil was",
    "architect The redesigned building the after realizing that foundation the was",
    "doctor The prescribed medication a new after the test results showed",
    "teacher The explained mathematical the proof step by step until the students",
    "general The retreated forces his when scouts the reported that the enemy",
    "chemist The adjusted pH the level of the solution and observed that",
    "programmer The debugged code the by tracing error the back to",
    "historian The analyzed ancient the text and concluded that the civilization",
    "biologist The discovered species a new in the rainforest that had evolved",
    "economist The predicted market the crash based on the indicators that",
    "pilot The diverted flight the after the weather radar detected",
    "philosopher The argued that consciousness emerges when the brain processes",
    "musician The composed symphony the by combining the melody with",
]

RND_PROMPTS_20 = [
    "purple elephant singing quantum banana telescope philosophical motorcycle dictionary",
    "seven forty algorithm whispering cylinder harmonious blanket satellite vocabulary",
    "triangular philosophy dancing refrigerator seventeen orchestral pavement gravity",
    "hexagonal nostalgia climbing umbrella forty-two symphonic cardboard longitude",
    "cylindrical democracy juggling thermometer seventeen philosophical horizon alphabet",
    "crimson saxophone calculating meadow hexadecimal nocturnal pavement dictionary",
    "spherical metaphor boiling calendar thirty-seven barometric cardboard telescope",
    "octagonal paradox swimming lantern twenty-three harmonic velvet algorithm",
    "pentagonal symphony running telescope forty-one meteorological canvas paradox",
    "rhomboid epistemology flying compass thirty-six orchestral marble nostalgia",
    "elliptical taxonomy walking microscope forty-five harmonic granite algorithm",
    "conical ontology jumping thermometer twenty-nine symphonic limestone paradox",
    "pyramidal phenomenology hiking compass thirty-eight barometric sandstone metaphor",
    "fractal cosmology sailing lantern forty-four meteorological basalt taxonomy",
    "logarithmic eschatology climbing microscope thirty-one harmonic shale ontology",
    "exponential teleology running telescope forty-seven barometric quartz phenomenology",
    "sinusoidal hermeneutics walking compass thirty-three meteorological granite cosmology",
    "hyperbolic dialectics jumping lantern forty-one orchestral marble eschatology",
    "polynomial exegesis hiking microscope thirty-five harmonic limestone teleology",
    "logistical apologetics sailing compass thirty-nine barometric sandstone hermeneutics",
]


def load_minicpm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    print("  Loading MiniCPM5-1B (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_trajectory(model, tokenizer, prompt, max_new_tokens=25, collect_prefill=False):
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


def run_ab_experiment(model, tokenizer, alpha_star, prompts_by_type, collect_prefill=False):
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
            traj = generate_trajectory(model, tokenizer, p, max_new_tokens=25, collect_prefill=collect_prefill)
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


def run_affine_compression(model, tokenizer, alpha_star, prompts):
    print("\n[待办5] Affine compression online calibration (MiniCPM5-1B)...")

    optimizer = RidgeOptimizer(variance_threshold=0.95, gamma=0.01)

    cal_traj = generate_trajectory(model, tokenizer, prompts[0], max_new_tokens=50)
    if cal_traj["decode"] is None or cal_traj["decode"].size(0) < 20:
        print("  Calibration trajectory too short, skipping")
        return None

    cal_hs = cal_traj["decode"]
    ridge_info = optimizer.auto_find_ridge(cal_hs)
    print(f"  Ridge: r={ridge_info['ridge_dim']}, compression={ridge_info['compression_ratio']:.3f}, R2={ridge_info['explained_variance']:.4f}")

    cal_list = [cal_hs[-3], cal_hs[-2], cal_hs[-1]]
    alpha_info = optimizer.auto_calibrate_alpha(cal_list)
    print(f"  Alpha*: {alpha_info['alpha_star']:.4f}")

    test_traj = generate_trajectory(model, tokenizer, prompts[1], max_new_tokens=50)
    if test_traj["decode"] is None or test_traj["decode"].size(0) < 20:
        print("  Test trajectory too short, skipping")
        return None

    test_hs = test_traj["decode"]
    cosine_sims = []
    mses = []
    pc_ratios = []

    engine = ThermodynamicEngine(alpha_star=alpha_star, gamma=0.01)

    for t in range(2, min(test_hs.size(0), 50)):
        h_curr = test_hs[t:t+1]
        h_recon, metrics = optimizer.step_decode(h_curr)
        cosine_sims.append(metrics["cosine_sim"])
        mses.append(metrics["mse"])
        if metrics["pc_ratio"] > 0:
            pc_ratios.append(metrics["pc_ratio"])

    per_token_r = engine.compute_per_token_ratio(test_hs)

    return {
        "ridge_dim": ridge_info["ridge_dim"],
        "compression_ratio": ridge_info["compression_ratio"],
        "explained_variance": ridge_info["explained_variance"],
        "alpha_star_calibrated": alpha_info["alpha_star"],
        "cosine_sim_mean": float(np.mean(cosine_sims)) if cosine_sims else None,
        "mse_mean": float(np.mean(mses)) if mses else None,
        "pc_ratio_monitor_mean": float(np.mean(pc_ratios)) if pc_ratios else None,
        "per_token_pc_ratio": per_token_r.get("per_token_ratio"),
        "cross_prompt": True,
    }


def main():
    print("=" * 70)
    print("v22 Step 1: MiniCPM5-1B 完整实验 (待办1+2+3+5)")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    model, tokenizer = load_minicpm()

    prompts = {
        "pos": POS_PROMPTS_20,
        "scr": SCR_PROMPTS_20,
        "rnd": RND_PROMPTS_20,
    }

    print("\n[待办1+2+3] MiniCPM5-1B bf16 (20 prompts, prefill vs decode)...")
    results = run_ab_experiment(model, tokenizer, 1.46, prompts, collect_prefill=True)

    affine = run_affine_compression(model, tokenizer, 1.46, POS_PROMPTS_20)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "library_version": "0.3.0",
        "minicpm5_1b_bf16_20prompts": results,
    }
    if affine:
        all_results["minicpm5_1b_affine_compression"] = affine

    report_path = REPORT_DIR / "v22_step1_minicpm_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("MiniCPM5-1B 实验报告")
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
        print(f"\n  [待办3] Prefill vs Decode r(0.95):")
        for t in ["pos", "scr", "rnd"]:
            if t in data and data[t].get("prefill_r_095_mean"):
                pf = data[t]["prefill_r_095_mean"]
                dc = data[t].get("decode_r_095_mean") or data[t].get("r_095_mean")
                print(f"    {t}: prefill={pf:.1f}, decode={dc:.1f}, ratio={pf/(dc+1e-8):.1f}x")

    if affine:
        print(f"\n--- [待办5] 仿射子空间压缩 ---")
        print(f"  Ridge dim: {affine['ridge_dim']}, Compression: {affine['compression_ratio']:.3f}")
        print(f"  R2: {affine['explained_variance']:.4f}, Cosine sim: {affine['cosine_sim_mean']:.4f}")
        print(f"  P_c/P_raw (monitor): {affine['pc_ratio_monitor_mean']:.4f}")
        print(f"  Per-token P_c/P_raw: {affine['per_token_pc_ratio']:.4f}")

    print(f"\nReport saved to: {report_path}")
    return all_results


if __name__ == "__main__":
    main()