import torch
import json
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Ornith-1___0-9B"
REPORT_PATH = Path(__file__).parent / "v20_ornith_thermo_report.json"

PROMPTS = [
    "Explain the concept of entropy in thermodynamics.",
    "What is the relationship between symmetry and conservation laws?",
    "Describe the process of photosynthesis in plants.",
    "How does machine learning differ from traditional programming?",
    "What are the fundamental forces of nature?",
    "Explain the theory of general relativity briefly.",
    "What is the role of DNA in genetics?",
    "How do vaccines work to protect against disease?",
    "Describe the water cycle in nature.",
]

def compute_effective_rank(H, threshold=0.95):
    if H.shape[0] < 2:
        return 0, H.shape[1]
    H_centered = H - H.mean(dim=0, keepdim=True)
    try:
        U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
        total = S.sum()
        if total < 1e-10:
            return 0, H.shape[1]
        cumsum = torch.cumsum(S, dim=0)
        r = int((cumsum / total >= threshold).nonzero()[0, 0]) + 1
        return r, H.shape[1]
    except Exception:
        return 0, H.shape[1]

def main():
    print("=" * 60)
    print("Ornith-1.0-9B Thermodynamic Monitoring Test")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Model: {type(model).__name__}")

    inner = model.model
    layers = inner.layers
    num_layers = len(layers)
    print(f"  Layers: {num_layers}")

    config = inner.config if hasattr(inner, 'config') else model.config
    if hasattr(config, 'text_config'):
        text_config = config.text_config
    else:
        text_config = config
    layer_types = text_config.layer_types if hasattr(text_config, 'layer_types') else None

    full_attn_layers = []
    linear_attn_layers = []
    if layer_types:
        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                full_attn_layers.append(i)
            else:
                linear_attn_layers.append(i)
        print(f"  Full attention layers: {full_attn_layers}")
        print(f"  Linear attention layers: {len(linear_attn_layers)}")
    else:
        full_attn_layers = list(range(num_layers))

    hook_layers = full_attn_layers + [num_layers - 1] if (num_layers - 1) not in full_attn_layers else full_attn_layers
    hook_layers = sorted(set(hook_layers))
    print(f"  Hook layers: {hook_layers}")

    print("\n[2/5] Setting up hooks...")
    hidden_states_store = defaultdict(list)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() == 3 and h.shape[1] == 1:
                h_detached = h[:, 0, :].detach().float().cpu()
                hidden_states_store[layer_idx].append(h_detached)
        return hook_fn

    hooks = []
    for li in hook_layers:
        hook = layers[li].register_forward_hook(make_hook(li))
        hooks.append(hook)
    print(f"  Registered {len(hooks)} hooks")

    print("\n[3/5] Running generation with thermodynamic monitoring...")
    all_results = {}

    for pidx, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pidx+1}/{len(PROMPTS)}: {prompt[:50]}...")
        hidden_states_store.clear()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        num_generated = new_tokens.shape[0]
        print(f"    Generated {num_generated} tokens")

        prompt_result = {
            "prompt": prompt[:60],
            "num_generated_tokens": num_generated,
            "generation_preview": decoded[:80],
            "per_layer": {},
        }

        for li in hook_layers:
            states = hidden_states_store.get(li, [])
            if len(states) < 3:
                print(f"    Layer {li}: only {len(states)} states, skipping")
                continue

            H = torch.cat(states, dim=0)
            T = H.shape[0]
            d = H.shape[1]

            velocities = H[1:] - H[:-1]
            if T < 3:
                continue
            accelerations = velocities[1:] - velocities[:-1]
            velocities = velocities[:-1]

            vel_norms = torch.norm(velocities, dim=1)
            acc_norms = torch.norm(accelerations, dim=1)

            P_c = (velocities * accelerations).sum(dim=1)
            P_raw = vel_norms * acc_norms

            mask = (vel_norms > 1e-8) & (acc_norms > 1e-8)
            if mask.sum() > 0:
                cos_Pc = P_c[mask] / (P_raw[mask] + 1e-12)
                mean_Pc = P_c[mask].mean().item()
                mean_Praw = P_raw[mask].mean().item()
                ratio = abs(mean_Pc) / (mean_Praw + 1e-12)
            else:
                cos_Pc = torch.tensor([])
                mean_Pc = 0.0
                mean_Praw = 0.0
                ratio = 0.0

            r_eff, d_full = compute_effective_rank(H, threshold=0.95)

            prompt_result["per_layer"][str(li)] = {
                "T": T,
                "d": d,
                "mean_vel_norm": vel_norms.mean().item(),
                "mean_acc_norm": acc_norms.mean().item(),
                "mean_Pc": mean_Pc,
                "mean_Praw": mean_Praw,
                "Pc_Praw_ratio": ratio,
                "cos_Pc_mean": cos_Pc.mean().item() if len(cos_Pc) > 0 else None,
                "cos_Pc_std": cos_Pc.std().item() if len(cos_Pc) > 1 else None,
                "effective_rank_r095": r_eff,
                "compression_ratio": 1.0 - r_eff / d_full if d_full > 0 else 0.0,
            }

        all_results[f"prompt_{pidx}"] = prompt_result

    print("\n[4/5] Computing cross-prompt statistics...")
    summary = {"model": "Ornith-1.0-9B", "quantization": "4bit-nf4", "per_layer_summary": {}}

    for li in hook_layers:
        li_str = str(li)
        Pc_ratios = []
        cos_vals = []
        ranks = []
        for pkey in all_results:
            lr = all_results[pkey]["per_layer"].get(li_str)
            if lr:
                Pc_ratios.append(lr["Pc_Praw_ratio"])
                if lr["cos_Pc_mean"] is not None:
                    cos_vals.append(lr["cos_Pc_mean"])
                ranks.append(lr["effective_rank_r095"])

        if Pc_ratios:
            summary["per_layer_summary"][li_str] = {
                "avg_Pc_Praw_ratio": sum(Pc_ratios) / len(Pc_ratios),
                "avg_cos_Pc": sum(cos_vals) / len(cos_vals) if cos_vals else None,
                "avg_effective_rank": sum(ranks) / len(ranks),
                "num_prompts": len(Pc_ratios),
            }

    print("\n  Layer | Pc/P_raw | cos(Pc) | r(0.95)")
    print("  " + "-" * 45)
    for li in hook_layers:
        s = summary["per_layer_summary"].get(str(li))
        if s:
            cos_str = f"{s['avg_cos_Pc']:.4f}" if s['avg_cos_Pc'] is not None else "N/A"
            print(f"  {li:5d} | {s['avg_Pc_Praw_ratio']:.4f}   | {cos_str}  | {s['avg_effective_rank']:.1f}")

    print("\n[5/5] Saving report...")
    report = {
        "model": "Ornith-1.0-9B",
        "quantization": "4bit-nf4",
        "num_layers": num_layers,
        "full_attention_layers": full_attn_layers,
        "hook_layers": hook_layers,
        "num_prompts": len(PROMPTS),
        "summary": summary,
        "detailed_results": all_results,
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to {REPORT_PATH}")

    for h in hooks:
        h.remove()

    print("\nORNITH THERMODYNAMIC MONITORING COMPLETE")
    return report

if __name__ == "__main__":
    main()