import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Ornith-1___0-9B"
REPORT_PATH = Path(__file__).parent / "v20_ornith_per_token_Pc_report.json"

ALPHA_STAR_QWEN = 1.41
ALPHA_STAR_ORNITH = None
GAMMA = 0.01

PROMPT = "Explain the concept of entropy in thermodynamics in two sentences."

def calibrate_alpha(H, gamma=GAMMA):
    velocities = H[1:] - H[:-1]
    accelerations = velocities[1:] - velocities[:-1]
    velocities = velocities[:-1]
    F_res = accelerations + gamma * velocities
    P_raw = (F_res * velocities).sum(dim=1)
    v_sq = (velocities * velocities).sum(dim=1)
    mask = v_sq > 1e-8
    if mask.sum() > 0:
        alpha = P_raw[mask].sum() / v_sq[mask].sum()
        return alpha.item()
    return 1.41

def main():
    print("=" * 60)
    print("Ornith-1.0-9B: Per-Token P_c (v17 method, 4-bit)")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/4] Loading model (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.eval()

    inner = model.model
    layers = inner.layers
    num_layers = len(layers)

    config = model.config
    text_config = config.text_config if hasattr(config, 'text_config') else config
    layer_types = text_config.layer_types

    print("\n[2/4] Setting up hooks on all layers...")
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
    for li in range(num_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))

    print("\n[3/4] Running generation...")
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)

    num_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"  Generated {num_generated} tokens")

    for h in hooks:
        h.remove()

    print("\n[4/4] Per-token P_c calculation...")
    print(f"  γ = {GAMMA}")
    print()

    results = {}
    print(f"  {'Layer':>5} | {'Type':>17} | {'α*':>6} | {'Pc/Praw':>8} | {'cos(v,a)':>8} | {'r(0.95)':>7}")
    print("  " * 1 + "-" * 70)

    for li in range(num_layers):
        states = hidden_states_store.get(li, [])
        lt = layer_types[li] if layer_types else "unknown"

        if len(states) < 3:
            continue

        H = torch.cat(states, dim=0)
        T = H.shape[0]

        alpha_star = calibrate_alpha(H)
        per_token_ratios = []
        per_token_cos_va = []

        for t in range(2, T):
            h_t = H[t]
            h_t1 = H[t-1]
            h_t2 = H[t-2]

            v_t = h_t - h_t1
            a_t = h_t - 2 * h_t1 + h_t2

            F_res = a_t + GAMMA * v_t
            P_raw = torch.sum(F_res * v_t).item()
            P_active = alpha_star * torch.sum(v_t * v_t).item()
            P_c = P_raw - P_active
            ratio = abs(P_c) / (abs(P_raw) + 1e-8)
            per_token_ratios.append(ratio)

            v_norm = torch.norm(v_t).item()
            a_norm = torch.norm(a_t).item()
            if v_norm > 1e-8 and a_norm > 1e-8:
                cos_va = torch.sum(v_t * a_t).item() / (v_norm * a_norm)
                per_token_cos_va.append(cos_va)

        mean_ratio = sum(per_token_ratios) / len(per_token_ratios) if per_token_ratios else 0.0
        mean_cos = sum(per_token_cos_va) / len(per_token_cos_va) if per_token_cos_va else 0.0

        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            total = S.sum()
            cumsum = torch.cumsum(S, dim=0)
            r_eff = int((cumsum / total >= 0.95).nonzero()[0, 0]) + 1
        except Exception:
            r_eff = 0

        results[str(li)] = {
            "layer_type": lt,
            "alpha_star": alpha_star,
            "Pc_Praw_mean": mean_ratio,
            "cos_v_a_mean": mean_cos,
            "effective_rank": r_eff,
        }

        print(f"  {li:5d} | {lt:>17} | {alpha_star:6.2f} | {mean_ratio:8.4f} | {mean_cos:8.4f} | {r_eff:7d}")

    report = {
        "model": "Ornith-1.0-9B",
        "quantization": "4bit-nf4",
        "gamma": GAMMA,
        "prompt": PROMPT,
        "num_generated": num_generated,
        "per_layer": results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("ORNITH PER-TOKEN P_c COMPLETE")

if __name__ == "__main__":
    main()