import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
REPORT_PATH = Path(__file__).parent / "v20_qwen25_correct_Pc_report.json"

ALPHA_STAR = 1.41
GAMMA = 0.01

PROMPT = "Explain the concept of entropy in thermodynamics in two sentences."

def main():
    print("=" * 60)
    print("Qwen2.5-7B: Correct P_c Calculation (Langevin decomposition)")
    print("=" * 60)

    print("\n[1/3] Loading model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.eval()

    inner = model.model
    layers = inner.layers
    num_layers = len(layers)

    print("\n[2/3] Setting up hooks + generating...")
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

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)

    num_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"  Generated {num_generated} tokens")

    for h in hooks:
        h.remove()

    print("\n[3/3] Computing P_c with CORRECT Langevin decomposition...")
    print(f"  P_c = (a + γv)·v - α*|v|²")
    print(f"  α* = {ALPHA_STAR}, γ = {GAMMA}")
    print()

    results = {}
    print(f"  {'Layer':>5} | {'P_c/P_raw':>10} | {'cos(v,a)':>8} | {'cos(v,Fc)':>9} | {'r(0.95)':>7}")
    print("  " + "-" * 55)

    for li in range(num_layers):
        states = hidden_states_store.get(li, [])
        if len(states) < 3:
            continue

        H = torch.cat(states, dim=0)
        T = H.shape[0]

        velocities = H[1:] - H[:-1]
        accelerations = velocities[1:] - velocities[:-1]
        velocities = velocities[:-1]

        F_res = accelerations + GAMMA * velocities
        P_raw = (F_res * velocities).sum(dim=1)
        P_active = ALPHA_STAR * (velocities * velocities).sum(dim=1)
        P_c = P_raw - P_active

        vel_norms = torch.norm(velocities, dim=1)
        acc_norms = torch.norm(accelerations, dim=1)
        Fc_norms = torch.norm(F_res - ALPHA_STAR * velocities, dim=1)

        mask = (vel_norms > 1e-8)
        if mask.sum() > 0:
            ratio = (P_c[mask].abs().mean() / (P_raw[mask].abs().mean() + 1e-12)).item()
            cos_va = ((velocities[mask] * accelerations[mask]).sum(dim=1) / (vel_norms[mask] * acc_norms[mask] + 1e-12)).mean().item()
            F_c_vec = F_res[mask] - ALPHA_STAR * velocities[mask]
            cos_vFc = ((velocities[mask] * F_c_vec).sum(dim=1) / (vel_norms[mask] * torch.norm(F_c_vec, dim=1) + 1e-12)).mean().item()
        else:
            ratio = 0.0
            cos_va = 0.0
            cos_vFc = 0.0

        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            total = S.sum()
            cumsum = torch.cumsum(S, dim=0)
            r_eff = int((cumsum / total >= 0.95).nonzero()[0, 0]) + 1
        except Exception:
            r_eff = 0

        results[str(li)] = {
            "Pc_Praw_ratio": ratio,
            "cos_v_a": cos_va,
            "cos_v_Fc": cos_vFc,
            "effective_rank": r_eff,
        }

        print(f"  {li:5d} | {ratio:10.4f} | {cos_va:8.4f} | {cos_vFc:9.4f} | {r_eff:7d}")

    report = {
        "model": "Qwen2.5-7B-Instruct",
        "quantization": "bf16",
        "alpha_star": ALPHA_STAR,
        "gamma": GAMMA,
        "prompt": PROMPT,
        "num_generated": num_generated,
        "per_layer": results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("CORRECT P_c CALCULATION COMPLETE")

if __name__ == "__main__":
    main()