import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Ornith-1___0-9B"
REPORT_PATH = Path(__file__).parent / "v20_ornith_bf16_thermo_report.json"

PROMPT = "Explain the concept of entropy in thermodynamics in two sentences."

def main():
    print("=" * 60)
    print("Ornith-1.0-9B BF16 CPU-Offload Thermodynamic Test")
    print("=" * 60)

    print("\n[1/4] Loading model (bf16, CPU offload)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()

    device_map = model.hf_device_map
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    print(f"  GPU layers: {gpu_layers}, CPU layers: {cpu_layers}")

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
        hook = layers[li].register_forward_hook(make_hook(li))
        hooks.append(hook)

    print("\n[3/4] Running generation (slow due to CPU offload)...")
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False, temperature=1.0, top_p=1.0)

    num_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"  Generated {num_generated} tokens")

    print("\n[4/4] Computing per-layer thermodynamics...")
    results = {}
    print(f"\n  {'Layer':>5} | {'Type':>17} | {'Pc/Praw':>8} | {'cos(Pc)':>8} | {'r(0.95)':>7} | {'|v|':>9} | {'|a|':>9}")
    print("  " + "-" * 80)

    for li in range(num_layers):
        states = hidden_states_store.get(li, [])
        lt = layer_types[li] if layer_types else "unknown"

        if len(states) < 3:
            print(f"  {li:5d} | {lt:>17} | SKIPPED ({len(states)} states)")
            continue

        H = torch.cat(states, dim=0)
        T = H.shape[0]
        d = H.shape[1]

        velocities = H[1:] - H[:-1]
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
            ratio = 0.0

        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            total = S.sum()
            cumsum = torch.cumsum(S, dim=0)
            r_eff = int((cumsum / total >= 0.95).nonzero()[0, 0]) + 1
        except Exception:
            r_eff = 0

        cos_mean = cos_Pc.mean().item() if len(cos_Pc) > 0 else 0.0
        results[str(li)] = {
            "layer_type": lt,
            "Pc_Praw_ratio": ratio,
            "cos_Pc_mean": cos_mean,
            "effective_rank": r_eff,
            "mean_vel_norm": vel_norms.mean().item(),
            "mean_acc_norm": acc_norms.mean().item(),
        }

        print(f"  {li:5d} | {lt:>17} | {ratio:8.4f} | {cos_mean:8.4f} | {r_eff:7d} | {vel_norms.mean().item():9.2e} | {acc_norms.mean().item():9.2e}")

    for h in hooks:
        h.remove()

    report = {
        "model": "Ornith-1.0-9B",
        "quantization": "bf16_cpu_offload",
        "prompt": PROMPT,
        "num_generated": num_generated,
        "per_layer": results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("ORNITH BF16 THERMODYNAMIC TEST COMPLETE")

if __name__ == "__main__":
    main()