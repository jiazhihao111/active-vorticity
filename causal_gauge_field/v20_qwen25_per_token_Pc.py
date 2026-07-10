import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
REPORT_PATH = Path(__file__).parent / "v20_qwen25_per_token_Pc_report.json"

ALPHA_STAR = 1.41
GAMMA = 0.01

PROMPT = "Explain the concept of entropy in thermodynamics in two sentences."

def main():
    print("=" * 60)
    print("Qwen2.5-7B: Per-Token P_c (v17 method, batch hook)")
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
    for li in [0, 7, 14, 21, 27]:
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

    print("\n[3/3] Per-token P_c calculation (v17 method)...")
    print(f"  α* = {ALPHA_STAR}, γ = {GAMMA}")
    print()

    for li in [0, 7, 14, 21, 27]:
        states = hidden_states_store.get(li, [])
        if len(states) < 3:
            print(f"  Layer {li}: insufficient data")
            continue

        H = torch.cat(states, dim=0)
        T = H.shape[0]

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
            P_active = ALPHA_STAR * torch.sum(v_t * v_t).item()
            P_c = P_raw - P_active
            ratio = abs(P_c) / (abs(P_raw) + 1e-8)
            per_token_ratios.append(ratio)

            v_norm = torch.norm(v_t).item()
            a_norm = torch.norm(a_t).item()
            if v_norm > 1e-8 and a_norm > 1e-8:
                cos_va = torch.sum(v_t * a_t).item() / (v_norm * a_norm)
                per_token_cos_va.append(cos_va)

        mean_ratio = sum(per_token_ratios) / len(per_token_ratios)
        mean_cos = sum(per_token_cos_va) / len(per_token_cos_va) if per_token_cos_va else 0.0

        print(f"  Layer {li}: P_c/P_raw mean={mean_ratio:.4f} (min={min(per_token_ratios):.4f} max={max(per_token_ratios):.4f}), cos(v,a)={mean_cos:.4f}")

    report = {
        "model": "Qwen2.5-7B-Instruct",
        "quantization": "bf16",
        "alpha_star": ALPHA_STAR,
        "gamma": GAMMA,
        "prompt": PROMPT,
        "num_generated": num_generated,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("PER-TOKEN P_c CALCULATION COMPLETE")

if __name__ == "__main__":
    main()