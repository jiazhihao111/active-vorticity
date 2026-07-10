import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
REPORT_PATH = Path(__file__).parent / "v20_qwen25_hidden_states_method_report.json"

PROMPT = "Explain the concept of entropy in thermodynamics in two sentences."

def main():
    print("=" * 60)
    print("Qwen2.5-7B: Hidden States Method vs Hook Method Comparison")
    print("=" * 60)

    print("\n[1/3] Loading model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    print("\n[2/3] Method A: output_hidden_states (prefill)...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    print(f"  Got {len(hidden_states)} layer hidden states")
    print(f"  Shape per layer: {hidden_states[0].shape}")

    print("\n  Prefill hidden states thermodynamics:")
    print(f"  {'Layer':>5} | {'Pc/Praw':>8} | {'cos(Pc)':>8} | {'r(0.95)':>7}")
    print("  " + "-" * 40)

    prefill_results = {}
    for li in [0, 7, 14, 21, 27]:
        H = hidden_states[li][0].float()
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
            ratio = abs(P_c[mask].mean().item()) / (P_raw[mask].mean().item() + 1e-12)
            cos_mean = cos_Pc.mean().item()
        else:
            ratio = 0.0
            cos_mean = 0.0

        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            total = S.sum()
            cumsum = torch.cumsum(S, dim=0)
            r_eff = int((cumsum / total >= 0.95).nonzero()[0, 0]) + 1
        except Exception:
            r_eff = 0

        prefill_results[str(li)] = {"Pc_Praw_ratio": ratio, "cos_Pc_mean": cos_mean, "effective_rank": r_eff}
        print(f"  {li:5d} | {ratio:8.4f} | {cos_mean:8.4f} | {r_eff:7d}")

    print("\n[3/3] Method B: Hook during generate (decode)...")
    from collections import defaultdict
    hook_states = defaultdict(list)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() == 3 and h.shape[1] == 1:
                h_detached = h[:, 0, :].detach().float().cpu()
                hook_states[layer_idx].append(h_detached)
        return hook_fn

    hooks = []
    for li in [0, 7, 14, 21, 27]:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)

    for h in hooks:
        h.remove()

    print(f"\n  Decode hook thermodynamics:")
    print(f"  {'Layer':>5} | {'Pc/Praw':>8} | {'cos(Pc)':>8} | {'r(0.95)':>7}")
    print("  " + "-" * 40)

    decode_results = {}
    for li in [0, 7, 14, 21, 27]:
        states = hook_states.get(li, [])
        if len(states) < 3:
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
            ratio = abs(P_c[mask].mean().item()) / (P_raw[mask].mean().item() + 1e-12)
            cos_mean = cos_Pc.mean().item()
        else:
            ratio = 0.0
            cos_mean = 0.0

        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            total = S.sum()
            cumsum = torch.cumsum(S, dim=0)
            r_eff = int((cumsum / total >= 0.95).nonzero()[0, 0]) + 1
        except Exception:
            r_eff = 0

        decode_results[str(li)] = {"Pc_Praw_ratio": ratio, "cos_Pc_mean": cos_mean, "effective_rank": r_eff}
        print(f"  {li:5d} | {ratio:8.4f} | {cos_mean:8.4f} | {r_eff:7d}")

    report = {
        "model": "Qwen2.5-7B-Instruct",
        "quantization": "bf16",
        "prompt": PROMPT,
        "method_A_prefill": prefill_results,
        "method_B_decode_hook": decode_results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("METHOD COMPARISON COMPLETE")

if __name__ == "__main__":
    main()