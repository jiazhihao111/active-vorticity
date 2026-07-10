"""
GUIT-TRT Phase 1: Qwen2.5-7B CausalKV + ThermoProbe verification
"""

import torch
import time
import json
from collections import deque
from typing import List, Tuple

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"


class ThermoProbe:
    def __init__(self, alpha_star=1.41, gamma=0.01):
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.h_history = deque(maxlen=3)
        self.ratio_history = []
        self.causal_impact_history = []

    def reset(self):
        self.h_history.clear()
        self.ratio_history.clear()
        self.causal_impact_history.clear()

    def step(self, h_curr):
        if h_curr.dim() == 3:
            h_curr = h_curr.squeeze(1)
        self.h_history.append(h_curr.detach().float())
        if len(self.h_history) < 3:
            return 0.0, 0.0
        h_t, h_t1, h_t2 = self.h_history[2], self.h_history[1], self.h_history[0]
        v_t = h_t - h_t1
        a_t = h_t - 2 * h_t1 + h_t2
        F_res = a_t + self.gamma * v_t
        P_raw = torch.sum(F_res * v_t, dim=-1).mean().item()
        P_active = self.alpha_star * torch.sum(v_t * v_t, dim=-1).mean().item()
        P_c = P_raw - P_active
        ratio = abs(P_c) / (abs(P_raw) + 1e-8)
        v_norm_sq = torch.sum(v_t * v_t, dim=-1, keepdim=True) + 1e-8
        a_par = (torch.sum(a_t * v_t, dim=-1, keepdim=True) / v_norm_sq) * v_t
        a_perp = a_t - a_par
        causal_impact = torch.norm(a_perp, dim=-1).mean().item()
        self.ratio_history.append(ratio)
        self.causal_impact_history.append(causal_impact)
        return ratio, causal_impact


def run_baseline(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed, vram


def run_guit(model, tokenizer, prompt, max_new_tokens=80, max_capacity=9999):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    probe = ThermoProbe(alpha_star=1.41)
    causal_scores = []
    eviction_count = 0
    sink_tokens = 4
    recent_tokens = 10

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generated_ids = inputs["input_ids"].clone()
    t0 = time.time()

    with torch.no_grad():
        past_kv = None
        for step_i in range(max_new_tokens):
            if step_i == 0:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                seq_len = inputs["input_ids"].shape[1] + step_i
                attn_mask = torch.ones(1, seq_len, device=model.device, dtype=torch.long)
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            h_last = outputs.hidden_states[-1][:, -1:, :]
            ratio, ci = probe.step(h_last)
            causal_scores.append(ci)

            past_kv = outputs.past_key_values

            if max_capacity < 9999 and len(causal_scores) > max_capacity:
                n_evict = len(causal_scores) - max_capacity
                es = sink_tokens
                ee = max(es, len(causal_scores) - recent_tokens)
                if ee > es:
                    ev = causal_scores[es:ee]
                    si = sorted(range(len(ev)), key=lambda i: ev[i])
                    eg = {idx + es for idx in set(si[:n_evict])}
                    keep = [i for i in range(len(causal_scores)) if i not in eg]
                    kt = torch.tensor(keep, device=model.device)
                    from transformers import DynamicCache
                    new_cache = DynamicCache()
                    legacy = past_kv.to_legacy_cache() if hasattr(past_kv, 'to_legacy_cache') else past_kv
                    for li, (k, v) in enumerate(legacy):
                        new_cache.update(k[:, :, kt, :], v[:, :, kt, :], li)
                    past_kv = new_cache
                    causal_scores = [causal_scores[i] for i in keep]
                    eviction_count += n_evict

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    stats = {
        "eviction_count": eviction_count,
        "ratio_mean": float(torch.tensor(probe.ratio_history).mean()) if probe.ratio_history else 0,
        "ratio_std": float(torch.tensor(probe.ratio_history).std()) if probe.ratio_history else 0,
        "ratio_min": float(min(probe.ratio_history)) if probe.ratio_history else 0,
        "ratio_max": float(max(probe.ratio_history)) if probe.ratio_history else 0,
        "causal_impact_mean": float(torch.tensor(probe.causal_impact_history).mean()) if probe.causal_impact_history else 0,
        "causal_impact_std": float(torch.tensor(probe.causal_impact_history).std()) if probe.causal_impact_history else 0,
    }
    return text, elapsed, vram, stats


def main():
    print("=" * 60)
    print("GUIT-TRT Phase 1 Verification: Qwen2.5-7B")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    prompt = (
        "小明早上起床，发现外面在下大雨。他决定取消原定的爬山计划。"
        "他走到厨房，打开冰箱，发现没有牛奶了。于是他拿起钱包和雨伞，"
        "出门去了楼下的便利店。在便利店，他买了一盒牛奶和两个面包。"
        "结账时，他遇到了邻居王阿姨。王阿姨告诉他，小区门口的路因为积水被封了。"
        "小明听后，决定绕道从后门回家。回到家后，他一边吃面包一边喝牛奶，"
        "看着窗外的雨，觉得这个周末的早晨也很惬意。\n\n"
        "问题：小明最终是如何回家的？他买了什么？"
    )
    max_new = 80

    results = {}

    print("\n--- Baseline ---")
    bt, btime, bvram = run_baseline(model, tokenizer, prompt, max_new)
    print(f"  VRAM={bvram:.2f}GB Time={btime:.2f}s")
    results["baseline"] = {"vram_gb": bvram, "time_s": btime, "text": bt}

    print("\n--- Probe (no eviction) ---")
    try:
        pt, ptime, pvram, ps = run_guit(model, tokenizer, prompt, max_new, 9999)
        print(f"  VRAM={pvram:.2f}GB Time={ptime:.2f}s")
        print(f"  P_c/P_raw: mean={ps['ratio_mean']:.4f} std={ps['ratio_std']:.4f} "
              f"range=[{ps['ratio_min']:.4f},{ps['ratio_max']:.4f}]")
        print(f"  Causal impact: mean={ps['causal_impact_mean']:.4f} std={ps['causal_impact_std']:.4f}")
        results["probe"] = {"vram_gb": pvram, "time_s": ptime, "stats": ps, "text": pt}
    except Exception as e:
        import traceback; traceback.print_exc()
        results["probe"] = {"error": str(e)}

    for cap in [128, 64]:
        label = f"KV-{cap}"
        print(f"\n--- {label} ---")
        try:
            t, tm, vm, st = run_guit(model, tokenizer, prompt, max_new, cap)
            print(f"  VRAM={vm:.2f}GB Time={tm:.2f}s Evictions={st['eviction_count']}")
            print(f"  P_c/P_raw: mean={st['ratio_mean']:.4f}")
            results[label] = {"vram_gb": vm, "time_s": tm, "stats": st, "text": t}
        except Exception as e:
            import traceback; traceback.print_exc()
            results[label] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    for k, v in results.items():
        if "error" in v:
            print(f"  {k}: ERROR - {v['error']}")
        else:
            print(f"  {k}: VRAM={v['vram_gb']:.2f}GB Time={v['time_s']:.2f}s")
            if "stats" in v:
                s = v["stats"]
                print(f"    P_c/P_raw: mean={s['ratio_mean']:.4f} [{s['ratio_min']:.4f},{s['ratio_max']:.4f}]")
    print("\nText comparison (last 50 chars):")
    for k in ["baseline", "probe", "KV-128", "KV-64"]:
        if k in results and "text" in results[k]:
            print(f"  {k}: ...{results[k]['text'][-50:]}")

    with open("v17_guit_vram_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("\nReport saved.")


if __name__ == "__main__":
    main()
