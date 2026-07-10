"""
GUIT-TRT Phase 1 v2: MiniCPM5-1B Hook注入验证

修复v1三大问题：
1. Hook注入提取隐状态 → 消除观察者效应（不改变generate流程）
2. 4k+长文本 → KV Cache成为VRAM瓶颈
3. KV capacity > input_length → 避免上下文截断
"""

import torch
import time
import json
from collections import deque
from typing import List, Tuple

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"


class ThermoProbe:
    def __init__(self, alpha_star=1.46, gamma=0.01):
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


class HookThermoMonitor:
    """Hook注入式热力学监控器——不改变generate流程"""

    def __init__(self, model, alpha_star=1.46, gamma=0.01):
        self.probe = ThermoProbe(alpha_star, gamma)
        self.model = model
        self._hook = None
        self._registered = False

    def register(self, layer_index=-1):
        self.remove()
        layers = self.model.model.layers
        idx = layer_index if layer_index >= 0 else len(layers) + layer_index
        target = layers[idx]
        self._hook = target.register_forward_hook(self._hook_fn)
        self._registered = True
        return idx

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._registered = False

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if h.dim() == 3 and h.shape[1] == 1:
            self.probe.step(h[:, -1:, :])

    def get_stats(self):
        r = self.probe.ratio_history
        c = self.probe.causal_impact_history
        if not r:
            return {}
        return {
            "ratio_mean": float(torch.tensor(r).mean()),
            "ratio_std": float(torch.tensor(r).std()),
            "ratio_min": float(min(r)),
            "ratio_max": float(max(r)),
            "causal_impact_mean": float(torch.tensor(c).mean()) if c else 0,
            "causal_impact_std": float(torch.tensor(c).std()) if c else 0,
            "num_decode_steps": len(r),
        }


def build_long_prompt(tokenizer, target_tokens=4096):
    """构造4k+ token的长因果叙事"""
    base_story = (
        "在一个偏远的山村里，住着一个叫李明的年轻人。"
        "他每天早上五点起床，先去后山砍柴，然后把柴背到镇上的集市去卖。"
        "从村子到镇上要走两个小时的山路，沿途要经过三座桥和一片竹林。"
        "第一座桥是石拱桥，建于清朝，桥下是一条清澈的小溪。"
        "第二座桥是木桥，年久失修，每次走上去都会吱嘎作响。"
        "第三座桥是新建的水泥桥，是去年政府出资修建的。"
        "穿过竹林后，就能看到镇上的集市了。"
        "集市每周三和周六开市，李明通常在周三去卖柴。"
        "他每担柴能卖三十块钱，一个月下来能赚四五百块。"
        "这笔钱他要用来给生病的母亲买药，还要供妹妹上学。"
        "李明的母亲患有风湿病，每到下雨天就疼得厉害。"
        "镇上的赤脚医生王大夫说，需要一种山里罕见的草药才能根治。"
        "这种草药只生长在后山最高峰的悬崖上，采摘非常危险。"
        "李明决定冒险去采药。他选了一个晴朗的早晨出发。"
        "出发前，邻居张婶给了他一壶热茶和两个馒头。"
        "他沿着平时砍柴的小路走了大约一个小时，然后转向了一条人迹罕至的山径。"
        "山径越来越窄，两旁是茂密的灌木丛，偶尔能看到野兔和山鸡。"
        "走了两个小时后，他来到了悬崖脚下。"
        "抬头望去，悬崖高约百米，几乎垂直，只有零星的手抓点和脚踏点。"
        "他深吸一口气，开始攀爬。每一步都小心翼翼。"
        "爬到大约六十米的时候，他看到了那株草药，长在一个小岩缝里。"
        "他伸手去够，差了一点。他调整了一下姿势，终于把草药拔了出来。"
        "就在这时，他踩的一块石头松动了，他差点滑下去。"
        "他紧紧抓住旁边的岩缝，心跳加速。等了几分钟，才缓过神来。"
        "他慢慢往下爬，比上来时更加小心。终于安全回到地面。"
        "他带着草药回到村子，交给了王大夫。王大夫配了药，他母亲吃了以后果然好了很多。"
    )
    question = "\n\n根据以上故事，回答以下问题：\n1. 李明从村子到镇上要经过几座桥？分别是什么桥？\n2. 李明要去哪里采药？为什么危险？\n3. 谁给了李明出发前的食物？\n4. 李明的母亲患什么病？"

    prompt = base_story + question
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= target_tokens:
        return prompt

    while len(tokens) < target_tokens:
        prompt = base_story + "\n" + prompt
        tokens = tokenizer.encode(prompt)

    return prompt


def run_baseline(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    input_len = inputs["input_ids"].shape[1]
    return text, elapsed, vram, input_len


def run_with_monitor(model, tokenizer, prompt, max_new_tokens=100):
    """Hook注入监控——不改变generate流程"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    monitor = HookThermoMonitor(model, alpha_star=1.46)
    layer_idx = monitor.register(layer_index=-1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    stats = monitor.get_stats()
    monitor.remove()
    input_len = inputs["input_ids"].shape[1]
    return text, elapsed, vram, stats, input_len


def run_causal_kv(model, tokenizer, prompt, max_capacity, max_new_tokens=100):
    """CausalKV淘汰——手动逐token生成（仅用于KV淘汰测试）"""
    from transformers import DynamicCache
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if max_capacity < 100:
        return None, None, None, {"error": f"max_capacity({max_capacity}) too small"}, input_len

    probe = ThermoProbe(alpha_star=1.46)
    causal_scores = []
    eviction_count = 0
    sink_tokens = 4
    recent_tokens = 10

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generated_ids = inputs["input_ids"].clone()
    t0 = time.time()

    with torch.no_grad():
        cache = DynamicCache()
        for step_i in range(max_new_tokens):
            if step_i == 0:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                seq_len = input_len + step_i
                attn_mask = torch.ones(1, seq_len, device=model.device, dtype=torch.long)
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            h_last = outputs.hidden_states[-1][:, -1:, :]
            ratio, ci = probe.step(h_last)
            causal_scores.append(ci)

            if len(causal_scores) > max_capacity:
                n_evict = len(causal_scores) - max_capacity
                es = sink_tokens
                ee = max(es, len(causal_scores) - recent_tokens)
                if ee > es:
                    ev = causal_scores[es:ee]
                    si = sorted(range(len(ev)), key=lambda i: ev[i])
                    eg = {idx + es for idx in set(si[:n_evict])}
                    keep = [i for i in range(len(causal_scores)) if i not in eg]
                    kt = torch.tensor(keep, device=model.device)
                    legacy = cache.to_legacy_cache()
                    new_cache = DynamicCache()
                    for li, (k, v) in enumerate(legacy):
                        new_cache.update(k[:, :, kt, :], v[:, :, kt, :], li)
                    cache = new_cache
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
        "causal_impact_mean": float(torch.tensor(probe.causal_impact_history).mean()) if probe.causal_impact_history else 0,
    }
    return text, elapsed, vram, stats, input_len


def main():
    print("=" * 60)
    print("GUIT-TRT Phase 1 v2: MiniCPM5-1B Hook Verification")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    prompt = build_long_prompt(tokenizer, target_tokens=4096)
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"  Prompt length: {prompt_tokens} tokens")

    max_new = 100
    results = {}

    # 1. Baseline
    print("\n--- Baseline (native generate) ---")
    bt, btime, bvram, blen = run_baseline(model, tokenizer, prompt, max_new)
    print(f"  Input={blen} tokens | VRAM={bvram:.2f}GB | Time={btime:.2f}s")
    results["baseline"] = {"vram_gb": bvram, "time_s": btime, "input_len": blen, "text": bt}

    # 2. Hook监控（消除观察者效应）
    print("\n--- Hook Monitor (no eviction, same generate path) ---")
    ht, htime, hvram, hstats, hlen = run_with_monitor(model, tokenizer, prompt, max_new)
    print(f"  Input={hlen} tokens | VRAM={hvram:.2f}GB | Time={htime:.2f}s")
    if hstats:
        print(f"  P_c/P_raw: mean={hstats['ratio_mean']:.4f} std={hstats['ratio_std']:.4f} "
              f"range=[{hstats.get('ratio_min',0):.4f},{hstats.get('ratio_max',0):.4f}]")
        print(f"  Decode steps monitored: {hstats.get('num_decode_steps',0)}")
    results["hook_monitor"] = {"vram_gb": hvram, "time_s": htime, "stats": hstats, "text": ht}

    # 3. CausalKV (淘汰decode阶段生成的tokens)
    for cap in [2048, 1024, 512]:
        label = f"KV-{cap}"
        print(f"\n--- {label} ---")
        text, tm, vm, st, il = run_causal_kv(model, tokenizer, prompt, cap, max_new)
        if "error" in st:
            print(f"  SKIPPED: {st['error']}")
            results[label] = st
        else:
            print(f"  Input={il} tokens | VRAM={vm:.2f}GB | Time={tm:.2f}s | Evictions={st['eviction_count']}")
            print(f"  P_c/P_raw: mean={st['ratio_mean']:.4f}")
            results[label] = {"vram_gb": vm, "time_s": tm, "stats": st, "text": text, "input_len": il}

    # 4. VRAM分解估算
    print("\n--- KV Cache VRAM Estimation ---")
    n_layers = len(model.model.layers)
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2  # bf16=2bytes
    total_kv_vram = blen * kv_bytes_per_token / 1024**3
    print(f"  Layers={n_layers} KV_heads={n_kv_heads} head_dim={head_dim}")
    print(f"  KV per token: {kv_bytes_per_token/1024:.1f} KB")
    print(f"  Total KV ({blen} tokens): {total_kv_vram:.3f} GB")
    print(f"  KV as % of peak VRAM: {total_kv_vram/bvram*100:.1f}%")

    kv_est = {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "kv_bytes_per_token": kv_bytes_per_token,
        "total_kv_vram_gb": total_kv_vram,
        "kv_pct_of_peak": total_kv_vram / bvram * 100,
    }
    results["kv_estimation"] = kv_est

    # Report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v2")
    print("=" * 60)

    print(f"\n{'Metric':<25} | {'Baseline':<12} | {'Hook':<12} | {'KV-2048':<12} | {'KV-1024':<12} | {'KV-512':<12}")
    print("-" * 100)

    def sf(d, key, fmt=".2f"):
        if isinstance(d, dict) and key in d and not isinstance(d[key], str):
            return f"{d[key]:{fmt}}"
        return "SKIP"

    print(f"{'Peak VRAM (GB)':<25} | {sf(results['baseline'],'vram_gb')} | "
          f"{sf(results.get('hook_monitor',{}),'vram_gb')} | "
          f"{sf(results.get('KV-2048',{}),'vram_gb')} | "
          f"{sf(results.get('KV-1024',{}),'vram_gb')} | "
          f"{sf(results.get('KV-512',{}),'vram_gb')}")
    print(f"{'Gen Time (s)':<25} | {sf(results['baseline'],'time_s')} | "
          f"{sf(results.get('hook_monitor',{}),'time_s')} | "
          f"{sf(results.get('KV-2048',{}),'time_s')} | "
          f"{sf(results.get('KV-1024',{}),'time_s')} | "
          f"{sf(results.get('KV-512',{}),'time_s')}")

    print(f"\nKV Cache VRAM: {total_kv_vram:.3f} GB ({total_kv_vram/bvram*100:.1f}% of peak)")

    print("\nText quality (last 80 chars):")
    for k in ["baseline", "hook_monitor", "KV-4096", "KV-2048", "KV-1024"]:
        if k in results and "text" in results[k]:
            print(f"  {k}: ...{results[k]['text'][-80:]}")

    with open("v17b_guit_vram_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v17b_guit_vram_verification_report.json")


if __name__ == "__main__":
    main()