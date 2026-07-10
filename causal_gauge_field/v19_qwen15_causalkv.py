"""
GUIT-TRT Phase 1 v6: Qwen1.5-7B-Chat CausalKV验证（非GQA模型）

关键差异 vs 之前验证：
- Qwen1.5-7B-Chat: 32 KV heads (非GQA), hidden_size=4096, 32层
- Qwen2.5-7B: 4 KV heads (GQA), hidden_size=3584, 28层
- MiniCPM5-1B: 2 KV heads (GQA), hidden_size=1536, 24层

非GQA模型的KV Cache是GQA的8倍，CausalKV淘汰应有显著VRAM节省。
"""

import torch
import time
import json
import gc
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\qwen\Qwen1___5-7B-Chat"
ALPHA_STAR = 1.41
GAMMA = 0.01


class ThermoProbe:
    def __init__(self, alpha_star=ALPHA_STAR, gamma=GAMMA):
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
    def __init__(self, model, alpha_star=ALPHA_STAR, gamma=GAMMA):
        self.probe = ThermoProbe(alpha_star, gamma)
        self.model = model
        self._hook = None

    def register(self, layer_index=-1):
        self.remove()
        layers = self.model.model.layers
        idx = layer_index if layer_index >= 0 else len(layers) + layer_index
        target = layers[idx]
        self._hook = target.register_forward_hook(self._hook_fn)
        return idx

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

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
            "num_decode_steps": len(r),
        }


def build_long_prompt(tokenizer, target_tokens=4096):
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    monitor = HookThermoMonitor(model, alpha_star=ALPHA_STAR)
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


def _evict_cache(cache, causal_scores, max_capacity, sink_tokens=4, recent_tokens=10):
    """对KV Cache执行CausalKV淘汰，返回(cache, causal_scores, n_evicted)"""
    total_len = len(causal_scores)
    if total_len <= max_capacity:
        return cache, causal_scores, 0

    n_evict = total_len - max_capacity
    es = sink_tokens
    ee = max(es, total_len - recent_tokens)
    if ee <= es:
        return cache, causal_scores, 0

    ev = causal_scores[es:ee]
    si = sorted(range(len(ev)), key=lambda i: ev[i])
    eg = {idx + es for idx in set(si[:n_evict])}
    keep = [i for i in range(total_len) if i not in eg]
    kt = torch.tensor(keep, device="cpu")
    legacy = cache.to_legacy_cache()
    new_cache = DynamicCache()
    for li, (k, v) in enumerate(legacy):
        new_cache.update(k[:, :, kt, :].to(k.device), v[:, :, kt, :].to(v.device), li)
    new_scores = [causal_scores[i] for i in keep]
    return new_cache, new_scores, n_evict


def run_causal_kv(model, tokenizer, prompt, max_capacity, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if max_capacity < 100:
        return None, None, None, {"error": f"max_capacity({max_capacity}) too small"}, input_len

    probe = ThermoProbe(alpha_star=ALPHA_STAR)
    eviction_count = 0
    sink_tokens = 4
    recent_tokens = 10

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generated_ids = inputs["input_ids"].clone()
    t0 = time.time()

    with torch.no_grad():
        cache = DynamicCache()

        # Prefill: 获取所有input tokens的隐状态，计算causal_scores
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # 对input tokens计算causal impact（用最后3个隐状态的差分近似）
        h_all = outputs.hidden_states[-1].squeeze(0).detach().float()  # [seq_len, D]
        causal_scores = []
        if h_all.shape[0] >= 3:
            for i in range(h_all.shape[0]):
                if i < 2:
                    causal_scores.append(0.0)
                else:
                    v_i = h_all[i] - h_all[i-1]
                    v_prev = h_all[i-1] - h_all[i-2]
                    a_i = h_all[i] - 2*h_all[i-1] + h_all[i-2]
                    v_norm_sq = torch.sum(v_i * v_i) + 1e-8
                    a_par = (torch.sum(a_i * v_i) / v_norm_sq) * v_i
                    a_perp = a_i - a_par
                    causal_scores.append(torch.norm(a_perp).item())
        else:
            causal_scores = [0.0] * h_all.shape[0]

        # Prefill后立即淘汰input tokens的KV Cache
        if len(causal_scores) > max_capacity:
            cache, causal_scores, n_ev = _evict_cache(
                cache, causal_scores, max_capacity, sink_tokens, recent_tokens
            )
            eviction_count += n_ev
            print(f"    [Prefill eviction] {n_ev} tokens evicted, cache: {input_len} -> {len(causal_scores)}")

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        del outputs, h_all
        torch.cuda.empty_cache()

        # Decode阶段
        for step_i in range(1, max_new_tokens):
            cache_len = len(causal_scores)
            attn_mask = torch.ones(1, cache_len + 1, device=model.device, dtype=torch.long)
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
                cache, causal_scores, n_ev = _evict_cache(
                    cache, causal_scores, max_capacity, sink_tokens, recent_tokens
                )
                eviction_count += n_ev

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
    print("GUIT-TRT Phase 1 v6: Qwen1.5-7B-Chat CausalKV (Non-GQA)")
    print("=" * 60)

    print("\nLoading Qwen1.5-7B-Chat...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_attn_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_attn_heads
    is_gqa = n_kv_heads < n_attn_heads

    print(f"\n  Model: Qwen1.5-7B-Chat")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden: {hidden_size}")
    print(f"  Attn heads: {n_attn_heads}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  GQA: {is_gqa}")
    print(f"  KV per token: {2 * n_layers * n_kv_heads * head_dim * 2 / 1024:.1f} KB")

    prompt = build_long_prompt(tokenizer, target_tokens=2048)
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\n  Prompt length: {prompt_tokens} tokens")

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    kv_total_gb = prompt_tokens * kv_bytes_per_token / 1024**3
    print(f"  KV Cache estimate (input only): {kv_total_gb:.3f} GB")

    max_new = 50
    results = {
        "model": "Qwen1.5-7B-Chat",
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "n_attn_heads": n_attn_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "is_gqa": is_gqa,
        "kv_bytes_per_token": kv_bytes_per_token,
        "prompt_tokens": prompt_tokens,
    }

    # 1. Baseline
    print("\n--- Baseline (native generate) ---")
    bt, btime, bvram, blen = run_baseline(model, tokenizer, prompt, max_new)
    print(f"  Input={blen} tokens | VRAM={bvram:.2f}GB | Time={btime:.2f}s")
    print(f"  KV as % of peak: {kv_total_gb/bvram*100:.1f}%")
    results["baseline"] = {"vram_gb": bvram, "time_s": btime, "input_len": blen, "text": bt}

    # 2. Hook监控
    print("\n--- Hook Monitor ---")
    ht, htime, hvram, hstats, hlen = run_with_monitor(model, tokenizer, prompt, max_new)
    print(f"  Input={hlen} tokens | VRAM={hvram:.2f}GB | Time={htime:.2f}s")
    if hstats:
        print(f"  P_c/P_raw: mean={hstats['ratio_mean']:.4f} std={hstats['ratio_std']:.4f}")
        print(f"  Decode steps: {hstats.get('num_decode_steps', 0)}")
    results["hook_monitor"] = {"vram_gb": hvram, "time_s": htime, "stats": hstats, "text": ht}

    # 3. CausalKV淘汰（非GQA模型，KV Cache大，应能触发淘汰并节省VRAM）
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
        gc.collect()
        torch.cuda.empty_cache()

    # 4. 对比表
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v6: Qwen1.5-7B-Chat (Non-GQA)")
    print("=" * 60)

    print(f"\nModel: Qwen1.5-7B-Chat | KV heads: {n_kv_heads} | GQA: {is_gqa}")
    print(f"KV per token: {kv_bytes_per_token/1024:.1f} KB | Total KV ({blen}+{max_new}): {(blen+max_new)*kv_bytes_per_token/1024**3:.3f} GB")

    print(f"\n{'Mode':<20} | {'VRAM(GB)':<10} | {'Time(s)':<10} | {'Evictions':<10} | {'P_c/P_raw':<10}")
    print("-" * 70)

    for k in ["baseline", "hook_monitor", "KV-2048", "KV-1024", "KV-512"]:
        d = results.get(k, {})
        if not d or "error" in d:
            continue
        vram_s = f"{d.get('vram_gb', 0):.2f}"
        time_s = f"{d.get('time_s', 0):.2f}"
        evict = d.get("stats", {}).get("eviction_count", "-")
        ratio = d.get("stats", {}).get("ratio_mean", 0)
        ratio_s = f"{ratio:.4f}" if ratio else "-"
        print(f"{k:<20} | {vram_s:<10} | {time_s:<10} | {str(evict):<10} | {ratio_s:<10}")

    print(f"\nKV Cache proportion: {kv_total_gb:.3f}GB / {bvram:.2f}GB = {kv_total_gb/bvram*100:.1f}%")

    # 5. 与GQA模型对比
    print("\n--- Cross-Model KV Cache Comparison ---")
    models_compare = [
        ("Qwen1.5-7B-Chat", 32, 32, 4096, 32, 128),
        ("Qwen2.5-7B-Instruct", 4, 28, 3584, 28, 128),
        ("MiniCPM5-1B", 2, 24, 1536, 24, 64),
    ]
    print(f"{'Model':<25} | {'KV heads':<10} | {'Layers':<8} | {'KV/token(KB)':<12} | {'4k KV(GB)':<10}")
    print("-" * 70)
    for name, kv_h, layers, hsize, _, hdim in models_compare:
        kv_per_tok = 2 * layers * kv_h * hdim * 2
        kv_4k = 4096 * kv_per_tok / 1024**3
        print(f"{name:<25} | {kv_h:<10} | {layers:<8} | {kv_per_tok/1024:<12.1f} | {kv_4k:<10.3f}")

    print("\nText quality (last 80 chars):")
    for k in ["baseline", "hook_monitor", "KV-2048", "KV-1024", "KV-512"]:
        if k in results and "text" in results[k]:
            print(f"  {k}: ...{results[k]['text'][-80:]}")

    with open("v19_qwen15_causalkv_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v19_qwen15_causalkv_report.json")


if __name__ == "__main__":
    main()