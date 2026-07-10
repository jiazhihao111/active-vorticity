"""
GUIT-TRT Phase 1 v4: Needle-in-a-Haystack + CausalKV on Qwen2.5-7B

验证目标：
1. 32k上下文下KV Cache VRAM占比——确认压缩有实际意义
2. CausalKV淘汰在长上下文中的质量保持——needle能否被保留
3. 不同淘汰容量下的VRAM节省 vs 召回率权衡

方法：Hook注入监控 + 手动逐token生成（仅CausalKV模式需要）
"""

import torch
import time
import json
from collections import deque
from typing import List, Tuple

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
ALPHA_STAR = 1.46
GAMMA = 0.01


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


class HookThermoMonitor:
    def __init__(self, model, alpha_star=1.41, gamma=0.01):
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
        h = output[0] if isinstance(output, tuple) else output
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
            "ci_mean": float(torch.tensor(c).mean()) if c else 0,
            "num_decode_steps": len(r),
        }


def build_needle_haystack(tokenizer, needle="密码是XKCD-4096", target_tokens=8000, needle_position=0.5):
    filler_paragraphs = [
        "城市的交通系统在早高峰时段总是特别繁忙。地铁里挤满了赶着上班的人群，每个人都在低头看手机或者闭目养神。公交车站排起了长队，出租车也很难打到。共享单车成了很多人的选择，虽然骑行需要体力，但至少不用担心堵车。交通管理部门一直在尝试各种方案来缓解拥堵，比如限行、错峰上下班、发展公共交通等，但效果始终有限。",
        "图书馆是一个安静的地方，适合阅读和学习。书架上一排排的书籍涵盖了各种领域，从文学到科学，从历史到哲学。读者们可以在这里找到自己感兴趣的书籍，安静地阅读。图书馆还提供电子资源，方便读者在线查阅资料。管理员们认真地维护着图书馆的秩序，确保每个人都能有一个良好的阅读环境。",
        "秋天的公园里，落叶铺满了小路。金黄色的银杏叶和红色的枫叶交织在一起，形成了一幅美丽的画卷。老人们在长椅上晒太阳，孩子们在草地上奔跑嬉戏。湖面上倒映着岸边的树木，偶尔有几只野鸭游过。远处的山峦在薄雾中若隐若现，给人一种宁静祥和的感觉。这样的景色让人忘却了城市的喧嚣。",
        "现代科技的发展日新月异。人工智能正在改变我们的生活方式，从语音助手到自动驾驶，从医疗诊断到金融分析。量子计算机有望解决传统计算机无法处理的问题。基因编辑技术为治疗遗传疾病带来了新的希望。太空探索也在不断推进，人类对宇宙的认识越来越深入。这些技术的进步既带来了机遇，也带来了挑战。",
        "烹饪是一门艺术，也是一种生活技能。好的食材是美味佳肴的基础，新鲜的蔬菜、优质的肉类、香醇的调料缺一不可。火候的掌握至关重要，过火则老，欠火则生。调味需要恰到好处，咸淡适中才能让人回味无穷。摆盘也是一门学问，色香味俱全才能称得上是一道好菜。每个人都有自己的拿手菜，那是家的味道。",
        "音乐是人类最古老的艺术形式之一。从原始的鼓点到复杂的交响乐，音乐一直在陪伴着人类。不同的文化孕育了不同的音乐风格，古典、爵士、摇滚、电子，每种风格都有其独特的魅力。音乐可以表达语言无法描述的情感，可以抚慰受伤的心灵，也可以激发人们的斗志。无论是在欢乐还是悲伤的时刻，音乐都是最好的伴侣。",
        "海洋覆盖了地球表面的大部分面积，是生命的摇篮。深海中隐藏着无数未知的生物，有些甚至超出了我们的想象。珊瑚礁是海洋中的热带雨林，拥有丰富的生物多样性。洋流调节着全球的气候，影响着陆地上的天气模式。然而，海洋正面临着污染、过度捕捞和气候变化的威胁。保护海洋，就是保护我们自己的未来。",
        "建筑是凝固的音乐，也是时代的缩影。从古埃及的金字塔到现代的摩天大楼，建筑技术的进步反映了人类文明的发展。不同的建筑风格承载着不同的文化内涵，哥特式教堂的尖顶指向天堂，中国古典园林追求天人合一。现代建筑在追求功能性的同时，也越来越注重环保和可持续发展。绿色建筑、智能建筑正在成为新的趋势。",
    ]

    prompt_parts = []
    current_tokens = 0

    while current_tokens < target_tokens:
        for p in filler_paragraphs:
            prompt_parts.append(p)
            current_tokens = len(tokenizer.encode("\n".join(prompt_parts)))
            if current_tokens >= target_tokens:
                break

    full_text = "\n".join(prompt_parts)
    tokens = tokenizer.encode(full_text)

    needle_tokens = tokenizer.encode(needle)
    insert_pos = int(len(tokens) * needle_position)
    tokens = tokens[:insert_pos] + needle_tokens + tokens[insert_pos:]

    question = f"\n\n请回答：上面文本中隐藏的密码是什么？"
    tokens = tokens + tokenizer.encode(question)

    return tokenizer.decode(tokens), needle


def run_baseline(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0][-max_new_tokens:], skip_special_tokens=True)
    return text, elapsed, vram, input_len


def run_with_monitor(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    input_len = inputs["input_ids"].shape[1]

    monitor = HookThermoMonitor(model, alpha_star=ALPHA_STAR)
    layer_idx = monitor.register(layer_index=-1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0][-max_new_tokens:], skip_special_tokens=True)
    stats = monitor.get_stats()
    monitor.remove()
    return text, elapsed, vram, stats, input_len


def run_causal_kv(model, tokenizer, prompt, max_capacity, max_new_tokens=30):
    from transformers import DynamicCache
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if max_capacity < 100:
        return None, None, None, {"error": f"max_capacity({max_capacity}) too small"}, input_len

    probe = ThermoProbe(alpha_star=ALPHA_STAR)
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
    text = tokenizer.decode(generated_ids[0][-max_new_tokens:], skip_special_tokens=True)
    stats = {
        "eviction_count": eviction_count,
        "ratio_mean": float(torch.tensor(probe.ratio_history).mean()) if probe.ratio_history else 0,
        "causal_impact_mean": float(torch.tensor(probe.causal_impact_history).mean()) if probe.causal_impact_history else 0,
    }
    return text, elapsed, vram, stats, input_len


def check_recall(text, needle):
    return needle in text


def main():
    print("=" * 70)
    print("GUIT-TRT Phase 1 v4: Needle-in-a-Haystack (MiniCPM5-1B)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    hidden_dim = model.config.hidden_size
    print(f"  Layers={n_layers}, Hidden={hidden_dim}, KV_Heads={n_kv_heads}, Head_Dim={head_dim}")

    needle = "密码是XKCD-4096"
    target_tokens = 4096
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_new = 30
    results = {}

    for pos in positions:
        print(f"\n{'='*60}")
        print(f"Needle Position: {pos:.0%}")
        print(f"{'='*60}")

        prompt, _ = build_needle_haystack(tokenizer, needle=needle, target_tokens=target_tokens, needle_position=pos)
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"  Prompt: {prompt_tokens} tokens")

        kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
        kv_total = prompt_tokens * kv_bytes_per_token / 1024**3
        print(f"  KV Cache estimate: {kv_total:.3f} GB")

        pos_results = {"prompt_tokens": prompt_tokens, "kv_estimate_gb": kv_total}

        # Baseline
        print(f"  [Baseline] ", end="", flush=True)
        bt, btime, bvram, blen = run_baseline(model, tokenizer, prompt, max_new)
        b_recall = check_recall(bt, needle)
        print(f"VRAM={bvram:.2f}GB Time={btime:.1f}s Recall={'✅' if b_recall else '❌'} Text={bt[:60]}")
        pos_results["baseline"] = {"vram_gb": bvram, "time_s": btime, "recall": b_recall, "text": bt}

        # Hook Monitor
        print(f"  [Hook]     ", end="", flush=True)
        ht, htime, hvram, hstats, hlen = run_with_monitor(model, tokenizer, prompt, max_new)
        h_recall = check_recall(ht, needle)
        print(f"VRAM={hvram:.2f}GB Time={htime:.1f}s Recall={'✅' if h_recall else '❌'} "
              f"P_c/P_raw={hstats.get('ratio_mean',0):.4f}")
        pos_results["hook"] = {"vram_gb": hvram, "time_s": htime, "recall": h_recall, "stats": hstats}

        # CausalKV with different capacities
        for cap in [4096, 2048, 1024]:
            label = f"KV-{cap}"
            print(f"  [{label}]  ", end="", flush=True)
            text, tm, vm, st, il = run_causal_kv(model, tokenizer, prompt, cap, max_new)
            if "error" in st:
                print(f"SKIPPED: {st['error']}")
                pos_results[label] = st
            else:
                recall = check_recall(text, needle) if text else False
                print(f"VRAM={vm:.2f}GB Time={tm:.1f}s Evictions={st['eviction_count']} "
                      f"Recall={'✅' if recall else '❌'} Text={text[:60] if text else 'N/A'}")
                pos_results[label] = {"vram_gb": vm, "time_s": tm, "recall": recall, "stats": st, "text": text}

        results[f"pos_{pos:.1f}"] = pos_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Needle-in-a-Haystack Recall Matrix")
    print(f"{'='*70}")
    print(f"{'Position':<12} | {'Baseline':<10} | {'Hook':<10} | {'KV-4096':<10} | {'KV-2048':<10} | {'KV-1024':<10}")
    print("-" * 70)
    for pos in positions:
        key = f"pos_{pos:.1f}"
        pr = results[key]
        cells = [f"{pos:.0%}"]
        for mode in ["baseline", "hook", "KV-4096", "KV-2048", "KV-1024"]:
            if mode in pr and isinstance(pr[mode], dict):
                cells.append("Y" if pr[mode].get("recall") else "N")
            else:
                cells.append("N/A")
        print(f"  {cells[0]:<12} | {cells[1]:<10} | {cells[2]:<10} | {cells[3]:<10} | {cells[4]:<10} | {cells[5]:<10}")

    print(f"\nKV Cache VRAM at {target_tokens} tokens: {kv_total:.3f} GB")
    print(f"KV as % of peak: {kv_total/bvram*100:.1f}%")

    with open("v18_niah_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v18_niah_report.json")


if __name__ == "__main__":
    main()