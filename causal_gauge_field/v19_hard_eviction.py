"""
GUIT-TRT Phase 1 v8: Hard Eviction + Hook隐状态提取（Qwen1.5-7B-Chat, 非GQA）

核心方案：
1. 手动decode循环（不用model.generate()），每步只forward一个token
2. Hook提取隐状态计算causal impact（不开output_hidden_states）
3. Hard eviction：实际裁剪DynamicCache + 同步调整attention_mask
4. VRAM测量：对比Baseline(model.generate) vs HookKV的peak VRAM
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


class HookHiddenStateExtractor:
    """Hook提取最后一层隐状态，不开output_hidden_states"""

    def __init__(self, model):
        self.model = model
        self._hook = None
        self.last_hidden = None

    def register(self):
        self.remove()
        layers = self.model.model.layers
        target = layers[-1]
        self._hook = target.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        self.last_hidden = h.detach()

    def get_last_hidden(self):
        return self.last_hidden


class CausalImpactTracker:
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
        self.h_history.append(h_curr.float())
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
        ci = torch.norm(a_perp, dim=-1).mean().item()
        self.ratio_history.append(ratio)
        self.causal_impact_history.append(ci)
        return ratio, ci


def hard_evict_cache(cache, causal_scores, max_capacity, sink_tokens=4, recent_tokens=10):
    """Hard eviction: 裁剪DynamicCache + 返回新的causal_scores"""
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
    new_legacy = []
    for k, v in legacy:
        new_legacy.append((k[:, :, kt, :], v[:, :, kt, :]))
    new_cache = DynamicCache.from_legacy_cache(tuple(new_legacy))
    new_scores = [causal_scores[i] for i in keep]
    return new_cache, new_scores, n_evict


def build_long_prompt(tokenizer, target_tokens=2048):
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
    while len(tokens) < target_tokens:
        prompt = base_story + "\n" + prompt
        tokens = tokenizer.encode(prompt)
    return prompt


def run_baseline(model, tokenizer, prompt, max_new_tokens=50):
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


def run_hook_hard_eviction(model, tokenizer, prompt, max_capacity, max_new_tokens=50):
    """手动decode循环 + Hook隐状态提取 + Hard eviction"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    hook_ext = HookHiddenStateExtractor(model)
    hook_ext.register()
    tracker = CausalImpactTracker()

    sink_tokens = 4
    recent_tokens = 10
    eviction_count = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generated_ids = inputs["input_ids"].clone()
    t0 = time.time()

    with torch.no_grad():
        cache = DynamicCache()

        # Prefill
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # 初始化causal_scores（prefill tokens用0占位）
        causal_scores = [0.0] * input_len

        # Prefill后淘汰（如果input_len > max_capacity）
        if input_len > max_capacity:
            cache, causal_scores, n_ev = hard_evict_cache(
                cache, causal_scores, max_capacity, sink_tokens, recent_tokens
            )
            eviction_count += n_ev

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        del outputs
        torch.cuda.empty_cache()

        # Decode
        for step_i in range(1, max_new_tokens):
            cache_len = len(causal_scores)
            attn_mask = torch.ones(1, cache_len + 1, device=model.device, dtype=torch.long)

            outputs = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

            # Hook提取隐状态
            h_last = hook_ext.get_last_hidden()
            if h_last is not None and h_last.dim() == 3 and h_last.shape[1] == 1:
                ratio, ci = tracker.step(h_last[:, -1:, :])
                causal_scores.append(ci)

                # 每5步检查淘汰
                if step_i % 5 == 0 and len(causal_scores) > max_capacity:
                    cache, causal_scores, n_ev = hard_evict_cache(
                        cache, causal_scores, max_capacity, sink_tokens, recent_tokens
                    )
                    eviction_count += n_ev
            else:
                causal_scores.append(0.0)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            del outputs

    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    hook_ext.remove()

    stats = {
        "eviction_count": eviction_count,
        "final_cache_len": len(causal_scores),
        "ratio_mean": float(torch.tensor(tracker.ratio_history).mean()) if tracker.ratio_history else 0,
        "ratio_std": float(torch.tensor(tracker.ratio_history).std()) if tracker.ratio_history else 0,
    }
    return text, elapsed, vram, stats, input_len


def main():
    print("=" * 60)
    print("GUIT-TRT Phase 1 v8: Hard Eviction + Hook (Non-GQA)")
    print("=" * 60)

    print("\nLoading Qwen1.5-7B-Chat...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_kv_heads = model.config.num_key_value_heads
    n_attn_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_attn_heads
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2

    print(f"\n  KV heads: {n_kv_heads} | GQA: {n_kv_heads < n_attn_heads}")
    print(f"  KV per token: {kv_bytes_per_token/1024:.1f} KB")

    prompt = build_long_prompt(tokenizer, target_tokens=2048)
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"  Prompt: {prompt_tokens} tokens")
    kv_gb = prompt_tokens * kv_bytes_per_token / 1024**3
    print(f"  KV Cache estimate: {kv_gb:.3f} GB")

    max_new = 50
    results = {
        "model": "Qwen1.5-7B-Chat",
        "n_kv_heads": n_kv_heads, "is_gqa": False,
        "kv_bytes_per_token": kv_bytes_per_token,
        "prompt_tokens": prompt_tokens,
    }

    # 1. Baseline
    print("\n--- Baseline ---")
    bt, btime, bvram, blen = run_baseline(model, tokenizer, prompt, max_new)
    print(f"  VRAM={bvram:.2f}GB | Time={btime:.2f}s | KV%={kv_gb/bvram*100:.1f}%")
    results["baseline"] = {"vram_gb": bvram, "time_s": btime, "input_len": blen, "text": bt}

    # 2. Hook Hard Eviction
    for cap in [2048, 1024, 512]:
        label = f"HookKV-{cap}"
        print(f"\n--- {label} ---")
        gc.collect()
        torch.cuda.empty_cache()
        text, tm, vm, st, il = run_hook_hard_eviction(model, tokenizer, prompt, cap, max_new)
        print(f"  VRAM={vm:.2f}GB | Time={tm:.2f}s | Evictions={st['eviction_count']} | FinalCache={st['final_cache_len']}")
        results[label] = {"vram_gb": vm, "time_s": tm, "stats": st, "text": text, "input_len": il}

    # Report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v8: Hard Eviction + Hook")
    print("=" * 60)

    bv = results["baseline"]["vram_gb"]
    print(f"\n{'Mode':<20} | {'VRAM(GB)':<10} | {'Time(s)':<10} | {'Evictions':<10} | {'Saved':<10}")
    print("-" * 60)
    for k in ["baseline", "HookKV-2048", "HookKV-1024", "HookKV-512"]:
        d = results.get(k, {})
        if not d: continue
        vram_s = f"{d.get('vram_gb', 0):.2f}"
        time_s = f"{d.get('time_s', 0):.2f}"
        evict = d.get("stats", {}).get("eviction_count", "-")
        saved = f"{bv - d.get('vram_gb', bv):.2f}GB"
        print(f"{k:<20} | {vram_s:<10} | {time_s:<10} | {str(evict):<10} | {saved:<10}")

    print("\nText quality (last 120 chars):")
    for k in ["baseline", "HookKV-2048", "HookKV-1024", "HookKV-512"]:
        if k in results and "text" in results[k]:
            print(f"  {k}: ...{results[k]['text'][-120:]}")

    with open("v19_hard_eviction_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v19_hard_eviction_report.json")


if __name__ == "__main__":
    main()