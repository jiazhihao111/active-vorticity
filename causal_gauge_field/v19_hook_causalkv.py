"""
GUIT-TRT Phase 1 v7: Hook注入式CausalKV淘汰（Qwen1.5-7B-Chat, 非GQA）

核心创新：在model.generate()流程中通过Hook提取隐状态，计算causal impact，
然后通过修改DynamicCache淘汰低贡献token。全程不开output_hidden_states。

对比v6（逐token生成+output_hidden_states=True）：
- v6: VRAM 15.44GB（hidden_states存储开销抵消了KV淘汰收益）
- v7: 预期VRAM < Baseline 14.14GB（Hook零开销+KV淘汰释放显存）

实现策略：
1. 注册Hook到最后一层，decode阶段提取h_t计算causal impact
2. 每N步检查cache大小，超过max_capacity则淘汰
3. 通过DynamicCache的内部结构直接修改KV（不重建）
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


class CausalKVHookManager:
    """Hook注入式CausalKV淘汰管理器"""

    def __init__(self, model, max_capacity, alpha_star=ALPHA_STAR, gamma=GAMMA,
                 sink_tokens=4, recent_tokens=10, check_interval=10):
        self.model = model
        self.max_capacity = max_capacity
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.check_interval = check_interval

        self.h_history = deque(maxlen=3)
        self.causal_scores = []
        self.eviction_count = 0
        self.decode_step = 0
        self._hook = None
        self._cache = None
        self._attention_mask_len = 0

    def register(self, cache, attention_mask_len):
        self.remove()
        self._cache = cache
        self._attention_mask_len = attention_mask_len
        layers = self.model.model.layers
        target = layers[-1]
        self._hook = target.register_forward_hook(self._hook_fn)
        return len(layers) - 1

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def reset(self):
        self.h_history.clear()
        self.causal_scores = []
        self.eviction_count = 0
        self.decode_step = 0

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # 只在decode阶段（seq_len=1）提取隐状态
        if h.dim() != 3 or h.shape[1] != 1:
            return

        h_curr = h[:, -1:, :].detach().float()
        self.h_history.append(h_curr)
        self.decode_step += 1

        if len(self.h_history) < 3:
            self.causal_scores.append(0.0)
            return

        h_t = self.h_history[2].squeeze(1)
        h_t1 = self.h_history[1].squeeze(1)
        h_t2 = self.h_history[0].squeeze(1)

        v_t = h_t - h_t1
        a_t = h_t - 2 * h_t1 + h_t2
        v_norm_sq = torch.sum(v_t * v_t, dim=-1, keepdim=True) + 1e-8
        a_par = (torch.sum(a_t * v_t, dim=-1, keepdim=True) / v_norm_sq) * v_t
        a_perp = a_t - a_par
        ci = torch.norm(a_perp, dim=-1).mean().item()
        self.causal_scores.append(ci)

        # 每check_interval步检查是否需要淘汰
        if self.decode_step % self.check_interval == 0:
            self._try_evict()

    def _try_evict(self):
        total_len = len(self.causal_scores)
        if total_len <= self.max_capacity:
            return

        n_evict = total_len - self.max_capacity
        es = self.sink_tokens
        ee = max(es, total_len - self.recent_tokens)
        if ee <= es:
            return

        ev = self.causal_scores[es:ee]
        si = sorted(range(len(ev)), key=lambda i: ev[i])
        eg = [idx + es for idx in set(si[:n_evict])]

        # Soft eviction: 将淘汰token的KV值置零（保持cache长度不变，避免attention_mask不匹配）
        legacy = self._cache.to_legacy_cache()
        new_legacy = []
        for k, v in legacy:
            k_new = k.clone()
            v_new = v.clone()
            for idx in eg:
                if idx < k.shape[2]:
                    k_new[:, :, idx, :] = 0
                    v_new[:, :, idx, :] = 0
            new_legacy.append((k_new, v_new))
        new_cache = DynamicCache.from_legacy_cache(tuple(new_legacy))
        self._cache.layers = new_cache.layers

        # 标记已淘汰的score为inf（避免重复淘汰）
        for idx in eg:
            if idx < len(self.causal_scores):
                self.causal_scores[idx] = float('inf')
        self.eviction_count += n_evict

    def get_stats(self):
        return {
            "eviction_count": self.eviction_count,
            "decode_steps": self.decode_step,
            "final_cache_len": len(self.causal_scores),
            "causal_impact_mean": float(torch.tensor(self.causal_scores).mean()) if self.causal_scores else 0,
        }


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


def run_hook_causal_kv(model, tokenizer, prompt, max_capacity, max_new_tokens=50):
    """Hook注入式CausalKV淘汰——全程使用model.generate()，不开output_hidden_states"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    cache = DynamicCache()
    mgr = CausalKVHookManager(
        model, max_capacity,
        sink_tokens=4, recent_tokens=10,
        check_interval=5,
    )
    mgr.register(cache, input_len)

    # Prefill: 初始化causal_scores为0（input tokens的causal impact在prefill时未知）
    mgr.causal_scores = [0.0] * input_len

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    stats = mgr.get_stats()
    mgr.remove()
    return text, elapsed, vram, stats, input_len


def main():
    print("=" * 60)
    print("GUIT-TRT Phase 1 v7: Hook-Injected CausalKV (Non-GQA)")
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

    # 2. Hook CausalKV
    for cap in [2048, 1024, 512]:
        label = f"HookKV-{cap}"
        print(f"\n--- {label} ---")
        gc.collect()
        torch.cuda.empty_cache()
        text, tm, vm, st, il = run_hook_causal_kv(model, tokenizer, prompt, cap, max_new)
        print(f"  VRAM={vm:.2f}GB | Time={tm:.2f}s | Evictions={st['eviction_count']} | FinalCache={st['final_cache_len']}")
        results[label] = {"vram_gb": vm, "time_s": tm, "stats": st, "text": text, "input_len": il}

    # Report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v7: Hook-Injected CausalKV")
    print("=" * 60)

    print(f"\n{'Mode':<20} | {'VRAM(GB)':<10} | {'Time(s)':<10} | {'Evictions':<10} | {'VRAM Saved':<10}")
    print("-" * 65)

    bv = results["baseline"]["vram_gb"]
    for k in ["baseline", "HookKV-2048", "HookKV-1024", "HookKV-512"]:
        d = results.get(k, {})
        if not d:
            continue
        vram_s = f"{d.get('vram_gb', 0):.2f}"
        time_s = f"{d.get('time_s', 0):.2f}"
        evict = d.get("stats", {}).get("eviction_count", "-")
        saved = f"{(bv - d.get('vram_gb', bv)):.2f}GB" if d.get('vram_gb', 0) < bv else "+0"
        print(f"{k:<20} | {vram_s:<10} | {time_s:<10} | {str(evict):<10} | {saved:<10}")

    print("\nText quality (last 100 chars):")
    for k in ["baseline", "HookKV-2048", "HookKV-1024", "HookKV-512"]:
        if k in results and "text" in results[k]:
            print(f"  {k}: ...{results[k]['text'][-100:]}")

    with open("v19_hook_causalkv_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v19_hook_causalkv_report.json")


if __name__ == "__main__":
    main()