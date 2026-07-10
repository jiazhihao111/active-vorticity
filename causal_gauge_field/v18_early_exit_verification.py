"""
GUIT-TRT Phase 1 v3: Early Exit Verification on Qwen2.5-7B

验证目标：
1. 多层级P_c/P_raw分布画像——约束力信号是否随深度变化
2. 浅层logits一致性——低P_c token的浅层top-1是否与深层一致
3. 可退出比例估算——不同阈值下的理论FLOPs节省上限

方法：Hook注入（已验证无观察者效应），后验分析（不实际中断forward）
"""

import torch
import time
import json
from collections import deque
from typing import List, Dict, Tuple

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
ALPHA_STAR = 1.41
GAMMA = 0.01
PROBE_LAYERS = [8, 16, 20, 27]


class MultiLayerProbe:
    def __init__(self, alpha_star: float, gamma: float, layer_indices: List[int]):
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.layer_indices = layer_indices
        self.h_history = {idx: deque(maxlen=3) for idx in layer_indices}
        self.records = {idx: [] for idx in layer_indices}

    def reset(self):
        for idx in self.layer_indices:
            self.h_history[idx].clear()
            self.records[idx].clear()

    def step(self, layer_idx: int, h_curr: torch.Tensor):
        if h_curr.dim() == 3:
            h_curr = h_curr.squeeze(1)
        h = h_curr.detach().float()
        self.h_history[layer_idx].append(h)
        if len(self.h_history[layer_idx]) < 3:
            self.records[layer_idx].append({
                "ratio": 0.0, "causal_impact": 0.0, "vel_norm": 0.0
            })
            return
        ht, ht1, ht2 = self.h_history[layer_idx][2], self.h_history[layer_idx][1], self.h_history[layer_idx][0]
        v_t = ht - ht1
        a_t = ht - 2 * ht1 + ht2
        F_res = a_t + self.gamma * v_t
        P_raw = torch.sum(F_res * v_t, dim=-1).mean().item()
        P_active = self.alpha_star * torch.sum(v_t * v_t, dim=-1).mean().item()
        P_c = P_raw - P_active
        ratio = abs(P_c) / (abs(P_raw) + 1e-8)
        v_norm_sq = torch.sum(v_t * v_t, dim=-1, keepdim=True) + 1e-8
        a_par = (torch.sum(a_t * v_t, dim=-1, keepdim=True) / v_norm_sq) * v_t
        a_perp = a_t - a_par
        causal_impact = torch.norm(a_perp, dim=-1).mean().item()
        vel_norm = torch.norm(v_t, dim=-1).mean().item()
        self.records[layer_idx].append({
            "ratio": ratio, "causal_impact": causal_impact, "vel_norm": vel_norm
        })


class EarlyExitProfiler:
    """多层级Hook注入 + 逐step浅层logits后验分析"""

    def __init__(self, model, alpha_star: float, gamma: float, probe_layers: List[int]):
        self.model = model
        self.probe = MultiLayerProbe(alpha_star, gamma, probe_layers)
        self.probe_layers = probe_layers
        self._hooks = []
        self.shallow_top1 = {idx: [] for idx in probe_layers[:-1]}
        self.deep_top1 = []
        self.shallow_kl = {idx: [] for idx in probe_layers[:-1]}

    def register_hooks(self):
        self.remove_hooks()
        self.probe.reset()
        self.shallow_top1 = {idx: [] for idx in self.probe_layers[:-1]}
        self.deep_top1 = []
        self.shallow_kl = {idx: [] for idx in self.probe_layers[:-1]}
        layers = self.model.model.layers
        for idx in self.probe_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3 and h.shape[1] == 1:
                self.probe.step(layer_idx, h[:, -1:, :])
                with torch.no_grad():
                    norm_h = self.model.model.norm(h)
                    logits = self.model.lm_head(norm_h)
                    top1 = logits.argmax(dim=-1).item()
                    if layer_idx == self.probe_layers[-1]:
                        self.deep_top1.append(top1)
                        deep_logits = logits
                        for shallow_idx in self.probe_layers[:-1]:
                            if shallow_idx in self._pending_shallow:
                                s_top1, s_logits = self._pending_shallow[shallow_idx]
                                self.shallow_top1[shallow_idx].append(s_top1)
                                p_deep = torch.softmax(deep_logits.float(), dim=-1)
                                p_shallow = torch.softmax(s_logits.float(), dim=-1)
                                kl = torch.nn.functional.kl_div(
                                    p_shallow.log(), p_deep, reduction="sum"
                                ).item()
                                self.shallow_kl[shallow_idx].append(kl)
                    else:
                        if not hasattr(self, '_pending_shallow'):
                            self._pending_shallow = {}
                        self._pending_shallow[layer_idx] = (top1, logits)
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if hasattr(self, '_pending_shallow'):
            self._pending_shallow.clear()


def build_diverse_prompts(tokenizer, target_tokens=512):
    prompts = []

    story = (
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
    q1 = "\n\n根据以上故事回答：李明从村子到镇上要经过几座桥？分别是什么桥？"
    q2 = "\n\n根据以上故事回答：李明要去哪里采药？为什么危险？谁给了他出发前的食物？"
    q3 = "\n\n根据以上故事回答：李明的母亲患什么病？谁治好了她？用什么治好的？"

    for q in [q1, q2, q3]:
        p = story + q
        prompts.append(p)

    return prompts


def run_profiling(model, tokenizer, prompts, max_new_tokens=80):
    profiler = EarlyExitProfiler(model, ALPHA_STAR, GAMMA, PROBE_LAYERS)
    profiler.register_hooks()

    all_layer_stats = {idx: {"ratios": [], "top1_matches": [], "kl_divs": []} for idx in PROBE_LAYERS}
    baseline_texts = []
    total_decode_steps = 0

    for pi, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        input_len = inputs["input_ids"].shape[1]

        profiler.probe.reset()
        profiler.shallow_top1 = {idx: [] for idx in PROBE_LAYERS[:-1]}
        profiler.deep_top1 = []
        profiler.shallow_kl = {idx: [] for idx in PROBE_LAYERS[:-1]}
        if hasattr(profiler, '_pending_shallow'):
            profiler._pending_shallow = {}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        baseline_texts.append(text)

        for idx in PROBE_LAYERS:
            recs = profiler.probe.records[idx]
            for r in recs:
                all_layer_stats[idx]["ratios"].append(r["ratio"])

        for idx in PROBE_LAYERS[:-1]:
            s_top1s = profiler.shallow_top1[idx]
            s_kls = profiler.shallow_kl[idx]
            for i, (st, dk) in enumerate(zip(s_top1s, profiler.deep_top1)):
                all_layer_stats[idx]["top1_matches"].append(1 if st == dk else 0)
            for kl in s_kls:
                all_layer_stats[idx]["kl_divs"].append(kl)

        n_decode = len(profiler.probe.records[27])
        total_decode_steps += n_decode
        print(f"  Prompt {pi+1}: input={input_len} tokens, decode={n_decode} steps")

    profiler.remove_hooks()
    return all_layer_stats, baseline_texts, total_decode_steps


def analyze_exit_potential(layer_stats: Dict) -> Dict:
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    results = {}

    for idx in PROBE_LAYERS[:-1]:
        ratios = layer_stats[idx]["ratios"]
        matches = layer_stats[idx]["top1_matches"]
        kl_divs = layer_stats[idx]["kl_divs"]

        if not ratios:
            results[idx] = {}
            continue

        ratios_t = torch.tensor(ratios)
        layer_result = {
            "ratio_distribution": {
                "mean": float(ratios_t.mean()),
                "std": float(ratios_t.std()),
                "p25": float(ratios_t.quantile(0.25)),
                "p50": float(ratios_t.quantile(0.50)),
                "p75": float(ratios_t.quantile(0.75)),
            },
            "exit_analysis": [],
        }

        for tau in thresholds:
            mask = ratios_t < tau
            exit_frac = float(mask.float().mean())
            exit_indices = mask.nonzero(as_tuple=True)[0].tolist()

            exit_matches = [matches[i] for i in exit_indices if i < len(matches)]
            exit_kls = [kl_divs[i] for i in exit_indices if i < len(kl_divs)]

            top1_acc = float(torch.tensor(exit_matches, dtype=torch.float).mean()) if exit_matches else 0.0
            mean_kl = float(torch.tensor(exit_kls).mean()) if exit_kls else 0.0

            total_layers = 28
            saved_frac = exit_frac * (1.0 - (idx + 1) / total_layers)

            layer_result["exit_analysis"].append({
                "threshold": tau,
                "exit_fraction": exit_frac,
                "top1_accuracy": top1_acc,
                "mean_kl_divergence": mean_kl,
                "flops_saved_fraction": saved_frac,
            })

        results[idx] = layer_result

    return results


def main():
    print("=" * 70)
    print("GUIT-TRT Phase 1 v3: Early Exit Verification (Qwen2.5-7B)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading Qwen2.5-7B from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    n_kv_heads = model.config.num_key_value_heads
    print(f"  Layers={n_layers}, Hidden={hidden_dim}, KV_Heads={n_kv_heads}")
    print(f"  Probe layers: {PROBE_LAYERS}")

    prompts = build_diverse_prompts(tokenizer, target_tokens=2048)
    for i, p in enumerate(prompts):
        print(f"  Prompt {i+1}: {len(tokenizer.encode(p))} tokens")

    print("\n--- Running Multi-Layer Profiling ---")
    layer_stats, baseline_texts, total_decode = run_profiling(model, tokenizer, prompts, max_new_tokens=50)

    print("\n--- Per-Layer P_c/P_raw Distribution ---")
    for idx in PROBE_LAYERS:
        ratios = layer_stats[idx]["ratios"]
        if not ratios:
            print(f"  Layer {idx}: NO DATA")
            continue
        rt = torch.tensor(ratios)
        print(f"  Layer {idx:2d}: mean={rt.mean():.4f} std={rt.std():.4f} "
              f"p25={rt.quantile(0.25):.4f} p50={rt.quantile(0.50):.4f} p75={rt.quantile(0.75):.4f} "
              f"range=[{rt.min():.4f},{rt.max():.4f}]")

    print("\n--- Shallow Layer Top-1 Agreement with Layer 27 ---")
    for idx in PROBE_LAYERS[:-1]:
        matches = layer_stats[idx]["top1_matches"]
        kl_divs = layer_stats[idx]["kl_divs"]
        if not matches:
            print(f"  Layer {idx}: NO DATA")
            continue
        mt = torch.tensor(matches, dtype=torch.float)
        kt = torch.tensor(kl_divs)
        print(f"  Layer {idx:2d}: top1_acc={mt.mean():.4f} mean_kl={kt.mean():.4f} "
              f"kl_p50={kt.quantile(0.50):.4f}")

    print("\n--- Exit Potential Analysis ---")
    exit_results = analyze_exit_potential(layer_stats)

    for idx in PROBE_LAYERS[:-1]:
        if idx not in exit_results or not exit_results[idx]:
            continue
        print(f"\n  Layer {idx} (exit here saves {28-idx-1}/{28} layers = {(28-idx-1)/28*100:.0f}% FLOPs):")
        print(f"  {'Threshold':<12} {'Exit%':<10} {'Top1-Acc':<10} {'Mean-KL':<12} {'FLOPs-Saved%':<14}")
        print(f"  {'-'*58}")
        for ea in exit_results[idx]["exit_analysis"]:
            print(f"  {ea['threshold']:<12.2f} {ea['exit_fraction']*100:<10.1f} "
                  f"{ea['top1_accuracy']:<10.4f} {ea['mean_kl_divergence']:<12.4f} "
                  f"{ea['flops_saved_fraction']*100:<14.1f}")

    report = {
        "model": "Qwen2.5-7B-Instruct",
        "alpha_star": ALPHA_STAR,
        "probe_layers": PROBE_LAYERS,
        "total_decode_steps": total_decode,
        "per_layer_stats": {},
        "exit_analysis": exit_results,
    }

    for idx in PROBE_LAYERS:
        ratios = layer_stats[idx]["ratios"]
        matches = layer_stats[idx]["top1_matches"]
        kl_divs = layer_stats[idx]["kl_divs"]
        report["per_layer_stats"][str(idx)] = {
            "num_steps": len(ratios),
            "ratio_mean": float(torch.tensor(ratios).mean()) if ratios else 0,
            "ratio_std": float(torch.tensor(ratios).std()) if ratios else 0,
            "top1_accuracy": float(torch.tensor(matches, dtype=torch.float).mean()) if matches else 0,
            "mean_kl_divergence": float(torch.tensor(kl_divs).mean()) if kl_divs else 0,
        }

    with open("v18_early_exit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v18_early_exit_report.json")

    print("\n--- Key Findings Summary ---")
    best_exit = None
    best_savings = 0
    for idx in PROBE_LAYERS[:-1]:
        if idx not in exit_results or not exit_results[idx]:
            continue
        for ea in exit_results[idx]["exit_analysis"]:
            if ea["top1_accuracy"] >= 0.95 and ea["flops_saved_fraction"] > best_savings:
                best_savings = ea["flops_saved_fraction"]
                best_exit = (idx, ea)

    if best_exit:
        idx, ea = best_exit
        print(f"  Best Early Exit: Layer {idx} at threshold {ea['threshold']}")
        print(f"    Exit fraction: {ea['exit_fraction']*100:.1f}%")
        print(f"    Top-1 accuracy: {ea['top1_accuracy']:.4f}")
        print(f"    FLOPs saved: {ea['flops_saved_fraction']*100:.1f}%")
    else:
        print("  No viable early exit configuration found (top1_acc < 0.95 for all thresholds)")
        best_overall = None
        best_acc = 0
        for idx in PROBE_LAYERS[:-1]:
            if idx not in exit_results or not exit_results[idx]:
                continue
            for ea in exit_results[idx]["exit_analysis"]:
                if ea["top1_accuracy"] > best_acc and ea["exit_fraction"] > 0.1:
                    best_acc = ea["top1_accuracy"]
                    best_overall = (idx, ea)
        if best_overall:
            idx, ea = best_overall
            print(f"  Closest: Layer {idx} at threshold {ea['threshold']}")
            print(f"    Exit fraction: {ea['exit_fraction']*100:.1f}%, Top-1 accuracy: {ea['top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()