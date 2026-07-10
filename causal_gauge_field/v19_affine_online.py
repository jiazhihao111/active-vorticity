"""
GUIT-TRT Phase 1 v9: 仿射子空间压缩在线验证

两阶段验证：
1. 校准阶段：Hook收集decode阶段隐状态 → SVD得到投影基U_r
2. 在线阶段：生成时Hook将隐状态投影到r维子空间再重构 → 验证生成质量

关键验证点：
- 重构隐状态能否维持正确的token选择？
- 生成文本质量是否退化？
- 不同秩r(16/32/64)的trade-off
- VRAM节省：只存储r个坐标 vs D个坐标
"""

import torch
import time
import json
import gc
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
ALPHA_STAR = 1.41
GAMMA = 0.01
TARGET_LAYER = -1
RANKS = [16, 32, 64]


class HookHiddenStateCollector:
    """阶段1：收集decode阶段最后一层隐状态"""

    def __init__(self, model):
        self.model = model
        self._hook = None
        self.hidden_states = []

    def register(self):
        self.remove()
        layers = self.model.model.layers
        target = layers[TARGET_LAYER]
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
        if h.dim() == 3 and h.shape[1] == 1:
            self.hidden_states.append(h[:, -1, :].detach().float().cpu().squeeze(0))  # [D]

    def get_matrix(self):
        if not self.hidden_states:
            return None
        return torch.stack(self.hidden_states, dim=0)  # [N, D]


class HookAffineProjector:
    """阶段2：在线投影+重构+替换隐状态"""

    def __init__(self, model, U_r):
        self.model = model
        self.U_r = U_r.to(model.device)  # [D, r]
        self._hook = None
        self.original_logits_list = []
        self.reconstructed_logits_list = []
        self.top1_match_count = 0
        self.total_decode_steps = 0
        self.kl_divergences = []
        self._replace = False

    def register(self, replace=False):
        self.remove()
        self._replace = replace
        layers = self.model.model.layers
        target = layers[TARGET_LAYER]
        self._hook = target.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _hook_fn(self, module, input, output):
        if not self._replace:
            return
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        if h.dim() != 3 or h.shape[1] != 1:
            return

        h_t = h[:, -1, :].detach().float()  # [1, D]

        # 投影到低维子空间
        z = h_t @ self.U_r  # [1, r]
        # 重构
        h_rec = z @ self.U_r.mT  # [1, D]

        # 计算logits对比
        with torch.no_grad():
            norm = self.model.model.norm
            lm_head = self.model.lm_head
            dtype = norm.weight.dtype

            logits_orig = lm_head(norm(h_t.to(dtype)))
            logits_rec = lm_head(norm(h_rec.to(dtype)))

            top1_orig = logits_orig.argmax(dim=-1).item()
            top1_rec = logits_rec.argmax(dim=-1).item()

            if top1_orig == top1_rec:
                self.top1_match_count += 1

            # KL散度
            p = torch.softmax(logits_orig.float(), dim=-1)
            q = torch.softmax(logits_rec.float(), dim=-1)
            kl = (p * (p.log() - q.log())).sum(dim=-1).item()
            self.kl_divergences.append(kl)

            self.original_logits_list.append(top1_orig)
            self.reconstructed_logits_list.append(top1_rec)

        self.total_decode_steps += 1

        # 替换隐状态（修改output）
        h_rec_out = h_rec.to(h.dtype).unsqueeze(1)  # [1, 1, D]
        if isinstance(output, tuple):
            return (h_rec_out,) + output[1:]
        return h_rec_out

    def get_stats(self):
        if self.total_decode_steps == 0:
            return {}
        return {
            "top1_match_rate": self.top1_match_count / self.total_decode_steps,
            "kl_mean": float(torch.tensor(self.kl_divergences).mean()),
            "kl_max": float(torch.tensor(self.kl_divergences).max()),
            "total_steps": self.total_decode_steps,
            "rank": self.U_r.shape[1],
        }


def build_prompt(tokenizer):
    prompt = (
        "请详细解释量子纠缠的物理原理，包括贝尔不等式的意义和实验验证。"
        "然后讨论量子纠缠在量子计算和量子通信中的应用前景。"
        "最后，分析量子纠缠与经典物理学的基本区别。"
    )
    return prompt


def main():
    print("=" * 60)
    print("GUIT-TRT Phase 1 v9: Affine Subspace Online Verification")
    print("=" * 60)

    print("\nLoading Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Hidden: {hidden_dim} | Layers: {n_layers}")

    prompt = build_prompt(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {input_len} tokens")

    max_new = 100

    # ============================================================
    # 阶段1：校准——收集隐状态 + SVD
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 1: Calibration — Collect hidden states + SVD")
    print("=" * 60)

    collector = HookHiddenStateCollector(model)
    collector.register()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out_baseline = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
    elapsed_baseline = time.time() - t0
    vram_baseline = torch.cuda.max_memory_allocated() / 1024**3
    text_baseline = tokenizer.decode(out_baseline[0], skip_special_tokens=True)

    collector.remove()

    H = collector.get_matrix()
    print(f"  Collected {H.shape[0]} hidden states, dim={H.shape[1]}")

    # SVD
    H_centered = H - H.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    # Vh: [min(N,D), D], 每行是一个主成分
    # 投影基: Vh[:r, :].T → [D, r]

    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var

    print(f"\n  SVD results:")
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        r = (cumvar < threshold).sum().item() + 1
        print(f"    r({threshold}) = {r}")

    # 计算各rank的重构误差
    print(f"\n  Reconstruction error by rank:")
    for r in RANKS:
        U_r = Vh[:r, :].mT  # [D, r]
        z = H @ U_r  # [N, r]
        H_rec = z @ U_r.mT  # [N, D]
        mse = ((H - H_rec) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(H, H_rec, dim=-1).mean().item()
        print(f"    rank={r}: MSE={mse:.6f}, cos_sim={cos_sim:.6f}")

    # ============================================================
    # 阶段2：在线验证——投影+重构+替换
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 2: Online Verification — Project + Reconstruct + Replace")
    print("=" * 60)

    results = {
        "model": "Qwen2.5-7B-Instruct",
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "prompt_tokens": input_len,
        "max_new_tokens": max_new,
        "baseline": {
            "vram_gb": vram_baseline,
            "time_s": elapsed_baseline,
            "text": text_baseline,
        },
        "svd_ranks": {},
    }

    for threshold in [0.90, 0.95, 0.99]:
        r = (cumvar < threshold).sum().item() + 1
        results["svd_ranks"][str(threshold)] = r

    # 2a. 纯监控模式（不替换，只对比logits）
    print("\n--- Monitor Mode (no replacement, logits comparison) ---")
    for r in RANKS:
        U_r = Vh[:r, :].mT  # [D, r]
        projector = HookAffineProjector(model, U_r)
        projector.register(replace=False)

        # 需要手动decode来获取logits对比
        gc.collect()
        torch.cuda.empty_cache()

        # 先做一次正常生成（不替换），但Hook中计算logits对比
        # 由于replace=False，Hook不会修改output，但也不会计算logits
        # 需要改成replace=True但手动forward
        projector.remove()

        # 手动decode：每步提取h_t，投影重构，计算logits对比
        probe_h_history = deque(maxlen=3)
        causal_impact_list = []
        top1_matches = 0
        kl_divs = []
        cos_sims = []
        total_steps = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generated_ids = inputs["input_ids"].clone()
        t0 = time.time()

        from transformers import DynamicCache
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
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            del outputs

            # Decode
            for step_i in range(1, max_new):
                cache_len = input_len + step_i - 1
                attn_mask = torch.ones(1, cache_len + 1, device=model.device, dtype=torch.long)
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )

                # 提取最后一层隐状态（通过model.model.norm前的输出）
                # 我们需要获取最后一层decoder的输出
                # 使用Hook方式更干净，但这里直接用output_hidden_states
                # 为了避免output_hidden_states的开销，我们用另一种方式：
                # 通过model.model.norm + lm_head获取logits
                # 但我们需要原始hidden state来做对比

                # 实际上outputs没有hidden_states，我们用Hook
                # 让我换一种方式：直接从model.model的输出获取

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                del outputs

        # 上面的手动decode没有Hook来提取隐状态
        # 让我改用Hook方式
        pass

    # 重新实现：用Hook提取 + 对比
    print("\n--- Using Hook for online projection verification ---")

    for r in RANKS:
        U_r = Vh[:r, :].mT.to(model.device)  # [D, r]  # [D, r]
        label = f"rank-{r}"
        print(f"\n  === {label} ===")

        # 注册Hook：提取隐状态 + 投影重构 + logits对比
        top1_matches = 0
        kl_divs = []
        cos_sims = []
        mse_list = []
        total_steps = 0
        hook_h = [None]

        def make_hook_fn(U_r_local, stats_dict):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                if h.dim() != 3 or h.shape[1] != 1:
                    return
                h_t = h[:, -1, :].detach().float()  # [1, D]

                # 投影+重构
                z = h_t @ U_r_local  # [1, r]
                h_rec = z @ U_r_local.mT  # [1, D]

                # 对比
                with torch.no_grad():
                    norm_w = model.model.norm
                    lm_w = model.lm_head
                    dt = norm_w.weight.dtype

                    logits_orig = lm_w(norm_w(h_t.to(dt)))
                    logits_rec = lm_w(norm_w(h_rec.to(dt)))

                    top1_o = logits_orig.argmax(dim=-1).item()
                    top1_r = logits_rec.argmax(dim=-1).item()
                    if top1_o == top1_r:
                        stats_dict["top1_match"] += 1

                    p = torch.softmax(logits_orig.float(), dim=-1)
                    q = torch.softmax(logits_rec.float(), dim=-1)
                    kl = (p * (p.log() - q.log())).sum(dim=-1).item()
                    stats_dict["kl"].append(kl)

                    cos = torch.nn.functional.cosine_similarity(h_t, h_rec, dim=-1).item()
                    stats_dict["cos"].append(cos)

                    mse = ((h_t - h_rec) ** 2).mean().item()
                    stats_dict["mse"].append(mse)

                stats_dict["total"] += 1
                hook_h[0] = h_t  # 保存用于后续
            return hook_fn

        stats = {"top1_match": 0, "kl": [], "cos": [], "mse": [], "total": 0}
        layers = model.model.layers
        target_layer = layers[TARGET_LAYER]
        hook_handle = target_layer.register_forward_hook(make_hook_fn(U_r, stats))

        # 运行正常generate（Hook只监控不修改）
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**3
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        hook_handle.remove()

        # 统计
        n = stats["total"]
        match_rate = stats["top1_match"] / n if n > 0 else 0
        kl_mean = float(torch.tensor(stats["kl"]).mean()) if stats["kl"] else 0
        kl_max = float(torch.tensor(stats["kl"]).max()) if stats["kl"] else 0
        cos_mean = float(torch.tensor(stats["cos"]).mean()) if stats["cos"] else 0
        mse_mean = float(torch.tensor(stats["mse"]).mean()) if stats["mse"] else 0

        print(f"    Steps: {n}")
        print(f"    Top-1 match rate: {match_rate:.4f} ({stats['top1_match']}/{n})")
        print(f"    KL divergence: mean={kl_mean:.6f}, max={kl_max:.6f}")
        print(f"    Cosine similarity: {cos_mean:.6f}")
        print(f"    MSE: {mse_mean:.6f}")
        print(f"    VRAM: {vram:.2f}GB | Time: {elapsed:.2f}s")
        print(f"    Compression: {hidden_dim}D → {r}D ({r/hidden_dim*100:.1f}%)")

        results[label] = {
            "rank": r,
            "top1_match_rate": match_rate,
            "kl_mean": kl_mean,
            "kl_max": kl_max,
            "cosine_sim_mean": cos_mean,
            "mse_mean": mse_mean,
            "total_steps": n,
            "vram_gb": vram,
            "time_s": elapsed,
            "text": text,
            "compression_ratio": r / hidden_dim,
        }

    # ============================================================
    # 阶段3：在线替换验证——重构隐状态替换原始隐状态
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 3: Online Replacement — Replace h_t with reconstructed h_rec")
    print("=" * 60)

    for r in [32, 64]:
        U_r = Vh[:r, :].mT.to(model.device)  # [D, r]
        label = f"replace-r{r}"
        print(f"\n  === {label} ===")

        # 注册替换Hook
        def make_replace_hook(U_r_local):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                if h.dim() != 3 or h.shape[1] != 1:
                    return
                h_t = h[:, -1, :].detach().float()
                z = h_t @ U_r_local
                h_rec = z @ U_r_local.mT
                h_rec_out = h_rec.to(h.dtype).unsqueeze(1)
                if isinstance(output, tuple):
                    return (h_rec_out,) + output[1:]
                return h_rec_out
            return hook_fn

        hook_handle = target_layer.register_forward_hook(make_replace_hook(U_r))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**3
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        hook_handle.remove()

        # 与baseline对比
        baseline_text = results["baseline"]["text"]
        # 计算token级匹配
        baseline_tokens = tokenizer.encode(baseline_text)
        replaced_tokens = tokenizer.encode(text)
        min_len = min(len(baseline_tokens), len(replaced_tokens))
        token_match = sum(1 for i in range(min_len) if baseline_tokens[i] == replaced_tokens[i]) / min_len if min_len > 0 else 0

        print(f"    VRAM: {vram:.2f}GB | Time: {elapsed:.2f}s")
        print(f"    Token match with baseline: {token_match:.4f}")
        print(f"    Baseline len: {len(baseline_tokens)} | Replaced len: {len(replaced_tokens)}")
        print(f"    Text (last 120): ...{text[-120:]}")

        results[label] = {
            "rank": r,
            "vram_gb": vram,
            "time_s": elapsed,
            "text": text,
            "token_match_with_baseline": token_match,
            "baseline_text_len": len(baseline_tokens),
            "replaced_text_len": len(replaced_tokens),
        }

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v9: Affine Subspace Online Verification")
    print("=" * 60)

    print(f"\nModel: Qwen2.5-7B | Hidden: {hidden_dim} | Layer: {n_layers-1}")
    print(f"Prompt: {input_len} tokens | Max new: {max_new}")

    print(f"\n--- Monitor Mode (no replacement) ---")
    print(f"{'Rank':<8} | {'Top-1 Match':<12} | {'KL(mean)':<12} | {'Cos Sim':<10} | {'MSE':<10} | {'Compress':<10}")
    print("-" * 70)
    for r in RANKS:
        d = results.get(f"rank-{r}", {})
        if not d: continue
        print(f"{r:<8} | {d['top1_match_rate']:<12.4f} | {d['kl_mean']:<12.6f} | {d['cosine_sim_mean']:<10.6f} | {d['mse_mean']:<10.6f} | {d['compression_ratio']:<10.3f}")

    print(f"\n--- Replacement Mode (h_t → h_rec) ---")
    print(f"{'Rank':<8} | {'Token Match':<12} | {'VRAM(GB)':<10} | {'Time(s)':<10}")
    print("-" * 45)
    for r in [32, 64]:
        d = results.get(f"replace-r{r}", {})
        if not d: continue
        print(f"{r:<8} | {d['token_match_with_baseline']:<12.4f} | {d['vram_gb']:<10.2f} | {d['time_s']:<10.2f}")

    print(f"\nBaseline: VRAM={vram_baseline:.2f}GB | Time={elapsed_baseline:.2f}s")

    # VRAM节省估算
    print(f"\n--- VRAM Saving Estimation ---")
    for r in RANKS:
        # 每个token的隐状态存储：D * 2 bytes (bf16) → r * 2 bytes
        savings_per_token = (hidden_dim - r) * 2  # bytes
        savings_pct = (1 - r / hidden_dim) * 100
        # 假设4k tokens
        savings_4k = 4096 * savings_per_token / 1024**2
        print(f"  rank={r}: {hidden_dim}D→{r}D, save {savings_pct:.1f}%, 4k tokens save {savings_4k:.1f}MB")

    with open("v19_affine_online_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v19_affine_online_report.json")


if __name__ == "__main__":
    main()