"""
GUIT-TRT Phase 1 v9b: 仿射子空间压缩在线验证（简化版）

两阶段：
1. 校准：Hook收集decode隐状态 → SVD
2. 在线：Hook在generate()中投影+重构+对比logits（监控模式）
3. 替换：Hook在generate()中替换隐状态为重构版（替换模式）
"""

import torch
import time
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
RANKS = [32, 64, 128]


def main():
    print("=" * 60)
    print("GUIT-TRT v9b: Affine Subspace Online Verification")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Hidden: {hidden_dim} | Layers: {n_layers}")

    prompt = "请详细解释量子纠缠的物理原理，包括贝尔不等式的意义和实验验证。然后讨论量子纠缠在量子计算和量子通信中的应用前景。"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    max_new = 50
    print(f"  Prompt: {input_len} tokens | Max new: {max_new}")

    # ============================================================
    # 阶段1：校准
    # ============================================================
    print("\n=== Phase 1: Calibration ===")
    collected_h = []

    def collect_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if h.dim() == 3 and h.shape[1] == 1:
            collected_h.append(h[:, -1, :].detach().float().cpu().squeeze(0))

    handle = model.model.layers[-1].register_forward_hook(collect_hook)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out_baseline = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
    elapsed_bl = time.time() - t0
    vram_bl = torch.cuda.max_memory_allocated() / 1024**3
    text_bl = tokenizer.decode(out_baseline[0], skip_special_tokens=True)

    handle.remove()

    H = torch.stack(collected_h, dim=0)  # [N, D]
    N, D = H.shape
    print(f"  Collected: {N} hidden states, dim={D}")

    # SVD
    H_centered = H - H.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var

    for thr in [0.90, 0.95, 0.99]:
        r = (cumvar < thr).sum().item() + 1
        print(f"  r({thr}) = {r}")

    # ============================================================
    # 阶段2：监控模式（不替换，只对比logits）
    # ============================================================
    print("\n=== Phase 2: Monitor Mode (logits comparison) ===")

    results = {
        "model": "Qwen2.5-7B-Instruct", "hidden_dim": D, "n_layers": n_layers,
        "n_calibration_steps": N, "prompt_tokens": input_len,
        "baseline": {"vram_gb": vram_bl, "time_s": elapsed_bl, "text_len": len(tokenizer.encode(text_bl))},
    }

    for r in RANKS:
        U_r = Vh[:r, :].mT.to(model.device)  # [D, r]
        label = f"monitor-r{r}"
        print(f"\n  --- {label} ---")

        stats = {"top1_match": 0, "kl": [], "cos": [], "mse": [], "total": 0, "step": 0}

        def make_monitor_hook(U_r_local, stats_dict):
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

                stats_dict["step"] += 1
                cos = torch.nn.functional.cosine_similarity(h_t, h_rec, dim=-1).item()
                mse = ((h_t - h_rec) ** 2).mean().item()
                stats_dict["cos"].append(cos)
                stats_dict["mse"].append(mse)

                if stats_dict["step"] % 10 == 0:
                    with torch.no_grad():
                        dt = model.model.norm.weight.dtype
                        logits_o = model.lm_head(model.model.norm(h_t.to(dt)))
                        logits_r = model.lm_head(model.model.norm(h_rec.to(dt)))
                        if logits_o.argmax(dim=-1).item() == logits_r.argmax(dim=-1).item():
                            stats_dict["top1_match"] += 1
                        p = torch.softmax(logits_o.float(), dim=-1)
                        q = torch.softmax(logits_r.float(), dim=-1)
                        kl = (p * (p.log() - q.log())).sum(dim=-1).item()
                        stats_dict["kl"].append(kl)
                        stats_dict["total"] += 1
            return hook_fn

        h_handle = model.model.layers[-1].register_forward_hook(make_monitor_hook(U_r, stats))

        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)

        h_handle.remove()

        n = stats["total"]
        mr = stats["top1_match"] / n if n > 0 else 0
        kl_m = float(torch.tensor(stats["kl"]).mean()) if stats["kl"] else 0
        cos_m = float(torch.tensor(stats["cos"]).mean()) if stats["cos"] else 0
        mse_m = float(torch.tensor(stats["mse"]).mean()) if stats["mse"] else 0

        print(f"    Steps={n} | Top-1={mr:.4f} | KL={kl_m:.6f} | Cos={cos_m:.6f} | MSE={mse_m:.6f}")

        results[label] = {
            "rank": r, "top1_match_rate": mr, "kl_mean": kl_m,
            "cosine_sim_mean": cos_m, "mse_mean": mse_m, "total_steps": n,
            "compression_ratio": r / D,
        }

    # ============================================================
    # 阶段3：替换模式（h_t → h_rec，验证生成质量）
    # ============================================================
    print("\n=== Phase 3: Replacement Mode (h_t → h_rec) ===")

    for r in [32, 64, 128]:
        U_r = Vh[:r, :].mT.to(model.device)
        label = f"replace-r{r}"
        print(f"\n  --- {label} ---")

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
                h_out = h_rec.to(h.dtype).unsqueeze(1)
                if isinstance(output, tuple):
                    return (h_out,) + output[1:]
                return h_out
            return hook_fn

        h_handle = model.model.layers[-1].register_forward_hook(make_replace_hook(U_r))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**3
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        h_handle.remove()

        # Token级对比
        bl_tokens = tokenizer.encode(text_bl)
        rp_tokens = tokenizer.encode(text)
        min_len = min(len(bl_tokens), len(rp_tokens))
        token_match = sum(1 for i in range(min_len) if bl_tokens[i] == rp_tokens[i]) / min_len if min_len > 0 else 0

        print(f"    Token match={token_match:.4f} | VRAM={vram:.2f}GB | Time={elapsed:.2f}s")
        print(f"    Text (last 80): ...{text[-80:]}")

        results[label] = {
            "rank": r, "token_match_with_baseline": token_match,
            "vram_gb": vram, "time_s": elapsed,
            "baseline_len": len(bl_tokens), "replaced_len": len(rp_tokens),
            "text": text,
        }

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT v9b")
    print("=" * 60)

    print(f"\nModel: Qwen2.5-7B | Hidden: {D} | Calibration: {N} steps")
    print(f"\n--- Monitor Mode ---")
    print(f"{'Rank':<8} | {'Top-1':<8} | {'KL':<12} | {'Cos':<10} | {'MSE':<10} | {'Compress':<10}")
    print("-" * 65)
    for r in RANKS:
        d = results.get(f"monitor-r{r}", {})
        if not d: continue
        print(f"{r:<8} | {d['top1_match_rate']:<8.4f} | {d['kl_mean']:<12.6f} | {d['cosine_sim_mean']:<10.6f} | {d['mse_mean']:<10.6f} | {d['compression_ratio']:<10.4f}")

    print(f"\n--- Replacement Mode ---")
    print(f"{'Rank':<8} | {'TokenMatch':<12} | {'VRAM(GB)':<10} | {'Time(s)':<10}")
    print("-" * 45)
    for r in [32, 64, 128]:
        d = results.get(f"replace-r{r}", {})
        if not d: continue
        print(f"{r:<8} | {d['token_match_with_baseline']:<12.4f} | {d['vram_gb']:<10.2f} | {d['time_s']:<10.2f}")

    print(f"\nBaseline: VRAM={vram_bl:.2f}GB | Time={elapsed_bl:.2f}s")

    with open("v19_affine_online_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v19_affine_online_report.json")


if __name__ == "__main__":
    main()