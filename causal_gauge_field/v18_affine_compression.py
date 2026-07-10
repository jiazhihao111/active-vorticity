"""
GUIT-TRT Phase 1 v5: Affine Subspace Activation Compression Verification

验证目标：
1. 多层级隐状态有效秩画像——r(T) << D 是否成立
2. 低维投影→重构后的logits一致性——压缩是否保持语义
3. 逐token隐状态SVD——单步隐状态是否也在低维子空间内
4. VRAM节省估算——理论压缩率 vs 实际可行性

方法：Hook注入提取各层隐状态，后验SVD分析
"""

import torch
import time
import json
from collections import deque
from typing import List, Dict, Tuple

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
ALPHA_STAR = 1.41
GAMMA = 0.01
PROBE_LAYERS = [0, 8, 16, 20, 27]


class HiddenStateCollector:
    def __init__(self, model, probe_layers: List[int]):
        self.model = model
        self.probe_layers = probe_layers
        self._hooks = []
        self.h_states = {idx: [] for idx in probe_layers}

    def register_hooks(self):
        self.remove_hooks()
        self.h_states = {idx: [] for idx in self.probe_layers}
        layers = self.model.model.layers
        for idx in self.probe_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3 and h.shape[1] == 1:
                self.h_states[layer_idx].append(h[0, 0, :].detach().clone().float())
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_effective_rank(matrix: torch.Tensor, threshold: float = 0.95) -> int:
    if matrix.dim() != 2:
        return 0
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-10:
        return 0
    cum = torch.cumsum(S2, dim=0) / total
    rank = int(torch.searchsorted(cum, threshold)) + 1
    return min(rank, matrix.size(1))


def compute_rank_at_thresholds(matrix: torch.Tensor, thresholds=[0.90, 0.95, 0.99, 0.999]) -> Dict[float, int]:
    if matrix.dim() != 2:
        return {t: 0 for t in thresholds}
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-10:
        return {t: 0 for t in thresholds}
    cum = torch.cumsum(S2, dim=0) / total
    result = {}
    for t in thresholds:
        rank = int(torch.searchsorted(cum, t)) + 1
        result[t] = min(rank, matrix.size(1))
    return result


def test_lowrank_reconstruction(h_matrix: torch.Tensor, target_rank: int, model, layer_idx: int) -> Dict:
    U, S, Vt = torch.linalg.svd(h_matrix, full_matrices=False)
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vt_r = Vt[:target_rank, :]

    h_reconstructed = U_r @ torch.diag(S_r) @ Vt_r

    h_last_original = h_matrix[-1:, :]
    h_last_recon = h_reconstructed[-1:, :]

    recon_error = torch.norm(h_last_original - h_last_recon).item() / (torch.norm(h_last_original).item() + 1e-8)

    with torch.no_grad():
        h_orig_3d = h_last_original.unsqueeze(0).to(model.model.norm.weight.dtype)
        h_rec_3d = h_last_recon.unsqueeze(0).to(model.model.norm.weight.dtype)

        norm_orig = model.model.norm(h_orig_3d)
        logits_orig = model.lm_head(norm_orig)

        norm_rec = model.model.norm(h_rec_3d)
        logits_rec = model.lm_head(norm_rec)

        top1_orig = logits_orig.argmax(dim=-1).item()
        top1_rec = logits_rec.argmax(dim=-1).item()
        top1_match = 1 if top1_orig == top1_rec else 0

        p_orig = torch.softmax(logits_orig.float(), dim=-1)
        p_rec = torch.softmax(logits_rec.float(), dim=-1)
        kl = torch.nn.functional.kl_div(p_rec.log(), p_orig, reduction="sum").item()

        top5_orig = set(logits_orig.topk(5, dim=-1).indices.squeeze().tolist())
        top5_rec = set(logits_rec.topk(5, dim=-1).indices.squeeze().tolist())
        top5_overlap = len(top5_orig & top5_rec) / 5.0

    return {
        "recon_error": recon_error,
        "top1_match": top1_match,
        "kl_divergence": kl,
        "top5_overlap": top5_overlap,
    }


def build_prompt(tokenizer, target_tokens=512):
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
    )
    question = "\n\n根据以上故事回答：李明从村子到镇上要经过几座桥？分别是什么桥？"
    return story + question


def main():
    print("=" * 70)
    print("GUIT-TRT Phase 1 v5: Affine Subspace Compression (Qwen2.5-7B)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading Qwen2.5-7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  Layers={n_layers}, Hidden={hidden_dim}")

    prompt = build_prompt(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    input_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {input_len} tokens")

    collector = HiddenStateCollector(model, PROBE_LAYERS)
    collector.register_hooks()

    print("\n--- Collecting Hidden States (50 decode steps) ---")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=True)

    collector.remove_hooks()

    for idx in PROBE_LAYERS:
        n = len(collector.h_states[idx])
        print(f"  Layer {idx}: {n} hidden states collected")

    # ===== Analysis 1: Trajectory-level effective rank =====
    print("\n--- Analysis 1: Trajectory-Level Effective Rank ---")
    print(f"  (Stacking {50} decode-step hidden states into [T, D] matrix)")
    print(f"  {'Layer':<8} | {'r(0.90)':<10} | {'r(0.95)':<10} | {'r(0.99)':<10} | {'r(0.999)':<10} | {'D':<8} | {'Compression':<12}")
    print("  " + "-" * 70)

    traj_ranks = {}
    for idx in PROBE_LAYERS:
        h_list = collector.h_states[idx]
        if len(h_list) < 5:
            print(f"  Layer {idx}: insufficient data ({len(h_list)} steps)")
            continue
        h_matrix = torch.stack(h_list, dim=0)
        ranks = compute_rank_at_thresholds(h_matrix)
        traj_ranks[idx] = ranks
        r95 = ranks[0.95]
        compression = (1 - r95 / hidden_dim) * 100
        print(f"  {idx:<8} | {ranks[0.90]:<10} | {ranks[0.95]:<10} | {ranks[0.99]:<10} | {ranks[0.999]:<10} | {hidden_dim:<8} | {compression:.1f}%")

    # ===== Analysis 2: Single-step effective rank =====
    print("\n--- Analysis 2: Single-Step Hidden State Rank (SVD of [1, D]) ---")
    print("  Note: Single vector has rank 1 by definition.")
    print("  Instead, we measure: how much of h_t's energy is in the top-r subspace?")
    print("  Using the trajectory SVD basis from Analysis 1:")

    for idx in PROBE_LAYERS:
        if idx not in traj_ranks:
            continue
        h_list = collector.h_states[idx]
        h_matrix = torch.stack(h_list, dim=0)
        U, S, Vt = torch.linalg.svd(h_matrix, full_matrices=False)

        r95 = traj_ranks[idx][0.95]
        V_basis = Vt[:r95, :]

        projections = []
        for h in h_list:
            proj = torch.norm(V_basis @ h) / (torch.norm(h) + 1e-8)
            projections.append(proj.item())

        proj_t = torch.tensor(projections)
        print(f"  Layer {idx}: r95={r95}, projection ratio: "
              f"mean={proj_t.mean():.6f} std={proj_t.std():.6f} "
              f"min={proj_t.min():.6f} max={proj_t.max():.6f}")

    # ===== Analysis 3: Low-rank reconstruction quality =====
    print("\n--- Analysis 3: Low-Rank Reconstruction → Logits Quality ---")
    test_ranks = [8, 16, 32, 64, 128, 256]

    for idx in PROBE_LAYERS:
        if idx not in traj_ranks:
            continue
        h_list = collector.h_states[idx]
        if len(h_list) < 5:
            continue
        h_matrix = torch.stack(h_list, dim=0)

        print(f"\n  Layer {idx} (D={hidden_dim}, r95={traj_ranks[idx][0.95]}):")
        print(f"  {'Rank':<8} | {'Recon Err':<12} | {'Top1 Match':<12} | {'Top5 Overlap':<14} | {'KL Div':<10}")
        print("  " + "-" * 60)

        for r in test_ranks:
            if r >= hidden_dim:
                continue
            try:
                result = test_lowrank_reconstruction(h_matrix, r, model, idx)
                print(f"  {r:<8} | {result['recon_error']:<12.6f} | {result['top1_match']:<12} | "
                      f"{result['top5_overlap']:<14.2f} | {result['kl_divergence']:<10.4f}")
            except Exception as e:
                print(f"  {r:<8} | ERROR: {e}")

    # ===== Analysis 4: VRAM savings estimation =====
    print("\n--- Analysis 4: VRAM Savings Estimation ---")
    for idx in PROBE_LAYERS:
        if idx not in traj_ranks:
            continue
        r95 = traj_ranks[idx][0.95]
        r99 = traj_ranks[idx][0.99]

        original_bytes = hidden_dim * 2
        compressed_r95 = r95 * 2 + r95 * hidden_dim * 2
        compressed_r99 = r99 * 2 + r99 * hidden_dim * 2

        savings_r95 = (1 - compressed_r95 / (original_bytes * 1)) * 100
        savings_r99 = (1 - compressed_r99 / (original_bytes * 1)) * 100

        print(f"  Layer {idx}: D={hidden_dim}, r95={r95}, r99={r99}")
        print(f"    Per-token: {original_bytes}B → r95 basis: {compressed_r95}B, r99 basis: {compressed_r99}B")
        print(f"    Note: Need basis matrix V_r [{r95}x{hidden_dim}] = {r95*hidden_dim*2/1024:.1f}KB (one-time cost)")

    # Save report
    report = {
        "model": "Qwen2.5-7B-Instruct",
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "probe_layers": PROBE_LAYERS,
        "trajectory_ranks": {str(k): {str(t): v for t, v in ranks.items()} for k, ranks in traj_ranks.items()},
    }

    with open("v18_affine_compression_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to v18_affine_compression_report.json")


if __name__ == "__main__":
    main()