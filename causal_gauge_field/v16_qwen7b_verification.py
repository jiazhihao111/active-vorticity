import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import entropy, pearsonr
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_effective_rank(vel_matrix):
    U, S, Vt = np.linalg.svd(vel_matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-10:
        return 0.0
    p = S2 / total
    H = entropy(p, base=np.e)
    return np.exp(H)


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"

    print("[1] Loading Qwen2.5-7B-Instruct (4-bit quantization)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Check architecture
    cfg = model.config
    print(f"  hidden_size: {cfg.hidden_size}")
    print(f"  num_attention_heads: {cfg.num_attention_heads}")
    print(f"  num_key_value_heads: {getattr(cfg, 'num_key_value_heads', 'N/A')}")
    print(f"  intermediate_size: {cfg.intermediate_size}")
    print(f"  num_hidden_layers: {cfg.num_hidden_layers}")

    D = cfg.hidden_size

    pos_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。他打开宝箱，发现里面有一封古老的信件和一张地图。信上写着通往失落之城的路线，他决定按照地图出发寻找那座传说中的城市。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。回到家后，他把蔬菜分类整理，新鲜的留给自己吃，其余的装车准备明天一早送到集市上去卖。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。她把结果写成论文投稿到顶级期刊，经过三轮审稿，审稿人提出了一些修改意见。她认真回复了每个问题，补充了对照实验，论文最终被接收发表。",
        "侦探仔细检查了犯罪现场。他发现窗户上有指纹，地毯上有泥脚印。顺着线索，他找到了嫌疑人藏身的旅馆。在旅馆房间里，他发现了一件沾有血迹的外套和一把匕首。经过DNA比对，血迹与受害者吻合。嫌疑人最终认罪，案件告破。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他，给他准备了热腾腾的饭菜和干净的床铺。第二天一早，他告别村民继续上路，朝着下一个目的地前进。",
    ]

    scr_texts = [
        "钥匙他打开了走进宝箱，然后密室发现了小明。门后有一把信件，里面放着古老的地图。他决定路线出发，信上写着通往失落之城的宝箱拿起。",
        "蔬菜天刚亮就摘下了农夫，然后施肥浇了水。傍晚时分装进篮子，他把成熟的蔬菜分类整理。新鲜的送到集市上去卖，其余的留给自己吃。他满载而归，装车准备明天一早。",
        "论文失败了，她调整参数重试。第一次实验在实验室里反复，第二次结果更好。她继续优化终于成功了，写成投稿到顶级期刊。审稿人提出修改意见，她认真回复补充对照实验，论文最终被接收发表。",
        "指纹他发现窗户上有犯罪现场，地毯上有泥脚印。顺着线索血迹与受害者吻合，他找到了嫌疑人藏身的旅馆。DNA比对后外套和一把匕首，嫌疑人最终认罪案件告破。",
        "山脚下的村庄他翻过一座山，渡过一条河终于在天黑前到达。村民热情地招待了他，给他准备了热腾腾的饭菜和干净的床铺。旅行者背着行囊走在山路上，第二天一早告别村民继续上路。",
    ]

    np.random.seed(42)
    rnd_texts = []
    for _ in range(3):
        token_ids = np.random.randint(100, tokenizer.vocab_size - 100, size=30).tolist()
        rnd_texts.append(tokenizer.decode(token_ids))

    max_seq_len = 256
    gamma = 0.01

    # ============================================================
    # Extract hidden states
    # ============================================================
    print("\n[2] Extracting hidden states...")

    def extract_hidden(texts, label):
        results = []
        with torch.no_grad():
            for i, text in enumerate(texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
                input_ids = inputs["input_ids"].to(model.device)
                outputs = model(input_ids=input_ids, output_hidden_states=True)
                h = outputs.hidden_states[-1].squeeze(0).cpu().float()
                if h.size(0) >= 10:
                    results.append(h)
                if i == 0:
                    print(f"  {label}: T={h.size(0)}, d={h.size(1)}")
        return results

    pos_hidden = extract_hidden(pos_texts, "pos")
    scr_hidden = extract_hidden(scr_texts, "scr")
    rnd_hidden = extract_hidden(rnd_texts, "rnd")

    # ============================================================
    # Core v8.0 analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("Core v8.0 Analysis: P_c/P_raw, vel_norm, alpha*")
    print("=" * 80)

    # Estimate alpha* from pos
    alpha_estimates = []
    for h in pos_hidden:
        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        F_res = acc[:min_t] + gamma * v_for[:min_t]
        P_raw = (F_res * v_for[:min_t]).sum(dim=-1)
        P_active = (v_for[:min_t] * v_for[:min_t]).sum(dim=-1)
        if P_active.abs().mean() > 1e-10:
            alpha_estimates.append(P_raw.mean().item() / P_active.mean().item())
    alpha_star = np.mean(alpha_estimates)
    print(f"  alpha* = {alpha_star:.4f} (std={np.std(alpha_estimates):.4f})")

    # P_c/P_raw and vel_norm for each text type
    for label, hidden_list in [("pos", pos_hidden), ("scr", scr_hidden), ("rnd", rnd_hidden)]:
        pc_raw_ratios = []
        vel_norms = []
        fc_vel_cos = []

        for h in hidden_list:
            vel = h[1:] - h[:-1]
            acc = vel[1:] - vel[:-1]
            v_for = vel[1:]
            min_t = min(acc.size(0), v_for.size(0))

            F_raw = acc[:min_t] + gamma * v_for[:min_t]
            F_c = acc[:min_t] + (gamma - alpha_star) * v_for[:min_t]

            P_raw = (F_raw * v_for[:min_t]).sum(dim=-1)
            P_c = (F_c * v_for[:min_t]).sum(dim=-1)

            ratio = float(P_c.abs().mean() / (P_raw.abs().mean() + 1e-10))
            pc_raw_ratios.append(ratio)

            vel_norms.append(float(vel.norm(dim=-1).mean()))

            fc_dot_v = (F_c * v_for[:min_t]).sum(dim=-1)
            fc_n = F_c.norm(dim=-1)
            v_n = v_for[:min_t].norm(dim=-1)
            cos = float((fc_dot_v / (fc_n * v_n + 1e-10)).mean())
            fc_vel_cos.append(cos)

        print(f"  {label}: P_c/P_raw={np.mean(pc_raw_ratios):.4f}, vel_norm={np.mean(vel_norms):.2f}, F_c.vel_cos={np.mean(fc_vel_cos):.4f}")

    # ============================================================
    # Rank artifact test (raw_diff only, key comparison)
    # ============================================================
    print("\n" + "=" * 80)
    print("Rank Artifact Test (raw_diff)")
    print("=" * 80)

    T_list = [15, 30, 50, 80, 100, 150]

    for label, hidden_list in [("pos", pos_hidden)]:
        raw_by_T = defaultdict(list)

        for h in hidden_list:
            raw_h = h.numpy()
            T_max = raw_h.shape[0]

            for T in T_list:
                if T >= T_max:
                    continue
                h_sub = raw_h[:T, :]
                vel = h_sub[1:] - h_sub[:-1]
                vel_centered = vel - np.mean(vel, axis=0)

                if vel_centered.shape[0] < D:
                    gram = (vel_centered @ vel_centered.T) / T
                    eigvals = np.linalg.eigvalsh(gram)
                    eigvals = eigvals[eigvals > 1e-10]
                    if len(eigvals) > 0:
                        p = eigvals / eigvals.sum()
                        H = entropy(p, base=np.e)
                        rank = np.exp(H)
                    else:
                        rank = 0
                else:
                    rank = compute_effective_rank(vel_centered)

                raw_by_T[T].append(float(rank))

        print(f"  {label} raw_diff rank vs T:")
        T_vals = sorted(raw_by_T.keys())
        rank_means = [np.mean(raw_by_T[T]) for T in T_vals]
        for T, rm in zip(T_vals, rank_means):
            print(f"    T={T}: rank={rm:.2f} (rank/T={rm/T:.3f})")

        if len(T_vals) >= 3:
            corr, pval = pearsonr(T_vals, rank_means)
            print(f"  Pearson(T, rank): r={corr:.3f}, p={pval:.4f}")

    # ============================================================
    # Per-trajectory constraint parameterization (first trajectory)
    # ============================================================
    print("\n" + "=" * 80)
    print("Per-Trajectory Constraint Parameterization")
    print("=" * 80)

    for h in pos_hidden[:2]:
        raw_h = h.numpy()
        vel = raw_h[1:] - raw_h[:-1]
        N, d = vel.shape
        if N < 5:
            continue

        h_mid = raw_h[:-1][:N]

        U, S, Vt = np.linalg.svd(vel, full_matrices=True)
        total = (S ** 2).sum()
        cum = np.cumsum(S ** 2) / total
        r = int(np.searchsorted(cum, 0.95) + 1)
        null_basis = Vt[r:]

        r2_vals = []
        n_holo = 0
        n_test = min(null_basis.shape[0], 10)

        for i in range(n_test):
            n_i = null_basis[i]
            violations = vel @ n_i
            A = np.column_stack([h_mid, np.ones(N)])
            coeffs, _, _, _ = np.linalg.lstsq(A, violations, rcond=None)
            predicted = A @ coeffs
            residual = violations - predicted
            rms_res = np.sqrt(np.mean(residual ** 2))
            rms_tgt = np.sqrt(np.mean(violations ** 2))
            r2 = 1.0 - rms_res ** 2 / (rms_tgt ** 2 + 1e-10)
            r2_vals.append(r2)

            h_norm = np.linalg.norm(h_mid, axis=1)
            n_bins = 5
            bin_edges = np.percentile(h_norm, np.linspace(0, 100, n_bins + 1))
            bin_violations = []
            for b in range(n_bins):
                mask = (h_norm >= bin_edges[b]) & (h_norm < bin_edges[b + 1])
                if mask.sum() < 3:
                    continue
                bin_viol = vel[mask] @ n_i
                bin_violations.append(float(np.mean(np.abs(bin_viol))))
            cv = float(np.std(bin_violations) / (np.mean(bin_violations) + 1e-10)) if len(bin_violations) >= 2 else 0
            if cv < 0.15:
                n_holo += 1

        print(f"  T={N}, d={d}, vel_rank={r}, n_constraints={d-r}")
        print(f"  R2_linear: mean={np.mean(r2_vals):.6f}, min={np.min(r2_vals):.6f}, max={np.max(r2_vals):.6f}")
        print(f"  Holonomic: {n_holo}/{n_test}")

    # ============================================================
    # Cross-model comparison summary
    # ============================================================
    print("\n" + "=" * 80)
    print("Cross-Model Comparison: MiniCPM5-1B vs Qwen2.5-7B")
    print("=" * 80)

    comparison = {
        "model": "Qwen2.5-7B-Instruct",
        "hidden_size": D,
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, 'num_key_value_heads', 'N/A'),
        "alpha_star": alpha_star,
        "quantization": "4-bit NF4",
    }
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    # Save
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v16 Qwen2.5-7B Cross-Model Verification",
        "model_config": comparison,
        "alpha_star": alpha_star,
        "alpha_star_std": float(np.std(alpha_estimates)),
    }
    report_path = output_dir / "v16_qwen7b_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()