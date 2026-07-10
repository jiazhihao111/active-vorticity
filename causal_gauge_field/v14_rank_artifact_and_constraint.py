import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.signal import savgol_filter
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_hidden_states(model, tokenizer, texts, max_seq_len, device):
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1].squeeze(0).cpu().float()
            if h.size(0) >= 10:
                all_hidden.append(h)
    return all_hidden


# ============================================================
# P0-1: RankArtifactVerifier
# ============================================================

def compute_effective_rank(cov_matrix):
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]
    if len(eigenvalues) == 0:
        return 0.0
    p = eigenvalues / np.sum(eigenvalues)
    H = entropy(p, base=np.e)
    return np.exp(H)


def compute_effective_rank_svd(vel_matrix):
    U, S, Vt = np.linalg.svd(vel_matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-10:
        return 0.0
    p = S2 / total
    H = entropy(p, base=np.e)
    return np.exp(H)


def extract_velocity(raw_h, method='sg', window=11, polyorder=3):
    T, D = raw_h.shape
    vel = np.zeros_like(raw_h)

    if method == 'raw_diff':
        vel[:-1] = raw_h[1:] - raw_h[:-1]
        vel[-1] = vel[-2]
    elif method == 'sg':
        for d in range(D):
            vel[:, d] = savgol_filter(raw_h[:, d], window, polyorder, deriv=1, delta=1.0)
    elif method == 'central_diff':
        vel[1:-1] = (raw_h[2:] - raw_h[:-2]) / 2.0
        vel[0] = raw_h[1] - raw_h[0]
        vel[-1] = raw_h[-1] - raw_h[-2]

    return vel


def run_rank_sensitivity_test(hidden_list):
    results = []
    T_list = [15, 30, 60, 100]
    methods = [
        ('raw_diff', None, None),
        ('central_diff', None, None),
        ('sg', 5, 2),
        ('sg', 11, 3),
        ('sg', 21, 3),
    ]

    for traj_idx, h in enumerate(hidden_list):
        raw_h = h.numpy()
        T_max, D = raw_h.shape
        actual_T_list = [T for T in T_list if T <= T_max]

        for T in actual_T_list:
            h_sub = raw_h[:T, :]

            for method_name, w, p in methods:
                if method_name == 'sg' and w >= T:
                    continue

                vel = extract_velocity(h_sub, method=method_name, window=w, polyorder=p)
                vel_centered = vel - np.mean(vel, axis=0)

                if T < D:
                    gram = (vel_centered @ vel_centered.T) / T
                    rank = compute_effective_rank(gram)
                else:
                    cov = np.cov(vel_centered.T)
                    rank = compute_effective_rank(cov)

                results.append({
                    'traj_idx': traj_idx,
                    'T': T,
                    'method': f"{method_name}(w={w})" if method_name == 'sg' else method_name,
                    'effective_rank': round(float(rank), 2),
                    'T_max': T_max,
                })

    return results


# ============================================================
# P0-2: Per-Trajectory Constraint Parameterization
# ============================================================

def per_trajectory_constraint_analysis(hidden_list, alpha_star, gamma=0.01, mass=1.0):
    results = []

    for traj_idx, h in enumerate(hidden_list):
        if h.size(0) < 4:
            continue

        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        F_c = mass * acc[:min_t] + (gamma - alpha_star) * v_for[:min_t]

        h_mid = h[1:-1][:min_t]
        vel_np = v_for[:min_t].numpy()
        h_np = h_mid.numpy()
        Fc_np = F_c.numpy()

        N, d = vel_np.shape
        if N < 5:
            continue

        # Per-trajectory velocity SVD -> null basis
        U, S, Vt = np.linalg.svd(vel_np, full_matrices=True)
        total = (S ** 2).sum()
        if total < 1e-10:
            continue
        cum = np.cumsum(S ** 2) / total
        r = int(np.searchsorted(cum, 0.95) + 1)
        r = max(r, 1)
        null_basis = Vt[r:]
        n_constraints = d - r

        # P_c verification
        P_c = (Fc_np * vel_np).sum(axis=-1)
        P_c_mean = float(np.mean(P_c))
        P_c_abs_mean = float(np.mean(np.abs(P_c)))

        # F_c . vel cosine
        fc_dot_vel = np.sum(Fc_np * vel_np, axis=-1)
        fc_norm = np.linalg.norm(Fc_np, axis=-1)
        vel_norm = np.linalg.norm(vel_np, axis=-1)
        cos_mean = float(np.mean(fc_dot_vel / (fc_norm * vel_norm + 1e-10)))

        # Per-trajectory constraint parameterization
        constraint_params = []
        for i in range(min(n_constraints, 10)):
            n_i = null_basis[i]
            violations = vel_np @ n_i

            # Linear: n_i^T . vel ≈ w^T . h + b
            A = np.column_stack([h_np, np.ones(N)])
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, violations, rcond=None)
            predicted = A @ coeffs
            residual = violations - predicted
            rms_res = np.sqrt(np.mean(residual ** 2))
            rms_tgt = np.sqrt(np.mean(violations ** 2))
            r_squared = 1.0 - rms_res ** 2 / (rms_tgt ** 2 + 1e-10)

            # Quadratic
            h_sq = np.column_stack([h_np, h_np ** 2, np.ones(N)])
            coeffs2, _, _, _ = np.linalg.lstsq(h_sq, violations, rcond=None)
            pred2 = h_sq @ coeffs2
            res2 = violations - pred2
            rms_res2 = np.sqrt(np.mean(res2 ** 2))
            r_squared_quad = 1.0 - rms_res2 ** 2 / (rms_tgt ** 2 + 1e-10)

            # Holonomic check via bin CV
            h_norm = np.linalg.norm(h_np, axis=1)
            n_bins = 5
            bin_edges = np.percentile(h_norm, np.linspace(0, 100, n_bins + 1))
            bin_violations = []
            for b in range(n_bins):
                mask = (h_norm >= bin_edges[b]) & (h_norm < bin_edges[b + 1])
                if mask.sum() < 3:
                    continue
                bin_viol = vel_np[mask] @ n_i
                bin_violations.append(float(np.mean(np.abs(bin_viol))))
            cv_bins = float(np.std(bin_violations) / (np.mean(bin_violations) + 1e-10)) if len(bin_violations) >= 2 else 0
            is_holonomic = cv_bins < 0.15

            constraint_params.append({
                "idx": i,
                "r_squared_linear": float(r_squared),
                "r_squared_quad": float(r_squared_quad),
                "delta_r2": float(r_squared_quad - r_squared),
                "cv_bins": cv_bins,
                "is_holonomic": is_holonomic,
                "rms_violation": float(rms_tgt),
            })

        results.append({
            "traj_idx": traj_idx,
            "T": N,
            "d": d,
            "vel_rank": r,
            "n_constraints": n_constraints,
            "P_c_mean": P_c_mean,
            "P_c_abs_mean": P_c_abs_mean,
            "fc_vel_cosine": cos_mean,
            "constraint_params": constraint_params,
        })

    return results


# ============================================================
# Main
# ============================================================

def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Loading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    pos_texts_long = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。他打开宝箱，发现里面有一封古老的信件和一张地图。信上写着通往失落之城的路线，他决定按照地图出发寻找那座传说中的城市。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。回到家后，他把蔬菜分类整理，新鲜的留给自己吃，其余的装车准备明天一早送到集市上去卖。邻居们都说他的菜是全村最好的。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。她把结果写成论文投稿到顶级期刊，经过三轮审稿，审稿人提出了一些修改意见。她认真回复了每个问题，补充了对照实验，论文最终被接收发表。",
        "厨师先准备好食材：鸡蛋、面粉和糖。他把面粉倒入碗中，加入鸡蛋和糖搅拌，然后放进烤箱。三十分钟后，蛋糕做好了。他品尝了一小块，觉得甜度不够，于是又做了一批加了更多糖的。新蛋糕甜而不腻，他满意地把它切成块，装进精美的礼盒准备送给朋友。",
        "侦探仔细检查了犯罪现场。他发现窗户上有指纹，地毯上有泥脚印。顺着线索，他找到了嫌疑人藏身的旅馆。在旅馆房间里，他发现了一件沾有血迹的外套和一把匕首。经过DNA比对，血迹与受害者吻合。嫌疑人最终认罪，案件告破。侦探因此获得了局长的嘉奖。",
        "学生们在教室里安静地考试。小李认真读题，仔细计算，把答案写在答题卡上。考试结束后，他检查了一遍才交卷。几天后成绩公布，他考了全班第一名。老师表扬了他的努力，同学们也纷纷祝贺。他决定继续保持这样的学习状态，争取在期末考试中再创佳绩。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他，给他准备了热腾腾的饭菜和干净的床铺。第二天一早，他告别村民继续上路，朝着下一个目的地前进。一路上风景如画，他拍了很多照片留作纪念。",
        "医生询问了病人的症状后，安排了血液检查。检查结果显示感染，医生开了抗生素。病人按时服药，一周后康复了。出院那天，医生叮嘱他注意休息和饮食，定期复查。病人感激地握着医生的手说谢谢。回家后，他按照医嘱调整了生活习惯，身体越来越好，再也没有复发过。",
        "建筑师先画了设计图，然后计算了承重结构。施工队按图纸打地基、砌墙、封顶。半年后，一栋大楼拔地而起。验收时，工程师对每个房间都进行了仔细检查，发现几处小问题，施工队立即整改。最终大楼通过了所有安全检测，业主非常满意，决定把下一个项目也交给这支团队。",
        "渔夫清晨划船出海。他撒下渔网，等了几个小时。收网时发现网里全是鱼，他高兴地把鱼运回港口卖了个好价钱。用这笔钱，他给妻子买了一条新裙子，给儿子买了一套画笔。妻子穿上裙子笑得合不拢嘴，儿子拿着画笔画了一幅全家福。渔夫看着这一切，觉得生活真美好。",
    ]

    max_seq_len = 256

    print("[2] Extracting hidden states (max_seq_len=256)...")
    pos_hidden = get_hidden_states(model, tokenizer, pos_texts_long, max_seq_len, device)

    seq_lens = [h.size(0) for h in pos_hidden]
    print(f"  Trajectory lengths: {seq_lens}")
    print(f"  Mean length: {np.mean(seq_lens):.1f}, Min: {min(seq_lens)}, Max: {max(seq_lens)}")

    # ============================================================
    # P0-1: Rank Artifact Verification
    # ============================================================
    print("\n" + "=" * 80)
    print("P0-1: Rank Artifact Verification")
    print("=" * 80)

    rank_results = run_rank_sensitivity_test(pos_hidden)

    # Aggregate by (T, method)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rank_results:
        key = (r['T'], r['method'])
        agg[key].append(r['effective_rank'])

    print(f"\n{'T':<6} | {'Method':<18} | {'Mean_Rank':<10} | {'Std':<8} | {'N_traj':<6}")
    print("-" * 60)
    for (T, method), ranks in sorted(agg.items()):
        print(f"{T:<6} | {method:<18} | {np.mean(ranks):<10.2f} | {np.std(ranks):<8.2f} | {len(ranks):<6}")

    # Per-trajectory detail for raw_diff at T=max
    print("\n--- Per-trajectory detail (raw_diff, T=full) ---")
    for r in rank_results:
        if r['method'] == 'raw_diff' and r['T'] == r['T_max']:
            print(f"  Traj {r['traj_idx']}: T={r['T']}, rank={r['effective_rank']:.2f}")

    # ============================================================
    # P0-2: Per-Trajectory Constraint Parameterization
    # ============================================================
    print("\n" + "=" * 80)
    print("P0-2: Per-Trajectory Constraint Parameterization")
    print("=" * 80)

    # Estimate alpha*
    gamma = 0.01
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
    print(f"  alpha* = {alpha_star:.4f}")

    constraint_results = per_trajectory_constraint_analysis(pos_hidden, alpha_star, gamma)

    print(f"\n  Per-trajectory summary:")
    print(f"  {'Traj':<5} | {'T':<5} | {'vel_rank':<8} | {'n_constr':<8} | {'P_c':<10} | {'cos':<8} | {'R2_mean':<8} | {'R2_min':<8} | {'holo%':<6}")
    print("  " + "-" * 80)

    all_r2_linear = []
    all_r2_quad = []
    n_holonomic_total = 0
    n_constraints_total = 0

    for cr in constraint_results:
        r2_vals = [cp['r_squared_linear'] for cp in cr['constraint_params']]
        r2q_vals = [cp['r_squared_quad'] for cp in cr['constraint_params']]
        n_holo = sum(1 for cp in cr['constraint_params'] if cp['is_holonomic'])
        n_total = len(cr['constraint_params'])

        all_r2_linear.extend(r2_vals)
        all_r2_quad.extend(r2q_vals)
        n_holonomic_total += n_holo
        n_constraints_total += n_total

        r2_mean = np.mean(r2_vals) if r2_vals else 0
        r2_min = np.min(r2_vals) if r2_vals else 0
        holo_pct = n_holo / n_total * 100 if n_total > 0 else 0

        print(f"  {cr['traj_idx']:<5} | {cr['T']:<5} | {cr['vel_rank']:<8} | {cr['n_constraints']:<8} | {cr['P_c_mean']:<10.4f} | {cr['fc_vel_cosine']:<8.4f} | {r2_mean:<8.4f} | {r2_min:<8.4f} | {holo_pct:<6.1f}")

    print(f"\n  === Global Summary ===")
    print(f"  R²_linear: mean={np.mean(all_r2_linear):.4f}, std={np.std(all_r2_linear):.4f}, min={np.min(all_r2_linear):.4f}, max={np.max(all_r2_linear):.4f}")
    print(f"  R²_quad:   mean={np.mean(all_r2_quad):.4f}, std={np.std(all_r2_quad):.4f}")
    print(f"  ΔR²(quad-lin): mean={np.mean([q-l for q,l in zip(all_r2_quad, all_r2_linear)]):.6f}")
    print(f"  Holonomic: {n_holonomic_total}/{n_constraints_total} = {n_holonomic_total/n_constraints_total*100:.1f}%")

    # ============================================================
    # Diagnosis Matrix
    # ============================================================
    print("\n" + "=" * 80)
    print("DIAGNOSIS: Rank=12 Artifact Check")
    print("=" * 80)

    raw_ranks_T100 = [r['effective_rank'] for r in rank_results if r['method'] == 'raw_diff' and r['T'] >= 60]
    sg11_ranks_T100 = [r['effective_rank'] for r in rank_results if r['method'] == 'sg(w=11)' and r['T'] >= 60]
    raw_ranks_T30 = [r['effective_rank'] for r in rank_results if r['method'] == 'raw_diff' and r['T'] == 30]
    sg11_ranks_T30 = [r['effective_rank'] for r in rank_results if r['method'] == 'sg(w=11)' and r['T'] == 30]

    print(f"\n  T=30: raw_diff rank={np.mean(raw_ranks_T30):.2f} vs sg(w=11) rank={np.mean(sg11_ranks_T30):.2f}")
    if raw_ranks_T100:
        print(f"  T≥60: raw_diff rank={np.mean(raw_ranks_T100):.2f} vs sg(w=11) rank={np.mean(sg11_ranks_T100):.2f}")

    # Check if rank grows with T
    raw_by_T = defaultdict(list)
    for r in rank_results:
        if r['method'] == 'raw_diff':
            raw_by_T[r['T']].append(r['effective_rank'])

    print(f"\n  raw_diff rank vs T:")
    for T in sorted(raw_by_T.keys()):
        print(f"    T={T}: rank={np.mean(raw_by_T[T]):.2f} ± {np.std(raw_by_T[T]):.2f}")

    # Diagnosis
    if raw_by_T:
        T_vals = sorted(raw_by_T.keys())
        rank_vals = [np.mean(raw_by_T[T]) for T in T_vals]
        if len(T_vals) >= 3:
            from scipy.stats import pearsonr
            corr, pval = pearsonr(T_vals, rank_vals)
            print(f"\n  Pearson(T, rank): r={corr:.3f}, p={pval:.4f}")
            if corr > 0.8 and pval < 0.05:
                print("  ⚠️ DIAGNOSIS: Rank grows with T → likely SAMPLE TRUNCATION artifact")
            elif corr < 0.3:
                print("  ✅ DIAGNOSIS: Rank saturates → likely PHYSICAL/ARCHITECTURAL ground truth")
            else:
                print("  ⚠️ DIAGNOSIS: Inconclusive → need more T values")

    # ============================================================
    # Save report
    # ============================================================
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v14 Rank Artifact Verification + Per-Trajectory Constraint Parameterization",
        "alpha_star": alpha_star,
        "max_seq_len": max_seq_len,
        "trajectory_lengths": seq_lens,
        "rank_sensitivity": rank_results,
        "rank_aggregate": {f"T={k[0]},{k[1]}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                           for k, v in agg.items()},
        "per_trajectory_constraints": constraint_results,
        "global_constraint_summary": {
            "r2_linear_mean": float(np.mean(all_r2_linear)),
            "r2_linear_std": float(np.std(all_r2_linear)),
            "r2_linear_min": float(np.min(all_r2_linear)),
            "r2_quad_mean": float(np.mean(all_r2_quad)),
            "holonomic_fraction": float(n_holonomic_total / n_constraints_total) if n_constraints_total > 0 else 0,
        },
    }

    report_path = output_dir / "v14_rank_artifact_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()