import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.stats import entropy, kstest
from collections import defaultdict

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
# P1-1: RMT Zero Hypothesis Test
# ============================================================

def compute_local_jacobians(h_traj, vel_traj, k_neighbors=30, alpha_reg=1e-3, pca_dim=None):
    T, D = h_traj.shape
    if pca_dim and pca_dim < D:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim)
        h_pca = pca.fit_transform(h_traj)
        vel_pca = pca.transform(vel_traj)
    else:
        h_pca = h_traj
        vel_pca = vel_traj
        pca_dim = D

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, T))
    nn.fit(h_pca)

    jacobians = []
    for i in range(T):
        distances, indices = nn.kneighbors([h_pca[i]])
        indices = indices[0]

        H_local = h_pca[indices] - h_pca[i]
        V_local = vel_pca[indices] - vel_pca[i]

        reg = Ridge(alpha=alpha_reg, fit_intercept=False)
        reg.fit(H_local, V_local)
        J = reg.coef_.T

        jacobians.append(J)

    return jacobians, pca_dim


def rmt_analysis(jacobians, label=""):
    n_jac = len(jacobians)
    d = jacobians[0].shape[0]

    all_anti_eigvals = []
    all_sym_eigvals = []
    vorticity_norms = []
    dissipation_norms = []

    for J in jacobians:
        J_sym = (J + J.T) / 2.0
        J_anti = (J - J.T) / 2.0

        vorticity_norms.append(np.linalg.norm(J_anti, 'fro'))
        dissipation_norms.append(np.linalg.norm(J_sym, 'fro'))

        anti_eigvals = np.linalg.eigvalsh(J_anti)
        all_anti_eigvals.extend(anti_eigvals.tolist())

        sym_eigvals = np.linalg.eigvalsh(J_sym)
        all_sym_eigvals.extend(sym_eigvals.tolist())

    all_anti_eigvals = np.array(all_anti_eigvals)
    all_sym_eigvals = np.array(all_sym_eigvals)

    # GOE reference: Wigner semicircle distribution
    # For anti-symmetric matrix, eigenvalues come in ±pairs
    # The distribution of |eigenvalues| should follow Marchenko-Pastur if random
    # For anti-symmetric Gaussian random matrix, eigenvalue distribution is symmetric around 0
    # with variance = sigma^2 / d where sigma is the std of matrix entries

    anti_std = np.std(jacobians[0] - jacobians[0].T) / np.sqrt(2)
    n_anti = len(all_anti_eigvals)

    # Generate GOE reference (anti-symmetric Gaussian random matrix)
    n_ref = 100
    ref_anti_eigvals = []
    for _ in range(n_ref):
        R = np.random.randn(d, d) * anti_std
        R_anti = (R - R.T) / 2.0
        ref_anti_eigvals.extend(np.linalg.eigvalsh(R_anti).tolist())
    ref_anti_eigvals = np.array(ref_anti_eigvals)

    # KS test: compare empirical anti-symmetric eigenvalue distribution vs random reference
    ks_stat, ks_pval = kstest(np.abs(all_anti_eigvals), np.abs(ref_anti_eigvals))

    # Vorticity vs Dissipation ratio
    mean_vort = np.mean(vorticity_norms)
    mean_diss = np.mean(dissipation_norms)
    vort_diss_ratio = mean_vort / (mean_diss + 1e-10)

    # Spectral gap: largest |eigenvalue| of anti-symmetric part
    max_anti_eig = np.max(np.abs(all_anti_eigvals))
    max_ref_eig = np.max(np.abs(ref_anti_eigvals))

    return {
        "label": label,
        "n_jacobians": n_jac,
        "pca_dim": d,
        "mean_vorticity": float(mean_vort),
        "mean_dissipation": float(mean_diss),
        "vort_diss_ratio": float(vort_diss_ratio),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "max_anti_eigenvalue": float(max_anti_eig),
        "max_ref_eigenvalue": float(max_ref_eig),
        "anti_eigenvalue_range": [float(np.percentile(all_anti_eigvals, 1)),
                                   float(np.percentile(all_anti_eigvals, 99))],
        "ref_eigenvalue_range": [float(np.percentile(ref_anti_eigvals, 1)),
                                  float(np.percentile(ref_anti_eigvals, 99))],
        "vorticity_norms_mean": float(np.mean(vorticity_norms)),
        "vorticity_norms_std": float(np.std(vorticity_norms)),
        "dissipation_norms_mean": float(np.mean(dissipation_norms)),
        "dissipation_norms_std": float(np.std(dissipation_norms)),
    }


# ============================================================
# P1-2: Architecture Isomorphism
# ============================================================

def extract_attention_subspaces(model, layer_idx=-1):
    """Extract QKV and output projection subspaces from the last transformer layer."""
    if hasattr(model, 'model'):
        base = model.model
    else:
        base = model

    layers = base.layers if hasattr(base, 'layers') else base.decoder.layers
    if layer_idx == -1:
        layer = layers[-1]
    else:
        layer = layers[layer_idx]

    self_attn = layer.self_attn

    q_proj = self_attn.q_proj.weight.detach().cpu().float().numpy()
    k_proj = self_attn.k_proj.weight.detach().cpu().float().numpy()
    v_proj = self_attn.v_proj.weight.detach().cpu().float().numpy()
    o_proj = self_attn.o_proj.weight.detach().cpu().float().numpy()

    num_heads = 16
    head_dim = 128
    hidden_size = 1536
    num_kv_heads = 2

    # Q: [1536, 1536] -> 16 heads, each [128, 1536]
    # Each head's row space is a 128-dim subspace of R^1536
    q_heads = []
    for i in range(num_heads):
        q_head = q_proj[i*head_dim:(i+1)*head_dim, :]
        q_heads.append(q_head)

    # V: [256, 1536] -> 2 KV heads, each [128, 1536]
    v_heads = []
    for i in range(num_kv_heads):
        v_head = v_proj[i*head_dim:(i+1)*head_dim, :]
        v_heads.append(v_head)

    # O: [1536, 1536] -> column space
    # Each head's column space is a 128-dim subspace of R^1536
    o_heads = []
    for i in range(num_heads):
        o_head = o_proj[:, i*head_dim:(i+1)*head_dim]
        o_heads.append(o_head)

    # FFN
    mlp = layer.mlp
    gate_proj = mlp.gate_proj.weight.detach().cpu().float().numpy()
    up_proj = mlp.up_proj.weight.detach().cpu().float().numpy()
    down_proj = mlp.down_proj.weight.detach().cpu().float().numpy()

    return {
        'q_heads': q_heads,
        'k_proj': k_proj,
        'v_heads': v_heads,
        'o_heads': o_heads,
        'gate_proj': gate_proj,
        'up_proj': up_proj,
        'down_proj': down_proj,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
    }


def compute_principal_angles(subspace_A, subspace_B):
    """Compute principal angles between two subspaces defined by their bases.
    subspace_A: [r1, d] - row vectors are basis of subspace A
    subspace_B: [r2, d] - row vectors are basis of subspace B
    """
    Q_A, _ = np.linalg.qr(subspace_A.T)
    Q_B, _ = np.linalg.qr(subspace_B.T)

    M = Q_A.T @ Q_B
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    angles = np.arccos(svals)
    return angles


def architecture_isomorphism_analysis(tangent_basis, arch_subspaces):
    """Compare tangent space basis with architecture subspaces."""
    W_r = tangent_basis  # [r, d]

    results = {}

    # 1. Per-head Q subspace comparison
    q_angles = []
    for i, q_head in enumerate(arch_subspaces['q_heads']):
        angles = compute_principal_angles(W_r, q_head)
        q_angles.append({
            'head_idx': i,
            'min_angle': float(np.min(angles)),
            'mean_angle': float(np.mean(angles)),
            'n_small_angles': int(np.sum(angles < np.pi/6)),
        })
    results['q_head_angles'] = q_angles

    # 2. Per-head V subspace comparison (KV heads)
    v_angles = []
    for i, v_head in enumerate(arch_subspaces['v_heads']):
        angles = compute_principal_angles(W_r, v_head)
        v_angles.append({
            'kv_head_idx': i,
            'min_angle': float(np.min(angles)),
            'mean_angle': float(np.mean(angles)),
        })
    results['v_head_angles'] = v_angles

    # 3. Per-head O subspace comparison
    o_angles = []
    for i, o_head in enumerate(arch_subspaces['o_heads']):
        angles = compute_principal_angles(W_r, o_head.T)
        o_angles.append({
            'head_idx': i,
            'min_angle': float(np.min(angles)),
            'mean_angle': float(np.mean(angles)),
            'n_small_angles': int(np.sum(angles < np.pi/6)),
        })
    results['o_head_angles'] = o_angles

    # 4. Full Q, K, V, O projection comparison
    for name, proj in [('Q_full', arch_subspaces['q_heads'][0][:0].T),
                        ('K_full', arch_subspaces['k_proj']),
                        ('V_full', arch_subspaces['v_heads'][0][:0].T)]:
        pass

    # Q full row space
    q_full = arch_subspaces['q_heads'][0]  # Just use first head shape as reference
    # Actually compute full Q projection subspace
    q_all = np.vstack(arch_subspaces['q_heads'])  # [1536, 1536]
    angles_q_full = compute_principal_angles(W_r, q_all)
    results['q_full'] = {
        'min_angle': float(np.min(angles_q_full)),
        'mean_angle': float(np.mean(angles_q_full)),
        'n_small_angles': int(np.sum(angles_q_full < np.pi/6)),
    }

    # O full column space
    o_all = np.hstack([oh for oh in arch_subspaces['o_heads']])  # [1536, 1536]
    angles_o_full = compute_principal_angles(W_r, o_all.T)
    results['o_full'] = {
        'min_angle': float(np.min(angles_o_full)),
        'mean_angle': float(np.mean(angles_o_full)),
        'n_small_angles': int(np.sum(angles_o_full < np.pi/6)),
    }

    # 5. FFN comparison
    # gate_proj: [4608, 1536] -> row space
    gate = arch_subspaces['gate_proj']
    angles_gate = compute_principal_angles(W_r, gate)
    results['gate_proj'] = {
        'min_angle': float(np.min(angles_gate)),
        'mean_angle': float(np.mean(angles_gate)),
        'n_small_angles': int(np.sum(angles_gate < np.pi/6)),
    }

    # down_proj: [1536, 4608] -> column space
    down = arch_subspaces['down_proj']
    angles_down = compute_principal_angles(W_r, down.T)
    results['down_proj'] = {
        'min_angle': float(np.min(angles_down)),
        'mean_angle': float(np.mean(angles_down)),
        'n_small_angles': int(np.sum(angles_down < np.pi/6)),
    }

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

    pos_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。他打开宝箱，发现里面有一封古老的信件和一张地图。信上写着通往失落之城的路线，他决定按照地图出发寻找那座传说中的城市。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。回到家后，他把蔬菜分类整理，新鲜的留给自己吃，其余的装车准备明天一早送到集市上去卖。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。她把结果写成论文投稿到顶级期刊，经过三轮审稿，审稿人提出了一些修改意见。她认真回复了每个问题，补充了对照实验，论文最终被接收发表。",
        "侦探仔细检查了犯罪现场。他发现窗户上有指纹，地毯上有泥脚印。顺着线索，他找到了嫌疑人藏身的旅馆。在旅馆房间里，他发现了一件沾有血迹的外套和一把匕首。经过DNA比对，血迹与受害者吻合。嫌疑人最终认罪，案件告破。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他，给他准备了热腾腾的饭菜和干净的床铺。第二天一早，他告别村民继续上路，朝着下一个目的地前进。",
    ]

    rnd_texts = []
    np.random.seed(42)
    for _ in range(3):
        token_ids = np.random.randint(100, 130560 - 100, size=30).tolist()
        rnd_texts.append(tokenizer.decode(token_ids))

    max_seq_len = 256

    print("[2] Extracting hidden states...")
    pos_hidden = get_hidden_states(model, tokenizer, pos_texts, max_seq_len, device)
    rnd_hidden = get_hidden_states(model, tokenizer, rnd_texts, max_seq_len, device)

    # ============================================================
    # P1-1: RMT Analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("P1-1: RMT Zero Hypothesis Test")
    print("=" * 80)

    pca_dim = 32
    rmt_results = {}

    for label, hidden_list in [("pos", pos_hidden), ("rnd", rnd_hidden)]:
        print(f"\n  Processing {label}...")
        all_vort = []
        all_diss = []

        for traj_idx, h in enumerate(hidden_list):
            raw_h = h.numpy()
            vel = raw_h[1:] - raw_h[:-1]
            h_for = raw_h[:-1]

            T_avail = h_for.shape[0]
            if T_avail < 20:
                continue

            actual_pca = min(pca_dim, T_avail - 1)

            jacobians, eff_dim = compute_local_jacobians(
                h_for, vel, k_neighbors=min(20, T_avail // 2),
                alpha_reg=1e-3, pca_dim=actual_pca
            )

            for J in jacobians:
                J_anti = (J - J.T) / 2.0
                J_sym = (J + J.T) / 2.0
                all_vort.append(np.linalg.norm(J_anti, 'fro'))
                all_diss.append(np.linalg.norm(J_sym, 'fro'))

            if traj_idx == 0:
                # Detailed RMT for first trajectory
                rmt_detail = rmt_analysis(jacobians, label=f"{label}_traj0")
                rmt_results[label] = rmt_detail

        print(f"  {label}: mean_vorticity={np.mean(all_vort):.4f}, mean_dissipation={np.mean(all_diss):.4f}")
        print(f"  {label}: vort/diss ratio={np.mean(all_vort)/(np.mean(all_diss)+1e-10):.4f}")

    for label, rmt in rmt_results.items():
        print(f"\n  RMT Detail [{label}]:")
        print(f"    KS stat: {rmt['ks_stat']:.4f}, p-value: {rmt['ks_pval']:.6f}")
        print(f"    Vorticity mean: {rmt['mean_vorticity']:.4f}")
        print(f"    Dissipation mean: {rmt['mean_dissipation']:.4f}")
        print(f"    Vort/Diss ratio: {rmt['vort_diss_ratio']:.4f}")
        print(f"    Max anti eigenvalue: {rmt['max_anti_eigenvalue']:.4f} vs ref: {rmt['max_ref_eigenvalue']:.4f}")
        if rmt['ks_pval'] < 0.05:
            print(f"    ✅ KS test REJECTS null hypothesis → vorticity is NOT random noise")
        else:
            print(f"    ❌ KS test FAILS to reject null hypothesis → vorticity could be noise")

    # ============================================================
    # P1-2: Architecture Isomorphism
    # ============================================================
    print("\n" + "=" * 80)
    print("P1-2: Architecture Isomorphism Analysis")
    print("=" * 80)

    # Extract tangent basis from pos trajectories
    print("\n  Extracting tangent basis from pos trajectories...")
    all_vel = []
    for h in pos_hidden:
        raw_h = h.numpy()
        vel = raw_h[1:] - raw_h[:-1]
        all_vel.append(vel)

    vel_cat = np.vstack(all_vel)
    U, S, Vt = np.linalg.svd(vel_cat, full_matrices=False)
    total = (S ** 2).sum()
    cum = np.cumsum(S ** 2) / total
    r_95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  Global vel rank (95% variance): {r_95}")
    print(f"  Top 20 singular values: {S[:20].tolist()}")

    # Use multiple rank cutoffs for comparison
    for r_test in [12, r_95, 30, 50]:
        if r_test > len(S):
            continue
        W_r = Vt[:r_test]  # [r, 1536]

        print(f"\n  --- Tangent rank = {r_test} ---")

        # Extract architecture subspaces from last layer
        arch = extract_attention_subspaces(model, layer_idx=-1)

        iso_results = architecture_isomorphism_analysis(W_r, arch)

        # Q heads
        print(f"  Q heads (min_angle per head):")
        q_mins = [ha['min_angle'] * 180 / np.pi for ha in iso_results['q_head_angles']]
        best_q = np.argmin(q_mins)
        print(f"    Range: [{min(q_mins):.1f}°, {max(q_mins):.1f}°], Best head: {best_q} ({q_mins[best_q]:.1f}°)")
        print(f"    Heads with angle < 30°: {sum(1 for a in q_mins if a < 30)}/16")

        # O heads
        print(f"  O heads (min_angle per head):")
        o_mins = [ha['min_angle'] * 180 / np.pi for ha in iso_results['o_head_angles']]
        best_o = np.argmin(o_mins)
        print(f"    Range: [{min(o_mins):.1f}°, {max(o_mins):.1f}°], Best head: {best_o} ({o_mins[best_o]:.1f}°)")
        print(f"    Heads with angle < 30°: {sum(1 for a in o_mins if a < 30)}/16")

        # V (KV) heads
        print(f"  V KV heads (min_angle):")
        for va in iso_results['v_head_angles']:
            print(f"    KV head {va['kv_head_idx']}: min_angle={va['min_angle']*180/np.pi:.1f}°")

        # Full projections
        print(f"  Full Q projection: min_angle={iso_results['q_full']['min_angle']*180/np.pi:.1f}°, n<30°={iso_results['q_full']['n_small_angles']}")
        print(f"  Full O projection: min_angle={iso_results['o_full']['min_angle']*180/np.pi:.1f}°, n<30°={iso_results['o_full']['n_small_angles']}")
        print(f"  Gate proj (FFN): min_angle={iso_results['gate_proj']['min_angle']*180/np.pi:.1f}°, n<30°={iso_results['gate_proj']['n_small_angles']}")
        print(f"  Down proj (FFN): min_angle={iso_results['down_proj']['min_angle']*180/np.pi:.1f}°, n<30°={iso_results['down_proj']['n_small_angles']}")

    # Also check layer 0 (shallow) and layer 12 (mid)
    print("\n  --- Cross-layer comparison (rank=30) ---")
    for layer_idx in [0, 12, 23]:
        arch_l = extract_attention_subspaces(model, layer_idx=layer_idx)
        W_r30 = Vt[:30]
        iso_l = architecture_isomorphism_analysis(W_r30, arch_l)
        q_mins = [ha['min_angle'] * 180 / np.pi for ha in iso_l['q_head_angles']]
        o_mins = [ha['min_angle'] * 180 / np.pi for ha in iso_l['o_head_angles']]
        print(f"  Layer {layer_idx}: Q best={min(q_mins):.1f}°, O best={min(o_mins):.1f}°, "
              f"Q<30°={sum(1 for a in q_mins if a < 30)}/16, O<30°={sum(1 for a in o_mins if a < 30)}/16")

    # ============================================================
    # Save
    # ============================================================
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v15 RMT + Architecture Isomorphism",
        "rmt_results": rmt_results,
        "vel_rank_95": r_95,
        "top_singular_values": S[:30].tolist(),
    }

    report_path = output_dir / "v15_rmt_architecture_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()