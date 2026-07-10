import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

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
            if h.size(0) >= 4:
                all_hidden.append(h)
    return all_hidden


def extract_constraint_manifold(hidden_list, alpha_star, gamma=0.01, mass=1.0):
    """提取约束流形的微分几何结构

    已知:
    - F_c = m·ḧ + (γ-α*)·ḣ, F_c ⊥ ḣ (P_c ≈ 0)
    - 约束方程: C_i(h, ḣ) = n_i^T · ḣ = 0, i=1,...,d-r
    - n_i是约束流形的法向量

    目标:
    1. 提取n_i作为h的函数: n_i(h) = A_i · h + b_i (线性参数化)
    2. 计算约束流形的度量张量 g_μν
    3. 计算约束流形的曲率
    4. 判别完整/非完整约束
    """

    all_vel = []
    all_h = []
    all_Fc = []

    for h in hidden_list:
        if h.size(0) < 4:
            continue
        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        F_c = mass * acc[:min_t] + (gamma - alpha_star) * v_for[:min_t]

        h_mid = h[1:-1] if h.size(0) > 2 else h[:1]
        h_trim = h_mid[:min_t]

        all_vel.append(v_for[:min_t])
        all_h.append(h_trim)
        all_Fc.append(F_c)

    # 合并所有轨迹
    vel_cat = torch.cat(all_vel, dim=0).numpy()
    h_cat = torch.cat(all_h, dim=0).numpy()
    Fc_cat = torch.cat(all_Fc, dim=0).numpy()

    d = vel_cat.shape[1]
    N = vel_cat.shape[0]

    # === Step 1: 提取速度空间的零空间基 (法向量) ===
    # 逐轨迹提取，然后取平均
    per_traj_null_bases = []
    per_traj_ranks = []

    for h in hidden_list:
        if h.size(0) < 4:
            continue
        vel = h[1:] - h[:-1]
        vel_np = vel.numpy()
        if vel_np.shape[0] < 2:
            continue

        cov = np.cov(vel_np.T)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        total = eigvals.sum()
        if total < 1e-10:
            continue
        cum = np.cumsum(eigvals) / total
        r = int(np.searchsorted(cum, 0.95) + 1)
        per_traj_ranks.append(r)

        U, S, Vt = np.linalg.svd(vel_np, full_matrices=True)
        null_basis = Vt[r:]
        per_traj_null_bases.append(null_basis)

    r_mean = int(np.mean(per_traj_ranks)) if per_traj_ranks else 12
    n_constraints = d - r_mean

    # 用全局速度矩阵提取法向量（更稳定）
    U_g, S_g, Vt_g = np.linalg.svd(vel_cat, full_matrices=True)
    total_g = (S_g ** 2).sum()
    cum_g = np.cumsum(S_g ** 2) / total_g
    r_global = int(np.searchsorted(cum_g, 0.95) + 1)
    null_basis_global = Vt_g[r_global:]

    # === Step 2: 法向量n_i作为h的函数 ===
    # 线性参数化: n_i(h) ≈ A_i · h + b_i
    # 约束: n_i(h)^T · ḣ = 0
    # 线性近似: (A_i · h + b_i)^T · ḣ = 0
    # 最小二乘: 找A_i, b_i使得(A_i·h+b_i)^T·ḣ ≈ 0

    # 简化: 对每个法向量n_i, 检验n_i^T·ḣ是否可以用h线性预测
    # 即: n_i^T·ḣ ≈ w^T·h + b
    # 如果R²高, 说明法向量可以参数化为h的函数(完整约束)
    # 如果R²低, 说明法向量依赖于ḣ(非完整约束)

    linear_param_results = []
    for i in range(min(n_constraints, 10)):
        n_i = null_basis_global[i]
        violations = vel_cat @ n_i

        A = np.column_stack([h_cat, np.ones(N)])
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, violations, rcond=None)
        predicted = A @ coeffs
        residual = violations - predicted
        rms_res = np.sqrt(np.mean(residual ** 2))
        rms_tgt = np.sqrt(np.mean(violations ** 2))
        r_squared = 1.0 - rms_res ** 2 / (rms_tgt ** 2 + 1e-10)

        # 非线性参数化尝试: 加入h的二次项
        h_sq = np.column_stack([h_cat, h_cat ** 2, np.ones(N)])
        coeffs2, _, _, _ = np.linalg.lstsq(h_sq, violations, rcond=None)
        pred2 = h_sq @ coeffs2
        res2 = violations - pred2
        rms_res2 = np.sqrt(np.mean(res2 ** 2))
        r_squared_quad = 1.0 - rms_res2 ** 2 / (rms_tgt ** 2 + 1e-10)

        # 非完整判别: Frobenius可积性条件
        # 如果n_i = ∇f_i, 则 ∂n_i^k/∂h^j = ∂n_i^j/∂h^k
        # 用bin检验: 将h空间分bin, 检验n_i^T·ḣ在不同bin中是否恒定
        h_norm = np.linalg.norm(h_cat, axis=1)
        n_bins = 5
        bin_edges = np.percentile(h_norm, np.linspace(0, 100, n_bins + 1))
        bin_violations = []
        for b in range(n_bins):
            mask = (h_norm >= bin_edges[b]) & (h_norm < bin_edges[b + 1])
            if mask.sum() < 5:
                continue
            bin_viol = vel_cat[mask] @ n_i
            bin_violations.append(float(np.mean(np.abs(bin_viol))))

        cv_bins = np.std(bin_violations) / (np.mean(bin_violations) + 1e-10) if len(bin_violations) >= 2 else 0
        is_holonomic = cv_bins < 0.15

        linear_param_results.append({
            "constraint_idx": i,
            "r_squared_linear": float(r_squared),
            "r_squared_quadratic": float(r_squared_quad),
            "improvement_quad": float(r_squared_quad - r_squared),
            "rms_violation": float(rms_tgt),
            "is_holonomic": is_holonomic,
            "cv_bins": float(cv_bins),
            "n_i_norm": float(np.linalg.norm(n_i)),
        })

    # === Step 3: 约束流形的度量张量 ===
    # 在速度切空间中, 度量由速度协方差给出
    # g_μν = ⟨v_μ · v_ν⟩ (速度的内积矩阵)
    vel_cov = np.cov(vel_cat.T)
    eigvals_vel = np.sort(np.abs(np.linalg.eigvalsh(vel_cov)))[::-1]

    # 在约束子空间中的度量
    # 投影到速度切空间的前r个主方向
    U_vel, S_vel, Vt_vel = np.linalg.svd(vel_cat, full_matrices=False)
    tangent_basis = Vt_vel[:r_global]

    # 诱导度量: g_induced = P^T · g · P, P = tangent_basis
    g_induced = tangent_basis @ vel_cov @ tangent_basis.T

    # === Step 4: 约束流形的曲率 ===
    # Gauss曲率 K = R_1212 / (g_11·g_22 - g_12^2)
    # 对于r维流形嵌入在d维空间中, 计算截面曲率
    # 简化: 用Gauss方程, K ≈ (||第二基本形式||^2 - ||第一基本形式||^2) / ...

    # 第二基本形式: II_μν = n_i · ∂²h/∂v_μ∂v_ν
    # 离散近似: 用加速度在法向量方向的投影
    acc_cat = torch.cat([h[2:] - 2 * h[1:-1] + h[:-2] for h in hidden_list if h.size(0) >= 4], dim=0).numpy()

    # 法向加速度: a_⊥ = Σ_i (a · n_i) n_i
    normal_acc_components = []
    for i in range(min(n_constraints, 10)):
        n_i = null_basis_global[i]
        proj = acc_cat @ n_i
        normal_acc_components.append(float(np.mean(proj ** 2)))

    # 切向加速度: a_∥ = Σ_μ (a · e_μ) e_μ
    tangent_acc_components = []
    for mu in range(min(r_global, 10)):
        e_mu = tangent_basis[mu]
        proj = acc_cat @ e_mu
        tangent_acc_components.append(float(np.mean(proj ** 2)))

    # 截面曲率近似: K ≈ ||a_⊥|| / ||a_∥|| (法向加速度/切向加速度)
    mean_normal_acc = np.mean(normal_acc_components) if normal_acc_components else 0
    mean_tangent_acc = np.mean(tangent_acc_components) if tangent_acc_components else 1
    curvature_approx = mean_normal_acc / (mean_tangent_acc + 1e-10)

    # === Step 5: 约束流形的拓扑 ===
    # Betti numbers via persistent homology approximation
    # 简化: 用速度空间的连通性分析
    # H0 = 连通分量数, H1 = 1维孔洞数

    # 用速度向量的聚类估计H0
    from sklearn.cluster import DBSCAN
    try:
        clustering = DBSCAN(eps=np.mean(vel_cat.std(axis=0)), min_samples=3).fit(vel_cat[:, :min(20, d)])
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    except:
        n_clusters = 1

    return {
        "manifold_dimensions": {
            "ambient_dim": d,
            "tangent_rank": r_global,
            "normal_rank": n_constraints,
            "compression_ratio": float(r_global / d),
        },
        "constraint_parameterization": linear_param_results,
        "induced_metric": {
            "g_induced_eigenvalues_top5": np.sort(np.abs(np.linalg.eigvalsh(g_induced)))[::-1][:5].tolist(),
            "g_induced_condition_number": float(np.max(np.abs(np.linalg.eigvalsh(g_induced))) / (np.min(np.abs(np.linalg.eigvalsh(g_induced))) + 1e-10)),
            "vel_cov_eigenvalues_top10": eigvals_vel[:10].tolist(),
        },
        "curvature": {
            "curvature_approx": float(curvature_approx),
            "mean_normal_acc": float(mean_normal_acc),
            "mean_tangent_acc": float(mean_tangent_acc),
            "normal_acc_per_constraint": normal_acc_components[:5],
            "tangent_acc_per_dim": tangent_acc_components[:5],
        },
        "topology": {
            "n_clusters_approx": n_clusters,
            "vel_effective_rank_per_traj": per_traj_ranks[:10],
            "vel_rank_mean": float(np.mean(per_traj_ranks)) if per_traj_ranks else 0,
        },
        "n_trajectories": len(hidden_list),
        "n_samples": N,
    }


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Loading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    pos_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。",
        "厨师先准备好食材：鸡蛋、面粉和糖。他把面粉倒入碗中，加入鸡蛋和糖搅拌，然后放进烤箱。三十分钟后，蛋糕做好了。",
        "侦探仔细检查了犯罪现场。他发现窗户上有指纹，地毯上有泥脚印。顺着线索，他找到了嫌疑人藏身的旅馆。",
        "学生们在教室里安静地考试。小李认真读题，仔细计算，把答案写在答题卡上。考试结束后，他检查了一遍才交卷。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他。",
        "医生询问了病人的症状后，安排了血液检查。检查结果显示感染，医生开了抗生素。病人按时服药，一周后康复了。",
        "建筑师先画了设计图，然后计算了承重结构。施工队按图纸打地基、砌墙、封顶。半年后，一栋大楼拔地而起。",
        "渔夫清晨划船出海。他撒下渔网，等了几个小时。收网时发现网里全是鱼，他高兴地把鱼运回港口卖了个好价钱。",
    ]

    max_seq_len = 64
    gamma = 0.01

    print("[2] Extracting hidden states...")
    pos_hidden = get_hidden_states(model, tokenizer, pos_texts, max_seq_len, device)

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

    print("[3] Extracting constraint manifold geometry...")
    results = extract_constraint_manifold(pos_hidden, alpha_star, gamma)

    # 打印结果
    print("\n" + "=" * 80)
    print("Constraint Manifold Differential Geometry")
    print("=" * 80)

    md = results["manifold_dimensions"]
    print(f"\n  Ambient dimension: {md['ambient_dim']}")
    print(f"  Tangent rank: {md['tangent_rank']}")
    print(f"  Normal rank (n_constraints): {md['normal_rank']}")
    print(f"  Compression ratio: {md['compression_ratio']:.4f}")

    print(f"\n  Constraint Parameterization:")
    for r in results["constraint_parameterization"]:
        holo = "holonomic" if r["is_holonomic"] else "nonholonomic"
        print(f"    C_{r['constraint_idx']}: R²_lin={r['r_squared_linear']:.4f}, "
              f"R²_quad={r['r_squared_quadratic']:.4f}, "
              f"ΔR²={r['improvement_quad']:.4f}, "
              f"CV={r['cv_bins']:.3f}, "
              f"type={holo}")

    print(f"\n  Induced Metric:")
    print(f"    Eigenvalues: {[f'{v:.4f}' for v in results['induced_metric']['g_induced_eigenvalues_top5']]}")
    print(f"    Condition number: {results['induced_metric']['g_induced_condition_number']:.2f}")
    print(f"    Vel cov top10: {[f'{v:.2f}' for v in results['induced_metric']['vel_cov_eigenvalues_top10']]}")

    print(f"\n  Curvature:")
    print(f"    Curvature approx (||a_⊥||/||a_∥||): {results['curvature']['curvature_approx']:.6f}")
    print(f"    Mean normal acceleration: {results['curvature']['mean_normal_acc']:.4f}")
    print(f"    Mean tangent acceleration: {results['curvature']['mean_tangent_acc']:.4f}")
    print(f"    Normal acc per constraint: {[f'{v:.4f}' for v in results['curvature']['normal_acc_per_constraint']]}")
    print(f"    Tangent acc per dim: {[f'{v:.4f}' for v in results['curvature']['tangent_acc_per_dim']]}")

    print(f"\n  Topology:")
    print(f"    Clusters (approx H0): {results['topology']['n_clusters_approx']}")
    print(f"    Vel rank per traj: {results['topology']['vel_effective_rank_per_traj']}")
    print(f"    Vel rank mean: {results['topology']['vel_rank_mean']:.1f}")

    # 保存
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v13 Constraint Manifold Differential Geometry",
        "alpha_star": alpha_star,
        "results": results,
    }
    report_path = output_dir / "v13_constraint_manifold_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()