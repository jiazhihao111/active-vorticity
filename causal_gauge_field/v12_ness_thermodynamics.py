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


def compute_ness_dynamics(hidden_list, alpha_star, gamma=0.01, mass=1.0):
    """计算NESS热力学特征

    运动方程: m·ḧ + γ·ḣ = α*·ḣ + F_c + ξ
    等价:     m·ḧ + γ_eff·ḣ = F_c + ξ,  γ_eff = γ - α*

    NESS五大量:
    1. 熵产生率 σ = ⟨P_active⟩ / T_eff
    2. 概率流 J = ⟨ḣ⟩ (稳态非零流)
    3. 有效温度 T_eff = ⟨δh²⟩ / d (涨落强度)
    4. 耗散功率 P_diss = ⟨γ·ḣ·ḣ⟩ (摩擦耗散)
    5. 自由能效率 η_F = P_c_useful / P_active (有用功/驱动功)
    """

    all_vel = []
    all_acc = []
    all_P_active = []
    all_P_raw = []
    all_P_c = []
    all_P_diss = []
    all_h_mid = []
    all_vel_norm = []

    for h in hidden_list:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h = h.unsqueeze(0)

        vel = h[:, 1:, :] - h[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))

        F_total = mass * acc[:, :min_t, :] + gamma * v_for[:, :min_t, :]
        F_active = alpha_star * v_for[:, :min_t, :]
        F_c = F_total - F_active

        P_raw = (F_total * v_for[:, :min_t, :]).sum(dim=-1)
        P_active = (F_active * v_for[:, :min_t, :]).sum(dim=-1)
        P_c = (F_c * v_for[:, :min_t, :]).sum(dim=-1)
        P_diss = gamma * (v_for[:, :min_t, :] * v_for[:, :min_t, :]).sum(dim=-1)

        h_mid = h[:, 1:-1, :] if h.size(1) > 2 else h[:, :1, :]
        h_trim = h_mid[:, :min_t, :]

        all_vel.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))
        all_acc.append(acc[:, :min_t, :].reshape(-1, acc.size(-1)))
        all_P_active.extend(P_active.flatten().tolist())
        all_P_raw.extend(P_raw.flatten().tolist())
        all_P_c.extend(P_c.flatten().tolist())
        all_P_diss.extend(P_diss.flatten().tolist())
        all_h_mid.append(h_trim.reshape(-1, h_trim.size(-1)))
        all_vel_norm.extend(v_for[:, :min_t, :].norm(dim=-1).flatten().tolist())

    vel_cat = torch.cat(all_vel, dim=0).numpy()
    h_cat = torch.cat(all_h_mid, dim=0).numpy()

    # === 1. 熵产生率 ===
    # 对于Langevin系统: σ = ⟨P_active - P_c⟩ / T_eff
    # 简化: σ = ⟨P_diss⟩ / T_eff (摩擦耗散是不可逆熵产生的来源)
    # 但在NESS中, 总熵产生 = ⟨P_active⟩ / T_eff (驱动力的全部做功都转化为热)

    # === 3. 有效温度 ===
    # T_eff = ⟨δv²⟩ / d (速度涨落的均方值/维度)
    # 这里用速度的协方差迹来估计
    vel_mean = vel_cat.mean(axis=0)
    vel_centered = vel_cat - vel_mean
    T_eff_per_dim = np.mean(vel_centered ** 2)
    T_eff = T_eff_per_dim

    # 也可以用隐状态的涨落估计
    h_mean = h_cat.mean(axis=0)
    h_centered = h_cat - h_mean
    T_eff_h = np.mean(h_centered ** 2) / h_cat.shape[1]

    # === 1. 熵产生率 (最终计算) ===
    P_active_mean = np.mean(all_P_active)
    P_diss_mean = np.mean(all_P_diss)
    sigma_active = P_active_mean / (T_eff + 1e-10)
    sigma_diss = P_diss_mean / (T_eff + 1e-10)

    # === 2. 概率流 (细致平衡破缺) ===
    # J = ⟨ḣ⟩ (稳态非零速度)
    J_mean = vel_mean
    J_norm = np.linalg.norm(J_mean)
    J_per_dim = J_norm / len(J_mean)

    # 细致平衡检验: 前向转移 vs 反向转移
    # 如果细致平衡成立, p(h→h') = p(h'→h)·exp(-ΔE/T)
    # 简化检验: 速度分布是否关于0对称
    vel_per_dim = vel_cat[:, 0]  # 取第一维
    skewness = float(stats.skew(vel_per_dim))
    kurtosis = float(stats.kurtosis(vel_per_dim))

    # 更强的细致平衡检验: 每个维度的速度均值是否显著非零
    n_dims = vel_cat.shape[1]
    nonzero_vel_dims = 0
    vel_mean_pvals = []
    for d in range(min(n_dims, 50)):
        t_stat, p_val = stats.ttest_1samp(vel_cat[:, d], 0)
        vel_mean_pvals.append(p_val)
        if p_val < 0.05:
            nonzero_vel_dims += 1

    # === 4. 耗散功率 ===
    P_diss_total = P_diss_mean

    # === 5. 自由能效率 ===
    P_c_mean = np.mean(all_P_c)
    eta_F = abs(P_c_mean) / (abs(P_active_mean) + 1e-10)

    # === 额外: 隐状态范数演化 (NESS特征: 宏观量稳定) ===
    h_norms = np.linalg.norm(h_cat, axis=-1)
    h_norm_mean = np.mean(h_norms)
    h_norm_std = np.std(h_norms)
    h_norm_cv = h_norm_std / (h_norm_mean + 1e-10)

    # === 额外: 速度自相关 (NESS特征: 指数衰减) ===
    # 对单条轨迹计算速度自相关
    autocorr_results = []
    for h in hidden_list[:5]:
        if h.dim() == 2:
            h2 = h.unsqueeze(0)
        else:
            h2 = h
        vel = h2[0, 1:, :] - h2[0, :-1, :]
        if vel.size(0) < 5:
            continue
        vel_np = vel.numpy()
        # 对每个维度计算自相关, 取平均
        acfs = []
        for d in range(min(vel_np.shape[1], 20)):
            v = vel_np[:, d]
            v = v - v.mean()
            if np.std(v) < 1e-10:
                continue
            acf = np.correlate(v, v, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]
            acfs.append(acf)
        if acfs:
            min_len = min(len(a) for a in acfs)
            avg_acf = np.mean([a[:min_len] for a in acfs], axis=0)
            # 拟合指数衰减: ACF(τ) ≈ exp(-τ/τ_c)
            if len(avg_acf) > 2:
                tau_c_estimate = 1.0
                for k in range(1, min(5, len(avg_acf))):
                    if avg_acf[k] > 0:
                        tau_c_estimate = -1.0 / np.log(max(avg_acf[k], 1e-10))
                        break
                autocorr_results.append({
                    "acf_5": avg_acf[:min(5, len(avg_acf))].tolist(),
                    "tau_c_estimate": float(tau_c_estimate),
                })

    return {
        "entropy_production": {
            "sigma_active": float(sigma_active),
            "sigma_diss": float(sigma_diss),
            "P_active_mean": float(P_active_mean),
            "P_diss_mean": float(P_diss_mean),
            "P_c_mean": float(P_c_mean),
        },
        "probability_current": {
            "J_norm": float(J_norm),
            "J_per_dim": float(J_per_dim),
            "n_nonzero_vel_dims": nonzero_vel_dims,
            "n_dims_tested": min(n_dims, 50),
            "vel_skewness": float(skewness),
            "vel_kurtosis": float(kurtosis),
            "detailed_balance_broken": nonzero_vel_dims > min(n_dims, 50) * 0.1,
        },
        "effective_temperature": {
            "T_eff_vel": float(T_eff),
            "T_eff_h": float(T_eff_h),
            "h_norm_mean": float(h_norm_mean),
            "h_norm_std": float(h_norm_std),
            "h_norm_cv": float(h_norm_cv),
        },
        "dissipation": {
            "P_diss_mean": float(P_diss_mean),
            "P_diss_per_dim": float(P_diss_mean / n_dims),
        },
        "free_energy_efficiency": {
            "eta_F": float(eta_F),
            "P_c_P_active_ratio": float(abs(P_c_mean) / (abs(P_active_mean) + 1e-10)),
        },
        "autocorrelation": autocorr_results[:3],
        "alpha_star": float(alpha_star),
        "gamma_eff": float(gamma - alpha_star),
        "n_trajectories": len(hidden_list),
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

    scr_texts = [
        "小明走进房间，宝箱突然出现了。他打开门，发现桌上有一把钥匙。钥匙用他打开了宝箱，然后密室走进了他。",
        "天刚亮，蔬菜就摘下了农夫。他先装进篮子，然后浇了水，最后满载而归。施肥把他放进烤箱，蛋糕走到田里。",
        "科学家在实验室里成功了。她调整了第一次，优化了第二次。实验反复失败，终于参数重试了。结果更好了实验。",
        "厨师把烤箱倒进面粉中。鸡蛋准备好了食材，糖搅拌了他。然后碗加入蛋糕，三十分钟后，面粉做好了厨师。",
        "侦探找到了旅馆，他检查了嫌疑人。窗户上有泥脚印，地毯上有指纹。犯罪现场顺着线索，他藏身在旅馆里发现。",
        "考试在教室里安静了学生。小李写在答题卡上，认真读题，仔细计算。交卷后他检查了一遍，答案才考试结束。",
        "旅行者背着村庄走在山路上。他翻过一条河，渡过一座山，终于到达了行囊。村民天黑前招待了他，山脚下热情地走了。",
        "医生开了血液检查，病人安排了抗生素。检查结果显示症状，医生询问了一周后康复。病人按时感染，服药了结果。",
        "建筑师画了施工队，然后计算了设计图。大楼按图纸打地基，承重结构砌墙封顶。半年后，一栋设计图拔地而起。",
        "渔夫清晨卖了个好价钱。他撒下渔网，等了鱼。收网时发现港口全是网，他高兴地把鱼运回海里划船了几个小时。",
    ]

    # 随机token
    np.random.seed(42)
    rnd_texts = []
    for _ in range(10):
        ids = np.random.randint(100, tokenizer.vocab_size - 100, 30).tolist()
        rnd_texts.append(tokenizer.decode(ids))

    max_seq_len = 64
    gamma = 0.01

    print("[2] Extracting hidden states...")
    pos_hidden = get_hidden_states(model, tokenizer, pos_texts, max_seq_len, device)
    scr_hidden = get_hidden_states(model, tokenizer, scr_texts, max_seq_len, device)
    rnd_hidden = get_hidden_states(model, tokenizer, rnd_texts, max_seq_len, device)

    # 估计α*
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

    print("[3] Computing NESS dynamics...")
    pos_ness = compute_ness_dynamics(pos_hidden, alpha_star, gamma)
    scr_ness = compute_ness_dynamics(scr_hidden, alpha_star, gamma)
    rnd_ness = compute_ness_dynamics(rnd_hidden, alpha_star, gamma)

    # 打印结果
    print("\n" + "=" * 80)
    print("NESS Thermodynamics Results")
    print("=" * 80)

    for label, ness in [("pos", pos_ness), ("scr", scr_ness), ("rnd", rnd_ness)]:
        ep = ness["entropy_production"]
        pc = ness["probability_current"]
        te = ness["effective_temperature"]
        di = ness["dissipation"]
        fe = ness["free_energy_efficiency"]

        print(f"\n--- {label} ---")
        print(f"  Entropy production:")
        print(f"    sigma_active = {ep['sigma_active']:.4f}")
        print(f"    sigma_diss   = {ep['sigma_diss']:.4f}")
        print(f"    P_active     = {ep['P_active_mean']:.2f}")
        print(f"    P_diss       = {ep['P_diss_mean']:.2f}")
        print(f"    P_c          = {ep['P_c_mean']:.4f}")
        print(f"  Probability current:")
        print(f"    J_norm       = {pc['J_norm']:.4f}")
        print(f"    J_per_dim    = {pc['J_per_dim']:.6f}")
        print(f"    nonzero dims = {pc['n_nonzero_vel_dims']}/{pc['n_dims_tested']}")
        print(f"    DB broken    = {pc['detailed_balance_broken']}")
        print(f"    vel skewness = {pc['vel_skewness']:.4f}")
        print(f"  Effective temperature:")
        print(f"    T_eff_vel    = {te['T_eff_vel']:.4f}")
        print(f"    T_eff_h      = {te['T_eff_h']:.4f}")
        print(f"    h_norm_cv    = {te['h_norm_cv']:.4f}")
        print(f"  Free energy efficiency:")
        print(f"    eta_F        = {fe['eta_F']:.6f}")
        print(f"  gamma_eff = {ness['gamma_eff']:.4f}")

    # NESS判定
    print("\n" + "=" * 80)
    print("NESS Verdict")
    print("=" * 80)

    for label, ness in [("pos", pos_ness), ("scr", scr_ness), ("rnd", rnd_ness)]:
        ep = ness["entropy_production"]
        pc = ness["probability_current"]
        te = ness["effective_temperature"]

        c1_sigma = ep["sigma_active"] > 0
        c2_J = pc["J_norm"] > 0
        c3_DB = pc["detailed_balance_broken"]
        c4_cv = te["h_norm_cv"] < 0.5

        n_pass = sum([c1_sigma, c2_J, c3_DB, c4_cv])
        print(f"\n  {label}: sigma>0={c1_sigma}, J>0={c2_J}, DB_broken={c3_DB}, CV<0.5={c4_cv} -> {n_pass}/4")

        if n_pass >= 3:
            print(f"    -> NESS CONFIRMED")
        elif n_pass >= 2:
            print(f"    -> NESS LIKELY")
        else:
            print(f"    -> NOT NESS")

    # 保存
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v12 NESS Thermodynamics",
        "alpha_star": alpha_star,
        "gamma": gamma,
        "gamma_eff": gamma - alpha_star,
        "results": {"pos": pos_ness, "scr": scr_ness, "rnd": rnd_ness},
    }
    report_path = output_dir / "v12_ness_thermodynamics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()