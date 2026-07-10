import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.newton.active_force_analyzer import ActiveForceAnalyzer


POSITIVE_TEXTS = [
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
    "小女孩在花园里种下一颗种子。她每天浇水，细心照料。几周后，嫩芽破土而出，最终开出了一朵美丽的花。",
    "消防员接到报警后迅速出发。他们用水枪扑灭了大火，救出了被困在楼里的居民。事后，他们检查了火灾原因。",
    "程序员发现代码有bug，他先用调试器定位问题。找到原因后，他修改了代码并运行测试。测试通过后，他提交了修复。",
    "母亲在厨房准备晚餐。她洗菜、切肉、炒菜，一道道菜端上桌。全家人围坐在一起，享用了一顿温馨的晚餐。",
    "登山队员从大本营出发，经过一号营地、二号营地，逐步适应高海拔。最终，他们在第五天成功登顶了雪山。",
]

NEGATIVE_TEXTS = [
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
    "种子在花园里种下了小女孩。她每天照料，细心浇水。几周后，花朵破土而出，最终嫩芽开出了一颗美丽的嫩芽。",
    "消防员检查了火灾原因后迅速出发。他们用居民扑灭了大火，救出了被困在楼里的水枪。事后，他们报警了火灾。",
    "程序员提交了修复，他先用测试定位问题。找到原因后，他修改了调试器并运行代码。bug通过后，他代码了提交。",
    "母亲在厨房准备晚餐。她炒菜、洗肉、切菜，一道道菜端上桌。全家人围坐在一起，享用了一顿晚餐的厨房。",
    "登山队员从山顶出发，经过二号营地、大本营，逐步适应低海拔。最终，他们在第五天成功登顶了一号营地。",
]


def extract_hidden_trajectories(model, tokenizer, texts, max_seq_len=64, device="cuda"):
    """从真实LLM提取隐状态轨迹"""
    trajectories = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # 取最后一层hidden states: shape [1, seq_len, hidden_size]
            hidden = outputs.hidden_states[-1].squeeze(0).cpu().float()  # [seq_len, hidden_size]

            if hidden.size(0) >= 4:
                trajectories.append(hidden)

    return trajectories


def compute_alpha_star(pos_hidden, gamma=0.01):
    """从轨迹数据直接估计α*"""
    alpha_estimates = []
    for h in pos_hidden:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h2 = h.unsqueeze(0)
        else:
            h2 = h
        vel = h2[:, 1:, :] - h2[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))
        F_res = acc[:, :min_t, :] + gamma * v_for[:, :min_t, :]
        P_raw = (F_res * v_for[:, :min_t, :]).sum(dim=-1)
        P_active = (v_for[:, :min_t, :] * v_for[:, :min_t, :]).sum(dim=-1)
        if P_active.abs().mean() > 1e-10:
            alpha_local = P_raw.mean().item() / P_active.mean().item()
            alpha_estimates.append(alpha_local)
    if not alpha_estimates:
        return None, None
    return float(np.mean(alpha_estimates)), float(np.std(alpha_estimates))


def analyze_layer_dynamics(model, tokenizer, pos_texts, neg_texts, max_seq_len, device, gamma=0.01):
    """逐层分析隐状态动力学"""
    layer_results = {}
    n_layers = model.config.num_hidden_layers

    model.eval()
    pos_all_hidden = {l: [] for l in range(n_layers + 1)}
    neg_all_hidden = {l: [] for l in range(n_layers + 1)}

    with torch.no_grad():
        for text in pos_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l].squeeze(0).cpu().float()
                if h.size(0) >= 4:
                    pos_all_hidden[l].append(h)

        for text in neg_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l].squeeze(0).cpu().float()
                if h.size(0) >= 4:
                    neg_all_hidden[l].append(h)

    for l in range(n_layers + 1):
        pos_h = pos_all_hidden[l]
        neg_h = neg_all_hidden[l]
        if len(pos_h) < 3 or len(neg_h) < 3:
            continue

        alpha_mean, alpha_std = compute_alpha_star(pos_h, gamma=gamma)
        if alpha_mean is None:
            continue

        analyzer = ActiveForceAnalyzer(mass=1.0, friction=gamma)
        v8_results = analyzer.full_analysis(pos_h, neg_h, method="D", alpha=alpha_mean)

        # 约束子空间
        F_c_all = []
        vel_all = []
        for h in pos_h:
            if h.size(0) < 4:
                continue
            if h.dim() == 2:
                h = h.unsqueeze(0)
            vel = h[:, 1:, :] - h[:, :-1, :]
            acc = vel[:, 1:, :] - vel[:, :-1, :]
            v_for = vel[:, 1:, :]
            min_t = min(acc.size(1), v_for.size(1))
            F_c = acc[:, :min_t, :] + (gamma - alpha_mean) * v_for[:, :min_t, :]
            F_c_all.append(F_c.reshape(-1, F_c.size(-1)))
            vel_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))

        if F_c_all:
            F_c_cat = torch.cat(F_c_all, dim=0).numpy()
            vel_cat = torch.cat(vel_all, dim=0).numpy()
            cov_fc = np.cov(F_c_cat.T)
            eigenvalues_fc = np.sort(np.abs(np.linalg.eigvalsh(cov_fc)))[::-1]
            total_var = eigenvalues_fc.sum()
            if total_var > 1e-10:
                cumulative = np.cumsum(eigenvalues_fc) / total_var
                effective_rank_fc = int(np.searchsorted(cumulative, 0.95) + 1)
            else:
                effective_rank_fc = 0
            orthogonality = float(np.mean(np.sum(F_c_cat * vel_cat, axis=-1)))

            vel_cov = np.cov(vel_cat.T)
            eigenvalues_vel = np.sort(np.abs(np.linalg.eigvalsh(vel_cov)))[::-1]
            total_var_vel = eigenvalues_vel.sum()
            if total_var_vel > 1e-10:
                cumulative_vel = np.cumsum(eigenvalues_vel) / total_var_vel
                effective_rank_vel = int(np.searchsorted(cumulative_vel, 0.95) + 1)
            else:
                effective_rank_vel = 0

            F_c_norm = float(np.mean(np.linalg.norm(F_c_cat, axis=-1)))
            vel_norm = float(np.mean(np.linalg.norm(vel_cat, axis=-1)))
        else:
            effective_rank_fc = 0
            effective_rank_vel = 0
            orthogonality = 0
            F_c_norm = 0
            vel_norm = 0

        v8_cp = v8_results.get("corrected_constraint_power", {})
        v8_ed = v8_results.get("energy_decomposition", {})
        v8_ov = v8_results.get("overall_verdict", {})

        layer_results[l] = {
            "alpha_star_mean": alpha_mean,
            "alpha_star_std": alpha_std,
            "pos_Pc_mean": v8_cp.get("pos_Pc_mean"),
            "neg_Pc_mean": v8_cp.get("neg_Pc_mean"),
            "pos_vs_zero_p": v8_cp.get("pos_vs_zero_p"),
            "pos_vs_neg_p": v8_cp.get("pos_vs_neg_p"),
            "criteria": v8_cp.get("criteria"),
            "verdict": v8_ov.get("verdict"),
            "active_fraction_pos": v8_ed.get("active_fraction_pos"),
            "active_fraction_neg": v8_ed.get("active_fraction_neg"),
            "F_c_effective_rank": effective_rank_fc,
            "vel_effective_rank": effective_rank_vel,
            "total_dim": model.config.hidden_size,
            "compression_ratio": float(effective_rank_fc / model.config.hidden_size) if model.config.hidden_size > 0 else 0,
            "F_c_orthogonality_to_vel": orthogonality,
            "F_c_norm": F_c_norm,
            "vel_norm": vel_norm,
            "F_c_to_vel_norm_ratio": float(F_c_norm / (vel_norm + 1e-10)),
        }

    return layer_results


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_config_path = Path(__file__).parent.parent / "causal_gauge_field" / "config.yaml"
    log_dir = "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("v10RealLLM", log_dir, "v10_real_llm.log")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info(f"设备: {device}")
    logger.info(f"模型: MiniCPM5-1B ({model_path})")

    # === 加载模型 ===
    logger.info("[阶段1] 加载MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  参数量: {n_params/1e9:.2f}B")
    logger.info(f"  层数: {model.config.num_hidden_layers}")
    logger.info(f"  隐状态维度: {model.config.hidden_size}")

    max_seq_len = 64

    # === 阶段2: 最终层验证 ===
    logger.info("\n[阶段2] 最终层隐状态提取与v8.0验证...")
    pos_hidden = extract_hidden_trajectories(model, tokenizer, POSITIVE_TEXTS, max_seq_len, device)
    neg_hidden = extract_hidden_trajectories(model, tokenizer, NEGATIVE_TEXTS, max_seq_len, device)
    logger.info(f"  轨迹: pos={len(pos_hidden)}, neg={len(neg_hidden)}")

    # 隐状态基本统计
    pos_norms = [h.norm(dim=-1).mean().item() for h in pos_hidden]
    neg_norms = [h.norm(dim=-1).mean().item() for h in neg_hidden]
    logger.info(f"  pos隐状态范数: mean={np.mean(pos_norms):.4f}, std={np.std(pos_norms):.4f}")
    logger.info(f"  neg隐状态范数: mean={np.mean(neg_norms):.4f}, std={np.std(neg_norms):.4f}")

    # α*估计
    alpha_mean, alpha_std = compute_alpha_star(pos_hidden, gamma=0.01)
    logger.info(f"  α* = {alpha_mean:.4f} ± {alpha_std:.4f}" if alpha_mean else "  α* 估计失败")

    # v8.0核心验证
    if alpha_mean:
        analyzer = ActiveForceAnalyzer(mass=1.0, friction=0.01)
        v8_results = analyzer.full_analysis(pos_hidden, neg_hidden, method="D", alpha=alpha_mean)

        v8_cp = v8_results.get("corrected_constraint_power", {})
        v8_ed = v8_results.get("energy_decomposition", {})
        v8_ov = v8_results.get("overall_verdict", {})

        logger.info(f"  === v8.0核心验证 ===")
        logger.info(f"  pos P_c = {v8_cp.get('pos_Pc_mean', 'N/A'):.6f}")
        logger.info(f"  neg P_c = {v8_cp.get('neg_Pc_mean', 'N/A'):.6f}")
        logger.info(f"  pos_vs_zero_p = {v8_cp.get('pos_vs_zero_p', 'N/A'):.4f}")
        logger.info(f"  pos_vs_neg_p = {v8_cp.get('pos_vs_neg_p', 'N/A'):.6f}")
        logger.info(f"  判决: {v8_ov.get('verdict', 'N/A')}")

        criteria = v8_cp.get("criteria", {})
        logger.info(f"  准则通过: {sum(1 for v in criteria.values() if v)}/3")
        for k, v in criteria.items():
            logger.info(f"    {k}: {v}")

        logger.info(f"  === 能量分解 ===")
        logger.info(f"  P_active占比(pos): {v8_ed.get('active_fraction_pos', 'N/A'):.4f}")
        logger.info(f"  P_active占比(neg): {v8_ed.get('active_fraction_neg', 'N/A'):.4f}")

    # 约束子空间
    logger.info(f"\n  === 约束子空间 ===")
    F_c_all = []
    vel_all = []
    for h in pos_hidden:
        if h.size(0) < 4:
            continue
        if h.dim() == 2:
            h = h.unsqueeze(0)
        vel = h[:, 1:, :] - h[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        v_for = vel[:, 1:, :]
        min_t = min(acc.size(1), v_for.size(1))
        F_c = acc[:, :min_t, :] + (0.01 - alpha_mean) * v_for[:, :min_t, :]
        F_c_all.append(F_c.reshape(-1, F_c.size(-1)))
        vel_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))

    if F_c_all:
        F_c_cat = torch.cat(F_c_all, dim=0).numpy()
        vel_cat = torch.cat(vel_all, dim=0).numpy()
        d = F_c_cat.shape[1]

        cov_fc = np.cov(F_c_cat.T)
        eigenvalues_fc = np.sort(np.abs(np.linalg.eigvalsh(cov_fc)))[::-1]
        total_var = eigenvalues_fc.sum()
        cumulative = np.cumsum(eigenvalues_fc) / total_var if total_var > 0 else np.zeros_like(eigenvalues_fc)
        effective_rank_fc = int(np.searchsorted(cumulative, 0.95) + 1) if total_var > 0 else 0

        orthogonality = float(np.mean(np.sum(F_c_cat * vel_cat, axis=-1)))

        vel_cov = np.cov(vel_cat.T)
        eigenvalues_vel = np.sort(np.abs(np.linalg.eigvalsh(vel_cov)))[::-1]
        total_var_vel = eigenvalues_vel.sum()
        cumulative_vel = np.cumsum(eigenvalues_vel) / total_var_vel if total_var_vel > 0 else np.zeros_like(eigenvalues_vel)
        effective_rank_vel = int(np.searchsorted(cumulative_vel, 0.95) + 1) if total_var_vel > 0 else 0

        F_c_norm = float(np.mean(np.linalg.norm(F_c_cat, axis=-1)))
        vel_norm = float(np.mean(np.linalg.norm(vel_cat, axis=-1)))

        logger.info(f"  F_c有效秩: {effective_rank_fc}/{d} (压缩比={effective_rank_fc/d:.3f})")
        logger.info(f"  vel有效秩: {effective_rank_vel}/{d}")
        logger.info(f"  F_c⊥ḣ: {orthogonality:.6f}")
        logger.info(f"  F_c/vel范数比: {F_c_norm/(vel_norm+1e-10):.4f}")

        # 约束方程提取
        U_vel, S_vel, Vt_vel = np.linalg.svd(vel_cat, full_matrices=True)
        r_vel = effective_rank_vel
        null_basis = Vt_vel[r_vel:]
        n_constraints = d - r_vel
        logger.info(f"  约束方程数: {n_constraints}")

        # pos vs neg约束违反
        neg_F_c_all = []
        neg_vel_all = []
        for h in neg_hidden:
            if h.size(0) < 4:
                continue
            if h.dim() == 2:
                h = h.unsqueeze(0)
            vel = h[:, 1:, :] - h[:, :-1, :]
            acc = vel[:, 1:, :] - vel[:, :-1, :]
            v_for = vel[:, 1:, :]
            min_t = min(acc.size(1), v_for.size(1))
            F_c = acc[:, :min_t, :] + (0.01 - alpha_mean) * v_for[:, :min_t, :]
            neg_F_c_all.append(F_c.reshape(-1, F_c.size(-1)))
            neg_vel_all.append(v_for[:, :min_t, :].reshape(-1, v_for.size(-1)))

        if neg_vel_all:
            neg_vel_cat = torch.cat(neg_vel_all, dim=0).numpy()
            pos_lower_count = 0
            for i in range(min(n_constraints, 10)):
                n_i = null_basis[i]
                v_pos = vel_cat @ n_i
                v_neg = neg_vel_cat @ n_i
                pos_rms = float(np.sqrt(np.mean(v_pos**2)))
                neg_rms = float(np.sqrt(np.mean(v_neg**2)))
                if pos_rms < neg_rms:
                    pos_lower_count += 1
            logger.info(f"  pos违反<neg违反: {pos_lower_count}/{min(n_constraints, 10)}")

    # === 阶段3: 逐层分析 ===
    logger.info(f"\n[阶段3] 逐层动力学分析...")
    layer_results = analyze_layer_dynamics(
        model, tokenizer, POSITIVE_TEXTS, NEGATIVE_TEXTS, max_seq_len, device, gamma=0.01
    )

    logger.info(f"\n{'='*80}")
    logger.info("逐层结果汇总")
    logger.info(f"{'='*80}")
    logger.info(f"{'层':>4} {'α*':>8} {'pos_Pc':>10} {'neg_Pc':>10} {'判决':>12} {'F_c秩':>6} {'vel秩':>6} {'压缩比':>8} {'F_c⊥ḣ':>10}")
    logger.info("-" * 90)
    for l in sorted(layer_results.keys()):
        r = layer_results[l]
        logger.info(
            f"{l:>4} {r['alpha_star_mean']:>8.4f} "
            f"{r['pos_Pc_mean']:>10.6f} {r['neg_Pc_mean']:>10.6f} "
            f"{r['verdict']:>12} "
            f"{r['F_c_effective_rank']:>6} {r['vel_effective_rank']:>6} "
            f"{r['compression_ratio']:>8.3f} "
            f"{r['F_c_orthogonality_to_vel']:>10.6f}"
        )

    # α*随层数变化趋势
    alphas = [(l, r["alpha_star_mean"]) for l, r in layer_results.items() if r.get("alpha_star_mean")]
    if len(alphas) >= 3:
        layers_arr = np.array([a[0] for a in alphas])
        alphas_arr = np.array([a[1] for a in alphas])
        corr, p_val = stats.pearsonr(layers_arr, alphas_arr)
        logger.info(f"\n  α*与层数的Pearson相关: r={corr:.4f}, p={p_val:.6f}")
        if corr < -0.5 and p_val < 0.05:
            logger.info(f"  → α*随层数显著递减（与NPNW跨模型结论一致）")
        elif corr > 0.5 and p_val < 0.05:
            logger.info(f"  → α*随层数显著递增（新发现！）")
        else:
            logger.info(f"  → α*与层数无显著线性关系")

    # === 保存报告 ===
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v10.0 Real LLM Verification (MiniCPM5-1B)",
        "model": {
            "name": "MiniCPM5-1B",
            "path": model_path,
            "n_params": n_params,
            "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "architecture": "LlamaForCausalLM",
        },
        "final_layer_results": {
            "alpha_star_mean": alpha_mean,
            "alpha_star_std": alpha_std,
            "v8_results": v8_results if alpha_mean else {},
            "constraint_subspace": {
                "F_c_effective_rank": effective_rank_fc if F_c_all else None,
                "vel_effective_rank": effective_rank_vel if F_c_all else None,
                "total_dim": model.config.hidden_size,
                "compression_ratio": float(effective_rank_fc / model.config.hidden_size) if F_c_all and model.config.hidden_size > 0 else None,
                "F_c_orthogonality_to_vel": orthogonality if F_c_all else None,
                "n_constraints": n_constraints if F_c_all else None,
            },
        },
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }

    report_path = output_dir / "v10_real_llm_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n报告已保存: {report_path}")

    # Markdown报告
    md_lines = [
        "# GUIT-TRT v10.0 真实LLM验证报告 (MiniCPM5-1B)",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n---\n",
        f"## 模型信息\n",
        f"- 架构: LlamaForCausalLM",
        f"- 参数量: {n_params/1e9:.2f}B",
        f"- 层数: {model.config.num_hidden_layers}",
        f"- 隐状态维度: {model.config.hidden_size}",
        f"\n---\n",
        f"## 最终层v8.0验证\n",
        f"- α* = {alpha_mean:.4f} ± {alpha_std:.4f}" if alpha_mean else "- α* 估计失败",
    ]

    if alpha_mean and v8_results:
        v8_cp = v8_results.get("corrected_constraint_power", {})
        v8_ed = v8_results.get("energy_decomposition", {})
        v8_ov = v8_results.get("overall_verdict", {})
        criteria = v8_cp.get("criteria", {})

        md_lines.extend([
            f"- pos P_c = {v8_cp.get('pos_Pc_mean', 'N/A'):.6f}",
            f"- neg P_c = {v8_cp.get('neg_Pc_mean', 'N/A'):.6f}",
            f"- pos_vs_zero_p = {v8_cp.get('pos_vs_zero_p', 'N/A'):.4f}",
            f"- pos_vs_neg_p = {v8_cp.get('pos_vs_neg_p', 'N/A'):.6f}",
            f"- 判决: **{v8_ov.get('verdict', 'N/A')}**",
            f"- 准则通过: {sum(1 for v in criteria.values() if v)}/3",
            f"- P_active占比(pos): {v8_ed.get('active_fraction_pos', 'N/A'):.4f}",
            f"- P_active占比(neg): {v8_ed.get('active_fraction_neg', 'N/A'):.4f}",
        ])

    if F_c_all:
        md_lines.extend([
            f"\n## 约束子空间\n",
            f"- F_c有效秩: {effective_rank_fc}/{model.config.hidden_size}",
            f"- vel有效秩: {effective_rank_vel}/{model.config.hidden_size}",
            f"- 压缩比: {effective_rank_fc/model.config.hidden_size:.3f}",
            f"- F_c⊥ḣ: {orthogonality:.6f}",
            f"- F_c/vel范数比: {F_c_norm/(vel_norm+1e-10):.4f}",
            f"- 约束方程数: {n_constraints}",
        ])

    md_lines.extend([
        f"\n---\n",
        f"## 逐层动力学\n",
        f"| 层 | α* | pos_Pc | neg_Pc | 判决 | F_c秩 | vel秩 | 压缩比 | F_c⊥ḣ |",
        f"|----|-----|--------|--------|------|-------|-------|--------|-------|",
    ])
    for l in sorted(layer_results.keys()):
        r = layer_results[l]
        md_lines.append(
            f"| {l} | {r['alpha_star_mean']:.4f} | {r['pos_Pc_mean']:.6f} | "
            f"{r['neg_Pc_mean']:.6f} | {r['verdict']} | {r['F_c_effective_rank']} | "
            f"{r['vel_effective_rank']} | {r['compression_ratio']:.3f} | {r['F_c_orthogonality_to_vel']:.6f} |"
        )

    if len(alphas) >= 3:
        corr, p_val = stats.pearsonr(layers_arr, alphas_arr)
        md_lines.extend([
            f"\n---\n",
            f"## α*层间趋势\n",
            f"- Pearson r = {corr:.4f}, p = {p_val:.6f}",
            f"- 趋势: {'递减' if corr < 0 else '递增' if corr > 0 else '无趋势'}",
        ])

    md_path = output_dir / "v10_real_llm_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown报告已保存: {md_path}")

    del model
    torch.cuda.empty_cache()
    logger.info(f"\n完成时间: {datetime.now().isoformat()}")

    return report


if __name__ == "__main__":
    main()