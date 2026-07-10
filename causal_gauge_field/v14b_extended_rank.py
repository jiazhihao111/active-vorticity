import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.signal import savgol_filter
from scipy.stats import entropy, pearsonr
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_effective_rank_svd(vel_matrix):
    U, S, Vt = np.linalg.svd(vel_matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-10:
        return 0.0
    p = S2 / total
    H = entropy(p, base=np.e)
    return np.exp(H)


def extract_velocity(raw_h, method='raw_diff', window=11, polyorder=3):
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


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Loading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    long_texts = [
        "在一个遥远的国度，有一位年轻的王子名叫阿尔弗雷德。他从小就对星空充满了好奇，每天晚上都会爬上城堡最高的塔楼，用望远镜观察星星。他的父亲国王并不理解他的爱好，希望他能更多地关注治国之道。但阿尔弗雷德坚信，星辰之中隐藏着宇宙的秘密。十八岁那年，他离开王宫，踏上了寻找传说中的星辰之石的旅程。他穿越了茂密的森林，翻越了终年积雪的高山，渡过了波涛汹涌的大河。在旅途中，他遇到了一位智慧的老者，老者告诉他星辰之石藏在世界的尽头，只有心怀纯粹好奇心的人才能找到它。阿尔弗雷德继续前行，经过无数艰难险阻，终于来到了一座被遗忘的古老神殿。神殿深处，一块散发着淡蓝色光芒的石头静静地悬浮在空中。当阿尔弗雷德伸手触碰它的那一刻，整片星空在他眼前展开，他终于理解了宇宙的奥秘——每一颗星星都是一个故事的起点，每一段旅程都是一次新的发现。他带着星辰之石回到了王国，用它照亮了整个国度，人们从此不再惧怕黑暗。",
        "李明是一位资深的软件工程师，在一家大型科技公司工作了十五年。他精通多种编程语言，从C++到Python再到Rust，每一种他都能运用自如。每天早上六点，他准时起床，先在跑步机上跑三十分钟，然后冲一杯黑咖啡，开始阅读最新的技术论文。他的团队正在开发一个基于大语言模型的智能助手系统，这个系统需要处理海量的自然语言数据。李明负责架构设计，他选择了微服务架构，将系统拆分为数据采集、预处理、模型推理和后处理四个模块。在开发过程中，他们遇到了一个棘手的性能瓶颈——模型推理的延迟过高。李明带领团队进行了深入分析，发现问题出在注意力机制的计算复杂度上。他提出了一个创新的稀疏注意力方案，将推理延迟降低了百分之六十。项目上线后，用户反馈非常积极，日活跃用户在一个月内就突破了百万。公司因此给了李明一个特别贡献奖，他谦虚地说这是整个团队共同努力的结果。",
        "张教授是一位著名的分子生物学家，她研究蛋白质折叠问题已经二十年了。在她的实验室里，有十名研究生和五名博士后，每个人都在不同的子课题上工作。最近，她们团队取得了一项突破性发现——一种新型分子伴侣蛋白能够显著加速蛋白质的正确折叠。这个发现源于一次偶然的实验失误：一名研究生在配置缓冲液时多加了一种盐，结果蛋白质的折叠效率提高了三倍。张教授没有忽视这个意外，而是立即组织团队进行系统研究。她们设计了严格的对照实验，排除了各种干扰因素，最终确认了这种盐离子与分子伴侣蛋白的协同作用机制。论文发表在Nature上后，引起了全球关注。多家制药公司联系她们，希望将这一发现应用于治疗阿尔茨海默病等蛋白质错误折叠相关的疾病。张教授对此非常谨慎，她认为从基础发现到临床应用还有很长的路要走，需要更多的安全性和有效性验证。她决定先成立一个跨学科合作小组，联合计算生物学家和临床医学专家，共同推进这项研究的转化工作。",
    ]

    max_seq_len = 512

    print("[2] Extracting hidden states (max_seq_len=512)...")
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for text in long_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1].squeeze(0).cpu().float()
            print(f"  Text length: {h.size(0)} tokens")
            all_hidden.append(h)

    # ============================================================
    # Rank sensitivity test with extended T range
    # ============================================================
    print("\n" + "=" * 80)
    print("Extended Rank Artifact Verification (long sequences)")
    print("=" * 80)

    T_list = [15, 30, 50, 80, 100, 150, 200]
    methods = [
        ('raw_diff', None, None),
        ('central_diff', None, None),
        ('sg', 5, 2),
        ('sg', 11, 3),
        ('sg', 21, 3),
    ]

    all_results = []

    for traj_idx, h in enumerate(all_hidden):
        raw_h = h.numpy()
        T_max, D = raw_h.shape
        actual_T_list = [T for T in T_list if T <= T_max - 5]

        for T in actual_T_list:
            h_sub = raw_h[:T, :]

            for method_name, w, p in methods:
                if method_name == 'sg' and w >= T:
                    continue

                vel = extract_velocity(h_sub, method=method_name, window=w, polyorder=p)
                vel_centered = vel - np.mean(vel, axis=0)

                if T < D:
                    gram = (vel_centered @ vel_centered.T) / T
                    eigvals = np.linalg.eigvalsh(gram)
                    eigvals = eigvals[eigvals > 1e-10]
                    if len(eigvals) == 0:
                        rank = 0.0
                    else:
                        p_dist = eigvals / eigvals.sum()
                        H = entropy(p_dist, base=np.e)
                        rank = np.exp(H)
                else:
                    cov = np.cov(vel_centered.T)
                    eigvals = np.linalg.eigvalsh(cov)
                    eigvals = eigvals[eigvals > 1e-10]
                    if len(eigvals) == 0:
                        rank = 0.0
                    else:
                        p_dist = eigvals / eigvals.sum()
                        H = entropy(p_dist, base=np.e)
                        rank = np.exp(H)

                all_results.append({
                    'traj_idx': traj_idx,
                    'T': T,
                    'method': f"{method_name}(w={w})" if method_name == 'sg' else method_name,
                    'effective_rank': round(float(rank), 2),
                    'T_max': T_max,
                })

    # Aggregate
    agg = defaultdict(list)
    for r in all_results:
        key = (r['T'], r['method'])
        agg[key].append(r['effective_rank'])

    print(f"\n{'T':<6} | {'Method':<18} | {'Mean_Rank':<10} | {'Std':<8} | {'N':<4}")
    print("-" * 55)
    for (T, method), ranks in sorted(agg.items()):
        print(f"{T:<6} | {method:<18} | {np.mean(ranks):<10.2f} | {np.std(ranks):<8.2f} | {len(ranks):<4}")

    # ============================================================
    # Key diagnosis: raw_diff rank vs T
    # ============================================================
    print("\n" + "=" * 80)
    print("DIAGNOSIS: Does rank saturate or grow with T?")
    print("=" * 80)

    raw_by_T = defaultdict(list)
    for r in all_results:
        if r['method'] == 'raw_diff':
            raw_by_T[r['T']].append(r['effective_rank'])

    print(f"\n  raw_diff effective rank vs T:")
    T_vals = sorted(raw_by_T.keys())
    rank_means = []
    for T in T_vals:
        mean_r = np.mean(raw_by_T[T])
        rank_means.append(mean_r)
        print(f"    T={T}: rank={mean_r:.2f} ± {np.std(raw_by_T[T]):.2f}")

    # Pearson correlation
    if len(T_vals) >= 3:
        corr, pval = pearsonr(T_vals, rank_means)
        print(f"\n  Pearson(T, rank): r={corr:.3f}, p={pval:.4f}")

        # Check saturation: compare slope in [15,80] vs [80,200]
        if len(T_vals) >= 5:
            mid = len(T_vals) // 2
            T_low = T_vals[:mid]
            r_low = rank_means[:mid]
            T_high = T_vals[mid:]
            r_high = rank_means[mid:]

            if len(T_low) >= 2 and len(T_high) >= 2:
                slope_low = (r_low[-1] - r_low[0]) / (T_low[-1] - T_low[0])
                slope_high = (r_high[-1] - r_high[0]) / (T_high[-1] - T_high[0]) if (T_high[-1] - T_high[0]) > 0 else 0
                print(f"  Slope [T={T_low[0]}-{T_low[-1]}]: {slope_low:.4f} rank/T")
                print(f"  Slope [T={T_high[0]}-{T_high[-1]}]: {slope_high:.4f} rank/T")

                if slope_high < slope_low * 0.3:
                    print("  ✅ DIAGNOSIS: Rank SATURATES at large T → likely PHYSICAL/ARCHITECTURAL ground truth")
                elif slope_high > slope_low * 0.7:
                    print("  ⚠️ DIAGNOSIS: Rank CONTINUES TO GROW → likely SAMPLE TRUNCATION artifact")
                else:
                    print("  ⚠️ DIAGNOSIS: Partial saturation → mixed effect (some physical + some truncation)")

    # Also check: ratio rank/T
    print(f"\n  rank/T ratio:")
    for T in T_vals:
        mean_r = np.mean(raw_by_T[T])
        print(f"    T={T}: rank/T={mean_r/T:.4f}")

    # ============================================================
    # SG smoothing effect quantification
    # ============================================================
    print("\n" + "=" * 80)
    print("SG Smoothing Effect")
    print("=" * 80)

    for T in [50, 100, 200]:
        if T not in [t for t in T_vals]:
            continue
        raw_r = np.mean([r['effective_rank'] for r in all_results if r['method'] == 'raw_diff' and r['T'] == T])
        sg5_r = np.mean([r['effective_rank'] for r in all_results if r['method'] == 'sg(w=5)' and r['T'] == T])
        sg11_r = np.mean([r['effective_rank'] for r in all_results if r['method'] == 'sg(w=11)' and r['T'] == T])
        sg21_r_vals = [r['effective_rank'] for r in all_results if r['method'] == 'sg(w=21)' and r['T'] == T]
        sg21_r = np.mean(sg21_r_vals) if sg21_r_vals else 0

        print(f"  T={T}: raw={raw_r:.2f}, sg5={sg5_r:.2f}({sg5_r/raw_r*100:.0f}%), sg11={sg11_r:.2f}({sg11_r/raw_r*100:.0f}%), sg21={sg21_r:.2f}({sg21_r/raw_r*100:.0f}%)" if sg21_r > 0 else f"  T={T}: raw={raw_r:.2f}, sg5={sg5_r:.2f}({sg5_r/raw_r*100:.0f}%), sg11={sg11_r:.2f}({sg11_r/raw_r*100:.0f}%)")

    # Save
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "v14b Extended Rank Artifact Verification",
        "trajectory_lengths": [h.size(0) for h in all_hidden],
        "rank_results": all_results,
        "rank_aggregate": {f"T={k[0]},{k[1]}": {"mean": float(np.mean(v)), "std": float(np.std(v))}
                           for k, v in agg.items()},
    }
    report_path = output_dir / "v14b_extended_rank_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()