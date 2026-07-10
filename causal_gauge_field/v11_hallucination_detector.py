import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HallucinationAlert:
    token_position: int
    pc_praw_ratio: float
    pc_value: float
    praw_value: float
    severity: str
    context_window: List[int] = field(default_factory=list)


class CausalConstraintMonitor:
    """基于P_c/P_raw的实时因果约束偏离监控器

    核心原理:
    - 合法叙事: P_c/P_raw ≈ 0.001 (约束力几乎不做功)
    - 语序打乱: P_c/P_raw ≈ 0.02 (约束力做功增大)
    - 随机token: P_c/P_raw ≈ 0.06 (约束力做功显著)

    检测方法:
    1. 滑动窗口计算P_c/P_raw
    2. 当P_c/P_raw超过阈值时触发警报
    3. 警报级别: LOW / MEDIUM / HIGH / CRITICAL
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        gamma: float = 0.01,
        mass: float = 1.0,
        window_size: int = 5,
        alpha: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.mass = mass
        self.window_size = window_size
        self.alpha = alpha

        if thresholds is None:
            self.thresholds = {
                "low": 0.01,
                "medium": 0.025,
                "high": 0.05,
                "critical": 0.10,
            }
        else:
            self.thresholds = thresholds

        self.hidden_buffer = []
        self.alerts = []
        self.pc_praw_history = []

    def calibrate_alpha(self, calibration_texts: List[str], max_seq_len: int = 64):
        """用校准文本估计α*"""
        alpha_estimates = []
        self.model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
                input_ids = inputs["input_ids"].to(self.device)
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                h = outputs.hidden_states[-1].squeeze(0).cpu().float()

                if h.size(0) < 4:
                    continue

                vel = h[1:] - h[:-1]
                acc = vel[1:] - vel[:-1]
                v_for = vel[1:]
                min_t = min(acc.size(0), v_for.size(0))
                F_res = self.mass * acc[:min_t] + self.gamma * v_for[:min_t]
                P_raw = (F_res * v_for[:min_t]).sum(dim=-1)
                P_active = (v_for[:min_t] * v_for[:min_t]).sum(dim=-1)
                if P_active.abs().mean() > 1e-10:
                    alpha_estimates.append(P_raw.mean().item() / P_active.mean().item())

        if alpha_estimates:
            self.alpha = float(np.mean(alpha_estimates))
            return self.alpha
        return None

    def _compute_step_pc_praw(self, h_prev: torch.Tensor, h_curr: torch.Tensor, h_next: torch.Tensor) -> Tuple[float, float]:
        """计算单步P_c和P_raw"""
        vel_prev = h_curr - h_prev
        vel_curr = h_next - h_curr
        acc = vel_curr - vel_prev

        F_total = self.mass * acc + self.gamma * vel_curr
        F_c = self.mass * acc + (self.gamma - self.alpha) * vel_curr

        P_raw = float((F_total * vel_curr).sum())
        P_c = float((F_c * vel_curr).sum())

        return P_c, P_raw

    def _classify_severity(self, ratio: float) -> str:
        if ratio >= self.thresholds["critical"]:
            return "CRITICAL"
        elif ratio >= self.thresholds["high"]:
            return "HIGH"
        elif ratio >= self.thresholds["medium"]:
            return "MEDIUM"
        elif ratio >= self.thresholds["low"]:
            return "LOW"
        return "NORMAL"

    def monitor_text(self, text: str, max_seq_len: int = 64) -> Dict:
        """监控整段文本的因果约束偏离

        Returns:
            {
                "alerts": [HallucinationAlert],
                "pc_praw_per_step": [(pos, ratio, P_c, P_raw)],
                "overall_ratio": float,
                "max_ratio": float,
                "n_alerts": int,
                "severity_counts": Dict,
            }
        """
        if self.alpha is None:
            raise ValueError("α*未校准，请先调用calibrate_alpha()")

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].squeeze(0).cpu().float()

        seq_len = hidden.size(0)
        if seq_len < 3:
            return {"error": "序列太短"}

        # 逐步计算P_c/P_raw
        pc_praw_per_step = []
        alerts = []

        for t in range(1, seq_len - 1):
            P_c, P_raw = self._compute_step_pc_praw(hidden[t-1], hidden[t], hidden[t+1])
            ratio = abs(P_c) / (abs(P_raw) + 1e-10)
            pc_praw_per_step.append((t, ratio, P_c, P_raw))

            severity = self._classify_severity(ratio)
            if severity != "NORMAL":
                alert = HallucinationAlert(
                    token_position=t,
                    pc_praw_ratio=ratio,
                    pc_value=P_c,
                    praw_value=P_raw,
                    severity=severity,
                )
                alerts.append(alert)

        # 滑动窗口平滑
        windowed_ratios = []
        windowed_alerts = []
        for i in range(len(pc_praw_per_step)):
            start = max(0, i - self.window_size + 1)
            window = pc_praw_per_step[start:i+1]
            avg_ratio = np.mean([r[1] for r in window])
            windowed_ratios.append((pc_praw_per_step[i][0], avg_ratio))

            severity = self._classify_severity(avg_ratio)
            if severity != "NORMAL":
                windowed_alerts.append(HallucinationAlert(
                    token_position=pc_praw_per_step[i][0],
                    pc_praw_ratio=avg_ratio,
                    pc_value=pc_praw_per_step[i][2],
                    praw_value=pc_praw_per_step[i][3],
                    severity=severity,
                    context_window=[pc_praw_per_step[j][0] for j in range(start, i+1)],
                ))

        all_ratios = [r[1] for r in pc_praw_per_step]
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for a in windowed_alerts:
            severity_counts[a.severity] += 1

        return {
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "seq_len": seq_len,
            "alpha": self.alpha,
            "pc_praw_per_step": [(p, round(r, 6), round(pc, 4), round(pr, 4)) for p, r, pc, pr in pc_praw_per_step],
            "windowed_ratios": [(p, round(r, 6)) for p, r in windowed_ratios],
            "alerts": windowed_alerts,
            "overall_ratio": float(np.mean(all_ratios)),
            "max_ratio": float(max(all_ratios)) if all_ratios else 0,
            "n_alerts": len(windowed_alerts),
            "severity_counts": severity_counts,
        }

    def batch_monitor(self, texts: List[str], labels: Optional[List[str]] = None, max_seq_len: int = 64) -> Dict:
        """批量监控多段文本"""
        results = []
        for i, text in enumerate(texts):
            label = labels[i] if labels else "unknown"
            r = self.monitor_text(text, max_seq_len)
            r["label"] = label
            results.append(r)

        label_stats = {}
        for r in results:
            label = r["label"]
            if label not in label_stats:
                label_stats[label] = {"ratios": [], "n_alerts": [], "max_ratios": []}
            label_stats[label]["ratios"].append(r["overall_ratio"])
            label_stats[label]["n_alerts"].append(r["n_alerts"])
            label_stats[label]["max_ratios"].append(r["max_ratio"])

        summary = {}
        for label, stats_dict in label_stats.items():
            summary[label] = {
                "mean_ratio": float(np.mean(stats_dict["ratios"])),
                "std_ratio": float(np.std(stats_dict["ratios"])),
                "mean_n_alerts": float(np.mean(stats_dict["n_alerts"])),
                "mean_max_ratio": float(np.mean(stats_dict["max_ratios"])),
            }

        if len(label_stats) >= 2:
            labels_list = list(label_stats.keys())
            ratios_0 = label_stats[labels_list[0]]["ratios"]
            ratios_1 = label_stats[labels_list[1]]["ratios"]
            if len(ratios_0) >= 2 and len(ratios_1) >= 2:
                t, p = stats.ttest_ind(ratios_0, ratios_1, equal_var=False)
                summary["ttest"] = {"t": float(t), "p": float(p)}

        return {"individual_results": results, "label_summary": summary}


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1] 加载MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # 校准文本（因果叙事）
    calibration_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。",
    ]

    monitor = CausalConstraintMonitor(model, tokenizer, device, window_size=3)

    print(f"[2] 校准α*...")
    alpha = monitor.calibrate_alpha(calibration_texts)
    print(f"  α* = {alpha:.4f}")

    # 测试文本
    pos_texts = [
        "学生们在教室里安静地考试。小李认真读题，仔细计算，把答案写在答题卡上。考试结束后，他检查了一遍才交卷。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他。",
        "医生询问了病人的症状后，安排了血液检查。检查结果显示感染，医生开了抗生素。病人按时服药，一周后康复了。",
        "建筑师先画了设计图，然后计算了承重结构。施工队按图纸打地基、砌墙、封顶。半年后，一栋大楼拔地而起。",
        "渔夫清晨划船出海。他撒下渔网，等了几个小时。收网时发现网里全是鱼，他高兴地把鱼运回港口卖了个好价钱。",
    ]

    scr_texts = [
        "考试在教室里安静了学生。小李写在答题卡上，认真读题，仔细计算。交卷后他检查了一遍，答案才考试结束。",
        "旅行者背着村庄走在山路上。他翻过一条河，渡过一座山，终于到达了行囊。村民天黑前招待了他，山脚下热情地走了。",
        "医生开了血液检查，病人安排了抗生素。检查结果显示症状，医生询问了一周后康复。病人按时感染，服药了结果。",
        "建筑师画了施工队，然后计算了设计图。大楼按图纸打地基，承重结构砌墙封顶。半年后，一栋设计图拔地而起。",
        "渔夫清晨卖了个好价钱。他撒下渔网，等了鱼。收网时发现港口全是网，他高兴地把鱼运回海里划船了几个小时。",
    ]

    # 混合文本：前半因果，后半打乱（模拟幻觉突变）
    mixed_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。宝箱突然钥匙了他，然后密室走进了他。门后是一间打开了，里面放着突然。",
        "农夫先浇了水，然后施肥。蔬菜摘下了农夫，装进篮子。施肥把他放进烤箱，蛋糕走到田里。傍晚时分，他烤箱了满载而归。",
        "科学家第一次失败了，她调整参数重试。实验成功了调整了第一次，优化了第二次。实验反复失败，终于参数重试了。结果更好了实验。",
    ]

    print(f"\n[3] 单文本监控演示...")
    for i, text in enumerate(mixed_texts):
        result = monitor.monitor_text(text)
        print(f"\n  混合文本 #{i+1}:")
        print(f"    overall P_c/P_raw = {result['overall_ratio']:.6f}")
        print(f"    max P_c/P_raw = {result['max_ratio']:.6f}")
        print(f"    alerts: {result['n_alerts']} ({result['severity_counts']})")
        # 显示每个step的ratio
        step_ratios = [f"t{p}:{r:.4f}" for p, r in result['windowed_ratios']]
        print(f"    per-step: {', '.join(step_ratios[:10])}...")

    print(f"\n[4] 批量监控...")
    all_texts = pos_texts + scr_texts
    all_labels = ["pos"] * len(pos_texts) + ["scr"] * len(scr_texts)
    batch_result = monitor.batch_monitor(all_texts, all_labels)

    print(f"\n  标签统计:")
    for label, stats in batch_result["label_summary"].items():
        if isinstance(stats, dict) and "mean_ratio" in stats:
            print(f"    {label}: mean_ratio={stats['mean_ratio']:.6f}, "
                  f"std={stats['std_ratio']:.6f}, "
                  f"mean_alerts={stats['mean_n_alerts']:.1f}")
    if "ttest" in batch_result["label_summary"]:
        tt = batch_result["label_summary"]["ttest"]
        print(f"    t-test: t={tt['t']:.4f}, p={tt['p']:.4f}")

    # 保存结果
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 序列化alerts
    for r in batch_result["individual_results"]:
        r["alerts"] = [{"pos": a.token_position, "ratio": a.pc_praw_ratio, "severity": a.severity} for a in r["alerts"]]

    report_path = output_dir / "v11_hallucination_detector_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("完成")


if __name__ == "__main__":
    main()