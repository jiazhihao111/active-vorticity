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
class ConstraintAlert:
    token_position: int
    signal_type: str
    signal_value: float
    baseline_value: float
    deviation_ratio: float
    severity: str


class CausalConstraintDetector:
    """多信号融合的因果约束偏离检测器

    三大检测信号:
    1. vel_norm突变: 速度范数骤降 → 模型对输入的"理解力"下降
    2. fc/vel ratio突变: 约束力/速度比增大 → 约束力相对增强
    3. P_c/P_raw整体: 整段轨迹的约束力做功占比 → 因果约束强度

    检测策略:
    - 信号1+2用于逐token定位突变点
    - 信号3用于整段文本的因果约束强度评估
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.mass = mass
        self.window_size = window_size
        self.alpha = alpha

    def calibrate(self, calibration_texts: List[str], max_seq_len: int = 64) -> Dict:
        """用校准文本建立基线"""
        alpha_estimates = []
        vel_norms = []
        fc_vel_ratios = []

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

                F_c = self.mass * acc[:min_t] + (self.gamma - alpha_estimates[-1] if alpha_estimates else 1.0) * v_for[:min_t]
                vel_norms.extend(v_for[:min_t].norm(dim=-1).tolist())
                fc_vel_ratios.extend((F_c.norm(dim=-1) / (v_for[:min_t].norm(dim=-1) + 1e-10)).tolist())

        self.alpha = float(np.mean(alpha_estimates)) if alpha_estimates else None
        self.baseline_vel_norm = float(np.mean(vel_norms)) if vel_norms else None
        self.baseline_fc_vel = float(np.mean(fc_vel_ratios)) if fc_vel_ratios else None

        return {
            "alpha": self.alpha,
            "baseline_vel_norm": self.baseline_vel_norm,
            "baseline_fc_vel": self.baseline_fc_vel,
            "n_calibration": len(alpha_estimates),
        }

    def detect(self, text: str, max_seq_len: int = 64) -> Dict:
        """检测文本中的因果约束偏离"""
        if self.alpha is None:
            raise ValueError("未校准，请先调用calibrate()")

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1].squeeze(0).cpu().float()

        seq_len = h.size(0)
        if seq_len < 3:
            return {"error": "序列太短"}

        vel = h[1:] - h[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))

        F_c = self.mass * acc[:min_t] + (self.gamma - self.alpha) * v_for[:min_t]
        F_total = self.mass * acc[:min_t] + self.gamma * v_for[:min_t]

        vel_norm = v_for[:min_t].norm(dim=-1)
        fc_norm = F_c.norm(dim=-1)
        fc_vel_ratio = fc_norm / (vel_norm + 1e-10)
        cos_sim = torch.nn.functional.cosine_similarity(F_c, v_for[:min_t], dim=-1)

        P_c = (F_c * v_for[:min_t]).sum(dim=-1)
        P_raw = (F_total * v_for[:min_t]).sum(dim=-1)

        # 滑动窗口平滑
        def smooth(tensor, w):
            result = torch.zeros_like(tensor)
            for i in range(len(tensor)):
                start = max(0, i - w + 1)
                result[i] = tensor[start:i+1].mean()
            return result

        smooth_vel = smooth(vel_norm, self.window_size)
        smooth_fc_vel = smooth(fc_vel_ratio, self.window_size)
        smooth_cos = smooth(cos_sim.abs(), self.window_size)

        # 检测突变点
        alerts = []
        for i in range(len(smooth_vel)):
            # 信号1: vel_norm低于基线
            if self.baseline_vel_norm:
                vel_deviation = (self.baseline_vel_norm - smooth_vel[i].item()) / self.baseline_vel_norm
                if vel_deviation > 0.2:
                    alerts.append(ConstraintAlert(
                        token_position=i,
                        signal_type="vel_norm_drop",
                        signal_value=smooth_vel[i].item(),
                        baseline_value=self.baseline_vel_norm,
                        deviation_ratio=vel_deviation,
                        severity="HIGH" if vel_deviation > 0.4 else "MEDIUM" if vel_deviation > 0.3 else "LOW",
                    ))

            # 信号2: fc/vel高于基线
            if self.baseline_fc_vel:
                fc_deviation = (smooth_fc_vel[i].item() - self.baseline_fc_vel) / self.baseline_fc_vel
                if fc_deviation > 0.15:
                    alerts.append(ConstraintAlert(
                        token_position=i,
                        signal_type="fc_vel_rise",
                        signal_value=smooth_fc_vel[i].item(),
                        baseline_value=self.baseline_fc_vel,
                        deviation_ratio=fc_deviation,
                        severity="HIGH" if fc_deviation > 0.3 else "MEDIUM" if fc_deviation > 0.2 else "LOW",
                    ))

        # 整体P_c/P_raw
        overall_pc_praw = abs(P_c.sum().item()) / (abs(P_raw.sum().item()) + 1e-10)

        # 段落级P_c/P_raw (前半 vs 后半)
        mid = len(P_c) // 2
        first_half_pc_praw = abs(P_c[:mid].sum().item()) / (abs(P_raw[:mid].sum().item()) + 1e-10)
        second_half_pc_praw = abs(P_c[mid:].sum().item()) / (abs(P_raw[mid:].sum().item()) + 1e-10)

        return {
            "text_preview": text[:60] + "..." if len(text) > 60 else text,
            "seq_len": seq_len,
            "alpha": self.alpha,
            "overall_pc_praw": overall_pc_praw,
            "first_half_pc_praw": first_half_pc_praw,
            "second_half_pc_praw": second_half_pc_praw,
            "mean_vel_norm": float(vel_norm.mean()),
            "mean_fc_vel": float(fc_vel_ratio.mean()),
            "mean_cos_sim": float(cos_sim.abs().mean()),
            "per_step": {
                "vel_norm": smooth_vel.tolist(),
                "fc_vel_ratio": smooth_fc_vel.tolist(),
                "cos_sim_abs": smooth_cos.tolist(),
            },
            "alerts": [{"pos": a.token_position, "type": a.signal_type, "value": round(a.signal_value, 4),
                        "baseline": round(a.baseline_value, 4), "dev": round(a.deviation_ratio, 4),
                        "severity": a.severity} for a in alerts],
            "n_alerts": len(alerts),
            "alert_positions": sorted(set(a.token_position for a in alerts)),
        }


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\MiniCPM5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Loading MiniCPM5-1B...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    calibration_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。门后是一间密室，里面放着宝箱。",
        "天刚亮，农夫就起床去田里。他先浇了水，然后施肥，最后把成熟的蔬菜摘下来装进篮子。傍晚时分，他满载而归。",
        "科学家在实验室里反复实验。第一次失败了，她调整参数重试。第二次结果更好，她继续优化。终于，实验成功了。",
    ]

    detector = CausalConstraintDetector(model, tokenizer, device, window_size=5)
    print("[2] Calibrating...")
    cal = detector.calibrate(calibration_texts)
    print("  alpha* = {:.4f}".format(cal["alpha"]))
    print("  baseline vel_norm = {:.2f}".format(cal["baseline_vel_norm"]))
    print("  baseline fc/vel = {:.4f}".format(cal["baseline_fc_vel"]))

    # Test texts
    pos_texts = [
        "学生们在教室里安静地考试。小李认真读题，仔细计算，把答案写在答题卡上。考试结束后，他检查了一遍才交卷。",
        "旅行者背着行囊走在山路上。他翻过一座山，渡过一条河，终于在天黑前到达了山脚下的村庄。村民热情地招待了他。",
        "医生询问了病人的症状后，安排了血液检查。检查结果显示感染，医生开了抗生素。病人按时服药，一周后康复了。",
    ]

    mixed_texts = [
        "小明走进房间，看到桌上有一把钥匙。他拿起钥匙，走到门前，用钥匙打开了门。宝箱突然钥匙了他，然后密室走进了他。门后是一间打开了，里面放着突然。",
        "农夫先浇了水，然后施肥。蔬菜摘下了农夫，装进篮子。施肥把他放进烤箱，蛋糕走到田里。傍晚时分，他烤箱了满载而归。",
        "科学家第一次失败了，她调整参数重试。实验成功了调整了第一次，优化了第二次。实验反复失败，终于参数重试了。结果更好了实验。",
    ]

    random_texts = []
    np.random.seed(42)
    for _ in range(3):
        ids = np.random.randint(100, tokenizer.vocab_size - 100, 30).tolist()
        random_texts.append(tokenizer.decode(ids))

    print("\n[3] Detection results:")
    all_results = {}

    for label, texts in [("pos", pos_texts), ("mixed", mixed_texts), ("random", random_texts)]:
        print(f"\n  === {label} ===")
        label_results = []
        for i, text in enumerate(texts):
            r = detector.detect(text)
            label_results.append(r)
            print(f"  #{i+1}: overall_pc_praw={r['overall_pc_praw']:.6f}, "
                  f"1st_half={r['first_half_pc_praw']:.6f}, 2nd_half={r['second_half_pc_praw']:.6f}, "
                  f"vel_norm={r['mean_vel_norm']:.2f}, fc/vel={r['mean_fc_vel']:.4f}, "
                  f"alerts={r['n_alerts']}")
        all_results[label] = label_results

    # Aggregate
    print("\n[4] Aggregate comparison:")
    for label in ["pos", "mixed", "random"]:
        results = all_results[label]
        vel_norms = [r["mean_vel_norm"] for r in results]
        fc_vels = [r["mean_fc_vel"] for r in results]
        pc_praw = [r["overall_pc_praw"] for r in results]
        print(f"  {label}: vel_norm={np.mean(vel_norms):.2f}, fc/vel={np.mean(fc_vels):.4f}, "
              f"P_c/P_raw={np.mean(pc_praw):.6f}")

    # Save
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v11_detector_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()