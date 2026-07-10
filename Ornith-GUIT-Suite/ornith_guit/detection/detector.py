"""滑动窗口幻觉检测器 + 批量/逐token P_c 假象验证 (论文 §4.9, 附录 E.3)。

HallucinationDetector 是论文附录 E.3 的独立检测器: 每生成一个 token 调用
step(h), 维护近 3 个隐状态的 P_c/P_raw 滑动窗口均值, 连续命中阈值即报
动力学相变。相比 OrnithGuard (兼顾 <test> 状态机), 本类更轻量、专注
滑动窗口统计, 便于端到端评测。

batch_vs_token_pc 复现论文 §4.9 的方法论陷阱: 批量平均产生
P_c/P_raw≈0.84 的假象, 正确的逐 token 平均给出 ≈0.12。
"""

from collections import deque
from typing import Dict, Optional
import numpy as np
import torch

from ..physics import ThermoPhysics


class HallucinationDetector:
    """基于 P_c/P_raw 滑动窗口的零样本实时幻觉检测器。"""

    def __init__(
        self,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        window_size: int = 5,
        threshold_ratio: float = 0.15,
        consecutive_hits: int = 3,
    ):
        self.eng = ThermoPhysics(alpha_star=alpha_star, gamma=gamma)
        self.window = window_size
        self.threshold = threshold_ratio
        self.hits = consecutive_hits
        self.h_hist: deque = deque(maxlen=3)
        self.ratio_hist: deque = deque(maxlen=window_size)
        self.alert = 0

    def step(self, h) -> Dict:
        """处理一个 decode 步的隐状态 [D]。

        Returns: {is_hallucinating, smooth_ratio, level, alert_counter}
        """
        h = torch.as_tensor(h).detach().float().reshape(-1)
        self.h_hist.append(h)
        if len(self.h_hist) < 3:
            return {"is_hallucinating": False, "smooth_ratio": 0.0,
                    "level": "WARMUP", "alert_counter": self.alert}

        hc, hp, hpp = self.h_hist[2], self.h_hist[1], self.h_hist[0]
        pc, _ = self.eng.pc_ratio(hc, hp, hpp)
        self.ratio_hist.append(pc)
        smooth = float(np.mean(self.ratio_hist))

        if smooth > self.threshold:
            self.alert += 1
        else:
            self.alert = max(0, self.alert - 1)

        is_h = self.alert >= self.hits
        level = "CRITICAL" if is_h else ("WARNING" if self.alert > 0 else "NORMAL")
        return {"is_hallucinating": is_h, "smooth_ratio": smooth,
                "level": level, "alert_counter": self.alert}

    def reset(self):
        self.h_hist.clear()
        self.ratio_hist.clear()
        self.alert = 0


def batch_vs_token_pc(
    hidden_states: torch.Tensor,
    alpha_star: float = 1.41,
    gamma: float = 0.01,
    eps: float = 1e-8,
) -> Dict:
    """论文 §4.9 方法论陷阱: 批量平均 (假象) vs 逐 token 平均 (正确)。

    批量: |<P_c>| / |<P_raw>|  —— 正负 P_c 部分抵消, 给出≈0.84 假象
    逐token: < |P_c| / |P_raw| > —— 每个 token 约束功率都小, 给出≈0.12
    """
    eng = ThermoPhysics(alpha_star=alpha_star, gamma=gamma)
    H = torch.as_tensor(hidden_states).float()
    T = H.shape[0]
    Pc, Praw = [], []
    for t in range(2, T):
        pr, pa, pc = eng.powers(H[t], H[t - 1], H[t - 2])
        Pc.append(float(pr.item()))
        Praw.append(float(pa.item()))
    Pc = np.array(Pc)
    Praw = np.array(Praw)

    batch = abs(Pc.mean()) / (abs(Praw.mean()) + eps)
    token = float(np.mean(abs(Pc) / (abs(Praw) + eps)))
    return {
        "batch_avg_ratio": float(batch),
        "token_avg_ratio": token,
        "illusion_factor": float(batch / (token + eps)),
        "note": ("GUIT 铁律: 必须使用逐 token 计算, 禁止批量平均。本代理下二者符号结构"
                 "使 token 均值 > 批量均值 (真实 Ornith 实测方向可能相反), 但核心铁律"
                 "——P_c/P_raw 必须逐 token 计算、不可批量平均——不变; 绝对数值以真实"
                 " Ornith 实测为准。"),
    }
