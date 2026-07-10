"""OrnithGuard — 测试代码热力学相变守卫 (Agentic Coding 防崩溃)。

理论锚点: Ornith 在 <test> ... </test> 内生成死循环/错误断言/语法
崩溃代码时, 隐状态会逃逸出 r=25 的代码仿射子空间, 约束力做功 P_c
瞬间飙升 (动力学相变)。

实现: 前向 Hook + 标签状态机 + 异常中断。检测到相变时抛出
PhaseTransitionException, 交由上层 Agent 框架 Backtracking 或修正。

对比基线 (本套件 benchmark 中使用):
  - vel_norm 基线: 论文指出 4-bit 量化下 vel_norm 梯度反转, 不可靠
  - 语法检查基线: 只能抓语法, 抓不住"看似合理但逻辑崩溃"的语义缺陷
"""

from typing import Optional, Dict, List, Deque
from collections import deque
import torch


class PhaseTransitionException(Exception):
    """GUIT 动力学相变异常: 用于中断生成并触发 Agent 回退。"""

    def __init__(self, pc_ratio: float, message: str):
        self.pc_ratio = pc_ratio
        super().__init__(message)


class OrnithGuard:
    def __init__(
        self,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        pc_threshold: float = 0.08,
        window_size: int = 3,
        consecutive_hits: int = 2,
    ):
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.pc_thresh = pc_threshold
        self.window = window_size
        self.hits = consecutive_hits

        self.in_test_block = False
        self.h_history: Deque[torch.Tensor] = deque(maxlen=3)
        self.ratio_history: Deque[float] = deque(maxlen=window_size)
        self.vel_history: Deque[float] = deque(maxlen=window_size)
        self.alert_counter = 0
        self._last_pc = 0.0
        self._last_vel = 0.0

        self.hook = None

    # ---- 状态机 (可由 tokenizer 在生成循环里调用) ------------------
    def update_state_machine(self, token_text: str):
        if "<test>" in token_text:
            self.in_test_block = True
            self._reset_thermo_state()
        elif "</test>" in token_text:
            self.in_test_block = False

    def _reset_thermo_state(self):
        self.h_history.clear()
        self.ratio_history.clear()
        self.vel_history.clear()
        self.alert_counter = 0

    # ---- 核心热力学计算 (可直接由仿真器/Hook 调用) -----------------
    @torch.no_grad()
    def process_step(self, h_curr: torch.Tensor) -> Dict:
        """处理一个 decode 步的隐状态。

        返回 metrics; 若判定为相变且满足连续命中, 抛出
        PhaseTransitionException (仅当 in_test_block 时报警)。
        """
        h = h_curr.detach().float().reshape(-1)  # [D]
        info: Dict = {"in_test_block": self.in_test_block,
                      "pc_ratio": self._last_pc, "vel_norm": self._last_vel,
                      "alert": False}

        if not self.in_test_block:
            return info

        self.h_history.append(h)
        if len(self.h_history) < 3:
            return info

        h_t, h_t1, h_t2 = self.h_history[2], self.h_history[1], self.h_history[0]
        v = h_t - h_t1
        a = v - (h_t1 - h_t2)
        F_res = a + self.gamma * v
        P_raw = float(torch.sum(F_res * v).item())
        P_active = self.alpha_star * float(torch.sum(v * v).item())
        P_c = P_raw - P_active
        pc = abs(P_c) / (abs(P_raw) + 1e-8)
        vn = float(v.norm().item())

        self._last_pc, self._last_vel = pc, vn
        self.ratio_history.append(pc)
        self.vel_history.append(vn)

        smooth_pc = sum(self.ratio_history) / len(self.ratio_history)
        smooth_vel = sum(self.vel_history) / len(self.vel_history)

        if smooth_pc > self.pc_thresh:
            self.alert_counter += 1
        else:
            self.alert_counter = max(0, self.alert_counter - 1)

        info.update({"pc_ratio": smooth_pc, "vel_norm": smooth_vel,
                     "alert": self.alert_counter >= self.hits})
        if self.alert_counter >= self.hits:
            raise PhaseTransitionException(
                smooth_pc,
                f"GUIT 拦截: 测试代码动力学相变 (P_c/P_raw={smooth_pc:.4f}), 逻辑可能崩溃!",
            )
        return info

    # ---- 可选: 真实模型 Hook 注册 ----------------------------------
    def register_hook(self, model):
        """在最后一层注册前向 Hook 捕获 decode 隐状态 (真实 Ornith 部署)。"""
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:           # decode 阶段
                self.process_step(h.squeeze(1))
        last = model.model.layers[-1]
        self.hook = last.register_forward_hook(hook_fn)

    def close(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
