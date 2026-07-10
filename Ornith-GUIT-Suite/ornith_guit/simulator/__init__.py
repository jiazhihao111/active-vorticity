"""Ornith 潜空间物理仿真器。

由于真实 Ornith-1.0-9B (9B, 消费级 GPU 难以加载) 无法在本环境直接运行,
本仿真器依据论文公布的 Ornith 热力学签名构建一个**忠实可控的潜空间生成器**:

  - 环境维度 D=256 (论文 4096 的缩放代理; 无量纲指标与尺度无关)
  - 脊线维度 r=16  (对应论文 r(0.95)=25, 压缩率 ~93.75%)
  - alpha* = 1.41 (Ornith bf16 校准值)
  - 仿射-非完整混合约束: 位置空间仿射骨架 (V_r 子空间) + 速度空间非完整分布

生成三类文本态 (与论文表3/4 一致):
  - pos: 因果叙事/合法代码 (off-ridge 噪声小 → P_c/P_raw 低)
  - scr: 语序打乱 (中等)
  - rnd: 随机 token (off-ridge 噪声大 → P_c/P_raw 高)

边界声明 (GUIT 铁律): 本仿真器是**理论信号的受控代理**, 用于在无 GPU 环境
下验证算法的相对优劣 (方法学对比); 绝对数值以真实 Ornith 实测为准。
"""

from typing import Tuple, List, Dict, Optional
import torch
import numpy as np


class OrnithLatentSimulator:
    def __init__(
        self,
        D: int = 256,
        r: int = 16,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        seed: int = 20260710,
    ):
        self.D = D
        self.r = r
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.gen = torch.Generator().manual_seed(seed)

        # 仿射骨架: 随机正交的脊线基底 V_r [D, r] 与补空间 V_perp [D, D-r]
        Q, _ = torch.linalg.qr(torch.randn(D, D, generator=self.gen))
        self.Vr = Q[:, :r].clone()            # [D, r]
        self.Vp = Q[:, r:].clone()            # [D, D-r]
        self.mu = torch.randn(D, generator=self.gen) * 0.1

        # 活性旋转生成器 J (反对称, 谱归一化) → 脊线上自洽涡旋 (active vorticity)
        A = torch.randn(r, r, generator=self.gen)
        J = A - A.T
        nrm = J.norm() + 1e-8
        self.J = J / nrm * 0.3   # 随机涡旋 (通用反对称, 与 RMT 零假设一致)

        # 结构化活性涡流 (论文 §3.4): 分块 2x2 旋转 → 反对称矩阵特征值幅度
        # 呈 δ 峰聚集, 系统性偏离随机反对称系综 (RMT 零假设)。pos 态使用。
        # 与随机 J 取相同 Frobenius 范数 (0.3), 仅*结构*不同, 不引入量级差异。
        S = torch.zeros(r, r)
        c = 1.0
        for b in range(0, r - (r % 2), 2):
            S[b, b + 1] = c
            S[b + 1, b] = -c
        s_norm = torch.norm(S) + 1e-8
        self.J_structured = S * (0.3 / s_norm)  # Frobenius=0.3, 与随机 J 一致
        self.omega = 0.3
        # 涡流分析专用: 更强/更干净的旋转, 使速度雅可比可被轨迹估计可靠恢复
        # (P_c 诊断所需的弱旋转会让雅可比被噪声淹没, 故两者解耦)。
        self.omega_vort = 2.0
        self.sigma_z_vort = 0.01



        # 各文本态 off-ridge 噪声幅度 (决定 P_c/P_raw 梯度, 论文表3: pos<scr<rnd)
        self.sigma_off = {"pos": 0.03, "scr": 0.60, "rnd": 2.50}

    # ------------------------------------------------------------------
    def _step_latent(self, z, sigma_off, sigma_z: float = 0.10, J: Optional[torch.Tensor] = None,
                     omega: Optional[float] = None):
        """演化一步脊线潜变量 z [r]。

        受控活性涡旋: z_{t+1} = z_t + omega*(J@z_t) + noise,
        J 已谱归一化, 故增量幅度 ~ omega*||z||, 形成稳定有界振荡。
        J 默认随机反对称 (通用); pos 态传入 J_structured (结构化涡流)。
        omega / sigma_z 可临时覆盖 (涡流分析专用强旋转)。
        """
        if J is None:
            J = self.J
        if omega is None:
            omega = self.omega
        z_new = z + omega * (J @ z) + sigma_z * torch.randn(
            self.r, generator=self.gen)
        return z_new

    def generate_trajectory(
        self,
        regime: str = "pos",
        length: int = 60,
        drift: bool = False,
        drift_rate: float = 0.0,
        halluc_steps: Optional[List[int]] = None,
        vorticity_mode: bool = False,
    ) -> Tuple[torch.Tensor, List[str]]:
        """生成一条 [length, D] 隐状态轨迹 + 每步标签。

        regime: pos / scr / rnd
        drift: 脊线子空间随时间从 V_r0 漂移到 V_r1 (测试压缩器)
        halluc_steps: 在这些步注入 off-ridge 大扰动 (动力学相变)
        vorticity_mode: 使用更强/更干净的旋转, 使速度雅可比可被轨迹估计
            可靠恢复 (用于活性涡流 RMT 分析; 不影响 P_c 诊断所需的弱旋转)。
        """
        if regime not in self.sigma_off:
            raise ValueError(f"未知 regime: {regime}")
        halluc_steps = set(halluc_steps or [])
        sigma_off = self.sigma_off[regime]
        vort_omega = self.omega_vort if vorticity_mode else self.omega
        vort_sigma_z = self.sigma_z_vort if vorticity_mode else 0.10

        # drift 目标子空间
        if drift:
            Q2, _ = torch.linalg.qr(torch.randn(self.D, self.D, generator=self.gen))
            Vr1 = Q2[:, :self.r].clone()

        z = torch.randn(self.r, generator=self.gen) * 0.5
        o_prev = torch.randn(self.D - self.r, generator=self.gen) * sigma_off
        H = []
        labels = []
        for t in range(length):
            # 脊线子空间 (drift 时插值)
            if drift:
                w = min(1.0, drift_rate * t)
                Vr_t = torch.linalg.qr(
                    (1 - w) * self.Vr + w * Vr1)[0]
            else:
                Vr_t = self.Vr

            if t in halluc_steps:
                # 持续离流形漂移 (动力学相变):
                # 冻结脊线旋转, 注入大幅且缓慢变化的 off-ridge 分量,
                # 使约束力做功 P_c 持续飙升 (与论文表3梯度一致)。
                o = o_prev + 1.5 * torch.randn(self.D - self.r, generator=self.gen)
                h = self.mu + Vr_t @ z + self.Vp @ (o * 4.0)
                labels.append("halluc")
            else:
                use_J = self.J_structured if regime == "pos" else self.J
                z = self._step_latent(z, sigma_off, sigma_z=vort_sigma_z, J=use_J,
                                      omega=vort_omega)
                o = torch.randn(self.D - self.r, generator=self.gen) * sigma_off
                h = self.mu + Vr_t @ z + self.Vp @ o
                labels.append(regime)
            o_prev = o
            H.append(h)

        return torch.stack(H, dim=0), labels

    # ------------------------------------------------------------------
    def generate_test_code_stream(
        self, length: int = 80, flaw_start: int = 40, flaw_len: int = 15
    ) -> Tuple[List[torch.Tensor], List[bool]]:
        """生成一段 <test> 代码生成的隐状态流。

        前段: 合法代码 (pos 动力学, 低 P_c)
        中段: 逻辑缺陷块 (rnd 动力学, 高 P_c) —— 被 GUIT 捕获的目标
        后段: 恢复正常

        Returns: (list of [D] tensors, is_flaw flags)
        """
        states, flags = [], []
        z = torch.randn(self.r, generator=self.gen) * 0.5
        o_prev = torch.randn(self.D - self.r, generator=self.gen) * self.sigma_off["pos"]
        for t in range(length):
            regime = "rnd" if (flaw_start <= t < flaw_start + flaw_len) else "pos"
            sigma_off = self.sigma_off[regime]
            z = self._step_latent(z, sigma_off)
            o = torch.randn(self.D - self.r, generator=self.gen) * sigma_off
            o_prev = o
            h = self.mu + self.Vr @ z + self.Vp @ o
            if regime == "rnd":
                h = h + 2.0 * torch.randn(self.D, generator=self.gen)
            states.append(h)
            flags.append(regime == "rnd")
        return states, flags

    # ------------------------------------------------------------------
    def generate_niah_context(
        self, num_tokens: int = 200, num_needles: int = 4, seed: int = 7
    ) -> Dict:
        """生成 NIAH 风格长上下文, 用于 KV 淘汰对比。

        返回: causal_score[seq], attention_score[seq], needle_mask[seq]
        - causal_score: 承重墙在 needle 位置及之后若干步出现高值 (因果转折)
        - attention_score: H2O 代理 = 近期衰减 + 噪声 (无因果语义)
        """
        g = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        causal = np.zeros(num_tokens, dtype=float)
        attn = np.zeros(num_tokens, dtype=float)
        needle = np.zeros(num_tokens, dtype=bool)

        positions = rng.choice(num_tokens - 20, size=num_needles, replace=False)
        for p in positions:
            needle[p] = True
            # 承重墙: needle 位置高, 并向后衰减 (因果依赖)
            for k in range(12):
                if p + k < num_tokens:
                    causal[p + k] += 1.0 * np.exp(-k / 4.0)
        # 基础因果背景噪声
        causal += rng.uniform(0.0, 0.15, size=num_tokens)

        # 注意力分数: 强烈偏向近期 (H2O 行为) + 噪声, 与 needle 无关
        for t in range(num_tokens):
            recency = np.exp(-(num_tokens - t) / 40.0)
            attn[t] = recency + rng.uniform(0.0, 0.1)
        attn = attn / attn.max()

        return {
            "causal_score": causal,
            "attention_score": attn,
            "needle_mask": needle,
            "num_needles": int(needle.sum()),
        }
