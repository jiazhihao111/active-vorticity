import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GaugeField(nn.Module):
    """因果规范场机制 (B-01/C-01/C-06/C-02/B-05/B-06)。

    将 token 序列的隐含轨迹视为纤维丛上的截面，计算：
      - 可学习规范势 A_t        (B-01)
      - 耦合常数 g              (C-01)
      - 场强 F_{μν}=∂A-∂A - g[A,A]   (C-02, 离散实值近似)
      - 由 F 派生的曲率度量 G^curve = (δ + η·Tr(F²))·I  (B-04, C-05 正定)
      - 叙事闭环用 Wilson 环量 W(γ)=Tr P exp(g∮A)        (B-06)

    重要(C-13/C-14): 规范场/纤维丛/H-geo/H-equiv 目前是候选隐喻，
    属待证假设，本模块仅提供机制实现，不宣称已验证统一理论。
    """

    def __init__(self, base_dim: int, coupling_init: float = 0.1):
        super().__init__()
        self.base_dim = base_dim
        # 规范势 A_t 由相邻隐含态差经线性映射得到 (B-01)
        self.A_head = nn.Linear(base_dim, base_dim, bias=False)
        self.g = nn.Parameter(torch.tensor(float(coupling_init)))   # 耦合常数 (C-01)
        self.curvature_delta = nn.Parameter(torch.tensor(0.1))      # δ
        self.curvature_eta = nn.Parameter(torch.tensor(0.05))       # η

    def connection(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, db) -> A: (B, T-1, db, db) 反对称联络矩阵
        diff = hidden[:, 1:, :] - hidden[:, :-1, :]          # (B, T-1, db)
        A_flat = self.A_head(diff)                           # (B, T-1, db)
        B, Tm1, db = A_flat.shape
        A = torch.zeros(B, Tm1, db, db, device=hidden.device)
        idx = torch.arange(db, device=hidden.device)
        A[:, :, idx, :] = A_flat.unsqueeze(-1)               # 用差分填充各行
        A = 0.5 * (A - A.transpose(-1, -2))                  # 反对称化
        return A

    def curvature(self, A: torch.Tensor) -> torch.Tensor:
        # F_{μν} ≈ A_{t+1} - A_t - g[A_t, A_{t+1}]  (离散实值近似, C-02)
        g = self.g
        dA = A[:, 1:, :, :] - A[:, :-1, :, :]                # (B, T-2, db, db)
        comm = (torch.matmul(A[:, :-1, :, :], A[:, 1:, :, :])
                - torch.matmul(A[:, 1:, :, :], A[:, :-1, :, :]))   # 交换子 [A_t, A_{t+1}]
        F = dA - g * comm
        return F

    def curvature_metric(self, hidden: torch.Tensor) -> torch.Tensor:
        # G^curve = (δ + η·Tr(F²))·I   (B-04, C-05 正定)
        A = self.connection(hidden)
        F = self.curvature(A)                                # (B, T-2, db, db)
        F2 = torch.matmul(F, F.transpose(-1, -2))            # (B, T-2, db, db)
        trF2 = torch.diagonal(F2, dim1=-2, dim2=-1).sum(-1) # (B, T-2)
        trF2_mean = trF2.mean(dim=-1).clamp(min=0)           # (B,)
        scale = (self.curvature_delta + self.curvature_eta * trF2_mean).clamp(min=1e-4)
        B = hidden.size(0)
        db = self.base_dim
        G_curve = (scale.unsqueeze(-1).unsqueeze(-1)
                   * torch.eye(db, device=hidden.device).unsqueeze(0))
        return G_curve                                      # (B, db, db)

    def wilson_loop(self, hidden: torch.Tensor) -> torch.Tensor:
        # W(γ) = Tr P exp(g∮A): 路径序指数 ∏_t exp(g A_t) 的迹 (B-06 标准离散化).
        # 注: 必须用 matrix_exp 而非 (I+gA)^n —— 后者对长序列(~60步)仅为一阶近似,
        # 会指数发散(实测平坦度爆炸到 1e4 量级). A 反对称 ⇒ exp(gA) 正交 ⇒ 乘积有界,
        # 这才是正确的 Wilson 环. C-11: Var[W]→0 仍为待实证声明.
        # 本方法沿整条序列的【开路径】乘积, 从未真正"闭合", 量的是开放轨迹的扭曲,
        # 无法检测叙事闭合. 真正闭合版本见 wilson_loop_closed(). 保留仅作消融对照.
        A = self.connection(hidden)                         # (B, T-1, db, db)
        g = self.g
        B, Tm1, db, _ = A.shape
        prod = torch.eye(db, device=A.device).unsqueeze(0).expand(B, -1, -1).clone()
        for t in range(Tm1):
            U = torch.linalg.matrix_exp(g * A[:, t, :, :])   # (B, db, db) 正交
            prod = torch.matmul(prod, U)
        W = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1)  # (B,)
        return W

    def connection_pair(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        # 两隐含态 h_a, h_b 间连边的反对称联络 (B, db, db).
        # 用于构造闭合边 (如 h_T -> h_0) 的规范势.
        diff = h_b - h_a                                    # (B, db)
        A_flat = self.A_head(diff)                          # (B, db)
        B, db = A_flat.shape
        A = torch.zeros(B, db, db, device=h_a.device)
        idx = torch.arange(db, device=h_a.device)
        A[:, idx, :] = A_flat.unsqueeze(-1)
        A = 0.5 * (A - A.transpose(-1, -2))
        return A

    def wilson_loop_closed(self, hidden: torch.Tensor) -> torch.Tensor:
        # 真正的闭合 Wilson 环 (B-06 的正确实现, 实验6主判据):
        #   先沿 h_0..h_T 做路径序指数 ∏_t exp(g A_t), 再补【闭合边】 A_close (h_T -> h_0),
        #   得到闭环 holonomy W(γ) ∈ (B, db, db). 每步 exp(gA) 正交 ⇒ 乘积有界(‖W‖_F=√db),
        #   故 ‖W - I‖_F 是 [0, 2√db] 上的有意义平坦度.
        # 规范场语义: 闭环叙事 h_T 回到与 h_0 一致的框架 ⇒ A_close≈0 ⇒ W≈I (平坦);
        #             破缺叙事 h_T 与 h_0 不一致 ⇒ A_close≠0 ⇒ W≠I (不平坦).
        # 这恰是 C-11「叙事闭环 ⇔ 规范场平坦」应被检验的几何量.
        A_seq = self.connection(hidden)                     # (B, T-1, db, db)
        g = self.g
        B, Tm1, db, _ = A_seq.shape
        prod = torch.eye(db, device=hidden.device).unsqueeze(0).expand(B, -1, -1).clone()
        for t in range(Tm1):
            U = torch.linalg.matrix_exp(g * A_seq[:, t, :, :])  # (B, db, db) 正交
            prod = torch.matmul(prod, U)
        A_close = self.connection_pair(hidden[:, -1, :], hidden[:, 0, :])  # (B, db, db)
        U_close = torch.linalg.matrix_exp(g * A_close)
        W = torch.matmul(prod, U_close)                     # (B, db, db)
        return W

    def holonomy_flatness(self, hidden: torch.Tensor) -> torch.Tensor:
        # 闭环 holonomy 到单位阵(平坦)的 Frobenius 距离; 闭环叙事应更小. (B,)
        W = self.wilson_loop_closed(hidden)                 # (B, db, db)
        I = torch.eye(W.size(-1), device=W.device).unsqueeze(0)
        return torch.norm(W - I, dim=(-2, -1))

    def closed_wilson_trace(self, hidden: torch.Tensor) -> torch.Tensor:
        # 闭环 holonomy 的迹; 平坦 ⇒ Tr(W)=db. 与 db 的偏差即不平坦度. (B,)
        W = self.wilson_loop_closed(hidden)
        return torch.diagonal(W, dim1=-2, dim2=-1).sum(-1)

    def loop_back_holonomy_flatness(
        self,
        hidden: torch.Tensor,
        min_gap: int = 3,
        max_loops: int = 5,
        sim_floor: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """差异化的『回环点』holonomy 平坦度 (实验7 主判据, §10.7 任务1).

        旧 wilson_loop_closed 对【整条序列】做 h_T→h_0 闭合, 是整段均值式量,
        正/负叙事一起被压缩, 无法区分 (exp6 结论). 这里改为:

          仅在『回环点』——即序列中某对 (s,t) 满足 h_t 与早先 h_s 高度相似
          (叙事性地『回到过去某个状态』)——处定义局部 holonomy:
              沿子路径 h_s..h_t 做路径序指数 ∏exp(gA), 再补闭合边 h_t→h_s.
          若该回环叙事性地连贯(正例的『真正回归』), 局部几何应平坦(W≈I);
          若只是轨迹巧合靠近(负例的『伪回归』/发散尾), 子路径内部会缠绕
          (非零曲率), 局部 holonomy 不平坦.

        聚合: 取每样本相似度最高的 max_loops 个回环, 对其平坦度取均值作为该
              样本的 loop_flatness; 无回环样本回退到整段闭合 holonomy.
        返回: flatness (B,), loop_count (B,) (诊断用: 正例应检到更多真实回环).
        """
        B, T, db = hidden.shape
        dev = hidden.device
        eye = torch.eye(db, device=dev)
        g = self.g
        out = torch.zeros(B, device=dev)
        counts = torch.zeros(B, device=dev)
        for b in range(B):
            h = hidden[b]                                       # (T, db)
            hn = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
            sim = hn @ hn.t()                                   # (T, T)
            idx = torch.arange(T, device=dev)
            mask = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= min_gap
            sim_m = sim.masked_fill(~mask, -1.0)
            flats = []
            k = min(max_loops, T * T)
            vals, flat_idx = torch.topk(sim_m.flatten(), k=k)
            for rank in range(k):
                s = int(flat_idx[rank] // T)
                t = int(flat_idx[rank] % T)
                if sim_m[s, t].item() < sim_floor:
                    continue
                sub = h[s:t]                                    # (L, db)
                if sub.size(0) < 2:
                    continue
                A_seq = self.connection(sub.unsqueeze(0))       # (1, L-1, db, db)
                prod = eye.clone().unsqueeze(0)
                for kk in range(A_seq.size(1)):
                    U = torch.linalg.matrix_exp(g * A_seq[0, kk])
                    prod = prod @ U
                A_close = self.connection_pair(h[t:t + 1], h[s:s + 1])  # (1, db, db)
                W = prod @ torch.linalg.matrix_exp(g * A_close[0])
                flats.append(torch.norm(W - eye))
            if flats:
                out[b] = torch.stack(flats).mean()
                counts[b] = len(flats)
            else:
                W = self.wilson_loop_closed(h.unsqueeze(0))[0]
                out[b] = torch.norm(W - eye)
                counts[b] = 0
        return out, counts

    def per_transition_connection_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """每 token 转移的局部非平坦度代理 (论文 §3.7 离散代理量, 与 token 转移一一对齐).

        取相邻隐含态联络 A_t 的 Frobenius 范数作为『局部曲率/非平坦度』离散代理:
          - 刚性几何要求连接平坦 (A≡0) ⇒ 该范数须近 0;
          - 柔性流形允许非零但有界 ⇒ 该范数允许 < τ.
        返回 (B, T-1), 与 per-token 转移对齐, 便于按刚柔分层掩码 (铁律八).
        """
        A = self.connection(hidden)                       # (B, T-1, db, db)
        return torch.norm(A, dim=(-2, -1))                # (B, T-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.curvature_metric(hidden)
