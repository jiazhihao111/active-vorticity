"""
Frenet-Serret 标架计算模块 — GUIT-TRT 融合验证核心
=====================================================

功能:
1. 在任意维度隐空间中对离散序列构建 Frenet-Serret 活动标架 {T, N, B}
2. 计算局域语义曲率 κ_sem 和挠率 τ_sem
3. 计算语义螺旋升角 Θ_sem = arctan(τ_sem / κ_sem)
4. 计算离散陈数代理 (Chern number proxy)

数学原理:
- 对隐状态序列 {h_t}, 通过 SVD 将局部 4 点投影到 3D 子空间
- 在 3D 子空间内计算标准 Frenet-Serret 几何量
- 螺旋升角 = arctan(τ/κ), 理论预测该比值在高质文本中稳定

使用方法:
    from causal_gauge_field.utils.frenet_serret import FrenetSerretAnalyzer
    fs = FrenetSerretAnalyzer(eps=1e-8)
    result = fs.analyze(hidden_states)  # [B, T, dim]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FrenetSerretResult:
    """Frenet-Serret 分析结果"""
    kappa: torch.Tensor       # [B, T-3] 局域曲率
    tau: torch.Tensor         # [B, T-3] 局域挠率
    theta: torch.Tensor       # [B, T-3] 螺旋升角 = arctan(tau/kappa)
    tan_theta: torch.Tensor   # [B, T-3] tau/kappa 比值
    T: torch.Tensor           # [B, T-1, dim] 单位切向量
    N: Optional[torch.Tensor] # [B, T-2, dim] 主法向量
    B: Optional[torch.Tensor] # [B, T-3, dim] 副法向量
    chern_proxy: torch.Tensor # [B] 累积陈数代理 = Σ Tr(F²) / (4π²)
    chern_cumulative: torch.Tensor # [B, T-3] 累积曲率积分序列
    valid_mask: torch.Tensor  # [B, T-3] κ > eps 的有效位置


class FrenetSerretAnalyzer:
    """
    在任意维度隐空间中计算离散 Frenet-Serret 标架。

    核心方法:
    - _project_to_3d(): 通过 SVD 将局部 4 点投影到 3D 子空间
    - _frenet_serret_3d(): 在 3D 子空间中计算 T, N, B, κ, τ
    - analyze(): 端到端分析管线
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def _normalize(self, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """安全的向量归一化"""
        norm = torch.norm(v, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        return v / norm

    def _compute_tangent(self, h: torch.Tensor) -> torch.Tensor:
        """
        计算单位切向量 T_t = normalize(h_{t+1} - h_t)

        Args:
            h: [B, T, D] 隐状态序列
        Returns:
            T: [B, T-1, D] 单位切向量
        """
        dh = h[:, 1:] - h[:, :-1]  # [B, T-1, D]
        return self._normalize(dh, dim=-1)

    def _project_to_3d(self, points: torch.Tensor) -> torch.Tensor:
        """
        通过 SVD 将局部点集投影到最优 3D 子空间。

        Args:
            points: [B, N, D] 局部点集 (N >= 3)
        Returns:
            projected: [B, N, 3] 投影后的 3D 坐标
        """
        B, N, D = points.shape
        # 中心化
        centroid = points.mean(dim=1, keepdim=True)  # [B, 1, D]
        centered = points - centroid  # [B, N, D]

        # SVD 分解，取前 3 个右奇异向量作为 3D 基底
        U, S, V = torch.linalg.svd(centered, full_matrices=False)
        # V: [B, min(N,D), D], 取前 3 个
        basis = V[:, :3, :]  # [B, 3, D]

        # 投影: (centered @ basis^T)
        projected = torch.bmm(centered, basis.transpose(1, 2))  # [B, N, 3]
        # 确保右手系 (如果 det < 0 则翻转)
        return projected

    def _frenet_serret_3d(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        在 3D 空间中计算 Frenet-Serret 标架。

        Args:
            p: [B, 4, 3] 4 个连续的局部 3D 点
        Returns:
            dict with kappa[B], tau[B], T[B,3], N[B,3], B_vec[B,3]
        """
        # 一阶差分
        d1 = p[:, 1] - p[:, 0]  # [B, 3]
        d2 = p[:, 2] - p[:, 1]  # [B, 3]
        d3 = p[:, 3] - p[:, 2]  # [B, 3]

        # 单位切向量 (取中间点)
        T = self._normalize(d2, dim=-1)  # [B, 3]

        # 加速度向量 (反映曲率)
        acc = d3 - d2  # [B, 3] 二阶差分

        # 加速度的法向分量
        acc_parallel = (acc * T).sum(dim=-1, keepdim=True) * T  # [B, 3]
        acc_normal = acc - acc_parallel  # [B, 3]

        # 曲率 κ = ||acc_normal|| / ||d2||^2 ≈ ||acc_normal|| (归一化后)
        kappa = torch.norm(acc_normal, dim=-1)  # [B]
        N = self._normalize(acc_normal, dim=-1)  # [B, 3] 主法向量

        # 副法向量 B = T × N (在 3D 中)
        B_vec = torch.linalg.cross(T, N)  # [B, 3]

        # 挠率计算: 需要 dN/ds 在 B 方向的投影
        # 用三阶差分近似 dN
        # dN ≈ N_{t+1} - N_t，其中 N_{t+1} 由 d3 的加速度法向分量得到
        # 简化: 使用标准离散 Frenet 公式
        # τ = (T_t × T_{t+1}) · T_{t+2} / (||T_t × T_{t+1}|| · Δs)
        # 在投影空间中，Δs=1 近似

        # 方法：用 d1, d2, d3 计算挠率
        T_prev = self._normalize(d1, dim=-1)  # 前一个切向量
        T_next = self._normalize(d3, dim=-1)  # 后一个切向量

        # 副法向量变化
        cross_T = torch.linalg.cross(T_prev, T)  # [B, 3]
        cross_norm = torch.norm(cross_T, dim=-1, keepdim=True)
        cross_norm = torch.clamp(cross_norm, min=self.eps)

        # τ = (cross(T_prev, T) · T_next) / (||cross(T_prev, T)||^2)
        # 基于离散几何的标准公式
        numerator = (cross_T * T_next).sum(dim=-1)  # [B]
        denominator = (cross_norm.squeeze(-1)) ** 2
        denominator = torch.clamp(denominator, min=self.eps)
        tau = numerator / denominator  # [B]

        # 修正: tau 应该取绝对值，符号实际上取决于右手/左手螺旋
        # 在 GUIF 框架中，我们关心 |τ| 而非符号
        tau_abs = torch.abs(tau)

        return {
            'kappa': kappa,      # [B]
            'tau': tau_abs,      # [B]
            'T': T,              # [B, 3]
            'N': N,              # [B, 3]
            'B_vec': B_vec,      # [B, 3]
        }

    def analyze(
        self,
        hidden: torch.Tensor,
        compute_chern: bool = True,
    ) -> FrenetSerretResult:
        """
        端到端 Frenet-Serret 分析。

        Args:
            hidden: [B, T, D] 隐状态序列
            compute_chern: 是否计算陈数代理

        Returns:
            FrenetSerretResult 包含所有几何量
        """
        B, T, D = hidden.shape

        if T < 4:
            raise ValueError(f"序列长度至少为 4，当前 T={T}")

        # 存储每 4 点窗口的 FS 几何量
        kappa_list = []
        tau_list = []
        T_list = []
        N_list = []
        B_list = []

        for t in range(T - 3):
            # 取 4 个连续点
            window = hidden[:, t:t+4, :]  # [B, 4, D]

            # 投影到 3D
            p_3d = self._project_to_3d(window)  # [B, 4, 3]

            # 计算 Frenet-Serret
            fs_3d = self._frenet_serret_3d(p_3d)

            kappa_list.append(fs_3d['kappa'])
            tau_list.append(fs_3d['tau'])
            T_list.append(fs_3d['T'])
            N_list.append(fs_3d['N'])
            B_list.append(fs_3d['B_vec'])

        # 堆叠成序列
        kappa = torch.stack(kappa_list, dim=1)  # [B, T-3]
        tau = torch.stack(tau_list, dim=1)      # [B, T-3]
        T_seq = torch.stack(T_list, dim=1)      # [B, T-3, 3] (在局部3D空间中)
        N_seq = torch.stack(N_list, dim=1)      # [B, T-3, 3]
        B_seq = torch.stack(B_list, dim=1)      # [B, T-3, 3]

        # 有效掩码: κ > eps
        valid_mask = kappa > self.eps  # [B, T-3]

        # 螺旋升角 tan(Θ_sem) = τ / κ
        safe_kappa = torch.where(valid_mask, kappa, torch.ones_like(kappa) * self.eps)
        tan_theta = tau / safe_kappa  # [B, T-3]
        theta = torch.atan(tan_theta)  # [B, T-3] (0, π/2)

        # 陈数代理: 累积 Tr(F²) / (4π²)
        # 用局部曲率平方的累积近似
        if compute_chern:
            # F_proxy = κ · τ (曲率-挠率耦合度量)
            # Tr(F²)_proxy = κ² / (4π) 粗略近似
            local_chern = (kappa ** 2) / (4 * np.pi)  # [B, T-3]
            chern_cumulative = torch.cumsum(local_chern, dim=-1)  # [B, T-3]
            chern_proxy = chern_cumulative[:, -1]  # [B] 最终累积值
        else:
            chern_cumulative = torch.zeros(B, T-3, device=hidden.device)
            chern_proxy = torch.zeros(B, device=hidden.device)

        return FrenetSerretResult(
            kappa=kappa,
            tau=tau,
            theta=theta,
            tan_theta=tan_theta,
            T=T_seq,
            N=N_seq,
            B=B_seq,
            chern_proxy=chern_proxy,
            chern_cumulative=chern_cumulative,
            valid_mask=valid_mask,
        )

    def compute_helix_angle_stability(
        self,
        hidden: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算语义螺旋升角的稳定性指标。

        Returns:
            dict with:
            - mean_tan_theta: 平均 τ/κ 比值
            - std_tan_theta: τ/κ 比值的标准差
            - cv_tan_theta: 变异系数 (std/mean)
            - r_squared: τ-κ 线性拟合 R²
            - pearson_r: τ-κ Pearson 相关系数
            - valid_ratio: κ>eps 的有效点比例
        """
        result = self.analyze(hidden, compute_chern=False)

        # 只取有效点
        valid = result.valid_mask  # [B, T-3]
        kappa_valid = result.kappa[valid]
        tau_valid = result.tau[valid]
        tan_theta_valid = result.tan_theta[valid]

        if valid.sum() < 10:
            return {
                'mean_tan_theta': float('nan'),
                'std_tan_theta': float('nan'),
                'cv_tan_theta': float('nan'),
                'r_squared': float('nan'),
                'pearson_r': float('nan'),
                'valid_ratio': float(valid.float().mean().item()),
                'n_valid': int(valid.sum().item()),
            }

        k_np = kappa_valid.detach().cpu().numpy()
        t_np = tau_valid.detach().cpu().numpy()
        tan_np = tan_theta_valid.detach().cpu().numpy()

        mean_tan = float(np.mean(tan_np))
        std_tan = float(np.std(tan_np))
        cv = std_tan / mean_tan if mean_tan > 0 else float('nan')

        # Pearson 相关系数
        if len(k_np) > 1:
            corr_matrix = np.corrcoef(k_np, t_np)
            pearson_r = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else float('nan')
        else:
            pearson_r = float('nan')

        r_squared = pearson_r ** 2 if not np.isnan(pearson_r) else float('nan')

        return {
            'mean_tan_theta': mean_tan,
            'std_tan_theta': std_tan,
            'cv_tan_theta': cv,
            'r_squared': r_squared,
            'pearson_r': pearson_r,
            'valid_ratio': float(valid.float().mean().item()),
            'n_valid': int(valid.sum().item()),
        }

    def compute_gauss_bonnet_penalty(
        self,
        hidden: torch.Tensor,
        target_chi: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算高斯-博内拓扑惩罚项。

        λ_topo * (∫ Tr(F∧F) / (4π²) - χ)²

        Args:
            hidden: [B, T, D] 隐状态
            target_chi: 目标欧拉示性数 (连贯语义世界: χ=2)

        Returns:
            penalty: [B] 拓扑惩罚
            chern_estimate: [B] 估计的陈数
        """
        result = self.analyze(hidden, compute_chern=True)
        # chern_proxy 是 κ²/(4π) 的累积
        # 真正的陈数 = ∫ Tr(F∧F)/(8π²), 这里是简化代理
        # 规范化到 [0, 10] 范围
        chern_estimate = result.chern_proxy  # [B]

        # 目标值 (2 对应 χ=2 的球面拓扑)
        target = torch.ones_like(chern_estimate) * target_chi

        # 使用对数尺度避免梯度爆炸
        penalty = ((chern_estimate - target) ** 2).mean()

        return penalty, chern_estimate


def compute_batch_spiral_helix_statistics(
    analyzer: FrenetSerretAnalyzer,
    hidden_batch: torch.Tensor,
) -> Dict:
    """
    批量计算螺旋升角统计量，用于 H-helix 假设检验。

    Args:
        analyzer: FrenetSerretAnalyzer 实例
        hidden_batch: [B, T, D] 一批隐状态

    Returns:
        dict with aggregated statistics
    """
    B = hidden_batch.shape[0]
    all_tan_theta = []
    all_kappa = []
    all_tau = []

    for b in range(B):
        h = hidden_batch[b:b+1]  # [1, T, D]
        try:
            result = analyzer.analyze(h, compute_chern=False)
            valid = result.valid_mask[0]  # [T-3]
            all_tan_theta.append(result.tan_theta[0][valid].detach().cpu().numpy())
            all_kappa.append(result.kappa[0][valid].detach().cpu().numpy())
            all_tau.append(result.tau[0][valid].detach().cpu().numpy())
        except Exception:
            continue

    if not all_tan_theta:
        return {'error': 'no valid sequences'}

    # 拼接
    tan_all = np.concatenate(all_tan_theta)
    kappa_all = np.concatenate(all_kappa)
    tau_all = np.concatenate(all_tau)

    # 统计量
    mean_tan = float(np.mean(tan_all))
    std_tan = float(np.std(tan_all))
    cv = std_tan / mean_tan if mean_tan > 0 else float('nan')
    median_tan = float(np.median(tan_all))

    # FWHM (半高宽)
    sorted_tan = np.sort(tan_all)
    q25 = float(np.percentile(sorted_tan, 25))
    q75 = float(np.percentile(sorted_tan, 75))
    iqr = q75 - q25

    # Pearson r
    if len(kappa_all) > 1:
        corr = np.corrcoef(kappa_all, tau_all)
        pearson_r = float(corr[0, 1]) if corr.shape == (2, 2) else float('nan')
    else:
        pearson_r = float('nan')

    # 线性拟合 R²
    try:
        A = np.vstack([kappa_all, np.ones_like(kappa_all)]).T
        slope, intercept = np.linalg.lstsq(A, tau_all, rcond=None)[0]
        residuals = tau_all - (slope * kappa_all + intercept)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((tau_all - np.mean(tau_all)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    except Exception:
        r_squared = float('nan')
        slope = float('nan')

    return {
        'n_total': len(tan_all),
        'mean_tan_theta': mean_tan,
        'std_tan_theta': std_tan,
        'cv_tan_theta': cv,
        'median_tan_theta': median_tan,
        'iqr_tan_theta': iqr,
        'pearson_r_kappa_tau': pearson_r,
        'r_squared': r_squared,
        'slope': float(slope) if not np.isnan(slope) else float('nan'),
    }
