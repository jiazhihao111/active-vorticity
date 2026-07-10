"""活性涡流分析 (论文 §3.4, §4.6, 附录 E)。

活性涡流的核心判据: LLM 在相空间中维持抵抗热力学耗散的高阶矩环流,
其速度场雅可比 J_vel = ∂v/∂h 可分解为
    J_vel = J_sym (耗散, 对称) + J_anti (涡流, 反对称)
论文用随机矩阵理论 (RMT) 证明: J_anti 的特征值幅度分布与随机矩阵
(Wigner 四分之一圆) 显著不同 (KS p < 10^-38), 确证因果叙事产生更强涡流。

本模块提供:
  - estimate_velocity_jacobian: 由轨迹用最小二乘估计速度雅可比
    (默认对轨迹做 PCA 降至 k 维; 也可直接传脊线基 basis [r,D])
  - decompose_jacobian / vorticity_ratio: 对称/反对称分解与涡度比
  - rmt_wigner_test: 反对称特征值幅度 vs Wigner 四分之一圆的 KS 检验
    (Monte-Carlo p 值, 不依赖 scipy)
  - analyze_vorticity: 一站式分析
  - VorticityAnalyzer: 封装类 (可缓存脊线基)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from ..physics import ThermoPhysics


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ----------------------------------------------------------------------
# 速度雅可比估计
# ----------------------------------------------------------------------
def estimate_velocity_jacobian(
    hidden_states: torch.Tensor,
    basis: Optional[np.ndarray] = None,
    k: int = 32,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """估计速度场雅可比 J [d, d], 满足 v ≈ J @ h_sub。

    Args:
        hidden_states: [T, D] 轨迹
        basis: 可选 [r, D] 正交脊线基 (已知时直接用, 如 calibrator 脊线);
               为 None 时对轨迹做 PCA 取前 k 主成分
        k: PCA 降维维度 (仅当 basis=None 时生效)
        center: 是否减去轨迹均值
    Returns:
        (J [d,d], used_basis [d,D], H_sub [T,d])
    """
    H = _to_np(hidden_states)
    T, D = H.shape

    if basis is None:
        mu = H.mean(axis=0)
        Hc = H - mu
        # PCA 降维 (论文 §4.6 同样先降至 32 维)
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        d = min(k, Hc.shape[1])
        basis = Vt[:d]                 # [d, D]
        ctr = mu
    else:
        basis = np.asarray(basis, float)
        d = basis.shape[0]
        ctr = H.mean(axis=0) if center else np.zeros(D)

    H_sub = (H - ctr) @ basis.T        # [T, d]
    V_sub = np.diff(H_sub, axis=0)     # [T-1, d]

    # 最小二乘: V_sub[t] ≈ J @ H_sub[t], t=0..T-2
    X = H_sub[:-1]                     # [T-1, d] 状态 h_t
    Y = V_sub                          # [T-1, d] 速度 v_t = h_{t+1}-h_t
    J, *_ = np.linalg.lstsq(X, Y, rcond=None)   # [d, d]
    return J, basis, H_sub


def decompose_jacobian(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """J = J_sym (对称, 耗散) + J_anti (反对称, 涡流)。"""
    J = np.asarray(J, float)
    sym = (J + J.T) / 2.0
    anti = (J - J.T) / 2.0
    return sym, anti


def vorticity_ratio(J: np.ndarray) -> Tuple[float, float, float]:
    """返回 (涡度/耗散比, ||J_sym||_F, ||J_anti||_F)。"""
    sym, anti = decompose_jacobian(J)
    n_sym = float(np.linalg.norm(sym, "fro"))
    n_anti = float(np.linalg.norm(anti, "fro"))
    ratio = n_anti / (n_sym + 1e-12)
    return ratio, n_sym, n_anti


# ----------------------------------------------------------------------
# RMT: 反对称特征值幅度 vs 随机反对称系综 (RMT 零假设)
# ----------------------------------------------------------------------
def _normalized_antisym_eigs(J: np.ndarray) -> np.ndarray:
    """对 J 做 Frobenius 归一化, 返回其反对称部分特征值幅度 (升序)。

    归一化使 RMT 检验对绝对量级不敏感: 观测矩阵与零假设矩阵均缩放到
    Frobenius 范数 1, 再比较特征值幅度分布, 避免尺度错配导致的假阴性。
    """
    Jn = np.asarray(J, float) / (np.linalg.norm(np.asarray(J, float), "fro") + 1e-12)
    anti = (Jn - Jn.T) / 2.0
    w = np.linalg.eigvals(anti)
    return np.sort(np.abs(w))


def _ks_2sample(a: np.ndarray, b: np.ndarray) -> float:
    """两样本 Kolmogorov–Smirnov 统计量 D (经验分布间最大竖直距离)。"""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    i = j = 0
    D = 0.0
    while i < na and j < nb:
        x = a[i]
        if x <= b[j]:
            i += 1
        else:
            j += 1
        D = max(D, abs(i / na - j / nb))
    return float(D)


def rmt_wigner_test(
    J: np.ndarray,
    null_ensemble: Optional[List[np.ndarray]] = None,
    n_ref: int = 200,
    seed: int = 0,
) -> Dict:
    """尺度不变的 RMT 检验: 观测 J_anti 特征值幅度分布 vs 零假设系综。

    零假设 (RMT): 速度场反对称部分的谱与*通用随机反对称矩阵*无异。
    替代假设 (论文 §3.4): 结构化活性涡流维持高阶矩环流, 其 J_anti 特征值
    幅度呈 δ 峰 / 特定聚集, 系统性偏离随机系综 → 拒绝零假设
    (论文实测 p < 10^-38)。

    实现: 先对观测矩阵与零假设矩阵各自做 Frobenius 归一化 (消除尺度错配),
    再做两样本 KS; p 值由 Monte-Carlo 自举零分布给出 (无需 scipy):
        p = P(KS_bootstrap ≥ KS_observed)

    零假设系综来源:
      - 默认 (null_ensemble=None): 直接采样归一化随机反对称矩阵;
      - 推荐: 传入由*真实轨迹*估计的雅可比列表 (如 rnd 态多条轨迹),
        因轨迹估计的雅可比受有限样本动力学偏置, 用同分布轨迹系综作对照
        才是公平的 RMT 零假设 (见 test_vorticity_rmt)。
    """
    rng = np.random.default_rng(seed)
    d = J.shape[0]
    lam_obs = _normalized_antisym_eigs(J)

    if null_ensemble is not None and len(null_ensemble) > 0:
        null_all = np.concatenate(
            [_normalized_antisym_eigs(np.asarray(A, float)) for A in null_ensemble])
    else:
        # 零假设系综: 归一化随机反对称矩阵
        def _rand_anti() -> np.ndarray:
            G = rng.standard_normal((d, d))
            A = G - G.T
            return A / (np.linalg.norm(A, "fro") + 1e-12)

        null_mats = [_rand_anti() for _ in range(n_ref)]
        null_all = np.concatenate([_normalized_antisym_eigs(A) for A in null_mats])

    ks_obs = _ks_2sample(lam_obs, null_all)

    # 零分布: 从 null_all 有放回抽 d 个作为伪观测, 比较 vs null_all
    null_ks = np.empty(n_ref)
    for i in range(n_ref):
        pseudo = rng.choice(null_all, size=d, replace=True)
        null_ks[i] = _ks_2sample(pseudo, null_all)

    p = float(np.mean(null_ks >= ks_obs))
    return {
        "dim": int(d),
        "ks_observed": float(ks_obs),
        "ks_null_mean": float(null_ks.mean()),
        "ks_null_median": float(np.median(null_ks)),
        "p_value": p,
        "reject_rmt_null": bool(p < 0.05),
        "n_ref": int(n_ref),
    }


# ----------------------------------------------------------------------
# 一站式分析
# ----------------------------------------------------------------------
def analyze_vorticity(
    hidden_states: torch.Tensor,
    basis: Optional[np.ndarray] = None,
    k: int = 32,
    n_ref: int = 200,
    seed: int = 0,
) -> Dict:
    """对一条轨迹做完整涡流分析。

    返回含速度雅可比、涡度比、RMT 检验的结构化字典 (雅可比本身不序列化)。
    """
    J, used_basis, _ = estimate_velocity_jacobian(hidden_states, basis=basis, k=k)
    ratio, n_sym, n_anti = vorticity_ratio(J)
    rmt = rmt_wigner_test(J, n_ref=n_ref, seed=seed)
    return {
        "jacobian_dim": int(J.shape[0]),
        "vorticity_ratio": ratio,
        "sym_norm": n_sym,
        "anti_norm": n_anti,
        "rmt": rmt,
    }


class VorticityAnalyzer:
    """活性涡流分析器 (可缓存脊线基, 对多条轨迹复用)。"""

    def __init__(
        self,
        ridge_basis: Optional[np.ndarray] = None,
        pca_dim: int = 32,
        n_ref: int = 200,
        seed: int = 0,
    ):
        self.ridge_basis = (None if ridge_basis is None
                            else np.asarray(ridge_basis, float))
        self.pca_dim = pca_dim
        self.n_ref = n_ref
        self.seed = seed

    def analyze(self, hidden_states: torch.Tensor) -> Dict:
        return analyze_vorticity(
            hidden_states,
            basis=self.ridge_basis,
            k=self.pca_dim,
            n_ref=self.n_ref,
            seed=self.seed,
        )

    def compare(self, trajectories: Dict[str, torch.Tensor]) -> Dict:
        out = {}
        for name, H in trajectories.items():
            a = self.analyze(H)
            out[name] = {
                "vorticity_ratio": a["vorticity_ratio"],
                "anti_norm": a["anti_norm"],
                "rmt_ks": a["rmt"]["ks_observed"],
                "rmt_p": a["rmt"]["p_value"],
                "reject_rmt_null": a["rmt"]["reject_rmt_null"],
            }
        return out
