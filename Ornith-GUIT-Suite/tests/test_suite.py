"""Ornith-GUIT-Suite 冒烟与单元测试 (无需真实模型 / GPU)。"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ornith_guit.physics import ThermoPhysics, calibrate_alpha_star
from ornith_guit.core.ornith_calibrator import OrnithAutoCalibrator
from ornith_guit.core.streaming_compressor import StreamingAffineCompressor
from ornith_guit.detection.ornith_guard import OrnithGuard, PhaseTransitionException
from ornith_guit.detection.detector import HallucinationDetector, batch_vs_token_pc
from ornith_guit.steering.causal_kv import DynamicKVCacheEvictor
from ornith_guit.steering.affine_projector import AffineConstraintProjector
from ornith_guit.thermo import NESSDiagnostics, VorticityAnalyzer, abnormal_curvature
from ornith_guit.simulator import OrnithLatentSimulator
from ornith_guit.loop import PhysicsInformedLoop, SimulatedCodingBackend


def test_physics_powers_shape():
    eng = ThermoPhysics(alpha_star=1.41)
    h0 = torch.randn(16)
    h1 = torch.randn(16)
    h2 = torch.randn(16)
    P_raw, P_active, P_c = eng.powers(h0, h1, h2)
    assert P_raw.shape == P_active.shape == P_c.shape
    pc, vn = eng.pc_ratio(h0, h1, h2)
    assert 0.0 <= pc
    print("✅ physics.powers OK")


def test_calibrator():
    sim = OrnithLatentSimulator(seed=1)
    cal = OrnithAutoCalibrator(hidden_dim=sim.D, target_r=sim.r)
    states = [sim.generate_trajectory("pos", length=40)[0] for _ in range(4)]
    rep = cal.calibrate(states)
    assert cal.code_ridge_basis.shape == (cal.r, sim.D)
    # 投影-重构应有界 (仿射压缩本身有损, 但 r 维应捕获 >95% 方差)
    h = states[0][5]
    coords = cal.project(h)
    h2 = cal.reconstruct(coords)
    rel_err = float((h - h2).norm() / (h.norm() + 1e-8))
    # r 维应捕获 >95% 方差 ⇒ rel_err < sqrt(0.05) ≈ 0.224; 留余量取 0.22
    assert rel_err < 0.22, f"重构误差过大: {rel_err:.4f}"
    print(f"✅ calibrator OK: r={rep['ridge_dim (r)']}, alpha*={rep['alpha_star']}, recon_err={rel_err:.4f}")


def test_streaming_compressor():
    sim = OrnithLatentSimulator(seed=2)
    H, _ = sim.generate_trajectory("pos", length=40, drift=True, drift_rate=0.01)
    comp = StreamingAffineCompressor(sim.D, ridge_dim=sim.r)
    errs = []
    for t in range(len(H)):
        h_rec, _ = comp(H[t][None, None, :])
        errs.append(float((H[t] - h_rec.reshape(-1)).norm() / H[t].norm()))
    assert np.mean(errs) < 0.5
    print(f"✅ streaming compressor OK: mean_err={np.mean(errs):.4f}")


def test_guard_detects_flaw():
    sim = OrnithLatentSimulator(seed=3)
    states, flags = sim.generate_test_code_stream()
    guard = OrnithGuard(alpha_star=1.41, pc_threshold=0.08)
    guard.in_test_block = True
    detected = False
    for h in states:
        try:
            guard.process_step(h)
        except PhaseTransitionException:
            detected = True
            guard._reset_thermo_state()
    assert detected, "Guard 应检测到逻辑缺陷块"
    print("✅ OrnithGuard OK: flaw detected")


def test_kv_evictor():
    sim = OrnithLatentSimulator(seed=4)
    ctx = sim.generate_niah_context(num_tokens=100, num_needles=3)
    ev = DynamicKVCacheEvictor(max_capacity=50, n_sink=4, n_recent=10)
    ev.causal_scores = list(ctx["causal_score"])
    mask = ev.get_eviction_mask()
    assert len(mask) == 100
    assert sum(mask) == 50
    print("✅ CausalKV evictor OK")


def test_simulator_signature():
    sim = OrnithLatentSimulator(seed=5)
    eng = ThermoPhysics(alpha_star=1.41)
    pos = sim.generate_trajectory("pos", length=40)[0]
    rnd = sim.generate_trajectory("rnd", length=40)[0]
    pc_pos = np.mean([eng.pc_ratio(pos[t], pos[t-1], pos[t-2])[0] for t in range(2, 40)])
    pc_rnd = np.mean([eng.pc_ratio(rnd[t], rnd[t-1], rnd[t-2])[0] for t in range(2, 40)])
    print(f"✅ simulator signature OK: pos_pc={pc_pos:.4f} < rnd_pc={pc_rnd:.4f} "
          f"({'梯度正确' if pc_pos < pc_rnd else '梯度异常!'})")
    assert pc_pos < pc_rnd


def test_pi_loop_circuit_break():
    """PI-LOOP: 有缺陷时应触发物理熔断并节省 token。"""
    sim = OrnithLatentSimulator(seed=6)
    be = SimulatedCodingBackend(sim, flaw_start=10, flaw_len=40, fix_after=1)
    # 仿真器校准阈值 (较真实 Ornith 0.08 高, 因代理噪声更大; GUIT 边界约束)
    pi = PhysicsInformedLoop(be, alpha_star=1.41, pc_threshold=0.15,
                             max_iterations=4, guard_kwargs={"consecutive_hits": 3})
    _out, tr = pi.run("写测试用例", max_new_tokens=100)
    assert tr["physics_interventions"] >= 1, "应至少发生一次物理熔断"
    assert tr["total_tokens_saved"] > 0, "熔断应节省 token"
    assert tr["converged"], "第 2 轮修好后应收敛"
    print(f"✅ PI-LOOP OK: interventions={tr['physics_interventions']}, "
          f"tokens_saved={tr['total_tokens_saved']}, "
          f"converged@iter={tr['final_iteration']}")


def test_pi_loop_clean_pass():
    """PI-LOOP: 无缺陷时应一次通过, 零物理干预。"""
    sim = OrnithLatentSimulator(seed=7)
    be = SimulatedCodingBackend(sim, fix_after=0)  # 首轮即无缺陷
    pi = PhysicsInformedLoop(be, alpha_star=1.41, pc_threshold=0.15,
                             guard_kwargs={"consecutive_hits": 3})
    _out, tr = pi.run("写测试用例", max_new_tokens=80)
    assert tr["physics_interventions"] == 0
    assert tr["final_iteration"] == 0
    print("✅ PI-LOOP clean-pass OK: 一次通过, 零熔断")


def test_calibrator_online_update():
    """OrnithAutoCalibrator 在线更新 (元层进化) 应返回有界漂移。"""
    sim = OrnithLatentSimulator(seed=8)
    cal = OrnithAutoCalibrator(hidden_dim=sim.D, target_r=sim.r)
    cal.calibrate([sim.generate_trajectory("pos", length=40)[0] for _ in range(3)])
    rep = cal.update_online([sim.generate_trajectory("pos", length=40)[0]])
    assert rep["updated"] and 0.0 <= rep["subspace_drift"] <= 1.0
    assert cal.code_ridge_basis.shape == (cal.r, sim.D)
    print(f"✅ online update OK: drift={rep['subspace_drift']:.4f}")


def test_thermo_ness():
    """NESS 诊断: 合法 pos 轨迹应判为 NESS, 异常曲率 K_sub 合法<缺陷。"""
    sim = OrnithLatentSimulator(seed=11)
    diag = NESSDiagnostics(alpha_star=1.41, gamma=sim.gamma)
    pos = sim.generate_trajectory("pos", length=80)[0]
    halluc = sim.generate_trajectory("pos", length=80,
                                     halluc_steps=list(range(40, 55)))[0]
    mp = diag.diagnose(pos)
    mh = diag.diagnose(halluc)
    assert mp["entropy_production_sigma"] > 0
    assert mp["probability_flow_J"] > 0
    assert mp["is_NESS"], "合法 pos 轨迹应判为 NESS 定态"
    Kp = abnormal_curvature(pos, 1.41, sim.gamma)
    Kh = abnormal_curvature(halluc, 1.41, sim.gamma)
    assert Kp < Kh, "合法轨迹异常曲率应低于缺陷轨迹"
    print(f"✅ thermo NESS OK: is_NESS(pos)={mp['is_NESS']}, "
          f"Ksub pos={Kp:.4f} < halluc={Kh:.4f}")


def test_vorticity_rmt():
    """活性涡流: pos 结构化轨迹拒绝 RMT 零假设, 随机涡流 (rnd) 自洽不拒绝。

    零假设 = 速度场反对称部分与*通用随机涡流*无异。由于轨迹估计的雅可比受
    有限样本动力学偏置, 公平的 RMT 对照应取同分布的 rnd 轨迹系综 (而非纯
    随机矩阵)。pos 态注入结构化分块旋转涡流 → 特征值幅度呈 δ 峰, 系统性
    偏离 rnd 系综 → 拒绝零假设; rnd 自身与之自洽 → 不拒绝。

    涡流分析使用 vorticity_mode (强/干净旋转), 使速度雅可比可被轨迹估计
    可靠恢复; 该模式不影响 P_c 诊断所需的弱旋转 (其他测试默认关闭)。
    """
    from ornith_guit.thermo import estimate_velocity_jacobian, rmt_wigner_test
    sim = OrnithLatentSimulator(seed=12)
    basis = sim.Vr.numpy().T
    r = sim.r

    # 零假设系综: 多条 rnd (随机涡流) 轨迹估计的雅可比 (vorticity_mode)
    rnd_Js = []
    for _ in range(5):
        Hr = sim.generate_trajectory("rnd", length=120, vorticity_mode=True)[0]
        Jr, _, _ = estimate_velocity_jacobian(Hr, basis=basis, k=r)
        rnd_Js.append(Jr)

    # 观测: pos (结构化涡流) 轨迹估计的雅可比 (vorticity_mode)
    pos = sim.generate_trajectory("pos", length=120, vorticity_mode=True)[0]
    J_pos, _, _ = estimate_velocity_jacobian(pos, basis=basis, k=r)

    res_pos = rmt_wigner_test(J_pos, null_ensemble=rnd_Js, n_ref=120, seed=2026)
    res_rnd = rmt_wigner_test(rnd_Js[0], null_ensemble=rnd_Js, n_ref=120, seed=2025)
    assert res_pos["reject_rmt_null"], "pos 结构化涡流应拒绝 RMT 零假设"
    assert not res_rnd["reject_rmt_null"], "随机涡流 (rnd) 应自洽不拒绝 RMT 零假设"
    print(f"✅ vorticity RMT OK: pos_reject={res_pos['reject_rmt_null']} "
          f"(p={res_pos['p_value']:.3g}), rnd_reject="
          f"{res_rnd['reject_rmt_null']} (p={res_rnd['p_value']:.3g})")


def test_affine_projector():
    """仿射投影: 对 off-ridge 缺陷块投影后 P_c/P_raw 应显著下降。

    使用 generate_trajectory(halluc_steps=...) —— 其缺陷块是纯 off-ridge
    (Vp 空间) 注入, 与仿射约束模型一致, 投影可精确清除。比较区间锁定缺陷块,
    因合法脊线旋转自身 a⊥v 使全轨迹 P_c/P_raw 偏高 (见模块说明), 全轨迹均值
    无参考意义。
    """
    sim = OrnithLatentSimulator(seed=13)
    H = sim.generate_trajectory("pos", length=80,
                                halluc_steps=list(range(40, 55)))[0]
    proj = AffineConstraintProjector(sim.Vr.numpy().T, center=sim.mu.numpy())
    ba = proj.pc_before_after(H, alpha_star=1.41, gamma=sim.gamma,
                              region=(40, 56))
    assert ba["reduction_ratio"] > 0.3, "投影应显著降低缺陷块的 P_c/P_raw"
    print(f"✅ affine projector OK: block pc {ba['pc_raw']:.4f} -> "
          f"{ba['pc_projected']:.4f} (降幅 {ba['reduction_ratio']*100:.1f}%)")


def test_pc_illusion():
    """P_c 批量平均假象: batch_avg 与 token_avg 应明显不同 (禁止批量平均)。"""
    sim = OrnithLatentSimulator(seed=14)
    H = sim.generate_trajectory("pos", length=100,
                                halluc_steps=list(range(40, 55)))[0]
    res = batch_vs_token_pc(H, alpha_star=1.41, gamma=sim.gamma)
    assert res["batch_avg_ratio"] != res["token_avg_ratio"]
    print(f"✅ P_c illusion OK: batch={res['batch_avg_ratio']:.4f} vs "
          f"token={res['token_avg_ratio']:.4f} (factor={res['illusion_factor']:.2f}x)")


def test_detector_sliding_window():
    """HallucinationDetector: 滑动窗口应能在缺陷块触发 (连续命中)。"""
    sim = OrnithLatentSimulator(seed=15)
    det = HallucinationDetector(alpha_star=1.41, threshold_ratio=0.15,
                                consecutive_hits=3)
    states, flags = sim.generate_test_code_stream(
        length=80, flaw_start=30, flaw_len=20)
    triggered = False
    for h in states:
        r = det.step(h)
        if r["is_hallucinating"]:
            triggered = True
            break
    assert triggered, "滑动窗口检测器应在缺陷块触发"
    print("✅ detector sliding-window OK: 缺陷块触发")


if __name__ == "__main__":
    test_physics_powers_shape()
    test_calibrator()
    test_streaming_compressor()
    test_guard_detects_flaw()
    test_kv_evictor()
    test_simulator_signature()
    test_pi_loop_circuit_break()
    test_pi_loop_clean_pass()
    test_calibrator_online_update()
    test_thermo_ness()
    test_vorticity_rmt()
    test_affine_projector()
    test_pc_illusion()
    test_detector_sliding_window()
    print("\n🎉 全部冒烟测试通过")
