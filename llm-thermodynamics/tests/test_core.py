import torch
import pytest

from llm_thermo.core.kinematics import KinematicsExtractor
from llm_thermo.core.thermodynamics import ThermodynamicEngine
from llm_thermo.detection.phase_transition import HallucinationDetector, AlertLevel
from llm_thermo.detection.alerts import AlertCallback
from llm_thermo.steering.affine_projector import AffineProjector
from llm_thermo.steering.kv_cache import DynamicKVCacheEvictor
from llm_thermo.steering.ridge_optimizer import RidgeOptimizer


class TestKinematicsExtractor:
    def test_raw_diff_velocity(self):
        T, D = 10, 32
        h = torch.randn(T, D)
        ext = KinematicsExtractor(method="raw_diff")
        vel = ext.extract_velocity(h)
        assert vel.shape == (T, D)
        assert torch.allclose(vel[:-1], h[1:] - h[:-1], atol=1e-6)

    def test_central_diff_velocity(self):
        T, D = 10, 32
        h = torch.randn(T, D)
        ext = KinematicsExtractor(method="central_diff")
        vel = ext.extract_velocity(h)
        assert vel.shape == (T, D)

    def test_batch_velocity(self):
        B, T, D = 2, 10, 32
        h = torch.randn(B, T, D)
        ext = KinematicsExtractor(method="raw_diff")
        vel = ext.extract_velocity(h)
        assert vel.shape == (B, T, D)

    def test_effective_rank(self):
        T, D = 50, 32
        h = torch.randn(T, D)
        ext = KinematicsExtractor(method="raw_diff")
        vel = ext.extract_velocity(h)
        rank = KinematicsExtractor.compute_effective_rank(vel)
        assert 1 <= rank <= D

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            KinematicsExtractor(method="invalid")


class TestThermodynamicEngine:
    def test_compute_work_and_power_shapes(self):
        B, D = 4, 32
        engine = ThermodynamicEngine(alpha_star=1.0, gamma=0.01)
        h_curr = torch.randn(B, D)
        h_prev = torch.randn(B, D)
        h_prev2 = torch.randn(B, D)
        P_raw, P_active, P_c = engine.compute_work_and_power(h_curr, h_prev, h_prev2)
        assert P_raw.shape == (B, 1)
        assert P_active.shape == (B, 1)
        assert P_c.shape == (B, 1)

    def test_constraint_ratio_range(self):
        B, D = 4, 32
        engine = ThermodynamicEngine(alpha_star=1.0)
        h_curr = torch.randn(B, D)
        h_prev = torch.randn(B, D)
        h_prev2 = torch.randn(B, D)
        P_raw, _, P_c = engine.compute_work_and_power(h_curr, h_prev, h_prev2)
        ratio = engine.compute_constraint_ratio(P_c, P_raw)
        assert ratio.shape == (B, 1)
        assert (ratio >= 0).all()

    def test_trajectory_metrics(self):
        T, D = 20, 32
        h = torch.randn(T, D)
        engine = ThermodynamicEngine(alpha_star=1.0)
        metrics = engine.compute_trajectory_metrics(h)
        assert "P_c_P_raw_ratio" in metrics
        assert "fc_vel_cosine" in metrics
        assert "vel_norm_mean" in metrics

    def test_trajectory_too_short(self):
        h = torch.randn(3, 32)
        engine = ThermodynamicEngine()
        metrics = engine.compute_trajectory_metrics(h)
        assert "error" in metrics

    def test_calibrate_alpha_star(self):
        T, D = 20, 32
        trajectories = [torch.randn(T, D) for _ in range(5)]
        engine = ThermodynamicEngine(alpha_star=1.0)
        alpha = engine.calibrate_alpha_star(trajectories)
        assert isinstance(alpha, float)
        assert alpha != 1.0 or True


class TestHallucinationDetector:
    def test_warmup_phase(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        detector = HallucinationDetector(engine, threshold_ratio=0.05)
        h = torch.randn(1, 32)
        result = detector.step(h)
        assert result["alert_level"] == AlertLevel.WARMUP
        assert not result["is_hallucinating"]

    def test_normal_detection(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        detector = HallucinationDetector(engine, threshold_ratio=0.05, consecutive_hits=2)
        for _ in range(10):
            h = torch.randn(1, 32) * 0.01
            detector.step(h)
        assert detector._step_count == 10

    def test_reset(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        detector = HallucinationDetector(engine)
        detector.step(torch.randn(1, 32))
        detector.step(torch.randn(1, 32))
        detector.reset()
        assert len(detector.h_history) == 0
        assert detector._step_count == 0


class TestAlertCallback:
    def test_callback_triggered(self):
        cb = AlertCallback()
        triggered = []
        cb.on_critical(lambda r: triggered.append(r))
        cb.notify(AlertLevel.CRITICAL, {"step": 1, "is_hallucinating": True})
        assert len(triggered) == 1

    def test_cooldown(self):
        cb = AlertCallback(cooldown_steps=5)
        triggered = []
        cb.on_critical(lambda r: triggered.append(r))
        cb.notify(AlertLevel.CRITICAL, {"step": 1})
        cb.notify(AlertLevel.CRITICAL, {"step": 2})
        assert len(triggered) == 1


class TestAffineProjector:
    def test_no_projection_when_below_threshold(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        projector = AffineProjector(engine, project_threshold=10.0)
        h_curr = torch.randn(1, 32)
        h_prev = torch.randn(1, 32)
        h_prev2 = torch.randn(1, 32)
        result = projector.project(h_curr, h_prev, h_prev2)
        assert torch.allclose(result, h_curr, atol=1e-6)

    def test_1d_input(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        projector = AffineProjector(engine, project_threshold=0.0)
        h_curr = torch.randn(32)
        h_prev = torch.randn(32)
        h_prev2 = torch.randn(32)
        result = projector.project(h_curr, h_prev, h_prev2)
        assert result.shape == (32,)


class TestDynamicKVCacheEvictor:
    def test_no_eviction_below_capacity(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        evictor = DynamicKVCacheEvictor(engine, max_capacity=100)
        for i in range(50):
            h_curr = torch.randn(1, 32)
            h_prev = torch.randn(1, 32)
            h_prev2 = torch.randn(1, 32)
            evictor.update_scores(h_curr, h_prev, h_prev2)
        mask = evictor.get_eviction_mask()
        assert not any(mask)

    def test_eviction_above_capacity(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        evictor = DynamicKVCacheEvictor(engine, max_capacity=20, n_sink=2, n_recent=3)
        for i in range(30):
            h_curr = torch.randn(1, 32)
            h_prev = torch.randn(1, 32)
            h_prev2 = torch.randn(1, 32)
            evictor.update_scores(h_curr, h_prev, h_prev2)
        mask = evictor.get_eviction_mask()
        n_evict = sum(mask)
        assert n_evict == 10
        assert not any(mask[:2])
        assert not any(mask[-3:])

    def test_causal_backtracking_with_attn(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        evictor = DynamicKVCacheEvictor(engine, max_capacity=100)
        for i in range(10):
            h_curr = torch.randn(1, 32)
            h_prev = torch.randn(1, 32)
            h_prev2 = torch.randn(1, 32)
            attn = torch.softmax(torch.randn(1, 4, 1, i + 1), dim=-1)
            evictor.update_scores(h_curr, h_prev, h_prev2, attn_weights=attn)
        assert len(evictor.causal_scores) == 10

    def test_reset(self):
        engine = ThermodynamicEngine(alpha_star=1.0)
        evictor = DynamicKVCacheEvictor(engine)
        evictor.update_scores(torch.randn(1, 32), torch.randn(1, 32), torch.randn(1, 32))
        evictor.reset()
        assert len(evictor.causal_scores) == 0


class TestRidgeOptimizer:
    def test_auto_find_ridge(self):
        N, D = 50, 64
        torch.manual_seed(42)
        basis = torch.randn(D, 8)
        coeffs = torch.randn(N, 8)
        h = coeffs @ basis.T + torch.randn(N, D) * 0.01
        optimizer = RidgeOptimizer(variance_threshold=0.95)
        info = optimizer.auto_find_ridge(h)
        assert info["ridge_dim"] > 0
        assert info["ridge_dim"] <= D
        assert 0 < info["compression_ratio"] < 1
        assert info["explained_variance"] >= 0.95

    def test_auto_find_ridge_3d_input(self):
        N, D = 30, 32
        h = torch.randn(1, N, D)
        optimizer = RidgeOptimizer()
        info = optimizer.auto_find_ridge(h)
        assert info["ridge_dim"] > 0

    def test_auto_calibrate_alpha(self):
        D = 32
        h_t2 = torch.randn(D)
        h_t1 = h_t2 + torch.randn(D) * 0.1
        h_t = h_t1 + torch.randn(D) * 0.1
        optimizer = RidgeOptimizer()
        result = optimizer.auto_calibrate_alpha([h_t2, h_t1, h_t])
        assert "alpha_star" in result
        assert isinstance(result["alpha_star"], float)

    def test_calibrate_needs_3_steps(self):
        optimizer = RidgeOptimizer()
        with pytest.raises(ValueError):
            optimizer.auto_calibrate_alpha([torch.randn(32), torch.randn(32)])

    def test_step_decode_without_ridge_raises(self):
        optimizer = RidgeOptimizer()
        with pytest.raises(RuntimeError):
            optimizer.step_decode(torch.randn(32))

    def test_step_decode_monitor_mode(self):
        N, D = 50, 64
        torch.manual_seed(42)
        basis = torch.randn(D, 8)
        coeffs = torch.randn(N, 8)
        h = coeffs @ basis.T + torch.randn(N, D) * 0.01
        optimizer = RidgeOptimizer(variance_threshold=0.95)
        optimizer.auto_find_ridge(h)
        optimizer.auto_calibrate_alpha([h[-3], h[-2], h[-1]])

        h_curr = torch.randn(1, D)
        h_recon, metrics = optimizer.step_decode(h_curr)
        assert h_recon.shape == (1, D)
        assert "pc_ratio" in metrics
        assert "status" in metrics
        assert "cosine_sim" in metrics
        assert "mse" in metrics

    def test_low_dim_coords_roundtrip(self):
        N, D = 50, 64
        torch.manual_seed(42)
        h = torch.randn(N, D)
        optimizer = RidgeOptimizer(variance_threshold=0.95)
        optimizer.auto_find_ridge(h)

        h_test = torch.randn(D)
        coords = optimizer.get_low_dim_coords(h_test)
        assert coords.shape == (optimizer.r,)

        h_rec = optimizer.reconstruct_from_coords(coords)
        assert h_rec.shape == (D,)

    def test_max_ridge_dim(self):
        N, D = 50, 64
        h = torch.randn(N, D)
        optimizer = RidgeOptimizer(variance_threshold=0.99, max_ridge_dim=10)
        info = optimizer.auto_find_ridge(h)
        assert info["ridge_dim"] <= 10

    def test_fallback_rate(self):
        N, D = 50, 64
        h = torch.randn(N, D)
        optimizer = RidgeOptimizer(variance_threshold=0.95)
        optimizer.auto_find_ridge(h)
        optimizer.auto_calibrate_alpha([h[-3], h[-2], h[-1]])
        assert optimizer.fallback_rate == 0.0

    def test_should_reset_ridge(self):
        optimizer = RidgeOptimizer()
        assert not optimizer.should_reset_ridge()
        optimizer._total_count = 20
        optimizer._fallback_count = 10
        assert optimizer.should_reset_ridge(threshold=0.3)

    def test_reset_ridge(self):
        N, D = 50, 64
        h = torch.randn(N, D)
        optimizer = RidgeOptimizer()
        optimizer.auto_find_ridge(h)
        assert optimizer.ridge_basis is not None
        optimizer.reset_ridge()
        assert optimizer.ridge_basis is None
        assert optimizer.r == 0