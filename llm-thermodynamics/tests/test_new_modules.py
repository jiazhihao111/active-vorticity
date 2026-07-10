import torch
import pytest
import numpy as np

from llm_thermo.core.ness import NESSEvaluator
from llm_thermo.core.sub_riemannian import SubRiemannianAnalyzer
from llm_thermo.core.rmt_vorticity import RMTVorticityAnalyzer
from llm_thermo.core.thermodynamics import ThermodynamicEngine
from llm_thermo.detection.quantization_guard import QuantizationGuard
from llm_thermo.config import get_preset, list_presets, PRESETS


class TestNESSEvaluator:
    def test_ness_evaluation_structure(self):
        T, D = 50, 32
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        evaluator = NESSEvaluator(alpha_star=1.41)
        result = evaluator.evaluate(h)
        assert "ness_verdict" in result
        assert "entropy_production_sigma" in result
        assert "J_FP_norm_per_dim" in result
        assert "macro_detailed_balance_broken" in result
        assert "coefficient_of_variation" in result
        assert "ness_pass_count" in result
        assert result["ness_pass_count"] >= 0

    def test_effective_temperature(self):
        T, D = 50, 32
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        evaluator = NESSEvaluator(alpha_star=1.41)
        T_eff = evaluator.effective_temperature(h)
        assert T_eff > 0

    def test_short_trajectory_error(self):
        h = torch.randn(5, 32)
        evaluator = NESSEvaluator(alpha_star=1.41)
        result = evaluator.evaluate(h)
        assert "error" in result

    def test_entropy_production_positive(self):
        T, D = 50, 32
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        evaluator = NESSEvaluator(alpha_star=1.41, gamma=0.01)
        result = evaluator.evaluate(h)
        assert result["entropy_production_sigma"] != 0


class TestSubRiemannianAnalyzer:
    def test_abnormal_curvature_shape(self):
        B, D = 4, 32
        analyzer = SubRiemannianAnalyzer(alpha_star=1.41)
        h_curr = torch.randn(B, D)
        h_prev = torch.randn(B, D)
        h_prev2 = torch.randn(B, D)
        K = analyzer.compute_abnormal_curvature(h_curr, h_prev, h_prev2)
        assert K.shape == (B, 1)
        assert (K >= 0).all()

    def test_abnormal_curvature_1d_input(self):
        D = 32
        analyzer = SubRiemannianAnalyzer(alpha_star=1.41)
        h_curr = torch.randn(D)
        h_prev = torch.randn(D)
        h_prev2 = torch.randn(D)
        K = analyzer.compute_abnormal_curvature(h_curr, h_prev, h_prev2)
        assert K.numel() == 1

    def test_trajectory_curvature(self):
        T, D = 30, 32
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        analyzer = SubRiemannianAnalyzer(alpha_star=1.41)
        result = analyzer.analyze_trajectory_curvature(h)
        assert "K_sub_mean" in result
        assert result["K_sub_mean"] >= 0

    def test_fit_affine_constraints(self):
        N, D = 50, 32
        torch.manual_seed(42)
        basis = torch.randn(D, 8)
        coeffs = torch.randn(N, 8)
        h = coeffs @ basis.T + torch.randn(N, D) * 0.01
        analyzer = SubRiemannianAnalyzer(alpha_star=1.41)
        result = analyzer.fit_affine_constraints(h)
        assert "R2_linear" in result
        assert result["R2_linear"] > 0.9
        assert result["effective_rank_r"] > 0

    def test_classify_constraints(self):
        N, D = 50, 32
        torch.manual_seed(42)
        h = torch.randn(N, D).cumsum(dim=0) * 0.1
        analyzer = SubRiemannianAnalyzer(alpha_star=1.41)
        result = analyzer.classify_constraints(h, per_trajectory=True)
        assert "n_holonomic" in result
        assert "n_nonholonomic" in result
        assert result["n_holonomic"] + result["n_nonholonomic"] == result["n_constraints"]


class TestRMTVorticityAnalyzer:
    def test_jacobian_computation(self):
        T, D = 50, 16
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        analyzer = RMTVorticityAnalyzer(pca_dim=None)
        result = analyzer.compute_jacobian(h)
        assert "J_vel" in result
        assert "J_symmetric" in result
        assert "J_antisymmetric" in result
        assert "vorticity_dissipation_ratio" in result

    def test_jacobian_with_pca(self):
        T, D = 50, 64
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        analyzer = RMTVorticityAnalyzer(pca_dim=16)
        result = analyzer.compute_jacobian(h)
        assert result["J_vel"].shape == (16, 16)
        assert result["pca_info"] is not None

    def test_wigner_test(self):
        d = 16
        torch.manual_seed(42)
        J_anti = torch.randn(d, d)
        J_anti = (J_anti - J_anti.T) / 2
        analyzer = RMTVorticityAnalyzer()
        result = analyzer.wigner_test(J_anti)
        assert "ks_statistic" in result
        assert "max_eigenvalue" in result

    def test_full_analysis(self):
        T, D = 50, 16
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        analyzer = RMTVorticityAnalyzer(pca_dim=None)
        result = analyzer.full_analysis(h)
        assert "J_vel" in result
        assert "ks_statistic" in result


class TestPerTokenRatio:
    def test_per_token_vs_batch(self):
        T, D = 50, 32
        torch.manual_seed(42)
        h = torch.randn(T, D).cumsum(dim=0) * 0.1
        engine = ThermodynamicEngine(alpha_star=1.41)
        result = engine.compute_per_token_ratio(h)
        assert "per_token_ratio" in result
        assert "batch_ratio_artifact" in result
        assert "method_warning" in result
        assert result["per_token_ratio"] >= 0
        assert result["batch_ratio_artifact"] >= 0

    def test_per_token_short_trajectory(self):
        h = torch.randn(3, 32)
        engine = ThermodynamicEngine(alpha_star=1.41)
        result = engine.compute_per_token_ratio(h)
        assert "error" in result


class TestQuantizationGuard:
    def test_check_rank_inflation_with_reference(self):
        T, D = 50, 64
        torch.manual_seed(42)
        h = torch.randn(T, D)
        guard = QuantizationGuard(bf16_r_095=32, max_inflation_rate=0.5)
        result = guard.check_rank_inflation(h)
        assert "current_r_095" in result
        assert "inflation_rate" in result
        assert "is_inflated" in result

    def test_check_rank_inflation_no_reference(self):
        T, D = 50, 64
        torch.manual_seed(42)
        h = torch.randn(T, D)
        guard = QuantizationGuard()
        result = guard.check_rank_inflation(h)
        assert result["bf16_reference_r_095"] is None
        assert not result["is_inflated"]

    def test_check_dtype_bf16(self):
        result = QuantizationGuard.check_dtype(torch.bfloat16)
        assert result["safe_for_ridge"] is True

    def test_check_dtype_float16(self):
        result = QuantizationGuard.check_dtype(torch.float16)
        assert result["safe_for_pc_ratio"] is True


class TestModelPresets:
    def test_get_preset_minicpm5(self):
        preset = get_preset("minicpm5-1b")
        assert preset.alpha_star == 1.46
        assert preset.hidden_dim == 1536

    def test_get_preset_qwen25(self):
        preset = get_preset("qwen2.5-7b")
        assert preset.alpha_star == 1.41
        assert preset.r_095_bf16 == 10

    def test_get_preset_ornith(self):
        preset = get_preset("ornith-1.0-9b")
        assert preset.alpha_star == 1.41
        assert "mixed" in preset.attention_type

    def test_list_presets(self):
        presets = list_presets()
        assert len(presets) >= 3

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent-model")