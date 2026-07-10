# llm-thermodynamics

Non-equilibrium thermodynamics of LLM latent spaces — hallucination detection, NESS evaluation, sub-Riemannian geometry, RMT vorticity, and causal manifold steering.

[![PyPI](https://img.shields.io/badge/version-0.3.0-blue)](pyproject.toml) [![Python](https://img.shields.io/badge/python-%E2%89%A53.10-green)](pyproject.toml) [![Tests](https://img.shields.io/badge/tests-56%20pass-brightgreen)]()

## What is this?

`llm-thermodynamics` is a **non-invasive, plug-and-play** framework for monitoring and steering LLM hidden state dynamics during inference. It treats the LLM's latent space as a thermodynamic system governed by an affine constraint manifold:

```
m·a + γ·v = α*·v + F_c + ξ
```

Where the constraint force power `P_c = F_c · v ≈ 0` for valid trajectories. When `P_c/P_raw` rises, the model is leaving the constraint manifold — a reliable hallucination signal.

### Validated across models

| Model | Params | Hidden Dim | α* | P_c/P_raw (pos) | r(0.95) | NESS |
|-------|--------|------------|-----|-----------------|---------|------|
| MiniCPM5-1B | 1.08B | 1536 | 1.46 | 0.140 | 15.2 | ✅ |
| Qwen2.5-7B | 7B | 3584 | 1.41 | 0.109 | 9.6 | ✅ |
| Ornith-1.0-9B | 9B | 4096 | 1.41 | 0.09-0.15 | 25 | ✅ |

## Installation

```bash
pip install -e .

# With HuggingFace integration:
pip install -e ".[hf]"

# Development:
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Thermodynamic Monitoring

```python
from llm_thermo import ThermodynamicEngine, get_preset

preset = get_preset("minicpm5-1b")
engine = ThermodynamicEngine(alpha_star=preset.alpha_star, gamma=preset.gamma)

# From hidden state trajectory [T, D]
result = engine.compute_per_token_ratio(hidden_states)
print(f"P_c/P_raw = {result['per_token_ratio']:.4f}")  # ~0.12 for valid text
```

### 2. Real-time Hallucination Detection

```python
from llm_thermo import ThermodynamicEngine, HallucinationDetector, ThermoHookManager

engine = ThermodynamicEngine(alpha_star=1.41)
detector = HallucinationDetector(engine, threshold_ratio=0.05)

# Non-invasive hook on HuggingFace model
hook_mgr = ThermoHookManager(model, detector)
hook_mgr.register_hooks(layer_index=-1)

# Generate with guardrail
outputs = hook_mgr.generate_with_guardrail(
    model, input_ids, max_new_tokens=50,
    on_hallucination_callback=lambda r: print("⚠️ Hallucination detected!", r)
)
```

### 3. NESS Evaluation

```python
from llm_thermo import NESSEvaluator

ness = NESSEvaluator(alpha_star=1.46)
result = ness.evaluate(hidden_states)  # [T, D]
print(result["ness_verdict"])  # "NESS" or "NOT_NESS"
# Returns: sigma, J_FP, T_eff, coefficient_of_variation, detailed_balance_verdict
```

### 4. Sub-Riemannian Geometry

```python
from llm_thermo import SubRiemannianAnalyzer

sub_riem = SubRiemannianAnalyzer(alpha_star=1.41)
result = sub_riem.analyze_trajectory(hidden_states)
# Returns: K_sub (curvature), holonomic_ratio, R_squared, constraint_type
```

### 5. RMT Vorticity Analysis

```python
from llm_thermo import RMTVorticityAnalyzer

rmt = RMTVorticityAnalyzer(pca_dim=32)
result = rmt.compute_jacobian(hidden_states)
# Returns: J_vel, symmetric_part, antisymmetric_part, wigner_ks_test
```

### 6. Affine Compression (Ridge Optimization)

```python
from llm_thermo import RidgeOptimizer

optimizer = RidgeOptimizer(variance_threshold=0.95)

# Auto-find ridge dimension
ridge_info = optimizer.auto_find_ridge(hidden_states)  # [N, D]
print(f"Ridge dim: {ridge_info['ridge_dim']}, R²: {ridge_info['explained_variance']:.4f}")

# Auto-calibrate alpha*
alpha_info = optimizer.auto_calibrate_alpha([h_t2, h_t1, h_t])

# Monitor mode (recommended over replacement)
h_recon, metrics = optimizer.step_decode(h_curr)
print(f"P_c/P_raw: {metrics['pc_ratio']:.4f}, cosine_sim: {metrics['cosine_sim']:.4f}")
```

### 7. Quantization Guard

```python
from llm_thermo import QuantizationGuard

guard = QuantizationGuard(bf16_r_095=33, max_inflation_rate=0.5)
result = guard.check_rank_inflation(hidden_states)
if result["inflation_detected"]:
    print(f"⚠️ r(0.95) inflated by {result['inflation_rate']:.1%}")
```

## Architecture

```
llm_thermo/
├── config.py                          # Model presets (MiniCPM5, Qwen2.5, Ornith)
├── core/
│   ├── kinematics.py                  # Velocity/acceleration extraction + effective rank
│   ├── thermodynamics.py              # Thermodynamic engine + per-token P_c/P_raw
│   ├── ness.py                        # NESS 4-criteria evaluator
│   ├── sub_riemannian.py              # Sub-Riemannian geometry (K_sub, affine constraints)
│   └── rmt_vorticity.py               # RMT vorticity (Jacobian decomposition, Wigner test)
├── detection/
│   ├── phase_transition.py            # Hallucination detector (multi-signal fusion)
│   ├── alerts.py                      # Callback mechanism
│   └── quantization_guard.py          # Quantization robustness guard
├── steering/
│   ├── affine_projector.py            # Affine constraint projection
│   ├── kv_cache.py                    # Dynamic KV cache eviction
│   └── ridge_optimizer.py             # Auto ridge extraction + thermodynamic optimization
└── integrations/
    └── huggingface.py                 # ThermoHookManager (non-invasive HF integration)
```

## Core Physics

### Motion Equation

```
m·a(t) + γ·v(t) = α*·v(t) + F_c(t) + ξ(t)
```

| Symbol | Meaning | LLM Interpretation |
|--------|---------|-------------------|
| `v = h_t - h_{t-1}` | Velocity | Hidden state displacement per token |
| `a = v_t - v_{t-1}` | Acceleration | Velocity change (curvature signal) |
| `α*` | Active dissipation | Calibrated from valid trajectories |
| `F_c` | Constraint force | Force maintaining trajectory on manifold |
| `P_c = F_c · v` | Constraint power | ≈ 0 for valid, > 0 for hallucinated |
| `ξ` | Noise | Stochastic exploration |

### Key Validated Results

1. **P_c/P_raw ≈ 0.10-0.15** across 1B-9B models (affine constraint holds)
2. **Causal gradient**: pos < scr < rnd (constraint weakest for random text)
3. **NESS confirmed**: σ > 0, J_FP > 0, macroscopic detailed balance, CV < 0.5
4. **R² = 1.0** (perfect linear parameterization of constraint manifold)
5. **K_sub ≈ 0** (extremely flat constraint manifold)
6. **Prefill r(0.95) << Decode r(0.95)** (dimensional separation between phases)

### Important Methodology Note

**P_c/P_raw must be computed per-token, not batch-averaged.** Batch averaging produces ~0.84 (artifact from sign cancellation of P_c across tokens). Per-token averaging gives the correct value ~0.12.

```python
# ❌ WRONG (batch artifact ~0.84)
batch_ratio = |mean(P_c)| / |mean(P_raw)|

# ✅ CORRECT (per-token ~0.12)
per_token_ratio = mean(|P_c(t)| / |P_raw(t)|)
```

## Model Presets

```python
from llm_thermo import get_preset, list_presets

print(list_presets())
# {'minicpm5-1b': 'MiniCPM5-1B (1536D, α*=1.46)',
#  'qwen2.5-7b': 'Qwen2.5-7B-Instruct (3584D, α*=1.41)',
#  'ornith-1.0-9b': 'Ornith-1.0-9B (4096D, α*=1.41)'}

preset = get_preset("qwen2.5-7b")
engine = ThermodynamicEngine(alpha_star=preset.alpha_star)
```

## Quantization Impact

| Model | bf16 r(0.95) | 4-bit r(0.95) | Inflation |
|-------|-------------|---------------|-----------|
| MiniCPM5-1B | ~15 | ~15 | ~0% (1B immune) |
| Qwen2.5-7B | 32-34 | 40-44 | 25-30% |
| Ornith-1.0-9B | 23-25 | 37-43 | 50-64% |

P_c/P_raw remains robust under 4-bit quantization (~0.12 per-token). Ridge extraction MUST use bf16.

## Testing

```bash
pytest tests/ -v
# 56 tests: 32 core + 24 new modules
```

## Citation

If you use this library, please cite:

```bibtex
@article{guit_trt_2026,
  title={Non-Equilibrium Thermodynamic Active Vortices in Large Language Model Latent Spaces},
  author={GUIT-TRT Collaboration},
  year={2026},
  note={From Affine Constraints to Active Vorticity}
}
```

## License

MIT