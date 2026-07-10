"""
llm-thermodynamics: Non-Equilibrium Thermodynamics of LLM Latent Spaces

Based on GUIT-TRT (Grand Unified Information Theory - Thermodynamic Reasoning
Toolkit): From Affine Constraints to Active Vorticity
"""

__version__ = "0.3.0"

from .core.thermodynamics import ThermodynamicEngine
from .core.kinematics import KinematicsExtractor
from .core.ness import NESSEvaluator
from .core.sub_riemannian import SubRiemannianAnalyzer
from .core.rmt_vorticity import RMTVorticityAnalyzer
from .detection.phase_transition import HallucinationDetector, AlertLevel
from .detection.alerts import AlertCallback
from .detection.quantization_guard import QuantizationGuard
from .steering.affine_projector import AffineProjector
from .steering.kv_cache import DynamicKVCacheEvictor
from .steering.ridge_optimizer import RidgeOptimizer
from .integrations.huggingface import ThermoHookManager
from .config import ModelPreset, get_preset, list_presets, PRESETS