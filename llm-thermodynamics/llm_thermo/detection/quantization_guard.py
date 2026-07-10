import torch
from typing import Dict, Optional
from ..core.kinematics import KinematicsExtractor


class QuantizationGuard:
    """Guard against quantization-induced metric inflation.

    Key findings from GUIT-TRT Phase 1:
    - 4-bit quantization inflates r(0.95) by 25-64%
      (Qwen2.5: 32->40, Ornith: 25->37)
    - P_c/P_raw remains robust under quantization (~0.12 per-token)
    - vel_norm may reverse gradient under 4-bit (Qwen2.5-7B)
    - Ridge extraction (RidgeOptimizer) MUST use bf16

    This guard monitors effective rank drift to detect when quantization
    is corrupting geometric measurements.
    """

    def __init__(
        self,
        bf16_r_095: Optional[int] = None,
        max_inflation_rate: float = 0.5,
    ):
        self.bf16_r_095 = bf16_r_095
        self.max_inflation_rate = max_inflation_rate

    def check_rank_inflation(
        self,
        hidden_states: torch.Tensor,
        variance_threshold: float = 0.95,
    ) -> Dict:
        """Check if effective rank is inflated by quantization.

        Args:
            hidden_states: [T, D] or [N, D] hidden states (current precision)
            variance_threshold: SVD variance threshold for r(0.95)

        Returns:
            Dict with inflation assessment
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {hidden_states.dim()}D")

        current_r = KinematicsExtractor.compute_effective_rank(
            hidden_states, threshold=variance_threshold
        )

        if self.bf16_r_095 is not None:
            inflation_rate = (current_r - self.bf16_r_095) / self.bf16_r_095
            is_inflated = inflation_rate > self.max_inflation_rate
            bf16_ref = self.bf16_r_095
        else:
            inflation_rate = None
            is_inflated = False
            bf16_ref = None

        return {
            "current_r_095": current_r,
            "bf16_reference_r_095": bf16_ref,
            "inflation_rate": inflation_rate,
            "is_inflated": is_inflated,
            "recommendation": (
                "Ridge extraction unreliable: use bf16 for RidgeOptimizer"
                if is_inflated
                else "P_c/P_raw robust under quantization; ridge extraction needs bf16"
            ),
            "safe_metrics": ["P_c/P_raw (per-token)", "fc_vel_cosine"],
            "unsafe_metrics_under_quantization": ["r(0.95)", "vel_norm (may reverse gradient)"],
        }

    @staticmethod
    def check_dtype(dtype: torch.dtype) -> Dict:
        """Check if the tensor dtype is safe for geometric measurements.

        Args:
            dtype: torch dtype of hidden states

        Returns:
            Dict with dtype safety assessment
        """
        is_bf16 = dtype in (torch.bfloat16, torch.float32, torch.float64)
        is_4bit = dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int4, torch.int8)

        if is_bf16:
            return {
                "dtype": str(dtype),
                "safe_for_ridge": True,
                "safe_for_pc_ratio": True,
                "safe_for_vel_norm": True,
                "recommendation": "Full precision: all metrics reliable",
            }
        else:
            return {
                "dtype": str(dtype),
                "safe_for_ridge": False,
                "safe_for_pc_ratio": True,
                "safe_for_vel_norm": False,
                "recommendation": "Quantized: P_c/P_raw robust, but r(0.95) inflated and vel_norm may reverse. Use bf16 for ridge extraction.",
            }