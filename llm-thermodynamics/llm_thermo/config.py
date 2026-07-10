from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ModelPreset:
    """Pre-calibrated thermodynamic parameters for a specific LLM.

    All values are from GUIT-TRT Phase 1 experimental validation.
    """
    name: str
    architecture: str
    hidden_dim: int
    n_layers: int
    kv_heads: int
    alpha_star: float
    gamma: float = 0.01
    mass: float = 1.0
    r_095_bf16: Optional[int] = None
    r_095_4bit: Optional[int] = None
    attention_type: str = "full"
    pc_raw_ratio_pos: Optional[float] = None
    pc_raw_ratio_scr: Optional[float] = None
    pc_raw_ratio_rnd: Optional[float] = None
    notes: str = ""


PRESETS: Dict[str, ModelPreset] = {
    "minicpm5-1b": ModelPreset(
        name="MiniCPM5-1B",
        architecture="LlamaForCausalLM",
        hidden_dim=1536,
        n_layers=24,
        kv_heads=2,
        alpha_star=1.46,
        r_095_bf16=15,
        attention_type="full (GQA)",
        pc_raw_ratio_pos=0.140,
        pc_raw_ratio_scr=0.149,
        pc_raw_ratio_rnd=0.144,
        notes="v22: 20-prompt per-token P_c/P_raw; prefill r(0.95)=6, decode r(0.95)=15",
    ),
    "qwen2.5-7b": ModelPreset(
        name="Qwen2.5-7B-Instruct",
        architecture="Qwen2ForCausalLM",
        hidden_dim=3584,
        n_layers=28,
        kv_heads=4,
        alpha_star=1.41,
        r_095_bf16=10,
        r_095_4bit=42,
        attention_type="full (GQA)",
        pc_raw_ratio_pos=0.109,
        pc_raw_ratio_scr=0.100,
        pc_raw_ratio_rnd=0.112,
        notes="v22: 5-prompt per-token P_c/P_raw; vel_norm reverses under 4-bit; prefill/decode r(0.95) gap small (0.8-1.0x)",
    ),
    "ornith-1.0-9b": ModelPreset(
        name="Ornith-1.0-9B",
        architecture="Qwen3_5ForConditionalGeneration",
        hidden_dim=4096,
        n_layers=32,
        kv_heads=4,
        alpha_star=1.41,
        r_095_bf16=25,
        r_095_4bit=40,
        attention_type="mixed (24 linear + 8 full, GQA)",
        pc_raw_ratio_pos=0.09,
        pc_raw_ratio_scr=None,
        pc_raw_ratio_rnd=0.15,
        notes="First coverage of mixed attention architecture; P_c/P_raw slightly elevated but affine constraint holds; lower r(0.95) than Qwen2.5 despite larger d",
    ),
}


def get_preset(model_id: str) -> ModelPreset:
    """Get model preset by ID (case-insensitive partial match)."""
    key = model_id.lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, v in PRESETS.items():
        if k.replace("-", "").replace("_", "") in key or key in k.replace("-", "").replace("_", ""):
            return v
    raise ValueError(
        f"Unknown model preset '{model_id}'. Available: {list(PRESETS.keys())}"
    )


def list_presets() -> Dict[str, str]:
    """List all available model presets with brief descriptions."""
    return {k: f"{v.name} ({v.hidden_dim}D, α*={v.alpha_star})" for k, v in PRESETS.items()}