import logging
from typing import Callable, Optional

import torch

from ..core.thermodynamics import ThermodynamicEngine
from ..detection.phase_transition import HallucinationDetector, AlertLevel
from ..detection.alerts import AlertCallback

logger = logging.getLogger(__name__)


class ThermoHookManager:
    """Non-invasive HuggingFace Transformers integration for thermodynamic monitoring.

    Registers forward hooks on specified transformer layers to extract hidden
    states during generation, feeding them into the HallucinationDetector.

    Usage:
        engine = ThermodynamicEngine(alpha_star=1.41)
        detector = HallucinationDetector(engine, threshold_ratio=0.08)
        hook_mgr = ThermoHookManager(model, detector)
        hook_mgr.register_hooks(layer_index=-1)

        outputs = hook_mgr.generate_with_guardrail(
            model, input_ids, max_new_tokens=50,
            on_hallucination_callback=lambda r: print("Warning!", r)
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        detector: HallucinationDetector,
        alert_callback: Optional[AlertCallback] = None,
    ):
        self.model = model
        self.detector = detector
        self.alert_callback = alert_callback
        self._hooks = []
        self._last_result: Optional[dict] = None

    def register_hooks(self, layer_index: int = -1):
        """Register forward hook on the specified transformer layer.

        Args:
            layer_index: Layer index to hook. -1 = last layer.
        """
        self.remove_hooks()
        self.detector.reset()

        target_layer = self._resolve_layer(layer_index)
        hook = target_layer.register_forward_hook(self._forward_hook_fn)
        self._hooks.append(hook)
        logger.info(f"Registered thermodynamic hook on {type(target_layer).__name__}")

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _resolve_layer(self, layer_index: int) -> torch.nn.Module:
        if not hasattr(self.model, "model"):
            raise ValueError("Model does not have expected .model attribute")

        inner = self.model.model
        layers = None

        for attr in ("layers", "encoder", "decoder", "blocks"):
            if hasattr(inner, attr):
                layers = getattr(inner, attr)
                break

        if layers is None:
            raise ValueError("Cannot find transformer layers in model structure")

        if layer_index < 0:
            layer_index = len(layers) + layer_index

        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"Layer index {layer_index} out of range [0, {len(layers)})")

        return layers[layer_index]

    def _forward_hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output

        if hidden_state.dim() == 3:
            h_t = hidden_state[:, -1, :]
        elif hidden_state.dim() == 2:
            h_t = hidden_state
        else:
            return

        h_t = h_t.detach().float()

        result = self.detector.step(h_t)
        self._last_result = result

        if self.alert_callback and result["alert_level"] in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            self.alert_callback.notify(result["alert_level"], result)

    def generate_with_guardrail(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        on_hallucination_callback: Optional[Callable[[dict], None]] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate with automatic hallucination detection.

        If on_hallucination_callback is provided, it will be called when
        hallucination is detected. Generation continues regardless.
        """
        self.detector.reset()
        self._last_result = None

        if on_hallucination_callback:
            if self.alert_callback is None:
                self.alert_callback = AlertCallback()
            self.alert_callback.on_critical(on_hallucination_callback)

        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        finally:
            if on_hallucination_callback and self.alert_callback is not None:
                self.alert_callback._callbacks[AlertLevel.CRITICAL].clear()

        return outputs

    @property
    def last_result(self) -> Optional[dict]:
        return self._last_result