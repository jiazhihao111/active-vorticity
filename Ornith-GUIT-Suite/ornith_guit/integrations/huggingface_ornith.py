"""HuggingFace Transformers 无侵入式集成 (Ornith 专属)。

3 行激活热力学监控: 在指定层注册前向 Hook, 捕获隐状态并交给
OrnithGuard / HallucinationDetector 进行实时相变监控与回退。

注意: 真实 Ornith 部署时使用; 本套件测试用 simulator 直接驱动组件。
"""

from typing import Optional, Callable
import torch


class ThermoHookManager:
    def __init__(self, model, alpha_star: float = 1.41, layer_index: int = -1,
                 on_phase_transition: Optional[Callable] = None):
        self.model = model
        self.alpha_star = alpha_star
        self.layer_index = layer_index
        self.callback = on_phase_transition
        self.hook = None
        self._h_prev2 = None
        self._h_prev = None

    def _make_hook(self):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() != 3 or h.shape[1] != 1:
                return  # 仅 decode 单步
            hc = h.squeeze(1).detach().float()
            if self._h_prev2 is not None and self._h_prev is not None:
                from ..physics import ThermoPhysics
                eng = ThermoPhysics(alpha_star=self.alpha_star)
                pc, vn = eng.pc_ratio(hc, self._h_prev, self._h_prev2)
                if pc > 0.08 and self.callback is not None:
                    self.callback(pc, vn)
            self._h_prev2 = self._h_prev
            self._h_prev = hc
        return hook_fn

    def register_hooks(self):
        layers = self.model.model.layers
        target = layers[self.layer_index] if self.layer_index < 0 else layers[self.layer_index]
        self.hook = target.register_forward_hook(self._make_hook())

    def generate_with_guardrail(self, *args, **kwargs):
        self.register_hooks()
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self.close()

    def close(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
