"""PhysicsInformedLoop — 物理感知自优化循环核心实现。

对齐套件真实 API (与文档示例的主要差异修正):
  - OrnithGuard 用 (alpha_star, gamma, pc_threshold, ...) 构造, 不含 model/tokenizer
  - update_state_machine 接收 token 文本 (str), 非 token id
  - 生成后端抽象为 GenerationBackend, 解耦具体模型, 支持无 GPU 仿真测试

核心流程 (双轨制闭环):
  run(task):
    for i in range(max_iterations+1):
      gen = generate_with_guard(prompt)          # 微观物理轨: 事中熔断
      evaluation = evaluate_with_physics(gen)    # 融合物理 + 语义反馈
      if evaluation.qualified: break             # 收敛
      prompt = revise_with_physics(...)          # 宏观语义轨: 定向修正
    online_calibrate(good_trajectory)            # 元层进化
"""

from __future__ import annotations

from typing import (Callable, Dict, List, Optional, Tuple, Iterator, Any)
from dataclasses import dataclass, field
import torch

from ..detection.ornith_guard import OrnithGuard, PhaseTransitionException


# =====================================================================
# 生成后端协议
# =====================================================================
class GenerationBackend:
    """生成后端抽象接口。

    stream(prompt, max_new_tokens) 逐 token 产出 (token_text, hidden_state)。
    hidden_state 为该 decode 步最后一层隐状态 [D] (float tensor)。
    真实部署实现 HFGenerationBackend; 测试用 SimulatedCodingBackend。
    """

    def stream(
        self, prompt: str, max_new_tokens: int
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError


# =====================================================================
# 生成结果
# =====================================================================
@dataclass
class GenerationResult:
    text: str
    hidden_states: List[torch.Tensor]
    tokens_generated: int
    max_new_tokens: int
    halted: bool                         # 是否被物理熔断截断
    physics_feedback: Optional[Dict] = None

    @property
    def tokens_saved(self) -> int:
        """相对"生成到底"策略节省的 token 数 (仅熔断时 > 0)。"""
        return (self.max_new_tokens - self.tokens_generated) if self.halted else 0


# =====================================================================
# PI-LOOP
# =====================================================================
class PhysicsInformedLoop:
    def __init__(
        self,
        backend: GenerationBackend,
        alpha_star: float = 1.41,
        gamma: float = 0.01,
        pc_threshold: float = 0.08,
        max_iterations: int = 3,
        quality_threshold: float = 8.5,
        calibrator: Optional[Any] = None,          # OrnithAutoCalibrator
        semantic_evaluator: Optional[Callable[[str, str], Dict]] = None,
        guard_kwargs: Optional[Dict] = None,
    ):
        self.backend = backend
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.calibrator = calibrator
        self.semantic_evaluator = semantic_evaluator

        gk = guard_kwargs or {}
        self.guard = OrnithGuard(
            alpha_star=alpha_star, gamma=gamma,
            pc_threshold=pc_threshold, **gk,
        )

        self.trace: Dict[str, Any] = {
            "iterations": [],
            "physics_interventions": 0,
            "total_tokens_generated": 0,
            "total_tokens_saved": 0,
            "converged": False,
            "final_iteration": -1,
        }

    # -----------------------------------------------------------------
    def generate_with_guard(
        self, prompt: str, max_new_tokens: int = 512,
        in_test_block: bool = True,
    ) -> GenerationResult:
        """带物理熔断的生成: Decode 过程中一旦相变立即截断。"""
        self.guard._reset_thermo_state()
        # 若上层已用 <test> 标签驱动状态机, 则不强制置位
        self.guard.in_test_block = in_test_block

        hidden: List[torch.Tensor] = []
        text_parts: List[str] = []
        halted = False
        feedback: Optional[Dict] = None
        n = 0
        try:
            for token_text, h in self.backend.stream(prompt, max_new_tokens):
                n += 1
                self.guard.update_state_machine(token_text)
                text_parts.append(token_text)
                h = h.detach().float().reshape(-1)
                hidden.append(h)
                self.guard.process_step(h)   # 相变时抛异常
        except PhaseTransitionException as e:
            halted = True
            self.trace["physics_interventions"] += 1
            partial = "".join(text_parts)
            feedback = {
                "type": "DYNAMICS_PHASE_TRANSITION",
                "pc_ratio": e.pc_ratio,
                "breakpoint_text": partial[-64:],
                "message": "底层物理探针检测到隐状态脱离因果流形, 逻辑链条已崩溃。",
            }
        finally:
            self.guard.in_test_block = False

        return GenerationResult(
            text="".join(text_parts),
            hidden_states=hidden,
            tokens_generated=n,
            max_new_tokens=max_new_tokens,
            halted=halted,
            physics_feedback=feedback,
        )

    # -----------------------------------------------------------------
    def evaluate_with_physics(
        self, task: str, gen: GenerationResult
    ) -> Dict:
        """融合物理熔断反馈 (最高优先级) 与语义评估。"""
        if gen.physics_feedback is not None:
            fb = gen.physics_feedback
            return {
                "overall_score": 2.0,
                "is_qualified": False,
                "source": "physics",
                "issues": [{
                    "dimension": "logic_cohesion",
                    "position": f"...{fb['breakpoint_text']}",
                    "description": (f"【物理级致命错误】{fb['message']} "
                                    f"(P_c/P_raw={fb['pc_ratio']:.3f})。"
                                    "断裂点之后的代码均为无效幻觉。"),
                    "severity": "critical",
                }],
                "revision_suggestions": [
                    "放弃断裂点之后的所有代码。",
                    "回到断裂点前重新审视变量状态与逻辑分支, 重构实现路径。",
                ],
            }
        # 无物理熔断 → 语义评估 (真实场景=沙盒执行/LLM 自评)
        if self.semantic_evaluator is not None:
            ev = self.semantic_evaluator(task, gen.text)
            ev.setdefault("source", "semantic")
            ev.setdefault("is_qualified",
                          ev.get("overall_score", 0) >= self.quality_threshold)
            ev.setdefault("revision_suggestions", [])
            return ev
        # 无评估器 → 默认通过 (无物理相变即视为合格)
        return {"overall_score": 9.0, "is_qualified": True,
                "source": "default", "issues": [], "revision_suggestions": []}

    # -----------------------------------------------------------------
    def revise_with_physics(
        self, task: str, gen: GenerationResult, evaluation: Dict
    ) -> str:
        """构造物理感知定向修正 Prompt。"""
        sug = evaluation.get("revision_suggestions") or ["重写"]
        issues = evaluation.get("issues") or []
        pos = issues[0]["position"] if issues else ""
        return (
            f"【任务】{task}\n"
            f"【上一版问题定位】{pos}\n"
            f"【修正指令】{' '.join(sug)}\n"
            f"要求: 删除断裂点后的无效代码, 保留正确上下文, 从断裂点重构, "
            f"确保因果逻辑严密, 避免再次触发物理相变。"
        )

    # -----------------------------------------------------------------
    def _online_calibrate(self, gen: GenerationResult):
        """元层进化: 用收敛的优质轨迹在线更新脊线基底。"""
        if self.calibrator is None or len(gen.hidden_states) < 4:
            return
        traj = torch.stack(gen.hidden_states, dim=0)   # [T, D]
        if hasattr(self.calibrator, "update_online"):
            self.calibrator.update_online([traj])
        elif hasattr(self.calibrator, "calibrate"):
            self.calibrator.calibrate([traj])

    # -----------------------------------------------------------------
    def run(self, task: str, max_new_tokens: int = 512) -> Tuple[str, Dict]:
        """执行 PI-LOOP 主循环, 返回 (最终代码, trace)。"""
        prompt = task
        final_output = ""
        last_gen: Optional[GenerationResult] = None

        for i in range(self.max_iterations + 1):
            gen = self.generate_with_guard(prompt, max_new_tokens)
            last_gen = gen
            final_output = gen.text
            self.trace["total_tokens_generated"] += gen.tokens_generated
            self.trace["total_tokens_saved"] += gen.tokens_saved

            evaluation = self.evaluate_with_physics(task, gen)
            self.trace["iterations"].append({
                "iter": i,
                "tokens_generated": gen.tokens_generated,
                "tokens_saved": gen.tokens_saved,
                "halted": gen.halted,
                "source": evaluation.get("source"),
                "score": evaluation.get("overall_score"),
                "qualified": evaluation.get("is_qualified"),
            })

            if evaluation.get("is_qualified"):
                self.trace["converged"] = True
                self.trace["final_iteration"] = i
                self._online_calibrate(gen)   # 元层进化
                break

            if i == self.max_iterations:
                self.trace["final_iteration"] = i
                break

            prompt = self.revise_with_physics(task, gen, evaluation)

        return final_output, self.trace


# =====================================================================
# 仿真生成后端 (无 GPU 可测)
# =====================================================================
class SimulatedCodingBackend(GenerationBackend):
    """用 OrnithLatentSimulator 仿真"代码生成"的隐状态流。

    模型"能力"随修正迭代提升: 每次 stream 被调用, flaw 出现得更晚,
    直到某轮不再出现 flaw (收敛)。用于对比 PI-LOOP (事中熔断) 与
    传统事后 LOOP (生成到底) 的 token 消耗与收敛速度。
    """

    def __init__(
        self,
        simulator,
        flaw_start: int = 12,
        flaw_len: int = 20,
        fix_after: int = 2,          # 第几次修正后彻底修好
        emit_test_tags: bool = True,
    ):
        self.sim = simulator
        self.flaw_start = flaw_start
        self.flaw_len = flaw_len
        self.fix_after = fix_after
        self.emit_test_tags = emit_test_tags
        self._call = 0

    def stream(
        self, prompt: str, max_new_tokens: int
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        has_flaw = self._call < self.fix_after
        self._call += 1

        sim = self.sim
        z = torch.randn(sim.r, generator=sim.gen) * 0.5
        pos_sigma = sim.sigma_off["pos"]
        o_prev = torch.randn(sim.D - sim.r, generator=sim.gen) * pos_sigma
        length = max_new_tokens
        if self.emit_test_tags:
            yield "<test>", (sim.mu + sim.Vr @ z).detach()

        for t in range(length):
            in_flaw = (has_flaw and self.flaw_start <= t
                       < self.flaw_start + self.flaw_len)
            if in_flaw:
                # 持续离流形漂移 (与 simulator.generate_trajectory 的 halluc 一致):
                # 冻结脊线旋转, 注入缓慢累积的大幅 off-ridge 分量,
                # 使约束力做功 P_c 持续飙升 → 可靠触发物理熔断。
                o = o_prev + 1.5 * torch.randn(sim.D - sim.r, generator=sim.gen)
                h = sim.mu + sim.Vr @ z + sim.Vp @ (o * 4.0)
                token = f"BAD_{t}"
            else:
                z = sim._step_latent(z, pos_sigma)
                o = torch.randn(sim.D - sim.r, generator=sim.gen) * pos_sigma
                h = sim.mu + sim.Vr @ z + sim.Vp @ o
                token = f"tok_{t}"
            o_prev = o
            yield token, h.detach()

        if self.emit_test_tags:
            yield "</test>", (sim.mu + sim.Vr @ z).detach()
