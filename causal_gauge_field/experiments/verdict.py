from typing import Dict


class VerdictMatrix:
    SUPPORT = "SUPPORT"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    OPPOSE = "OPPOSE"
    STRONG_OPPOSE = "STRONG_OPPOSE"
    INCONCLUSIVE = "INCONCLUSIVE"

    MATRIX = {
        (SUPPORT, SUPPORT, SUPPORT, SUPPORT): "理论核心得到初步验证，可推进至更大规模实验",
        (SUPPORT, SUPPORT, OPPOSE, INCONCLUSIVE): "部分支持，需修订记忆机制部分",
        (SUPPORT, OPPOSE, INCONCLUSIVE, INCONCLUSIVE): "核心训练主张可能错误，理论需重大修正",
        (OPPOSE, OPPOSE, OPPOSE, OPPOSE): "理论核心被严重削弱，类比可能只是修辞",
        (STRONG_OPPOSE, STRONG_OPPOSE, STRONG_OPPOSE, STRONG_OPPOSE): "理论应被视为证伪，需重新审视基本假设",
    }

    def __init__(self):
        self.exp1_verdict = self.INCONCLUSIVE
        self.exp2_verdict = self.INCONCLUSIVE
        self.exp3_verdict = self.INCONCLUSIVE
        self.exp4_verdict = self.INCONCLUSIVE
        # C-13/C-14: H-geo / H-equiv 仅为候选隐喻(待证假设)，不计入理论核心成败
        self.hypothesis_note = (
            "H-geo / H-equiv 为候选隐喻（C-13/C-14），属待证假设；"
            "其获得支持与否不计入理论核心成败，仅作为后续研究方向。"
        )

    def update(self, exp_id: int, verdict: str):
        if exp_id == 1:
            self.exp1_verdict = verdict
        elif exp_id == 2:
            self.exp2_verdict = verdict
        elif exp_id == 3:
            self.exp3_verdict = verdict
        elif exp_id == 4:
            self.exp4_verdict = verdict

    def render(self) -> Dict:
        key = (
            self._normalize(self.exp1_verdict),
            self._normalize(self.exp2_verdict),
            self._normalize(self.exp3_verdict),
            self._normalize(self.exp4_verdict),
        )
        conclusion = self._compute_conclusion(key)
        return {
            "exp1_curvature_correlation": self.exp1_verdict,
            "exp2_causal_loss_impact": self.exp2_verdict,
            "exp3_memory_kernel": self.exp3_verdict,
            "exp4_flatness": self.exp4_verdict,
            "overall_conclusion": conclusion,
            "hypothesis_caveat": self.hypothesis_note,
        }

    def _normalize(self, verdict: str) -> str:
        if verdict in ("SUPPORT", "STRONG_SUPPORT"):
            return self.SUPPORT
        elif verdict in ("OPPOSE", "STRONG_OPPOSE"):
            return self.OPPOSE
        elif verdict == "WEAK_SUPPORT":
            return self.WEAK_SUPPORT
        return self.INCONCLUSIVE

    def _compute_conclusion(self, key: tuple) -> str:
        if key in self.MATRIX:
            return self.MATRIX[key]
        support_count = sum(1 for k in key if k == self.SUPPORT)
        oppose_count = sum(1 for k in key if k == self.OPPOSE)
        if support_count >= 3:
            return "理论核心获得较强支持，建议继续推进"
        elif support_count >= 2:
            return "理论部分获得支持，需针对性修正薄弱环节"
        elif oppose_count >= 3:
            return "理论核心被严重削弱，需重大修正或放弃"
        elif oppose_count >= 2:
            return "理论存在显著问题，需重新审视核心假设"
        else:
            return "结果不明确，需更多数据或改进实验设计"