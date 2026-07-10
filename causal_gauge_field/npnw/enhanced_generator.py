import random
from typing import Dict, List, Optional, Tuple

from .world import NPNWWorld, ActionType, PersonalityType, WorldState
from .story_generator import Story, StoryStep


class EnhancedClosureGenerator:
    """更大、带真实闭环语义的语料 (实验7 用, §10.7 任务3).

    相对原 NPNW(单弧、极短、仅 key-door-treasure)的改进:
      - 叙事 = 出征(outbound) → 返回原点(return): 正例真正『闭环』(h_T≈h_0);
        负例出征后接一段发散尾(divergent tail), h_T 远离 h_0 (破缺闭环).
      - 正/负共享同一条出征路径 → 严格配对 (继承 exp5/6 的配对设计, 控制无关变量).
      - 人格一致性(勇敢/谨慎/贪婪)驱动出征动作选择, 更贴近叙事主体行为.
      - 数据量更大、序列更长 (由 runner 覆盖 config: max_stamina/max_seq_len).

    闭环语义: 『叙事闭环 ⇔ 回到起点(状态解析)』是 C-11 最自然的几何诠释,
    故用 return-home 作为正例的几何闭合标记. 这比原 NPNW 的 treasure 更接近
    『叙事完结』的几何意义, 且为 gauge field 提供可学习的几何区分.
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.world = NPNWWorld(config, seed=seed)
        self.rng = random.Random(seed + 7)
        self.min_steps = config["data"].get("enh_min_steps", 6)
        self.max_steps = config["data"].get("enh_max_steps", 9)
        self.grid_size = config["npnw"]["grid_size"]

    def _personality_filter(
        self, state: WorldState, legal: List[Tuple[ActionType, Dict]]
    ) -> List[Tuple[ActionType, Dict]]:
        p = state.character.personality
        s = state.character.stamina
        pref = []
        for at, params in legal:
            if p == PersonalityType.BRAVE:
                if at == ActionType.WAIT and s > 0:
                    continue
                pref.append((at, params))
            elif p == PersonalityType.CAUTIOUS:
                if at == ActionType.MOVE and s < 3:
                    continue
                pref.append((at, params))
            else:
                pref.append((at, params))
        return pref if pref else legal

    def _walk(self, state: WorldState, n_steps: int, rng: random.Random):
        steps: List[StoryStep] = []
        actions: List[Tuple[ActionType, Dict, Dict]] = []
        cur = self.world._copy_state(state)
        for _ in range(n_steps):
            if cur.step_count >= 30:
                break
            legal = self.world.get_legal_actions(cur)
            pref = self._personality_filter(cur, legal)
            at, params = rng.choice(pref)
            cur, record = self.world.execute_action(cur, at, params)
            steps.append(StoryStep(
                state=self._state_to_dict(cur),
                action=self._action_to_dict(at, params, cur),
                causal_labels=dict(record.causal_labels),
                physical_legal=record.physical_legal,
                narrative_legal=record.narrative_legal,
                psychological_legal=record.psychological_legal,
            ))
            actions.append((at, params, dict(record.causal_labels)))
        return cur, steps, actions

    def _reverse_actions(self, actions):
        rev = []
        for at, params, _cl in reversed(actions):
            if at == ActionType.MOVE:
                rev.append((ActionType.MOVE,
                            {"dx": -params.get("dx", 0), "dy": -params.get("dy", 0)}))
            else:
                rev.append((at, dict(params)))
        return rev

    def _apply_actions(self, state: WorldState, actions):
        steps: List[StoryStep] = []
        cur = self.world._copy_state(state)
        for at, params in actions:
            if at == ActionType.MOVE and cur.character.stamina <= 0:
                at, params = ActionType.WAIT, {}
            cur, record = self.world.execute_action(cur, at, params)
            steps.append(StoryStep(
                state=self._state_to_dict(cur),
                action=self._action_to_dict(at, params, cur),
                causal_labels=dict(record.causal_labels),
                physical_legal=record.physical_legal,
                narrative_legal=record.narrative_legal,
                psychological_legal=record.psychological_legal,
            ))
        return steps

    def generate_pair(self, story_id: int):
        personality = self.rng.choice(list(PersonalityType))
        state0 = self.world.generate_world(personality)
        n_out = self.rng.randint(self.min_steps, self.max_steps)
        end_state, out_steps, out_actions = self._walk(state0, n_out, self.rng)
        # 正例: 出征 + 原路返回 (闭环, h_T≈h_0)
        rev_actions = self._reverse_actions(out_actions)
        pos_return = self._apply_actions(end_state, rev_actions)
        pos_steps = out_steps + pos_return
        # 负例: 出征 + 一段发散尾 (不返回, 破缺, h_T 远离 h_0)
        div_rng = random.Random(story_id + 100003)
        _, neg_tail, _ = self._walk(
            end_state, self.rng.randint(self.min_steps, self.max_steps), div_rng)
        neg_steps = out_steps + neg_tail
        pos = Story(story_id=story_id, steps=pos_steps, personality=personality,
                    is_positive=True, violation_type=None)
        neg = Story(story_id=story_id, steps=neg_steps, personality=personality,
                    is_positive=False, violation_type="closure_broken")
        return pos, neg

    def generate_dataset(self, num_stories: int):
        pairs = []
        for i in range(num_stories):
            pos, neg = self.generate_pair(i)
            if pos is not None and neg is not None and len(pos.steps) >= 4:
                pairs.append((pos, neg))
        train_end = int(len(pairs) * self.config["data"]["train_ratio"])
        val_end = train_end + int(len(pairs) * self.config["data"]["val_ratio"])

        def split(pairs):
            return ([p for p, _ in pairs], [n for _, n in pairs])

        return split(pairs[:train_end]), split(pairs[train_end:val_end]), split(pairs[val_end:])

    def _state_to_dict(self, state: WorldState) -> Dict:
        return {
            "pos_x": state.character.pos_x,
            "pos_y": state.character.pos_y,
            "stamina": state.character.stamina,
            "holding": state.character.holding or "none",
            "key_seen": state.key_seen,
            "key_picked": state.key_picked,
            "door_opened": state.door_opened,
            "treasure_got": state.treasure_got,
            "step_count": state.step_count,
        }

    def _action_to_dict(self, at: ActionType, params: Dict, state: WorldState) -> Dict:
        return {
            "type": at.value,
            **params,
            "stamina": state.character.stamina,
            "holding": state.character.holding or "none",
        }
