import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .world import NPNWWorld, ActionType, PersonalityType, WorldState, ActionRecord
from .tokenizer import NPNWTokenizer


@dataclass
class StoryStep:
    state: Dict
    action: Dict
    causal_labels: Dict
    physical_legal: bool = True
    narrative_legal: bool = True
    psychological_legal: bool = True


@dataclass
class Story:
    story_id: int
    steps: List[StoryStep]
    personality: PersonalityType
    is_positive: bool = True
    violation_type: Optional[str] = None
    causal_graph: Dict = field(default_factory=dict)


# 刚柔分层 (论文 §3.7): 每步归属的 物理/柔性 层 与 违规标记, 供分层损失使用.
def step_layer_kind(step: "StoryStep") -> int:
    """0=物理(刚性)  1=叙事  2=心理/柔性默认."""
    cl = step.causal_labels or {}
    if cl.get("physical_continuous"):
        return 0
    if any(cl.get(k) for k in ("foreshadow_key", "pickup_key",
                               "encounter_door", "foreshadow_recover", "closure")):
        return 1
    return 2


def step_is_violated(step: "StoryStep") -> bool:
    return (step.physical_legal is False) or (step.narrative_legal is False) \
        or (step.psychological_legal is False)


class StoryGenerator:
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.world = NPNWWorld(config, seed=seed)
        self.tokenizer = NPNWTokenizer()
        self.rng = random.Random(seed + 1)
        self.min_steps = config["data"]["min_steps"]
        self.max_steps = config["data"]["max_steps"]

    def generate_positive_story(self, story_id: int) -> Optional[Story]:
        personality = self.rng.choice(list(PersonalityType))
        state = self.world.generate_world(personality)
        steps = []
        visited = set()
        max_attempts = 100
        for attempt in range(max_attempts):
            state = self.world.generate_world(personality)
            steps = []
            visited = set()
            for _ in range(self.max_steps):
                if self.world.is_terminal(state):
                    break
                legal_actions = self.world.get_legal_actions(state)
                if not legal_actions:
                    break
                preferred = self._personality_filter(state, legal_actions)
                action_type, params = self._choose_action(state, preferred, legal_actions)
                state, record = self.world.execute_action(state, action_type, params)
                step = StoryStep(
                    state=self._state_to_dict(state),
                    action=self._action_to_dict(action_type, params, state),
                    causal_labels=record.causal_labels,
                    physical_legal=record.physical_legal,
                    narrative_legal=record.narrative_legal,
                    psychological_legal=record.psychological_legal,
                )
                steps.append(step)
                visited.add((state.character.pos_x, state.character.pos_y))
            if len(steps) >= self.min_steps or state.treasure_got:
                break
        if not steps:
            return None
        causal_graph = self._build_causal_graph(steps)
        return Story(
            story_id=story_id,
            steps=steps,
            personality=personality,
            is_positive=True,
            violation_type=None,
            causal_graph=causal_graph,
        )

    def generate_negative_story(
        self, positive: Story, violation_type: str
    ) -> Story:
        steps = []
        for i, step in enumerate(positive.steps):
            if violation_type == "physical" and self._should_violate(i, len(positive.steps)):
                neg_step = self._violate_physical(step)
                steps.append(neg_step)
            elif violation_type == "narrative" and self._should_violate(i, len(positive.steps)):
                neg_step = self._violate_narrative(step)
                steps.append(neg_step)
            elif violation_type == "psychological" and self._should_violate(i, len(positive.steps)):
                neg_step = self._violate_psychological(step, positive.personality)
                steps.append(neg_step)
            else:
                steps.append(step)
        return Story(
            story_id=positive.story_id,
            steps=steps,
            personality=positive.personality,
            is_positive=False,
            violation_type=violation_type,
            causal_graph=positive.causal_graph,
        )

    def generate_dataset(
        self, num_stories: int, neg_per_positive: int = 3
    ) -> Tuple[List[Story], List[Story], List[Story]]:
        positives = []
        for i in range(num_stories):
            story = self.generate_positive_story(i)
            if story is not None:
                positives.append(story)
        negatives = []
        violation_types = ["physical", "narrative", "psychological"]
        for pos in positives:
            for _ in range(neg_per_positive):
                vtype = self.rng.choice(violation_types)
                neg = self.generate_negative_story(pos, vtype)
                negatives.append(neg)
        train_end = int(len(positives) * self.config["data"]["train_ratio"])
        val_end = train_end + int(len(positives) * self.config["data"]["val_ratio"])
        train_pos = positives[:train_end]
        val_pos = positives[train_end:val_end]
        test_pos = positives[val_end:]
        train_neg = negatives[: train_end * neg_per_positive]
        val_neg = negatives[train_end * neg_per_positive : val_end * neg_per_positive]
        test_neg = negatives[val_end * neg_per_positive :]
        return (train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg)

    def _personality_filter(
        self, state: WorldState, legal_actions: List[Tuple[ActionType, Dict]]
    ) -> List[Tuple[ActionType, Dict]]:
        p = state.character.personality
        s = state.character.stamina
        preferred = []
        for act_type, params in legal_actions:
            if p == PersonalityType.BRAVE:
                if act_type == ActionType.WAIT and s > 0:
                    continue
                preferred.append((act_type, params))
            elif p == PersonalityType.CAUTIOUS:
                if act_type == ActionType.MOVE and s < 3:
                    continue
                preferred.append((act_type, params))
            elif p == PersonalityType.GREEDY:
                preferred.append((act_type, params))
            else:
                preferred.append((act_type, params))
        return preferred if preferred else legal_actions

    def _choose_action(
        self,
        state: WorldState,
        preferred: List[Tuple[ActionType, Dict]],
        legal: List[Tuple[ActionType, Dict]],
    ) -> Tuple[ActionType, Dict]:
        if not state.key_picked:
            for act_type, params in preferred:
                if act_type == ActionType.MOVE:
                    dx, dy = params.get("dx", 0), params.get("dy", 0)
                    nx = state.character.pos_x + dx
                    ny = state.character.pos_y + dy
                    if 0 <= nx < self.world.grid_size and 0 <= ny < self.world.grid_size:
                        cell = state.grid[nx][ny]
                        if cell.value in ("key", "door", "treasure"):
                            return act_type, params
                elif act_type == ActionType.PICK_UP:
                    return act_type, params
        elif not state.door_opened:
            for act_type, params in preferred:
                if act_type == ActionType.USE and params.get("target") == "door":
                    return act_type, params
        if preferred:
            return self.rng.choice(preferred)
        return self.rng.choice(legal)

    def _should_violate(self, step_idx: int, total_steps: int) -> bool:
        return self.rng.random() < 0.3

    def _violate_physical(self, step: StoryStep) -> StoryStep:
        violated = StoryStep(
            state=dict(step.state),
            action=dict(step.action),
            causal_labels=dict(step.causal_labels),
            physical_legal=False,
            narrative_legal=step.narrative_legal,
            psychological_legal=step.psychological_legal,
        )
        act_type = step.action.get("type", "")
        if act_type == "move":
            violated.action["dx"] = self.rng.choice([-2, 2])
            violated.action["dy"] = self.rng.choice([-2, 2])
            violated.state["stamina"] = 0
        elif act_type == "pick_up":
            violated.state["stamina"] = 0
        elif act_type == "use":
            violated.action["holding_key"] = False
        return violated

    def _violate_narrative(self, step: StoryStep) -> StoryStep:
        violated = StoryStep(
            state=dict(step.state),
            action=dict(step.action),
            causal_labels=dict(step.causal_labels),
            physical_legal=step.physical_legal,
            narrative_legal=False,
            psychological_legal=step.psychological_legal,
        )
        if "pickup_key" in step.causal_labels:
            violated.causal_labels["pickup_key"] = False
            violated.action["type"] = "move"
        if "foreshadow_recover" in step.causal_labels:
            violated.causal_labels["foreshadow_recover"] = False
            violated.action["target"] = "wall"
        return violated

    def _violate_psychological(self, step: StoryStep, personality: PersonalityType) -> StoryStep:
        violated = StoryStep(
            state=dict(step.state),
            action=dict(step.action),
            causal_labels=dict(step.causal_labels),
            physical_legal=step.physical_legal,
            narrative_legal=step.narrative_legal,
            psychological_legal=False,
        )
        act_type = step.action.get("type", "")
        stamina = step.state.get("stamina", 5)
        if personality == PersonalityType.BRAVE and act_type != "wait":
            violated.action["type"] = "wait"
        elif personality == PersonalityType.CAUTIOUS and act_type == "move" and stamina < 3:
            violated.action["type"] = "move"
            violated.action["dx"] = 2
        elif personality == PersonalityType.GREEDY:
            violated.action["type"] = "wait"
        return violated

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

    def _action_to_dict(self, action_type: ActionType, params: Dict, state: WorldState) -> Dict:
        return {
            "type": action_type.value,
            **params,
            "stamina": state.character.stamina,
            "holding": state.character.holding or "none",
        }

    def _build_causal_graph(self, steps: List[StoryStep]) -> Dict:
        graph = {"physical_edges": [], "narrative_edges": [], "psychological_edges": []}
        key_step = None
        door_step = None
        treasure_step = None
        for i, step in enumerate(steps):
            if step.causal_labels.get("foreshadow_key"):
                key_step = i
            if step.causal_labels.get("pickup_key"):
                if key_step is not None:
                    graph["narrative_edges"].append((key_step, i, "key_seen_to_pickup"))
            if step.causal_labels.get("encounter_door"):
                door_step = i
            if step.causal_labels.get("foreshadow_recover"):
                if key_step is not None:
                    graph["narrative_edges"].append((key_step, i, "key_to_door"))
            if step.causal_labels.get("closure"):
                treasure_step = i
                if door_step is not None:
                    graph["narrative_edges"].append((door_step, i, "door_to_treasure"))
            if i > 0:
                graph["physical_edges"].append((i - 1, i, "state_transition"))
                graph["psychological_edges"].append((i - 1, i, "motivation_continuity"))
        return graph