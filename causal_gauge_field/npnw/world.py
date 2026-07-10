import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ActionType(Enum):
    MOVE = "move"
    PICK_UP = "pick_up"
    USE = "use"
    PUSH = "push"
    WAIT = "wait"
    SEE = "see"
    GET_TREASURE = "get_treasure"


class PersonalityType(Enum):
    BRAVE = "brave"
    CAUTIOUS = "cautious"
    GREEDY = "greedy"


class CellType(Enum):
    EMPTY = "empty"
    WALL = "wall"
    DOOR = "door"
    KEY = "key"
    TREASURE = "treasure"


@dataclass
class Character:
    pos_x: int = 0
    pos_y: int = 0
    stamina: int = 5
    holding: Optional[str] = None
    personality: PersonalityType = PersonalityType.BRAVE


@dataclass
class ActionRecord:
    action_type: ActionType
    params: Dict = field(default_factory=dict)
    physical_legal: bool = True
    narrative_legal: bool = True
    psychological_legal: bool = True
    causal_labels: Dict = field(default_factory=dict)


@dataclass
class WorldState:
    grid: List[List[CellType]] = field(default_factory=list)
    character: Character = field(default_factory=Character)
    key_seen: bool = False
    key_picked: bool = False
    door_opened: bool = False
    treasure_got: bool = False
    step_count: int = 0


class NPNWWorld:
    def __init__(self, config: dict, seed: Optional[int] = None):
        self.grid_size = config["npnw"]["grid_size"]
        self.max_stamina = config["npnw"]["max_stamina"]
        self.stamina_recover = config["npnw"]["stamina_recover_wait"]
        self.personality_rules = {
            "brave_stamina_threshold": config["npnw"]["brave_stamina_threshold"],
            "cautious_stamina_threshold": config["npnw"]["cautious_stamina_threshold"],
            "greedy_pick_up_always": config["npnw"]["greedy_pick_up_always"],
        }
        self.rng = random.Random(seed)

    def generate_world(self, personality: Optional[PersonalityType] = None) -> WorldState:
        grid = [[CellType.EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        num_walls = self.rng.randint(2, 4)
        for _ in range(num_walls):
            wx = self.rng.randint(1, self.grid_size - 2)
            wy = self.rng.randint(1, self.grid_size - 2)
            if grid[wx][wy] == CellType.EMPTY:
                grid[wx][wy] = CellType.WALL
        key_pos = self._place_item(grid, CellType.KEY)
        door_pos = self._place_item(grid, CellType.DOOR)
        treasure_pos = self._place_item(grid, CellType.TREASURE)
        if personality is None:
            personality = self.rng.choice(list(PersonalityType))
        character = Character(
            pos_x=0,
            pos_y=0,
            stamina=self.max_stamina,
            holding=None,
            personality=personality,
        )
        return WorldState(
            grid=grid,
            character=character,
            key_seen=False,
            key_picked=False,
            door_opened=False,
            treasure_got=False,
            step_count=0,
        )

    def execute_action(self, state: WorldState, action_type: ActionType, params: Dict = None) -> Tuple[WorldState, ActionRecord]:
        params = params or {}
        new_state = self._copy_state(state)
        record = ActionRecord(action_type=action_type, params=params)
        if action_type == ActionType.MOVE:
            dx, dy = params.get("dx", 0), params.get("dy", 0)
            new_x = new_state.character.pos_x + dx
            new_y = new_state.character.pos_y + dy
            record.physical_legal = self._check_move_legal(new_state, new_x, new_y)
            record.narrative_legal = True
            record.psychological_legal = self._check_move_psychological(new_state, dx, dy)
            if record.physical_legal:
                new_state.character.pos_x = new_x
                new_state.character.pos_y = new_y
                new_state.character.stamina -= 1
                cell = new_state.grid[new_x][new_y]
                if cell == CellType.KEY and not new_state.key_seen:
                    new_state.key_seen = True
                    record.causal_labels["foreshadow_key"] = True
                if cell == CellType.DOOR:
                    record.causal_labels["encounter_door"] = True
                if cell == CellType.TREASURE:
                    new_state.treasure_got = True
                    record.causal_labels["closure"] = True
            record.causal_labels["physical_continuous"] = record.physical_legal
        elif action_type == ActionType.PICK_UP:
            record.physical_legal = self._check_pickup_legal(new_state)
            record.narrative_legal = True
            record.psychological_legal = self._check_pickup_psychological(new_state)
            if record.physical_legal:
                cx, cy = new_state.character.pos_x, new_state.character.pos_y
                if new_state.grid[cx][cy] == CellType.KEY:
                    new_state.character.holding = "key"
                    new_state.key_picked = True
                    new_state.grid[cx][cy] = CellType.EMPTY
                    record.causal_labels["pickup_key"] = True
        elif action_type == ActionType.USE:
            target = params.get("target", "")
            record.physical_legal = self._check_use_legal(new_state, target)
            record.narrative_legal = new_state.key_picked if target == "door" else True
            record.psychological_legal = True
            if record.physical_legal and record.narrative_legal:
                if target == "door":
                    new_state.door_opened = True
                    record.causal_labels["foreshadow_recover"] = True
        elif action_type == ActionType.WAIT:
            new_state.character.stamina = min(
                new_state.character.stamina + self.stamina_recover,
                self.max_stamina,
            )
            record.physical_legal = True
            record.narrative_legal = True
            record.psychological_legal = self._check_wait_psychological(new_state)
        new_state.step_count += 1
        return new_state, record

    def get_legal_actions(self, state: WorldState) -> List[Tuple[ActionType, Dict]]:
        actions = []
        cx, cy = state.character.pos_x, state.character.pos_y
        if state.character.stamina > 0:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if state.grid[nx][ny] != CellType.WALL:
                        actions.append((ActionType.MOVE, {"dx": dx, "dy": dy}))
        cell = state.grid[cx][cy]
        if cell == CellType.KEY and state.character.holding is None and state.character.stamina > 0:
            actions.append((ActionType.PICK_UP, {}))
        if state.character.holding == "key":
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if state.grid[nx][ny] == CellType.DOOR:
                        actions.append((ActionType.USE, {"target": "door"}))
        actions.append((ActionType.WAIT, {}))
        return actions

    def is_terminal(self, state: WorldState) -> bool:
        return state.treasure_got or state.step_count >= 30

    def _place_item(self, grid: List[List[CellType]], item: CellType) -> Tuple[int, int]:
        while True:
            x = self.rng.randint(0, self.grid_size - 1)
            y = self.rng.randint(0, self.grid_size - 1)
            if grid[x][y] == CellType.EMPTY and (x, y) != (0, 0):
                grid[x][y] = item
                return (x, y)

    def _check_move_legal(self, state: WorldState, new_x: int, new_y: int) -> bool:
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            return False
        if state.grid[new_x][new_y] == CellType.WALL:
            return False
        if state.character.stamina <= 0:
            return False
        return True

    def _check_move_psychological(self, state: WorldState, dx: int, dy: int) -> bool:
        p = state.character.personality
        s = state.character.stamina
        if p == PersonalityType.BRAVE:
            return True
        if p == PersonalityType.CAUTIOUS:
            if s < self.personality_rules["cautious_stamina_threshold"]:
                return False
        return True

    def _check_pickup_legal(self, state: WorldState) -> bool:
        cx, cy = state.character.pos_x, state.character.pos_y
        if state.grid[cx][cy] != CellType.KEY:
            return False
        if state.character.holding is not None:
            return False
        if state.character.stamina <= 0:
            return False
        return True

    def _check_pickup_psychological(self, state: WorldState) -> bool:
        p = state.character.personality
        if p == PersonalityType.GREEDY:
            return True
        return True

    def _check_use_legal(self, state: WorldState, target: str) -> bool:
        if target == "door":
            if state.character.holding != "key":
                return False
            cx, cy = state.character.pos_x, state.character.pos_y
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if state.grid[nx][ny] == CellType.DOOR:
                        return True
            return False
        return True

    def _check_wait_psychological(self, state: WorldState) -> bool:
        p = state.character.personality
        s = state.character.stamina
        if p == PersonalityType.BRAVE:
            if s > self.personality_rules["brave_stamina_threshold"]:
                return False
        return True

    def _copy_state(self, state: WorldState) -> WorldState:
        import copy
        return copy.deepcopy(state)