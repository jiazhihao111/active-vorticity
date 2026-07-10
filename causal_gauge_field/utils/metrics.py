import numpy as np
from typing import List, Tuple


def physical_legal_rate(
    actions: List[dict],
    world_rules: dict,
) -> float:
    if not actions:
        return 0.0
    legal_count = 0
    for act in actions:
        if _check_physical_legal(act, world_rules):
            legal_count += 1
    return legal_count / len(actions)


def narrative_closure_rate(
    stories: List[List[dict]],
) -> float:
    # B-06/C-11: 启发式叙事闭环度量。论文主张用 Wilson 环量 W(γ) 的 Var→0
    # 作为几何闭环判据，但 Var[W]→0 仅为待实证声明，不可宣称已证。
    if not stories:
        return 0.0
    closed_count = 0
    for story in stories:
        if _check_narrative_closure(story):
            closed_count += 1
    return closed_count / len(stories)


def personality_consistency_rate(
    actions: List[dict],
    personality_type: str,
    personality_rules: dict,
) -> float:
    if not actions:
        return 0.0
    consistent_count = 0
    for act in actions:
        if _check_personality_consistent(act, personality_type, personality_rules):
            consistent_count += 1
    return consistent_count / len(actions)


def frchet_distance(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
) -> float:
    n = len(trajectory_a)
    m = len(trajectory_b)
    if n == 0 or m == 0:
        return float("inf")
    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(trajectory_a[0] - trajectory_b[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(trajectory_a[i] - trajectory_b[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(trajectory_a[0] - trajectory_b[j]))
    for i in range(1, n):
        for j in range(1, m):
            dist = np.linalg.norm(trajectory_a[i] - trajectory_b[j])
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist)
    return ca[n - 1, m - 1]


def discrete_curvature(
    hidden_states: np.ndarray,
) -> Tuple[float, float]:
    # C-03: hidden_states 应已是投影到底因基底 B 的空间；此处用与步长无关的
    # 转向角离散曲率（Menger 式）作为场强 F̃ 的代理曲率，在投影空间计算方与论文口径一致。
    #
    # 重要修正 (κ方向相反排查): 旧公式 κ=|Δv|/(|v_t|·|v_{t-1}|) 含 1/|v| 因子，
    # 与轨迹步长成反比——它量的是「反向速度」而非几何曲率。因果几何损失把相邻
    # 因果隐含态拉近，使步长被压缩约 3×，分母变小从而人为放大 κ（0.56→1.04），
    # 即使真实转向更少（更平坦）。这正是 κ 方向与 Wilson/理论预期相反的根因。
    # 改用相邻段向量的转向角 θ∈[0,π]：直线 θ=0（平坦），转弯越大 θ 越大，
    # 且 θ 与步长无关，正确反映「叙事闭环⇔平坦」的方向。
    if len(hidden_states) < 3:
        return 0.0, 0.0
    segments = np.diff(hidden_states, axis=0)          # (T-1, db) 段向量 v_t
    angles = []
    for t in range(1, len(segments)):
        a = segments[t]
        b = segments[t - 1]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cos_theta = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angles.append(np.arccos(cos_theta))        # 转向角，∈[0,π]，与步长无关
        else:
            angles.append(0.0)
    if not angles:
        return 0.0, 0.0
    return float(np.mean(angles)), float(np.max(angles))


def _check_physical_legal(action: dict, world_rules: dict) -> bool:
    act_type = action.get("type", "")
    if act_type == "move":
        if action.get("stamina", 0) <= 0:
            return False
        target = action.get("target_pos", (-1, -1))
        grid_size = world_rules.get("grid_size", 5)
        if target[0] < 0 or target[0] >= grid_size or target[1] < 0 or target[1] >= grid_size:
            return False
        if action.get("target_is_wall", False):
            return False
    elif act_type == "pick_up":
        if action.get("stamina", 0) <= 0:
            return False
        if not action.get("item_at_same_pos", False):
            return False
    elif act_type == "use":
        if not action.get("holding_required_item", False):
            return False
        if not action.get("target_adjacent", False):
            return False
    elif act_type == "wait":
        return True
    elif act_type == "push":
        if action.get("stamina", 0) <= 0:
            return False
    return True


def _check_narrative_closure(story: List[dict]) -> bool:
    has_key_encounter = False
    has_key_pickup = False
    has_door_open = False
    has_treasure = False
    for step in story:
        act = step.get("action", {})
        act_type = act.get("type", "")
        if act_type == "see_key":
            has_key_encounter = True
        elif act_type == "pick_up" and act.get("item") == "key":
            has_key_pickup = True
        elif act_type == "use" and act.get("target") == "door":
            has_door_open = True
        elif act_type == "get_treasure":
            has_treasure = True
    if has_treasure:
        return has_key_encounter and has_key_pickup and has_door_open
    return True


def _check_personality_consistent(
    action: dict,
    personality_type: str,
    personality_rules: dict,
) -> bool:
    act_type = action.get("type", "")
    stamina = action.get("stamina", 0)
    if personality_type == "brave":
        threshold = personality_rules.get("brave_stamina_threshold", 0)
        if act_type == "wait" and stamina > threshold:
            return False
    elif personality_type == "cautious":
        threshold = personality_rules.get("cautious_stamina_threshold", 3)
        if act_type == "move" and stamina < threshold:
            return False
    elif personality_type == "greedy":
        if act.get("see_item", False) and act_type != "pick_up":
            return False
    return True