from enum import Enum
from typing import Dict, List, Optional, Tuple


class SpecialToken(Enum):
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"


class NPNWTokenizer:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab()
        self.vocab_size = len(self.token_to_id)

    def _build_vocab(self):
        idx = 0
        for st in SpecialToken:
            self.token_to_id[st.value] = idx
            self.id_to_token[idx] = st.value
            idx += 1
        for action in ["move", "pick_up", "use", "push", "wait", "see", "get_treasure"]:
            self.token_to_id[f"ACT_{action}"] = idx
            self.id_to_token[idx] = f"ACT_{action}"
            idx += 1
        for direction in ["N", "S", "E", "W"]:
            self.token_to_id[f"DIR_{direction}"] = idx
            self.id_to_token[idx] = f"DIR_{direction}"
            idx += 1
        for i in range(6):
            self.token_to_id[f"POS_{i}"] = idx
            self.id_to_token[idx] = f"POS_{i}"
            idx += 1
        for i in range(21):
            self.token_to_id[f"STA_{i}"] = idx
            self.id_to_token[idx] = f"STA_{i}"
            idx += 1
        for item in ["none", "key"]:
            self.token_to_id[f"ITEM_{item}"] = idx
            self.id_to_token[idx] = f"ITEM_{item}"
            idx += 1
        for cell in ["empty", "wall", "door", "key", "treasure"]:
            self.token_to_id[f"CELL_{cell}"] = idx
            self.id_to_token[idx] = f"CELL_{cell}"
            idx += 1
        for personality in ["brave", "cautious", "greedy"]:
            self.token_to_id[f"PER_{personality}"] = idx
            self.id_to_token[idx] = f"PER_{personality}"
            idx += 1
        for causal in [
            "foreshadow_key", "pickup_key", "encounter_door",
            "foreshadow_recover", "closure", "physical_continuous",
        ]:
            self.token_to_id[f"CAUSAL_{causal}"] = idx
            self.id_to_token[idx] = f"CAUSAL_{causal}"
            idx += 1
        for label in ["legal", "illegal"]:
            self.token_to_id[f"LABEL_{label}"] = idx
            self.id_to_token[idx] = f"LABEL_{label}"
            idx += 1
        self.token_to_id["SEP"] = idx
        self.id_to_token[idx] = "SEP"
        idx += 1

    def encode_step(self, state_dict: dict, action_dict: dict, causal_labels: dict) -> List[int]:
        tokens = []
        tokens.append(self.token_to_id[f"POS_{state_dict.get('pos_x', 0)}"])
        tokens.append(self.token_to_id[f"POS_{state_dict.get('pos_y', 0)}"])
        tokens.append(self.token_to_id[f"STA_{state_dict.get('stamina', 0)}"])
        tokens.append(self.token_to_id[f"ITEM_{state_dict.get('holding', 'none')}"])
        tokens.append(self.token_to_id["SEP"])
        act_type = action_dict.get("type", "wait")
        tokens.append(self.token_to_id[f"ACT_{act_type}"])
        if act_type == "move":
            dx, dy = action_dict.get("dx", 0), action_dict.get("dy", 0)
            if dy > 0:
                tokens.append(self.token_to_id["DIR_N"])
            elif dy < 0:
                tokens.append(self.token_to_id["DIR_S"])
            elif dx > 0:
                tokens.append(self.token_to_id["DIR_E"])
            elif dx < 0:
                tokens.append(self.token_to_id["DIR_W"])
        tokens.append(self.token_to_id["SEP"])
        for k, v in causal_labels.items():
            if v:
                tokens.append(self.token_to_id[f"CAUSAL_{k}"])
        tokens.append(self.token_to_id["SEP"])
        return tokens

    def encode_story(self, steps: List[dict], with_step_ids: bool = False):
        tokens = [self.token_to_id[SpecialToken.SOS.value]]
        step_ids = [0]
        for si, step in enumerate(steps):
            step_tokens = self.encode_step(
                step.get("state", {}),
                step.get("action", {}),
                step.get("causal_labels", {}),
            )
            tokens.extend(step_tokens)
            if with_step_ids:
                step_ids.extend([si] * len(step_tokens))
        tokens.append(self.token_to_id[SpecialToken.EOS.value])
        if with_step_ids:
            step_ids.append(si)               # EOS 归属于最后一步
            return tokens, step_ids
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token.get(tid, SpecialToken.UNK.value) for tid in token_ids]

    def pad_sequence(self, token_ids: List[int], max_len: int) -> List[int]:
        pad_id = self.token_to_id[SpecialToken.PAD.value]
        if len(token_ids) >= max_len:
            return token_ids[:max_len]
        return token_ids + [pad_id] * (max_len - len(token_ids))