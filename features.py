# features.py
import numpy as np
import json

# Define vector size (e.g., 200 inputs)
INPUT_SIZE = 200 
# Max number of unique actions (for One-Hot encoding)
ACTION_SPACE_SIZE = 500 
_ACTION_TO_INDEX = {}
_INDEX_TO_ACTION = []
def featurize(state: dict, perspective: str) -> np.ndarray:
    """
    Convert game state dict to a fixed-size float32 numpy array.
    Normalize values between 0 and 1 where possible.
    """
    vec = []
    
    # 1. Resources (Normalized by 20)
    p = state['players'].get(perspective, {}).get('resources', {})
    vec.extend([p.get('mp',0)/20.0, p.get('ip',0)/20.0, p.get('pp',0)/20.0])

    # 2. Opinion Tracks (Normalized from -10...10 to 0...1)
    ops = state.get('opinion', {})
    dom = ops.get('domestic', {})
    vec.append((dom.get('israel', 0) + 10) / 20.0)
    vec.append((dom.get('iran', 0) + 10) / 20.0)
    
    # 3. Target Damage (Binary: 0=Fine, 1=Destroyed)
    # (You would loop through VALID_TARGETS here)
    # Placeholder filler
    current_len = len(vec)
    padding = INPUT_SIZE - current_len
    vec.extend([0.0] * padding)
    
    return np.array(vec[:INPUT_SIZE], dtype=np.float32)

def action_key(action: dict) -> int:
    """
    Map a specific JSON action to a unique integer index [0..ACTION_SPACE_SIZE).

    - 같은 action(JSON 문자열)이면 항상 같은 index
    - 다른 action은 다른 index
    - ACTION_SPACE_SIZE를 넘으면 RuntimeError (학습 때 upper bound 체크용)
    """
    s = json.dumps(action, sort_keys=True)

    # 이미 본 행동이면 기존 index 재사용
    idx = _ACTION_TO_INDEX.get(s)
    if idx is not None:
        return idx

    # 새 행동인데 슬롯 다 찼으면 에러
    if len(_ACTION_TO_INDEX) >= ACTION_SPACE_SIZE:
        raise RuntimeError(
            f"Exceeded ACTION_SPACE_SIZE={ACTION_SPACE_SIZE}. "
            f"Need to increase ACTION_SPACE_SIZE or predefine action mapping."
        )

    # 새 index 할당
    idx = len(_ACTION_TO_INDEX)
    _ACTION_TO_INDEX[s] = idx
    _INDEX_TO_ACTION.append(s)
    return idx
