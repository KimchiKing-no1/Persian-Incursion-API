# features.py
import numpy as np
import json
import hashlib
from rules_global import RULES

# --- 1. Define Fixed Vector Layout ---
# We calculate exact sizes to keep the vector fixed-length.
# This prevents "shape mismatch" errors during training.

_TARGET_LIST = sorted(list(RULES.get("targets", {}).keys()))
_SQUADRON_LIST_ISR = sorted(list(RULES.get("squadrons", {}).get("israel", {}).keys()))
_SQUADRON_LIST_IRN = sorted(list(RULES.get("squadrons", {}).get("iran", {}).keys()))

# Feature Counts
N_TARGETS = len(_TARGET_LIST)
N_SQ_ISR = len(_SQUADRON_LIST_ISR)
N_SQ_IRN = len(_SQUADRON_LIST_IRN)

# Offsets for vector slicing
# Global (Turn, Phase) + Resources(3*2) + Opinion(2) = 10
# Targets (2 bits each: Damaged, Destroyed)
# Squadrons (2 bits each: Strength, Status)
# Cards (River slots: 7 * Card_ID) - Simplified to "Hand Presence" for RL stability
INPUT_SIZE = 10 + (N_TARGETS * 2) + (N_SQ_ISR * 2) + (N_SQ_IRN * 2) + 20 
# Note: The +20 is a buffer for miscellaneous flags or card summaries.

# Action Space (Must allow saving/loading)
ACTION_SPACE_SIZE = 1000 
_ACTION_TO_INDEX = {}
_INDEX_TO_ACTION = []

def featurize(state: dict, perspective: str) -> np.ndarray:
    """
    Encodes the game state into a flat float32 vector.
    Perspective: 'israel' or 'iran' (flips friends/enemies if you want, 
    but for now we use absolute board state).
    """
    vec = []

    # --- A. Global & Resources (10 floats) ---
    # 1. Turn Number (Normalized 0-1)
    turn = state.get("turn", {}).get("turn_number", 1)
    vec.append(float(turn) / 21.0)
    
    # 2. Phase (One-hot or Scalar)
    phase = state.get("turn", {}).get("phase", "morning")
    vec.append({"morning": 0.0, "afternoon": 0.5, "night": 1.0}.get(phase, 0.0))

    # 3. Resources (Normalized)
    for side in ["israel", "iran"]:
        r = state.get("players", {}).get(side, {}).get("resources", {})
        vec.append(min(1.0, r.get("pp", 0) / 20.0))
        vec.append(min(1.0, r.get("ip", 0) / 20.0))
        vec.append(min(1.0, r.get("mp", 0) / 20.0))

    # 4. Opinions (Normalized -1.0 to 1.0)
    dom = state.get("opinion", {}).get("domestic", {})
    vec.append(dom.get("israel", 0) / 10.0)
    vec.append(dom.get("iran", 0) / 10.0)

    # --- B. Targets (N_TARGETS * 2 floats) ---
    # For every target: [Damage_Level, Is_Destroyed]
    # We read state["target_damage_status"]
    dmg_map = state.get("target_damage_status", {})
    
    for t_name in _TARGET_LIST:
        # Get damage info
        t_data = dmg_map.get(t_name, {})
        
        # Calculate aggregate damage (Sum of boxes hit)
        total_hits = 0
        if isinstance(t_data, dict):
            for comp_data in t_data.values():
                if isinstance(comp_data, dict):
                    total_hits += comp_data.get("damage_boxes_hit", 0)
                elif isinstance(comp_data, int):
                    total_hits += comp_data
        
        # Is it effectively destroyed? (Heuristic: >5 hits is bad)
        is_destroyed = 1.0 if total_hits >= 5 else 0.0
        
        vec.append(min(1.0, total_hits / 10.0)) # Normalized damage
        vec.append(is_destroyed)

    # --- C. Squadrons (N_SQ * 2 floats) ---
    # [Current_Strength_Pct, Is_Ready]
    
    # Helper to parse status
    def get_sq_feats(side, sq_list):
        feats = []
        # Get dynamic status from state
        sq_status_map = state.get("squadrons", {}).get(side, {})
        # Get OOB strength (losses)
        oob_map = state.get("oob", {}).get(side, {}).get("squadrons", {})
        
        for sq_id in sq_list:
            # Status: Ready=1.0, Flying/Resting=0.0
            st = sq_status_map.get(sq_id, "Ready")
            is_ready = 1.0 if st == "Ready" else 0.0
            
            # Strength: Default 100%
            # If state tracks losses, we subtract. 
            # (Assuming engine tracks losses in state['losses'] or oob)
            curr_str = 1.0 
            # Placeholder: Hook this up if your engine tracks exact plane counts per sq
            
            feats.append(curr_str)
            feats.append(is_ready)
        return feats

    vec.extend(get_sq_feats("israel", _SQUADRON_LIST_ISR))
    vec.extend(get_sq_feats("iran", _SQUADRON_LIST_IRN))

    # --- D. Padding ---
    # Fill remaining slots to reach INPUT_SIZE
    current_len = len(vec)
    if current_len > INPUT_SIZE:
        # If we exceeded, truncate (and log warning in real app)
        vec = vec[:INPUT_SIZE]
    else:
        padding = INPUT_SIZE - current_len
        vec.extend([0.0] * padding)

    return np.array(vec, dtype=np.float32)

def action_key(action: dict) -> int:
    """
    Returns a unique integer ID for an action.
    If the action hasn't been seen before, assigns a new ID.
    CRITICAL: This mapping must be saved/loaded!
    """
    # Create a deterministic string representation
    # Sorting keys ensures {"a":1, "b":2} == {"b":2, "a":1}
    s = json.dumps(action, sort_keys=True)
    
    if s in _ACTION_TO_INDEX:
        return _ACTION_TO_INDEX[s]
    
    # Assign new ID
    new_id = len(_ACTION_TO_INDEX)
    if new_id >= ACTION_SPACE_SIZE:
        # Fallback to a 'catch-all' or simple modulus to prevent crashing
        # But for training, we usually want to crash or expand.
        return new_id % ACTION_SPACE_SIZE 
        
    _ACTION_TO_INDEX[s] = new_id
    _INDEX_TO_ACTION.append(s)
    return new_id

def save_action_map(path="action_map.json"):
    with open(path, 'w') as f:
        json.dump(_INDEX_TO_ACTION, f)

def load_action_map(path="action_map.json"):
    global _INDEX_TO_ACTION, _ACTION_TO_INDEX
    try:
        with open(path, 'r') as f:
            _INDEX_TO_ACTION = json.load(f)
            _ACTION_TO_INDEX = {s: i for i, s in enumerate(_INDEX_TO_ACTION)}
        print(f"[Features] Loaded {_ACTION_TO_INDEX} actions.")
    except FileNotFoundError:
        print("[Features] No action map found, starting fresh.")
