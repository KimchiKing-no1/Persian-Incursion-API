# features.py
import numpy as np
import json

# Define vector size (e.g., 200 inputs)
INPUT_SIZE = 200 
# Max number of unique actions (for One-Hot encoding)
ACTION_SPACE_SIZE = 500 

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
    This allows the Neural Net to output a probability distribution.
    """
    # Simple hash-based bucketing for example (Collision prone but works for testing)
    # In production, you need a precise lookup table of all valid moves.
    s = json.dumps(action, sort_keys=True)
    return abs(hash(s)) % ACTION_SPACE_SIZE
