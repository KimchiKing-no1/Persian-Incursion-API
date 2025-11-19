# mcts.py
from __future__ import annotations
import math
import copy
import json
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# Optional: Imports for RL (will check availability at runtime)
try:
    import torch
    from model import PVModel
    from features import featurize, action_key
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    PVModel = None
    featurize = None
    action_key = None

GeminiCaller = Callable[..., str]

# ================================ N O D E ====================================

@dataclass
class Node:
    state: Dict[str, Any]
    parent: Optional["Node"] = None
    incoming_action: Optional[Dict[str, Any]] = None

    # MCTS stats
    N: int = 0          # visit count
    W: float = 0.0      # total value sum
    Q: float = 0.0      # mean value (W/N)
    P: float = 1.0      # prior probability (from Policy Network)

    # Tree structure
    children: List["Node"] = field(default_factory=list)
    unexpanded_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lookup key for transposition table
    key: str = ""

    def update(self, value: float) -> None:
        self.N += 1
        self.W += value
        self.Q = self.W / self.N if self.N else 0.0


# ============================== A G E N T ====================================

class MCTSAgent:
    def __init__(
        self,
        engine,
        side: str,
        simulations: int = 1000,
        c_uct: float = 1.4,
        model_path: Optional[str] = None, # Path to .pt file
        gemini: Optional[GeminiCaller] = None,
        seed: Optional[int] = None,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.25,
        verbose: bool = False
    ):
        self.engine = engine
        self.side = side.lower().strip()
        self.simulations = simulations
        self.c_uct = c_uct
        self.gemini = gemini
        self.rng = random.Random(seed)
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_eps = root_dirichlet_eps
        self.verbose = verbose
        self._transpo: Dict[str, Node] = {}

        # --- Load RL Model if available ---
        self.pv_model = None
        if model_path and RL_AVAILABLE:
            try:
                self.pv_model = PVModel.load(model_path)
                self.pv_model.eval() # Set to inference mode
                if self.verbose: print(f"[MCTS] Loaded RL Model from {model_path}")
            except Exception as e:
                print(f"[MCTS] WARNING: Failed to load RL model: {e}. Running in Heuristic Mode.")
        elif model_path and not RL_AVAILABLE:
            print("[MCTS] WARNING: 'torch' or 'features.py' missing. RL disabled.")

    # ----------------------------------------------------------------------
    # P U B L I C   A P I
    # ----------------------------------------------------------------------
    def choose_action(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
        self._transpo.clear()
        root = self._make_node(state)

        # If no legal moves or game over
        if not root.unexpanded_actions and not root.children:
            return {"type": "Pass"}, {}

        # 1. Add Exploration Noise to Root (AlphaZero style)
        if self.root_dirichlet_alpha and root.unexpanded_actions:
            self._inject_root_dirichlet(root)

        # 2. Run Simulations
        for _ in range(self.simulations):
            node = self._select(root)
            if node.unexpanded_actions:
                node = self._expand(node)
            
            # RL Value or Heuristic Rollout
            value = self._evaluate_leaf(node.state)
            self._backprop(node, value)

        # 3. Select best action (Visit Count)
        if not root.children:
            return root.unexpanded_actions[0], {}
            
        best_child = max(root.children, key=lambda n: n.N)
        
        # Generate policy map for training
        total_N = sum(c.N for c in root.children)
        policy = {json.dumps(c.incoming_action, sort_keys=True): (c.N / total_N) for c in root.children}

        if self.verbose:
            print(f"[MCTS] Best: {best_child.incoming_action} (Q={best_child.Q:.2f}, N={best_child.N})")

        return best_child.incoming_action, policy

    # ----------------------------------------------------------------------
    # C O R E   L O G I C
    # ----------------------------------------------------------------------
    def _select(self, node: Node) -> Node:
        # Navigate tree until we hit a leaf or unexpanded node
        while not node.unexpanded_actions and node.children:
            # PUCT Formula
            log_N = math.sqrt(node.N)
            def uct(n: Node) -> float:
                # Q + c * P * (sqrt(Parent_N) / (1 + Child_N))
                return n.Q + self.c_uct * n.P * (log_N / (1 + n.N))
            
            node = max(node.children, key=uct)
        return node

    def _expand(self, node: Node) -> Node:
        action = node.unexpanded_actions.pop(0)
        next_state = self._safe_apply(copy.deepcopy(node.state), action)
        child = self._make_node(next_state, parent=node, incoming_action=action)
        node.children.append(child)
        return child

    def _backprop(self, node: Node, value: float) -> None:
        # Value is always from Israel's perspective in this engine (-1 Iran, +1 Israel)
        # We assume the engine handles turn-switching, so we don't flip value here 
        # UNLESS your engine's apply_action doesn't swap perspectives. 
        # (Assuming Standard AlphaZero: Value is always relative to current player? 
        # No, usually relative to a fixed player in 2p zero-sum. We use Israel-Positive).
        cur = node
        while cur is not None:
            cur.update(value)
            cur = cur.parent

    # ----------------------------------------------------------------------
    # R L   I N T E G R A T I O N
    # ----------------------------------------------------------------------
    def _make_node(self, state: Dict[str, Any], parent=None, incoming_action=None) -> Node:
        """Creates a node and populates Priors (P) from the RL model if available."""
        legal = self._legal_actions(state)
        node = Node(
            state=state,
            parent=parent,
            incoming_action=incoming_action,
            unexpanded_actions=legal,
            key=self._state_key(state)
        )
        self._transpo[node.key] = node

        # --- RL POLICY PREDICTION ---
        if self.pv_model and RL_AVAILABLE:
            try:
                # 1. Featurize State
                turn_side = state.get("turn", {}).get("current_player", "israel").lower()
                feats = featurize(state, turn_side) # Must return np.array
                tensor_in = torch.from_numpy(feats).unsqueeze(0) # Add batch dim

                # 2. Model Inference
                with torch.no_grad():
                    p_logits, _ = self.pv_model(tensor_in) # Ignore value here, used in eval
                    p_probs = torch.softmax(p_logits, dim=1).numpy()[0]

                # 3. Map Probs to Actions
                # We assume 'action_key(action)' returns the index in the probability vector
                for act in node.unexpanded_actions:
                    idx = action_key(act) # Helper from features.py
                    if 0 <= idx < len(p_probs):
                        act['_prior'] = float(p_probs[idx])
                    else:
                        act['_prior'] = 0.0
                
                # 4. Sort actions by probability (highest first)
                node.unexpanded_actions.sort(key=lambda x: x.get('_prior', 0.0), reverse=True)
                
                # 5. Assign P to the node (Wait, P belongs to the CHILD edge, currently stored on the action dict)
                # When we expand, we will pass this P to the child node.
            except Exception as e:
                if self.verbose: print(f"[MCTS] Policy Error: {e}")
        
        # If no model, Uniform priors
        if not self.pv_model:
             for act in node.unexpanded_actions:
                act['_prior'] = 1.0 / len(node.unexpanded_actions)

        return node

    def _evaluate_leaf(self, state: Dict[str, Any]) -> float:
        """Returns value [-1, 1]. Uses RL model if present, else Heuristic Rollout."""
        winner = self.engine.is_game_over(state)
        if winner == "israel": return 1.0
        if winner == "iran": return -1.0

        # --- RL VALUE PREDICTION ---
        if self.pv_model and RL_AVAILABLE:
            try:
                turn_side = state.get("turn", {}).get("current_player", "israel").lower()
                feats = featurize(state, turn_side)
                tensor_in = torch.from_numpy(feats).unsqueeze(0)
                with torch.no_grad():
                    _, v_out = self.pv_model(tensor_in)
                return float(v_out.item())
            except Exception as e:
                if self.verbose: print(f"[MCTS] Value Error: {e}")
        
        # Fallback: Heuristic Rollout
        return self._heuristic_rollout(state)

    def _heuristic_rollout(self, state: Dict[str, Any]) -> float:
        """Fast random rollout if no neural net is available."""
        sim_state = copy.deepcopy(state)
        for _ in range(30): # Depth limit
            winner = self.engine.is_game_over(sim_state)
            if winner:
                return 1.0 if winner == "israel" else -1.0
            
            actions = self._legal_actions(sim_state)
            if not actions: break
            # Simple heuristic: prefer ops over passing
            ops = [a for a in actions if a.get("type") != "Pass"]
            move = random.choice(ops) if ops else actions[0]
            sim_state = self._safe_apply(sim_state, move)
        
        return 0.0 # Draw/Unfinished

    # ----------------------------------------------------------------------
    # H E L P E R S
    # ----------------------------------------------------------------------
    def _legal_actions(self, state):
        return self.engine.get_legal_actions(state)

    def _safe_apply(self, state, action):
        try:
            return self.engine.apply_action(state, action)
        except:
            return state

    def _state_key(self, state):
        # Strip non-serializable objects for caching
        clean = {k:v for k,v in state.items() if k not in ['_rng', 'log']}
        return json.dumps(clean, sort_keys=True, default=str)

    def _inject_root_dirichlet(self, root: Node):
        """Adds noise to root priors to ensure exploration."""
        count = len(root.unexpanded_actions)
        if count < 2: return
        noise = self.rng.dirichlet([self.root_dirichlet_alpha] * count)
        for i, act in enumerate(root.unexpanded_actions):
            act['_prior'] = (1 - self.root_dirichlet_eps) * act.get('_prior', 0.0) + self.root_dirichlet_eps * noise[i]
        # Re-sort after noise
        root.unexpanded_actions.sort(key=lambda x: x.get('_prior', 0.0), reverse=True)

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
