from __future__ import annotations
import math
import copy
import json
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# Optional: Imports for RL
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
    
    # Solver stats (Propagating definitive wins/losses)
    is_terminal: bool = False
    winner: Optional[str] = None

    # Tree structure
    children: List["Node"] = field(default_factory=list)
    unexpanded_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lookup key for transposition table
    key: str = ""

    def update(self, value: float) -> None:
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def best_child(self, c_uct: float = 1.4) -> Node:
        # PUCT Formula with stability epsilon
        sqrt_n = math.sqrt(max(1, self.N))
        def uct(n: Node) -> float:
            # If node represents a guaranteed loss for current player, avoid it
            if n.is_terminal and n.winner and n.winner != self._current_player_side():
                return -float('inf')
            exploit = n.Q
            explore = c_uct * n.P * (sqrt_n / (1 + n.N))
            return exploit + explore
        
        return max(self.children, key=uct)

    def _current_player_side(self) -> str:
        return self.state.get("turn", {}).get("current_player", "israel").lower()


# ============================== A G E N T ====================================

class MCTSAgent:
    def __init__(
        self,
        engine,
        side: str,
        simulations: int = 800,
        c_uct: float = 2.0,  # Higher exploration for complex wargames
        model_path: Optional[str] = None,
        gemini: Optional[GeminiCaller] = None,
        seed: Optional[int] = None,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.25,
        verbose: bool = False,
        reuse_tree: bool = True,  # Expert Feature: Keep tree between moves
        strict: bool = False,     # ← 여기 추가
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
        self.reuse_tree = reuse_tree
        self.strict = strict      # ← 여기 추가

        self._last_root: Optional[Node] = None
        self._transpo: Dict[str, Node] = {}

        # --- Load RL Model ---
        self.pv_model = None
        if model_path and RL_AVAILABLE:
            try:
                self.pv_model = PVModel.load(model_path)
                self.pv_model.eval()
                if self.verbose: print(f"[MCTS] Expert Mode: RL Model Loaded from {model_path}")
            except Exception as e:
                print(f"[MCTS] WARNING: RL load failed ({e}). Using Expert Heuristic Mode.")

    # ----------------------------------------------------------------------
    # P U B L I C   A P I
    # ----------------------------------------------------------------------
    def choose_action(self, state: Dict[str, Any], temperature: float = 0.5) -> tuple[Dict[str, Any], Dict[str, float]]:
        """
        Expert MCTS Entry Point.
        Handles Tree Reuse, Determinization, and Temperature sampling.
        """
        
        # 1. Determinization: Hide opponent information
        # In a real game, we don't know the enemy's cards. We must sample a plausible hand.
        root_state = self._determinize_state(state)
        
        # 2. Tree Reuse (Hot-Start)
        root = None
        if self.reuse_tree and self._last_root:
            # Try to find the new state in the old tree
            # Note: This is tricky in stochastic games; usually requires "Move Pruning"
            # For safety in this version, we rebuild if we can't find perfect match
            target_key = self._state_key(root_state)
            if target_key in self._transpo:
                root = self._transpo[target_key]
                # Prune parent to save memory
                root.parent = None 
                if self.verbose: print(f"[MCTS] Hot-start successful. Retained {root.N} simulations.")
        
        if root is None:
            self._transpo.clear()
            root = self._make_node(root_state)

        # If instant win/loss or no moves
        if not root.unexpanded_actions and not root.children:
            return {"type": "Pass"}, {}

        # 3. Add Noise to Root (Prevent rigid opening books)
        if self.root_dirichlet_alpha:
            self._inject_root_dirichlet(root)

        # 4. Search Loop
        start_time = time.time()
        for i in range(self.simulations):
            node = root
            
            # SELECT
            while not node.unexpanded_actions and node.children:
                node = node.best_child(self.c_uct)
            
            # EXPAND
            if node.unexpanded_actions:
                node = self._expand(node)
            
            # EVALUATE (RL or Expert Heuristic)
            value = self._evaluate_leaf(node.state)
            
            # BACKPROPAGATE
            self._backprop(node, value)

        # 5. Select Action based on Temperature
        # Temp -> 0 means argmax (Competitive play)
        # Temp -> 1 means proportional sampling (Exploration/Training)
        
        if not root.children:
             # Should technically pass, but if we have unexpanded, pick random
            if root.unexpanded_actions:
                return root.unexpanded_actions[0], {}
            return {"type": "Pass"}, {}

        counts = [(c.N, c.incoming_action) for c in root.children]
        total_N = sum(x[0] for x in counts)
        
        if temperature == 0:
            best_count, best_action = max(counts, key=lambda x: x[0])
        else:
            # Weighted random choice based on N^(1/temp)
            counts = [(x[0] ** (1/temperature), x[1]) for x in counts]
            total_temp = sum(x[0] for x in counts)
            if total_temp == 0: return max(counts)[1], {}
            probs = [x[0]/total_temp for x in counts]
            # Pick index
            idx = np.random.choice(len(counts), p=probs)
            best_action = counts[idx][1]

        # Cache root for next turn
        # We find the child node corresponding to the action we picked
        for c in root.children:
            if c.incoming_action == best_action:
                self._last_root = c
                break
        
        policy_map = {json.dumps(c.incoming_action, sort_keys=True): (c.N / total_N) for c in root.children}
        
        if self.verbose:
            print(f"[MCTS] Chosen: {best_action['type']} (WinRate: {self._last_root.Q:.2f}, Visits: {self._last_root.N})")

        return best_action, policy_map

    # ----------------------------------------------------------------------
    # D O M A I N   L O G I C
    # ----------------------------------------------------------------------

    def _determinize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expert Feature: Imperfect Information Handling.
        If playing as Israel, shuffle Iran's hand/river so we don't 'cheat'.
        """
        sim_state = copy.deepcopy(state)
        opponent = "iran" if self.side == "israel" else "israel"
        
        # We assume we can't see opponent's river. 
        # In a strict engine, 'river' might be masked. 
        # If it is visible, we should Randomize it to prevent overfitting to a specific draw.
        p_data = sim_state['players'].get(opponent, {})
        if 'deck' in p_data and 'river' in p_data:
            # Combine current river + deck, shuffle, and redeal
            # This simulates "I know they have cards, but not which ones"
            known_river = [c for c in p_data['river'] if c is not None] # Assuming None for holes
            remaining_deck = p_data.get('deck', [])
            
            # Pool them
            pool = known_river + remaining_deck
            self.rng.shuffle(pool)
            
            # Redeal
            new_river = []
            for _ in range(min(7, len(pool))):
                new_river.append(pool.pop(0))
            
            sim_state['players'][opponent]['river'] = new_river
            sim_state['players'][opponent]['deck'] = pool
            
        return sim_state

    def _expand(self, node: Node) -> Node:
        action = node.unexpanded_actions.pop(0)
        next_state = self._safe_apply(copy.deepcopy(node.state), action)
        
        child = self._make_node(next_state, parent=node, incoming_action=action)
        
        # Check terminality instantly (Solver optimization)
        winner = self.engine.is_game_over(next_state)
        if winner:
            child.is_terminal = True
            child.winner = winner
            # If this move wins immediately, boost Q massive amount
            if winner == self.side:
                child.Q = 1.0
                child.W = 1e6 # Hack to force selection
        
        node.children.append(child)
        return child

    def _evaluate_leaf(self, state: Dict[str, Any]) -> float:
        """
        Expert Evaluation:
        1. Game Over check
        2. Neural Net (if loaded)
        3. Expert Heuristic Rollout (Heavy Playout)
        """
        winner = self.engine.is_game_over(state)
        if winner:
            return 1.0 if winner == "israel" else -1.0

        # 1. Neural Net
        if self.pv_model and RL_AVAILABLE:
            try:
                turn_side = state.get("turn", {}).get("current_player", "israel").lower()
                feats = featurize(state, turn_side)
                tensor_in = torch.from_numpy(feats).unsqueeze(0)
                with torch.no_grad():
                    _, v_out = self.pv_model(tensor_in)
                return float(v_out.item())
            except Exception:
                pass # Fallback to rollout
        
        # 2. Heavy Rollout
        return self._heavy_rollout(state)

    def _heavy_rollout(self, state: Dict[str, Any]) -> float:
        """
        A 'Heavy' rollout uses domain knowledge instead of random moves.
        This significantly improves MCTS strength in wargames.
        """
        sim_state = copy.deepcopy(state)
        depth = 0
        max_depth = 40 
        
        while depth < max_depth:
            winner = self.engine.is_game_over(sim_state)
            if winner:
                return 1.0 if winner == "israel" else -1.0
            
            legal = self.engine.get_legal_actions(sim_state)
            if not legal: break
            
            # INTELLIGENT SELECTION
            current_player = sim_state.get("turn", {}).get("current_player", "israel")
            move = self._heuristic_policy_select(sim_state, legal, current_player)
            
            sim_state = self._safe_apply(sim_state, move)
            depth += 1
            
        # If depth exceeded, use static evaluation
        return self._static_eval(sim_state)

    def _heuristic_policy_select(self, state: Dict, legal: List[Dict], side: str) -> Dict:
        """
        Selects a move based on expert rules of thumb.
        """
        # 1. Always take lethal moves (not fully simulated here, but we prioritize attacks)
        
        ops = [a for a in legal if a['type'] not in ('Pass', 'End Impulse')]
        
        if not ops:
            return legal[0] # Pass/End Impulse

        # ISRAEL STRATEGY
        if side == 'israel':
            # Priority 1: Airstrikes on Nuclear targets if resources allow
            strikes = [a for a in ops if a['type'] == 'Order Airstrike']
            if strikes:
                # Sort by target value (Simple heuristic: Natanz > Arak > others)
                def target_score(a):
                    t = a.get('target', '').lower()
                    if 'natanz' in t: return 10
                    if 'arak' in t: return 8
                    return 1
                strikes.sort(key=target_score, reverse=True)
                return strikes[0]
            
            # Priority 2: Special Warfare to prep battlefield
            specwar = [a for a in ops if a['type'] == 'Order Special Warfare']
            if specwar:
                return specwar[0]

        # IRAN STRATEGY
        if side == 'iran':
            # Priority 1: Ballistic Missile retaliation
            bms = [a for a in ops if a['type'] == 'Order Ballistic Missile']
            if bms:
                return bms[0]
            
            # Priority 2: Terror Attacks to drain US opinion
            terror = [a for a in ops if a['type'] == 'Order Terror Attack']
            if terror:
                return terror[0]

        # Fallback: Random Op -> Random Card -> Pass
        cards = [a for a in legal if a['type'] == 'Play Card']
        if cards and self.rng.random() < 0.5:
            return self.rng.choice(cards)
            
        return self.rng.choice(ops) if ops else legal[0]

    def _static_eval(self, state: Dict[str, Any]) -> float:
        """
        Evaluate non-terminal state [-1, 1].
        Positive = Good for Israel.
        """
        score = 0.0
        
        # 1. Nuclear Damage (Primary Metric)
        targets = self.engine.rules.get("targets", {})
        for tname, tdata in state.get("target_damage_status", {}).items():
            # Check if it's a nuclear target
            trules = targets.get(tname, {})
            if "Nuclear" in trules.get("Target_Types", []):
                for comp, dam in tdata.items():
                    d = dam if isinstance(dam, int) else dam.get("damage_boxes_hit", 0)
                    score += (d * 0.1) # 0.1 points per box of nuclear damage

        # 2. Opinion Tracks
        dom = state.get("opinion", {}).get("domestic", {})
        score += (dom.get("israel", 0) * 0.05)
        score -= (dom.get("iran", 0) * 0.05)

        return max(-1.0, min(1.0, score))

    # ----------------------------------------------------------------------
    # H E L P E R S
    # ----------------------------------------------------------------------
    def _make_node(self, state: Dict[str, Any], parent=None, incoming_action=None) -> Node:
        legal = self._legal_actions(state)
        node = Node(
            state=state,
            parent=parent,
            incoming_action=incoming_action,
            unexpanded_actions=legal,
            key=self._state_key(state)
        )
        
        # Neural Net Priors
        if self.pv_model and RL_AVAILABLE:
            try:
                turn_side = state.get("turn", {}).get("current_player", "israel").lower()
                feats = featurize(state, turn_side)
                tensor_in = torch.from_numpy(feats).unsqueeze(0)
                with torch.no_grad():
                    p_logits, _ = self.pv_model(tensor_in)
                    p_probs = torch.softmax(p_logits, dim=1).numpy()[0]
                
                for act in node.unexpanded_actions:
                    idx = action_key(act)
                    if 0 <= idx < len(p_probs):
                        act['_prior'] = float(p_probs[idx])
                    else:
                        act['_prior'] = 0.001
                
                node.unexpanded_actions.sort(key=lambda x: x.get('_prior', 0.0), reverse=True)
            except Exception:
                pass # Fallback to uniform

        if not node.unexpanded_actions:
             node.is_terminal = True # No moves = game over usually

        return node


      def _legal_actions(self, state):
        try:
            return self.engine.get_legal_actions(state)
        except Exception as e:
            if self.strict:
                # 디버그 모드: 조용히 죽지 말고 바로 터뜨리기
                raise
            if self.verbose:
                print(f"[MCTS] WARNING: get_legal_actions failed, falling back to Pass. Error: {e}")
            return [{"type": "Pass"}]
   
    def _safe_apply(self, state, action):
        try:
            return self.engine.apply_action(state, action)
        except Exception as e:
            if self.strict:
                raise
            if self.verbose:
                print(f"[MCTS] WARNING: apply_action failed, returning unchanged state. Error: {e}")
            return state

    def _state_key(self, state):
        """
        Create a unique hash for the state to use in Transposition Tables.

        - Strip RNG + logs (비결정적 / 디버그용)
        - active_events_queue 는 '정규화(canonicalize)' 해서 포함
          → 같은 이벤트 세트면 순서 달라도 동일 key
        """
        def _canon_event(ev: Any) -> Any:
            if not isinstance(ev, dict):
                return ev
            # 불필요한 로그용 필드 있으면 여기서 빼도 됨
            return {k: ev[k] for k in sorted(ev.keys())}

        clean = {}
        for k, v in state.items():
            if k in ("_rng", "log"):
                continue
            if k == "active_events_queue":
                if isinstance(v, list):
                    canon_list = [_canon_event(e) for e in v]
                    canon_list = sorted(
                        canon_list,
                        key=lambda e: json.dumps(e, sort_keys=True, default=str)
                    )
                    clean[k] = canon_list
                else:
                    clean[k] = v
            else:
                clean[k] = v

        return json.dumps(clean, sort_keys=True, default=str)

    def _inject_root_dirichlet(self, root: Node):
        count = len(root.unexpanded_actions)
        if count < 2:
            return
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * count)
        for i, act in enumerate(root.unexpanded_actions):
            act['_prior'] = (
                (1 - self.root_dirichlet_eps) * act.get('_prior', 0.0)
                + self.root_dirichlet_eps * float(noise[i])
            )
        root.unexpanded_actions.sort(key=lambda x: x.get('_prior', 0.0), reverse=True)
