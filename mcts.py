from __future__ import annotations
import math
import copy
import json
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

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

    def best_child(self, c_uct: float = 1.4) -> "Node":
        # PUCT Formula with stability epsilon
        sqrt_n = math.sqrt(max(1, self.N))

        def uct(n: "Node") -> float:
            # If node represents a guaranteed loss for current player, avoid it
            if n.is_terminal and n.winner and n.winner != self._current_player_side():
                return -float("inf")
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
        strict: bool = False,
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
        self.strict = strict

        self._last_root: Optional[Node] = None
        self._transpo: Dict[str, Node] = {}

        # --- Load RL Model ---
        self.pv_model = None
        if model_path and RL_AVAILABLE:
            try:
                self.pv_model = PVModel.load(model_path)
                self.pv_model.eval()
                if self.verbose:
                    print(f"[MCTS] Expert Mode: RL Model Loaded from {model_path}")
            except Exception as e:
                print(f"[MCTS] WARNING: RL load failed ({e}). Using Expert Heuristic Mode.")

    # --------- NEW: safe / fast copy --------------------------------------
    def _fast_copy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        if "rules" not in state:
           
            return copy.deepcopy(state)


        rules_ref = state["rules"]
        new_state = copy.deepcopy(state)
        new_state["rules"] = rules_ref
        return new_state

    # ----------------------------------------------------------------------
    # P U B L I C   A P I
    # ----------------------------------------------------------------------
    def choose_action(
        self, state: Dict[str, Any], temperature: float = 0.5
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Expert MCTS Entry Point.
        Handles Tree Reuse, Determinization, and Temperature sampling.
        """
        # 1. Determinization: Hide opponent information
        root_state = self._determinize_state(state)

        # 2. Tree Reuse (Hot-Start)
        root: Optional[Node] = None
        if self.reuse_tree and self._last_root:
            target_key = self._state_key(root_state)
            if target_key in self._transpo:
                root = self._transpo[target_key]
                # Cut link to parent for memory
                root.parent = None
                if self.verbose:
                    print(
                        f"[MCTS] Hot-start successful. Retained {root.N} simulations."
                    )

        if root is None:
            self._transpo.clear()
            root = self._make_node(root_state)

        # If instant win/loss or no moves
        if not root.unexpanded_actions and not root.children:
            if self.verbose:
                print("[MCTS] No legal actions, returning Pass.")
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
        if not root.children:
            if root.unexpanded_actions:
                return root.unexpanded_actions[0], {}
            return {"type": "Pass"}, {}

        counts = [(c.N, c.incoming_action) for c in root.children]
        total_N = sum(x[0] for x in counts)

        if temperature == 0:
            best_count, best_action = max(counts, key=lambda x: x[0])
        else:
            # Weighted random choice based on N^(1/temp)
            temp_weights = [x[0] ** (1 / temperature) for x in counts]
            total_temp = sum(temp_weights)
            if total_temp == 0:
                best_action = max(counts, key=lambda x: x[0])[1]
            else:
                probs = [w / total_temp for w in temp_weights]
                idx = np.random.choice(len(counts), p=probs)
                best_action = counts[idx][1]

        # Cache root for next turn (tree reuse)
        for c in root.children:
            if c.incoming_action == best_action:
                self._last_root = c
                break

        policy_map = {
            json.dumps(c.incoming_action, sort_keys=True): (c.N / total_N)
            for c in root.children
        }

        if self.verbose and self._last_root is not None:
            print(
                f"[MCTS] Chosen: {best_action.get('type','?')} "
                f"(WinRate: {self._last_root.Q:.2f}, Visits: {self._last_root.N})"
            )

        return best_action, policy_map

    # ----------------------------------------------------------------------
    # D O M A I N   L O G I C
    # ----------------------------------------------------------------------

    def _determinize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expert Feature: Imperfect Information Handling.
        If playing as Israel, shuffle Iran's hand/river so we don't 'cheat'.
        """
        sim_state = self._fast_copy(state)
        opponent = "iran" if self.side == "israel" else "israel"

        p_data = sim_state.get("players", {}).get(opponent, {})
        if "deck" in p_data and "river" in p_data:
            known_river = [c for c in p_data["river"] if c is not None]
            remaining_deck = p_data.get("deck", [])

            pool = known_river + remaining_deck
            self.rng.shuffle(pool)

            new_river = []
            for _ in range(min(7, len(pool))):
                new_river.append(pool.pop(0))

            sim_state["players"][opponent]["river"] = new_river
            sim_state["players"][opponent]["deck"] = pool

        return sim_state

    def _expand(self, node: Node) -> Node:
        action = node.unexpanded_actions.pop(0)
        next_state = self._safe_apply(self._fast_copy(node.state), action)

        child = self._make_node(next_state, parent=node, incoming_action=action)

        # Check terminality instantly (Solver optimization)
        winner = self.engine.is_game_over(next_state)
        if winner:
            child.is_terminal = True
            child.winner = winner
            if winner == self.side:
                child.Q = 1.0
                child.W = 1e6  # Big bonus to force selection

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
                # Fallback to rollout
                pass

        # 2. Heavy Rollout
        return self._heavy_rollout(state)

    def _heavy_rollout(self, state: Dict[str, Any]) -> float:
        """
        A 'Heavy' rollout uses domain knowledge instead of random moves.
        This significantly improves MCTS strength in wargames.
        """
        sim_state = self._fast_copy(state)
        depth = 0
        max_depth = 40

        while depth < max_depth:
            winner = self.engine.is_game_over(sim_state)
            if winner:
                return 1.0 if winner == "israel" else -1.0

            current_player = sim_state.get("turn", {}).get("current_player", "israel")
            legal = self.engine.get_legal_actions(sim_state, side=current_player)

            if not legal:
                break

            current_player = sim_state.get("turn", {}).get("current_player", "israel")
            move = self._heuristic_policy_select(sim_state, legal, current_player)

            sim_state = self._safe_apply(sim_state, move)
            depth += 1

        return self._static_eval(sim_state)

    def _heuristic_policy_select(
        self, state: Dict[str, Any], legal: List[Dict[str, Any]], side: str
    ) -> Dict[str, Any]:
        """
        Selects a move based on expert rules of thumb.
        """
        ops = [a for a in legal if a.get("type") not in ("Pass", "End Impulse")]

        if not ops:
            return legal[0]

        # ISRAEL STRATEGY
        if side == "israel":
            current_op = (
                state.get("opinion", {})
                .get("domestic", {})
                .get("israel", 0)
            )
            if current_op < -5:
                cards = [a for a in legal if a.get("type") == "Play Card"]
                if cards:
                    return self.rng.choice(cards)
                return {"type": "Pass"}

            strikes = [a for a in ops if a.get("type") == "Order Airstrike"]
            if strikes:
                def target_score(a: Dict[str, Any]) -> int:
                    t = str(a.get("target", "")).lower()
                    if "natanz" in t:
                        return 10
                    if "arak" in t:
                        return 8
                    return 1

                strikes.sort(key=target_score, reverse=True)
                return strikes[0]

            recon = [a for a in ops if a.get("type") == "Recon"]
            if recon:
                return recon[0]

            specwar = [a for a in ops if a.get("type") == "Order Special Warfare"]
            if specwar:
                return specwar[0]

        # IRAN STRATEGY
        if side == "iran":
            current_op = (
                state.get("opinion", {})
                .get("domestic", {})
                .get("iran", 0)
            )
            if current_op < -5:
                cards = [a for a in legal if a.get("type") == "Play Card"]
                if cards:
                    return self.rng.choice(cards)
                return {"type": "Pass"}

            bms = [a for a in ops if a.get("type") == "Order Ballistic Missile"]
            if bms:
                return bms[0]

            terror = [a for a in ops if a.get("type") == "Order Terror Attack"]
            if terror:
                return terror[0]

        cards = [a for a in legal if a.get("type") == "Play Card"]
        if cards and self.rng.random() < 0.5:
            return self.rng.choice(cards)

        return self.rng.choice(ops) if ops else legal[0]

    def _static_eval(self, state: Dict[str, Any]) -> float:
        """
        Evaluate non-terminal state [-1, 1].
        Positive = Good for Israel.
        """
        score = 0.0

        targets_rules = getattr(self.engine, "rules", {}).get("targets", {})

        for tname, tdata in state.get("target_damage_status", {}).items():
            trules = targets_rules.get(tname, {})
            if "Nuclear" in trules.get("Target_Types", []):
                for comp, dam in tdata.items():
                    d = dam if isinstance(dam, int) else dam.get("damage_boxes_hit", 0)
                    score += d * 0.1

        dom = state.get("opinion", {}).get("domestic", {})
        isr_dom = float(dom.get("israel", 0))
        irn_dom = float(dom.get("iran", 0))

        score += isr_dom * 0.1
        score -= irn_dom * 0.1

        if isr_dom <= -6:
            score -= 0.5
        if isr_dom <= -8:
            score -= 1.5

        if irn_dom >= 6:
            score += 0.3

        players = state.get("players", {})
        isr_res = players.get("israel", {}).get("resources", {"mp": 0, "ip": 0, "pp": 0})
        score += (isr_res.get("mp", 0) + isr_res.get("ip", 0) + isr_res.get("pp", 0)) * 0.01

        losses = state.get("losses", {})
        score -= losses.get("israel_aircraft", 0) * 0.15

        return max(-1.0, min(1.0, score))

    # ----------------------------------------------------------------------
    # H E L P E R S
    # ----------------------------------------------------------------------
    def _make_node(
        self, state: Dict[str, Any], parent: Optional[Node] = None,
        incoming_action: Optional[Dict[str, Any]] = None
    ) -> Node:
        legal = self._legal_actions(state)
        node = Node(
            state=state,
            parent=parent,
            incoming_action=incoming_action,
            unexpanded_actions=legal,
            key=self._state_key(state),
        )

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
                        act["_prior"] = float(p_probs[idx])
                    else:
                        act["_prior"] = 0.001

                node.unexpanded_actions.sort(
                    key=lambda x: x.get("_prior", 0.0), reverse=True
                )
            except Exception:
                pass

        if not node.unexpanded_actions:
            node.is_terminal = True

        self._transpo[node.key] = node
        return node

    def _legal_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Prefer explicit side; fall back to state.turn.current_player
            side = self.side
            turn_side = (state.get("turn") or {}).get("current_player")
            if isinstance(turn_side, str):
                side = turn_side.lower()
            return self.engine.get_legal_actions(state, side=side)
        except Exception as e:
            if self.strict:
                raise
            if self.verbose:
                print(
                    f"[MCTS] WARNING: get_legal_actions failed, "
                    f"falling back to Pass. Error: {e}"
                )
            return [{"type": "Pass"}]


    def _safe_apply(self, state: Dict[str, Any], action: Dict[str, Any], side: str):
        """
        Apply action on a copy of `state` so MCTS never mutates the shared root.
        """
        try:
            if self.engine is None:
                return state
    
            # CRITICAL: never mutate the input state
            new_state = self._fast_copy(state)   # or copy.deepcopy(state)
            return self.engine.apply_action(new_state, action, side=side)
        except Exception as e:
            if self.logger:
                self.logger.error(f"_safe_apply failed: {e}")
            return state


    def _state_key(self, state: Dict[str, Any]) -> str:
        def _canon_event(ev: Any) -> Any:
            if not isinstance(ev, dict):
                return ev
            return {k: ev[k] for k in sorted(ev.keys())}

        clean: Dict[str, Any] = {}
        for k, v in state.items():
            if k in ("_rng", "log"):
                continue
            if k == "active_events_queue":
                if isinstance(v, list):
                    canon_list = [_canon_event(e) for e in v]
                    canon_list = sorted(
                        canon_list,
                        key=lambda e: json.dumps(e, sort_keys=True, default=str),
                    )
                    clean[k] = canon_list
                else:
                    clean[k] = v
            else:
                clean[k] = v

        return json.dumps(clean, sort_keys=True, default=str)

    def _inject_root_dirichlet(self, root: Node) -> None:
        count = len(root.unexpanded_actions)
        if count < 2:
            return
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * count)
        for i, act in enumerate(root.unexpanded_actions):
            act["_prior"] = (
                (1 - self.root_dirichlet_eps) * act.get("_prior", 0.0)
                + self.root_dirichlet_eps * float(noise[i])
            )
        root.unexpanded_actions.sort(
            key=lambda x: x.get("_prior", 0.0), reverse=True
        )

    # --------- NEW: backprop implementation --------------------------------
    def _backprop(self, node: Node, value: float) -> None:
        cur = node
        while cur is not None:
            cur.update(value)
            cur = cur.parent
