from __future__ import annotations
import math
import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# Optional Gemini callable signature:
#   def gemini(*, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str
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
    P: float = 1.0      # prior probability (optional)

    # Tree structure
    children: List["Node"] = field(default_factory=list)
    unexpanded_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Debug/lookup
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
        c_puct: Optional[float] = None,   # alias supported by run_dynamic.py
        rollout_depth: int = 32,
        time_budget_s: Optional[float] = None,
        gemini: Optional[GeminiCaller] = None,
        seed: Optional[int] = None,
        root_dirichlet_alpha: Optional[float] = None,
        root_dirichlet_eps: float = 0.25,
        verbose: bool = False,
        progress_every: int = 100,

        # === NEW (optional) hooks for value-function integration ===
        value_model: Optional[object] = None,   # torch.nn.Module with forward(x)->value in [-1,1]
        feature_fn: Optional[Callable[[Dict[str, Any]], "np.ndarray"]] = None,
    ):
        self.engine = engine
        self.side = side.lower().strip()  # 'israel' or 'iran'
        self.simulations = max(100, simulations)

        # ---- unify c_puct / c_uct ----
        if c_puct is not None:
            c_uct = float(c_puct)
        self.c_uct = float(c_uct)

        self.rollout_depth = rollout_depth
        self.time_budget_s = time_budget_s
        self.gemini = gemini
        self.rng = random.Random(seed)
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_eps = root_dirichlet_eps
        self.verbose = verbose
        self.progress_every = max(1, progress_every)

        # NEW: value fn hooks
        self.value_model = value_model
        self.feature_fn = feature_fn

        # Precompute rule-dependent constants for evaluation (safe if rules missing)
        self._oil_ref_total, self._oil_term_total = self._compute_total_oil_capacity()

        # Per-search caches
        self._transpo: Dict[str, Node] = {}

        # patch notice toggle
        self._patch_notice_shown = False


    # ----------------------------------------------------------------------
    # P U B L I C   A P I
    # ----------------------------------------------------------------------
    def choose_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a search tree from `state` and returns the action with highest visit count.
        """
        acts = self.engine.get_legal_actions(state, self.side)
        print(f"[legal/{self.side}] {len(acts)} -> {[a.get('type') for a in acts][:10]}")

        self._transpo.clear()

        root = self._make_node(state)
        if not root.unexpanded_actions and not root.children:
            root.unexpanded_actions = self._legal_actions(state)

        if not root.unexpanded_actions and not root.children:
            # no legal moves -> Pass if it's our turn; otherwise do nothing
            if state.get("turn", {}).get("current_player", "").lower() == self.side:
                return {"type": "Pass"}
            return {}

        # Root exploration noise
        if self.root_dirichlet_alpha and root.unexpanded_actions:
            self._inject_root_dirichlet(root)

        sims_done = 0
        t0 = time.time()
        while True:
            if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
                if self.verbose:
                    print(f"[MCTS] Time budget reached after {sims_done} sims.")
                break
            if self.time_budget_s is None and sims_done >= self.simulations:
                if self.verbose:
                    print(f"[MCTS] Simulation budget reached: {sims_done}.")
                break

            # ---- Determinize hidden info for THIS rollout
            det_state = self._determinize_hidden_info(copy.deepcopy(root.state))

            # ---- Selection
            leaf = self._select(root, det_state)

            # ---- Expansion
            if leaf.unexpanded_actions:
                child = self._expand(leaf)
            else:
                child = leaf

            # ---- Evaluation / Simulation
            # NEW: if a value model is provided, use it for leaf evaluation;
            #      otherwise, fall back to rollout simulation.
            value = self._leaf_value(child.state)

            # ---- Backprop
            self._backprop(child, value)

            sims_done += 1
            if self.verbose and (sims_done % self.progress_every == 0):
                best_q = max((c.Q for c in root.children), default=0.0)
                best_n = max((c.N for c in root.children), default=0)
                print(f"[MCTS] sims={sims_done} bestN={best_n} bestQ={best_q:.3f} children={len(root.children)}")

       # mcts_agent.py (AFTER)

            # If no expansions happened, return the first legal move with a 100% policy
            if not root.children:
                action = root.unexpanded_actions[0] if root.unexpanded_actions else {"type": "Pass"}
                policy = {json.dumps(action, sort_keys=True): 1.0}
                return action, policy
    
            # Calculate the policy from the visit counts of the root's children
            total_visits = sum(child.N for child in root.children)
            policy = {}
            if total_visits > 0:
                for child in root.children:
                    # The key is the JSON string of the action dictionary
                    action_key = json.dumps(child.incoming_action, sort_keys=True)
                    policy[action_key] = child.N / total_visits
    
            # The best action is still the one with the most visits
            best_child = max(root.children, key=lambda n: n.N)
            best_action = best_child.incoming_action
    
            if self.verbose and best_child is not None:
                print(f"[MCTS] Best action: {best_action} (N={best_child.N}, Q={best_child.Q:.3f})")
            
            # Return both the chosen action and the full policy
            return best_action, policy

    # ----------------------------------------------------------------------
    # M C T S   C O R E
    # ----------------------------------------------------------------------
    def _select(self, root: Node, rollout_state: Dict[str, Any]) -> Node:
        """
        Descend with UCT until a node with unexpanded actions or a leaf.
        Advance a rollout_state copy so keys align with determinization path.
        """
        node = root
        node.key = self._state_key(rollout_state)
        self._transpo[node.key] = node

        while True:
            if node.unexpanded_actions:
                return node
            if not node.children:
                return node

            log_N = math.log(max(1, node.N))
            def uct(n: Node) -> float:
                exploit = n.Q
                explore = self.c_uct * math.sqrt(log_N / max(1, n.N))
                return exploit + explore

            node = max(node.children, key=uct)

            if node.incoming_action:
                rollout_state = self._safe_apply(rollout_state, node.incoming_action)
                node.key = self._state_key(rollout_state)
                self._transpo[node.key] = node

            if self._safe_is_game_over(rollout_state) is not None:
                return node

    def _expand(self, node: Node) -> Node:
        """
        Pop one action and create a child by applying it.
        """
        action = node.unexpanded_actions.pop(0)
        child_state = self._safe_apply(copy.deepcopy(node.state), action)
        child = self._make_node(child_state, parent=node, incoming_action=action)
        node.children.append(child)
        return child

    # === NEW: leaf evaluation gate ==========================================
    def _leaf_value(self, state: Dict[str, Any]) -> float:
        """
        If a value model is present, evaluate the leaf with it.
        Otherwise, run the standard rollout simulation.
        """
        if self.value_model is not None and self.feature_fn is not None:
            return self._evaluate_leaf(state)
        return self._simulate(copy.deepcopy(state))

    def _evaluate_leaf(self, state: Dict[str, Any]) -> float:
        """
        Evaluate state using the provided value model (Israel-positive in [-1,1]).
        """
        try:
            import numpy as np
            import torch
        except Exception:
            # If torch/numpy not available, fall back.
            return self.evaluate_state(state)

        x = self.feature_fn(state)  # np.float32 array
        if not isinstance(x, np.ndarray):
            return self.evaluate_state(state)
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, D)
        self.value_model.eval()
        with torch.no_grad():
            v = self.value_model(xt).item()
        # clamp just in case
        if v > 1.0: v = 1.0
        if v < -1.0: v = -1.0
        return float(v)
    # ========================================================================

    def _simulate(self, sim_state: Dict[str, Any]) -> float:
        """
        Gemini-guided (or heuristic) rollout until terminal or depth cap.
        Returns a scalar in [-1, +1] from Israel's perspective.
        """
        depth = 0
        while depth < self.rollout_depth:
            winner = self._safe_is_game_over(sim_state)
            if winner is not None:
                return self._terminal_value(winner)

            legal = self._legal_actions(sim_state)
            if not legal:
                legal = [{"type": "Pass"}]

            move = self._gemini_rollout_move(sim_state, legal)
            sim_state = self._safe_apply(sim_state, move)
            depth += 1

        # Non-terminal -> evaluate heuristic
        return self.evaluate_state(sim_state)

    def _backprop(self, node: Node, value: float) -> None:
        """
        Backpropagate scalar value (Israel-positive) up to the root.
        """
        cur = node
        while cur is not None:
            cur.update(value)
            cur = cur.parent

    # ----------------------------------------------------------------------
    # N O D E / K E Y
    # ----------------------------------------------------------------------
    def _make_node(
        self,
        state: Dict[str, Any],
        parent: Optional[Node] = None,
        incoming_action: Optional[Dict[str, Any]] = None,
    ) -> Node:
        node = Node(
            state=state,
            parent=parent,
            incoming_action=incoming_action,
            unexpanded_actions=self._legal_actions(state),
            key=self._state_key(state),
        )
        self._transpo[node.key] = node
        return node


    def _state_key(self, state: Dict[str, Any]) -> str:
        """
        Deterministic, compact key. If state contains unserializable elements,
        fall back to a weaker key.
        """
        try:
            
            state_for_key = state.copy()
            
            
            state_for_key.pop('_rng', None)
            state_for_key.pop('_rng_seeded', None)

            return json.dumps(state_for_key, sort_keys=True, separators=(",", ":"))[:4096]
        except Exception:
            return f"{id(state)}:{state.get('turn', {}).get('turn_number')}"
        
    # ----------------------------------------------------------------------
    # L E G A L   M O V E S
    # ----------------------------------------------------------------------
    def _legal_actions(self, state):
        cur = (state.get("turn", {}).get("current_player") or "").lower()
        try:
            acts = self.engine.get_legal_actions(state, cur)
            return acts if acts else [{"type":"Pass"}]
        except Exception:
            return [{"type":"Pass"}]

    def _cost_extract(self, cost: str, letter: str) -> int:
        """
        Extract numeric requirement for a letter ('P','I','M') from a cost string.
        Robust to formats like '3P, 2I' or '1 M'.
        """
        letter = letter.upper()
        total = 0
        cur_num = ""
        for ch in cost:
            if ch.isdigit():
                cur_num += ch
            elif ch.isalpha():
                if cur_num and ch.upper() == letter:
                    total += int(cur_num)
                cur_num = ""
            else:
                cur_num = ""
        return total

    # ----------------------------------------------------------------------
    # D E T E R M I N I Z A T I O N  (Hidden info via Gemini)
    # ----------------------------------------------------------------------
    def _determinize_hidden_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For the opponent, if river unknown, ask Gemini to propose a plausible
        defensive river of 7 card names and map to IDs. Fallback to random.
        """
        cur_player = state.get("turn", {}).get("current_player", "").lower()
        opp = "iran" if cur_player == "israel" else "israel"

        river = state.get("players", {}).get(opp, {}).get("river")
        if isinstance(river, list) and river:
            return state  # already known

        state.setdefault("players", {}).setdefault(opp, {}).setdefault("river", [])
        desired = 7
        predicted_names: List[str] = []

        # Try Gemini
        summary = self._compact_state_summary(state)
        prompt = (
            f"You are an expert Persian Incursion player. Opponent is {opp}.\n"
            f"State summary:\n{summary}\n\n"
            f"Propose a plausible DEFENSIVE river of {desired} card names for {opp} as a pure JSON array of strings."
        )
        try:
            out = self._call_gemini(prompt, temperature=0.8, max_tokens=256)
            predicted_names = self._extract_json_list(out)
        except Exception:
            predicted_names = []

        # Map names→ids
        card_ids = self._map_card_names_to_ids(opp, predicted_names)

        # Fill up with random if needed
        if len(card_ids) < desired:
            all_ids = self._all_card_ids_for_side(opp)
            self.rng.shuffle(all_ids)
            for cid in all_ids:
                if cid not in card_ids:
                    card_ids.append(cid)
                if len(card_ids) >= desired:
                    break

        state["players"][opp]["river"] = card_ids[:desired]
        return state

    def _call_gemini(self, *, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        if self.gemini is None:
            return ""
        # Some callers use gemini(prompt=..., ...); we ensure kwargs.
        return self.gemini(prompt=prompt, temperature=temperature, max_tokens=max_tokens)

    def _compact_state_summary(self, state: Dict[str, Any]) -> str:
        t = state.get("turn", {})
        op = state.get("opinion", {})
        resources = {s: state.get("players", {}).get(s, {}).get("resources", {}) for s in ("israel", "iran")}
        return json.dumps(
            {
                "turn": t,
                "opinion": op,
                "resources": resources,
                "events_queued": len(state.get("active_events_queue", [])),
                "known_rivers": {
                    "israel": state.get("players", {}).get("israel", {}).get("river", []),
                    "iran": state.get("players", {}).get("iran", {}).get("river", []),
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def _extract_json_list(self, text: str) -> List[str]:
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                val = json.loads(text[start : end + 1])
                if isinstance(val, list):
                    return [str(x) for x in val]
        except Exception:
            pass
        return []

    def _map_card_names_to_ids(self, side: str, names: List[str]) -> List[int]:
        cards = self.engine.rules.get("cards", {}).get(side, {})
        result: List[int] = []
        if isinstance(cards, dict):
            name_to_id: Dict[str, Any] = {}
            for cid, c in cards.items():
                nm = (c.get("Name") or c.get("name") or "").strip().lower()
                if nm:
                    name_to_id[nm] = int(cid) if isinstance(cid, (int, str)) and str(cid).isdigit() else cid
            for n in names:
                key = (n or "").strip().lower()
                if key in name_to_id:
                    result.append(int(name_to_id[key]) if str(name_to_id[key]).isdigit() else name_to_id[key])
        else:
            # list of card dicts
            name_to_id = {}
            for c in cards:
                cid = c.get("No") or c.get("id")
                nm = (c.get("Name") or c.get("name") or "").strip().lower()
                if cid is not None and nm:
                    name_to_id[nm] = int(cid) if str(cid).isdigit() else cid
            for n in names:
                key = (n or "").strip().lower()
                if key in name_to_id:
                    result.append(name_to_id[key])
        return result

    def _all_card_ids_for_side(self, side: str) -> List[int]:
        cards = self.engine.rules.get("cards", {}).get(side, {})
        if isinstance(cards, dict):
            ids: List[int] = []
            for k in cards.keys():
                if isinstance(k, int) or str(k).isdigit():
                    ids.append(int(k))
            return ids
        else:
            ids = []
            for c in cards:
                if c.get("No") is not None and str(c["No"]).isdigit():
                    ids.append(int(c["No"]))
                elif c.get("id") is not None and str(c["id"]).isdigit():
                    ids.append(int(c["id"]))
            return ids

    # ----------------------------------------------------------------------
    # R O L L O U T   P O L I C Y  (Gemini or heuristic)
    # ----------------------------------------------------------------------
    def _gemini_rollout_move(self, state: Dict[str, Any], legal: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ask Gemini to choose among legal moves. If unavailable or unparsable,
        fall back to a strong, fast heuristic policy.
        """
        if not self.gemini:
            return self._fast_policy(state, legal)

        side_to_move = state.get("turn", {}).get("current_player", "").lower()
        summary = self._compact_state_summary(state)
        legal_text = json.dumps(legal, separators=(",", ":"))

        prompt = (
            f"You are {side_to_move} in Persian Incursion.\n"
            f"State: {summary}\n"
            f"Choose ONE good move from these legal actions (JSON list): {legal_text}\n"
            "Return ONLY the chosen action as a compact JSON object."
        )
        try:
            out = self._call_gemini(prompt=prompt, temperature=0.6, max_tokens=192)
            start = out.find("{")
            end = out.rfind("}")
            if start >= 0 and end > start:
                chosen = json.loads(out[start : end + 1])
                if isinstance(chosen, dict) and chosen.get("type"):
                    return chosen
        except Exception:
            pass

        return self._fast_policy(state, legal)

    def _fast_policy(self, state: Dict[str, Any], legal: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heuristic fallback:
          - Israel: prefer Airstrike (if affordable) > cheapest Play Card > SpecWar > Pass
          - Iran:   prefer Ballistic Missile > Terror > cheapest Play Card > Pass
        """
        side_to_move = state.get("turn", {}).get("current_player", "").lower()

        # Israel
        if side_to_move == "israel":
            res = state["players"]["israel"].get("resources", {})
            if res.get("mp", 0) >= 3 and res.get("ip", 0) >= 3:
                for a in legal:
                    if a.get("type") == "Order Airstrike":
                        return a
        # Iran
        if side_to_move == "iran":
            res = state["players"]["iran"].get("resources", {})
            if res.get("mp", 0) >= 1:
                for a in legal:
                    if a.get("type") == "Order Ballistic Missile":
                        return a
            if res.get("mp", 0) >= 1 and res.get("ip", 0) >= 1:
                for a in legal:
                    if a.get("type") == "Order Terror Attack":
                        return a

        # Cheapest Play Card
        best = None
        best_cost = 10**9
        cards_rules_map = self.engine.rules.get("cards", {}).get(side_to_move, {})
        for a in legal:
            if a.get("type") == "Play Card":
                cid = a.get("card_id")
                card = cards_rules_map.get(cid) if isinstance(cards_rules_map, dict) else None
                if card is None and isinstance(cards_rules_map, list):
                    for c in cards_rules_map:
                        if c.get("No") == cid or c.get("id") == cid:
                            card = c
                            break
                if card:
                    c = (card.get("Cost") or "").upper()
                    cost_val = (
                        self._cost_extract(c, "P")
                        + self._cost_extract(c, "I")
                        + self._cost_extract(c, "M")
                    )
                    if cost_val < best_cost:
                        best_cost = cost_val
                        best = a
        if best:
            return best

        # Otherwise Pass
        for a in legal:
            if a.get("type") == "Pass":
                return a
        return legal[0]

    # ----------------------------------------------------------------------
    # E V A L U A T I O N  (-1..+1, Israel-positive)
    # ----------------------------------------------------------------------
    def evaluate_state(self, state: Dict[str, Any]) -> float:
        """
        Composite score:
          + Opinion differential (scaled)
          + Nuclear primary destruction ratio
          + Oil capacity destroyed fraction (refinery+terminal)
          - Attrition penalty
        Tunable weights; output clamped to [-1, +1].
        """
        # Domestic opinion: Israel up, Iran down, normalized ~[-1,+1]
        dom = state.get("opinion", {}).get("domestic", {})
        is_dom = float(dom.get("israel", dom.get("Israel", 0)))
        ir_dom = float(dom.get("iran", dom.get("Iran", 0)))
        base = (is_dom - (-ir_dom)) / 20.0  # assumes ±10 scale; adjust if yours differs

        # Nuclear posture
        nuc_ratio = self._nuclear_destroy_ratio(state)

        # Oil damage
        oil_ratio = self._oil_destroy_fraction(state)

        # Attrition
        attr = self._attrition_penalty(state)

        # Weights
        score = (
            0.20 * base +        # Opinion is secondary to hard damage
            0.45 * nuc_ratio +   # Nuclear damage is a high priority
            0.35 * oil_ratio -   # Oil damage is also very important
            0.10 * attr          # Attrition is a minor penalty
        )

        return max(-1.0, min(1.0, score))

    def _terminal_value(self, winner: Optional[str]) -> float:
        if winner is None:
            return 0.0
        w = str(winner).lower()
        if w == "israel":
            return 1.0
        if w == "iran":
            return -1.0
        return 0.0

    # ---- Evaluation subroutines ---------------------------------------------
    def _nuclear_destroy_ratio(self, state: Dict[str, Any]) -> float:
        """
        Fraction of primary nuclear components destroyed (per rules data).
        """
        rules_targets = self.engine.rules.get("targets", {}) if hasattr(self.engine, "rules") else {}
        prim_total = 0
        prim_destroyed = 0

        for tname, trules in rules_targets.items():
            types = trules.get("Target_Types", [])
            if "Nuclear" not in types and "Nuclear_Infra" not in types:
                continue
            prims = trules.get("Primary_Targets", {}) or {}
            if not prims:
                continue
            prim_total += len(prims)

            dammap = state.get("target_damage_status", {}).get(tname, {})
            for comp_id, comp_rules in prims.items():
                destroyed_th = comp_rules.get("max_damage_for_destroyed") or comp_rules.get("destroyed_threshold") or comp_rules.get("damaged_threshold") or 10**9
                boxes = self._component_boxes(dammap.get(comp_id))
                if boxes >= destroyed_th:
                    prim_destroyed += 1

        if prim_total == 0:
            return 0.0
        return prim_destroyed / prim_total

    def _oil_destroy_fraction(self, state: Dict[str, Any]) -> float:
        """
        Average destroyed capacity fraction across refineries and terminals,
        using undamaged/damaged/destroyed → 100/50/0% capacity.
        """
        cur_ref, cur_term = self._current_oil_capacity(state)
        ref_frac = 0.0
        term_frac = 0.0
        if self._oil_ref_total > 0:
            ref_frac = 1.0 - (cur_ref / self._oil_ref_total)
        if self._oil_term_total > 0:
            term_frac = 1.0 - (cur_term / self._oil_term_total)
        return 0.5 * ref_frac + 0.5 * term_frac

    def _attrition_penalty(self, state: Dict[str, Any]) -> float:
        """
        Penalize Israel aircraft in 'Returning' and damage steps (capped to 1.0).
        """
        pen = 0.0
        sq = state.get("squadrons", {}).get("israel", {})
        pen += sum(1 for s in sq.values() if s == "Returning") * 0.02
        steps = state.get("aircraft_damage", {}).get("israel", {})
        pen += sum(max(0, int(v)) for v in steps.values()) * 0.03
        return min(1.0, pen)

    def _component_boxes(self, entry: Any) -> int:
        if isinstance(entry, dict):
            return int(entry.get("damage_boxes_hit", 0))
        if isinstance(entry, int):
            return int(entry)
        return 0

    def _compute_total_oil_capacity(self) -> Tuple[float, float]:
        """
        Sum barrels_per_day for Oil_Refinery and Oil_Terminal targets.
        """
        ref_total = 0.0
        term_total = 0.0
        rules_targets = self.engine.rules.get("targets", {}) if hasattr(self.engine, "rules") else {}
        for _, trules in rules_targets.items():
            prod = trules.get("Production", {}) or {}
            bpd = float(prod.get("barrels_per_day", 0) or 0)
            types = trules.get("Target_Types", []) or []
            if "Oil_Refinery" in types:
                ref_total += bpd
            if "Oil_Terminal" in types:
                term_total += bpd
        return ref_total, term_total

    def _current_oil_capacity(self, state: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute current effective capacity based on per-target status:
          undamaged=1.0, damaged=0.5, destroyed=0.0
        """
        cur_ref = 0.0
        cur_term = 0.0
        rules_targets = self.engine.rules.get("targets", {}) if hasattr(self.engine, "rules") else {}
        for tname, trules in rules_targets.items():
            prod = trules.get("Production", {}) or {}
            bpd = float(prod.get("barrels_per_day", 0) or 0)
            if hasattr(self.engine, "_get_target_status"):
                status = self.engine._get_target_status(state, tname)
            else:
                dammap = (state.get("target_damage_status", {}) or {}).get(tname, {})
                destroyed = any(
                    (isinstance(v, dict) and v.get("damage_boxes_hit", 0) >= v.get("max_damage_for_destroyed", 10**9))
                    or (isinstance(v, int) and v >= 99)
                    for v in dammap.values()
                )
                damaged = (not destroyed) and any(
                    (isinstance(v, dict) and v.get("damage_boxes_hit", 0) > 0)
                    or (isinstance(v, int) and v > 0)
                    for v in dammap.values()
                )
                status = "destroyed" if destroyed else ("damaged" if damaged else "undamaged")

            mult = 1.0 if status == "undamaged" else (0.5 if status == "damaged" else 0.0)
            if "Oil_Refinery" in trules.get("Target_Types", []):
                cur_ref += bpd * mult
            if "Oil_Terminal" in trules.get("Target_Types", []):
                cur_term += bpd * mult
        return cur_ref, cur_term

    # ----------------------------------------------------------------------
    # R O O T   E X P L O R A T I O N  (Dirichlet)
    # ----------------------------------------------------------------------
    def _inject_root_dirichlet(self, root: Node) -> None:
        """
        Blend Dirichlet noise into root priors; reorders expansion list accordingly.
        """
        K = max(1, len(root.unexpanded_actions))
        noise = self._dirichlet([self.root_dirichlet_alpha] * K)
        for i, a in enumerate(root.unexpanded_actions):
            a["_prior"] = (1.0 - self.root_dirichlet_eps) * (1.0 / K) + self.root_dirichlet_eps * noise[i]
        root.unexpanded_actions.sort(key=lambda x: x.get("_prior", 0.0), reverse=True)

    def _dirichlet(self, alphas: List[float]) -> List[float]:
        gs = [self._gamma(a) for a in alphas]
        s = sum(gs)
        if s <= 0:
            return [1.0 / len(gs)] * len(gs)
        return [g / s for g in gs]

    def _gamma(self, alpha: float) -> float:
        return self.rng.gammavariate(alpha, 1.0)

    # ----------------------------------------------------------------------
    # S A F E   E N G I N E   W R A P S
    # ----------------------------------------------------------------------
    def _safe_apply(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self.engine, "apply_action"):
            try:
                return self.engine.apply_action(state, action)
            except Exception as e:
                if self.verbose:
                    print(f"[MCTS] apply_action error {e}; falling back to no-op.")
        return state

    def _safe_is_game_over(self, state: Dict[str, Any]) -> Optional[str]:
        if hasattr(self.engine, "is_game_over"):
            try:
                return self.engine.is_game_over(state)
            except Exception as e:
                if self.verbose:
                    print(f"[MCTS] is_game_over error {e}; assuming not over.")
        return None


# === BEGIN PATCH: no-op warnings ===
def _safe_apply_patched(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    if not getattr(self, "_patch_notice_shown", False):
        print("[PATCH] _safe_apply_patched active")
        self._patch_notice_shown = True
    if hasattr(self.engine, "apply_action"):
        try:
            before = self._state_key(state)
            new_state = self.engine.apply_action(state, action)
            after = self._state_key(new_state)
            if before == after and action.get("type") != "Pass" and getattr(self, "verbose", False):
                print("[WARN] apply_action produced no state change:", action)
            return new_state
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"[MCTS] apply_action error {e}; falling back to no-op.")
    return state

def _expand_patched(self, node: Node) -> Node:
    # Pop one action and create a child by applying it.
    action = node.unexpanded_actions.pop(0)
    before = self._state_key(node.state)
    child_state = self._safe_apply(copy.deepcopy(node.state), action)
    after = self._state_key(child_state)
    if before == after and action.get("type") != "Pass" and getattr(self, "verbose", False):
        print("[WARN] EXPAND edge is no-op:", action)
    child = self._make_node(child_state, parent=node, incoming_action=action)
    node.children.append(child)
    return child

# Bind the patches
MCTSAgent._safe_apply = _safe_apply_patched
MCTSAgent._expand = _expand_patched
# === END PATCH: no-op warnings ===
# === BEGIN PATCH: policy-value integration ===
try:
    from model import PVModel           # your lightweight torch/nn model
    from features import featurize, action_key  # state/action featurizers
except Exception as _e:
    PVModel = None
    featurize = None
    action_key = None

def _attach_pv_model(self, model_path: str | None):
    """
    Call once to attach a trained policy-value model.
    Safe: if model or featurizer not present, it becomes a no-op.
    """
    self.pv_model = None
    if not model_path or PVModel is None or featurize is None or action_key is None:
        if getattr(self, "verbose", False):
            print("[PV] Model/featurizer not available; running heuristic-only.")
        return
    try:
        self.pv_model = PVModel.load(model_path)
        if getattr(self, "verbose", False):
            print(f"[PV] Loaded policy-value model from: {model_path}")
    except Exception as e:
        self.pv_model = None
        if getattr(self, "verbose", False):
            print(f"[PV] Failed to load model ({e}); running heuristic-only.")

def _pv_eval(self, state: Dict[str, Any]) -> tuple[dict, float] | None:
    """
    Returns (priors_map, value) or None if no model is attached.
    priors_map: {action_key_str: prob}
    value: float in [-1, 1] from Israel's perspective.
    """
    if not getattr(self, "pv_model", None):
        return None
    try:
        side = (state.get("turn", {}).get("current_player") or "").lower()
        feats = featurize(state, side)
        priors_map, value = self.pv_model.infer(feats)
        # Basic hygiene
        if not isinstance(priors_map, dict):
            priors_map = {}
        value = float(max(-1.0, min(1.0, value)))
        return priors_map, value
    except Exception as e:
        if getattr(self, "verbose", False):
            print(f"[PV] infer error {e}; ignoring model this node.")
        return None

# --- Hook priors into node creation ---
_old_make_node = MCTSAgent._make_node
def _make_node_pv(self, state: Dict[str, Any], parent=None, incoming_action=None):
    node = _old_make_node(self, state, parent=parent, incoming_action=incoming_action)

    # If we have a PV model, stamp priors onto unexpanded actions
    pv = _pv_eval(self, state)
    if pv is not None and node.unexpanded_actions:
        priors_map, _v = pv
        # map each concrete action -> prior via features.action_key
        raw = []
        for a in node.unexpanded_actions:
            k = action_key(a)
            p = float(priors_map.get(k, 0.0))
            a["_prior"] = p
            raw.append(p)
        s = sum(raw) if raw else 0.0
        if s > 0:
            # normalize to a distribution
            node.unexpanded_actions.sort(key=lambda x: x.get("_prior", 0.0), reverse=True)
        else:
            # fall back to uniform and let Dirichlet (if any) handle noise
            for a in node.unexpanded_actions:
                a["_prior"] = 1.0 / len(node.unexpanded_actions)
    return node
MCTSAgent._make_node = _make_node_pv

# --- Use value head as leaf evaluator in rollouts ---
_old_simulate = MCTSAgent._simulate
def _simulate_pv(self, sim_state: Dict[str, Any]) -> float:
    pv = _pv_eval(self, sim_state)
    if pv is not None:
        # Use the value head instantly (fast, AlphaZero-style leaf eval)
        _priors, v = pv
        return v
    # If no model, use original simulate (random/heuristic rollout)
    return _old_simulate(self, sim_state)

MCTSAgent._simulate = _simulate_pv

# --- convenience: allow passing model_path from runner without editing MCTSAgent signature ---
setattr(MCTSAgent, "_attach_pv_model", _attach_pv_model)
# === END PATCH: policy-value integration ===
