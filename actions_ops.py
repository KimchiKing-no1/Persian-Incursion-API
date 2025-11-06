# engine/actions_ops.py
from typing import Dict, Any, Tuple

class OpsLoggingMixin:
    """
    Mixin for GameEngine that provides:
      - cost calculation (with overrides from rules if present)
      - spend with before/after logging
      - concrete handlers for ops with clear logs
    Expect the host class to provide:
      - self._log(state, msg)
      - self.rng
      - state structure with state["players"][side]["resources"] dict (PP/IP/MP)
      - self._opponent(side): "israel" <-> "iran"
      - optional rules-driven overrides on self.rules (dict) or self.rules["costs"]
    """

    # -----------------------------
    # Cost helpers
    # -----------------------------
    DEFAULT_COSTS = {
        "airstrike": {"IP": 3, "MP": 3},  # ISR
        # Special warfare / Terror are min..max; we log the actual chosen costs
        "special_warfare_min": {"IP": 1, "MP": 1},  # ISR
        "special_warfare_max": {"IP": 4, "MP": 3},
        "terror_min": {"IP": 1, "MP": 1},          # IRN
        "terror_max": {"IP": 4, "MP": 3},
        "bm_per_battalion": {"IP": 0, "MP": 1},    # IRN (1 IP + 1 MP per battalion) – tune if your rules differ
        # Strait: MP 1..7 (+ optional IP up to 2). We’ll pass the chosen values in action.
        "strait_ip_cap": 2,
        "strait_mp_min": 1,
        "strait_mp_max": 7,
    }

    def _rules_cost(self, key: str, fallback: Dict[str, int]) -> Dict[str, int]:
        try:
            return dict(self.rules.get("costs", {}).get(key, fallback))
        except Exception:
            return dict(fallback)

    def _get_player_res(self, state: Dict, side: str) -> Dict[str, int]:
        return state["players"][side]["resources"]

    def _can_afford(self, res: Dict[str, int], cost: Dict[str, int]) -> Tuple[bool, str]:
        for k, v in cost.items():
            if v is None:  # ignore Nones
                continue
            if v < 0:
                return False, f"invalid cost {k}={v}"
            if res.get(k, 0) < v:
                return False, f"needs {k}≥{v}, has {res.get(k,0)}"
        return True, ""

    def _spend(self, state: Dict, side: str, cost: Dict[str, int]) -> None:
        res = self._get_player_res(state, side)
        before = dict(res)
        ok, reason = self._can_afford(res, cost)
        if not ok:
            self._log(state, f"[COST] {side} cannot afford {cost}: {reason}")
            raise ValueError(f"{side} cannot afford {cost}: {reason}")
        for k, v in cost.items():
            if v:
                res[k] = res.get(k, 0) - v
        self._log(state, f"[COST] Applied to {side}: {cost} | {before} → {res}")

    # -----------------------------
    # Resolution stubs (replace with your real combat/impact code)
    # Each must RETURN a short human string for the log.
    # -----------------------------
    def _resolve_bm_attack(self, state, side, target, battalions, missile_type) -> str:
        # ← replace with your real implementation (damage rolls, defenses, etc.)
        return f"{battalions}x {missile_type} launched at {target} (resolution stub)"

    def _resolve_special_warfare(self, state, side, mission_name, target, ip_spent, mp_spent) -> str:
        # ← replace with your real implementation
        return f"SW mission '{mission_name}' vs {target} using IP={ip_spent}, MP={mp_spent} (resolution stub)"

    def _resolve_terror_attack(self, state, side, target, ip_spent, mp_spent) -> str:
        # ← replace with your real implementation
        return f"Terror attack vs {target} using IP={ip_spent}, MP={mp_spent} (resolution stub)"

    def _resolve_airstrike(self, state, side, target) -> str:
        # ← replace with your real implementation
        return f"Airstrike on {target} (resolution stub)"

    def _resolve_close_strait(self, state, side, mp_spent, ip_spent, pp_spent_from_israel) -> str:
        # ← replace with your real implementation
        return (f"Close Strait attempt MP={mp_spent}, IP={ip_spent}, "
                f"ISR counter PP={pp_spent_from_israel} (resolution stub)")

    # -----------------------------
    # Public action handlers (call these from apply_action)
    # -----------------------------

    # IRAN – Ballistic Missiles
    def action_order_ballistic_missile(self, state, side, target: str, battalions: int, missile_type: str):
        if side != "iran":
            raise ValueError("BM orders are for Iran only.")
        per = self._rules_cost("bm_per_battalion", self.DEFAULT_COSTS["bm_per_battalion"])
        cost = {"IP": per.get("IP", 0) * battalions,
                "MP": per.get("MP", 0) * battalions}
        self._log(state, f"[ACTION] {side}: Order Ballistic Missile → target={target}, "
                          f"bns={battalions}, type={missile_type}, cost={cost}")
        self._spend(state, side, cost)
        result = self._resolve_bm_attack(state, side, target, battalions, missile_type)
        self._log(state, f"[RESULT] BM: {result}")

    # ISRAEL – Special Warfare
    def action_order_special_warfare(self, state, side, mission_name: str, target: str, ip_spent: int, mp_spent: int):
        if side != "israel":
            raise ValueError("Special Warfare orders are for Israel only.")
        # Validate spend within min/max
        mn = self._rules_cost("special_warfare_min", self.DEFAULT_COSTS["special_warfare_min"])
        mx = self._rules_cost("special_warfare_max", self.DEFAULT_COSTS["special_warfare_max"])
        if not (mn["IP"] <= ip_spent <= mx["IP"] and mn["MP"] <= mp_spent <= mx["MP"]):
            raise ValueError(f"SW spend out of bounds: IP {ip_spent} in [{mn['IP']},{mx['IP']}], "
                             f"MP {mp_spent} in [{mn['MP']},{mx['MP']}]")
        cost = {"IP": ip_spent, "MP": mp_spent}
        self._log(state, f"[ACTION] {side}: Order Special Warfare → mission={mission_name}, "
                          f"target={target}, cost={cost}")
        self._spend(state, side, cost)
        result = self._resolve_special_warfare(state, side, mission_name, target, ip_spent, mp_spent)
        self._log(state, f"[RESULT] SW: {result}")

    # IRAN – Terror
    def action_order_terror_attack(self, state, side, target: str, ip_spent: int, mp_spent: int):
        if side != "iran":
            raise ValueError("Terror attacks are for Iran only.")
        mn = self._rules_cost("terror_min", self.DEFAULT_COSTS["terror_min"])
        mx = self._rules_cost("terror_max", self.DEFAULT_COSTS["terror_max"])
        if not (mn["IP"] <= ip_spent <= mx["IP"] and mn["MP"] <= mp_spent <= mx["MP"]):
            raise ValueError(f"Terror spend out of bounds: IP {ip_spent} in [{mn['IP']},{mx['IP']}], "
                             f"MP {mp_spent} in [{mn['MP']},{mx['MP']}]")
        cost = {"IP": ip_spent, "MP": mp_spent}
        self._log(state, f"[ACTION] {side}: Order Terror Attack → target={target}, cost={cost}")
        self._spend(state, side, cost)
        result = self._resolve_terror_attack(state, side, target, ip_spent, mp_spent)
        self._log(state, f"[RESULT] Terror: {result}")

    # ISRAEL – Airstrike
    def action_order_airstrike(self, state, side, target: str):
        if side != "israel":
            raise ValueError("Airstrikes are for Israel only.")
        base_cost = self._rules_cost("airstrike", self.DEFAULT_COSTS["airstrike"])
        cost = {"IP": base_cost.get("IP", 0), "MP": base_cost.get("MP", 0)}
        self._log(state, f"[ACTION] {side}: Order Airstrike → target={target}, cost={cost}")
        self._spend(state, side, cost)
        result = self._resolve_airstrike(state, side, target)
        self._log(state, f"[RESULT] Airstrike: {result}")

    # IRAN – Close Strait of Hormuz (with optional Israeli PP counter)
    def action_close_strait(self, state, side, mp_spent: int, ip_spent: int, israel_pp_counter: int = 0):
        if side != "iran":
            raise ValueError("Close Strait is for Iran only.")
        cap = self._rules_cost("strait_ip_cap", {"cap": self.DEFAULT_COSTS["strait_ip_cap"]})
        mp_min = self._rules_cost("strait_mp_min", {"min": self.DEFAULT_COSTS["strait_mp_min"]})["min"]
        mp_max = self._rules_cost("strait_mp_max", {"max": self.DEFAULT_COSTS["strait_mp_max"]})["max"]

        if not (mp_min <= mp_spent <= mp_max):
            raise ValueError(f"Strait MP out of bounds: {mp_spent} in [{mp_min},{mp_max}]")
        if not (0 <= ip_spent <= (cap.get("cap") if isinstance(cap, dict) else cap)):
            raise ValueError(f"Strait IP out of bounds: {ip_spent} in [0,{cap}]")
        # Spend IRN cost
        cost_irn = {"MP": mp_spent, "IP": ip_spent}
        self._log(state, f"[ACTION] {side}: Attempt Close Strait → cost={cost_irn}; "
                          f"ISR counter-PP={israel_pp_counter}")
        self._spend(state, side, cost_irn)
        # Spend ISR counter PP if any
        if israel_pp_counter:
            self._log(state, f"[COUNTER] israel spends PP={israel_pp_counter} to counter strait attempt")
            self._spend(state, "israel", {"PP": israel_pp_counter})
        result = self._resolve_close_strait(state, side, mp_spent, ip_spent, israel_pp_counter)

        self._log(state, f"[RESULT] Strait: {result}")
