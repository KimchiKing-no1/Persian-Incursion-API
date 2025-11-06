 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/game_engine.py b/game_engine.py
index 7f17d83df4d6a04e437b3a63d7fd2aad245b3f93..3b02847f2f72a38c39a156049284b5098cb35d33 100644
--- a/game_engine.py
+++ b/game_engine.py
@@ -1,28 +1,28 @@
 # game_engine.py
 import copy, random, re
-from typing import Optional
+from typing import Any, Dict, List, Optional
 
 from mechanics import set_rules as mech_set_rules, opinion_roll
 from rules_global import RULES
 from actions_ops import OpsLoggingMixin
 
 # ===== OPINION → INCOME (rules-accurate) =====================================
 DOMESTIC_OPINION_INCOME = {
     "israel": [
         (9,  10, (6, 7, 10)),
         (5,   8, (5, 6, 10)),
         (2,   4, (4, 5, 10)),
         (-1,  1, (3, 5,  9)),
         (-4, -2, (2, 3,  8)),
         (-8, -5, (1, 1,  8)),
         (-10,-9, (0, 0,  6)),
     ],
     "iran": [
         (9,  10, (1, 0, 0)),
         (5,   8, (2, 1, 1)),
         (2,   4, (3, 2, 3)),
         (-1,  1, (4, 3, 5)),
         (-4, -2, (5, 4, 6)),  
         (-8, -5, (6, 5, 6)),
         (-10,-9, (7, 6, 6)),
     ],
@@ -1489,50 +1489,80 @@ class GameEngine:
         # (7) Make sure initial SAMs exist if scenario didn’t define them
         self._bootstrap_sams_if_missing(state)
 
         return state
 
       
     
     def _log_diff(self, before, after):
         def flat(d, prefix=""):
             out = {}
             for k, v in d.items():
                 if isinstance(v, dict):
                     for k2, v2 in flat(v, prefix=f"{prefix}{k}.").items():
                         out[k2] = v2
                 else:
                     out[f"{prefix}{k}"] = v
             return out
         b, a = flat(before), flat(after)
         diffs = []
         for key in sorted(set(b) | set(a)):
             if b.get(key) != a.get(key):
                 diffs.append(f"{key}: {b.get(key)} → {a.get(key)}")
         return "; ".join(diffs) if diffs else "no change"
 
     # ----------------------------- PUBLIC UTILITIES ----------------------------
    def apply_actions(self, state: Dict[str, Any], actions: List[Dict[str, Any]], side: Optional[str] = None):
        """Apply a sequence of actions, returning the new state and accumulated log."""
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list of dicts")

        working = copy.deepcopy(state)
        turn = working.setdefault("turn", {})
        if side:
            turn["current_player"] = side

        working.setdefault("log", [])
        start = len(working["log"])
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            try:
                updated = self.apply_action(working, action)
            except Exception as exc:
                raise ValueError(f"apply_actions[{idx}] failed for {action}: {exc}") from exc
            if isinstance(updated, dict):
                working = updated

        new_entries = []
        if isinstance(working.get("log"), list):
            new_entries = [str(entry) for entry in working["log"][start:]]

        return working, new_entries

     def apply_action(self, state, action):
         """
         Multi-action impulses:
         - You may issue multiple ops (Airstrike / Special Warfare / BM / Terror) in one impulse.
         - You may play AT MOST one card per impulse.
         - The impulse ends ONLY when the side does 'End Impulse' (or if no legal actions remain).
         - River/card rules: played card is removed → discard, river slides/right-align, refill from LEFT.
         """
         # Validate against legal generator for current side
         side_now = state.get("turn", {}).get("current_player", "israel")
         legal = self.get_legal_actions(state, side=side_now)
     
         def _same(a, b):
             if a.get("type") != b.get("type"):
                 return False
             # allow subset match (legal may omit optional fields)
             for k, v in a.items():
                 if k not in b or b[k] != v:
                     return False
             return True
     
         if not any(_same(action, a) for a in legal):
             self._log(state, f"[WARN] Illegal/blocked action refused: {action}")
             return state
     
 
EOF
)
