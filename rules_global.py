# rules_global.py — canonical shim (drop-in)
import importlib

_rb = importlib.import_module("rules_blob")

# Prefer the canonical RULES built in rules_blob (added above).
RULES = getattr(_rb, "RULES", {})

# ---- Friendly aliases for engine convenience (no data copying) ----
# Cards: map both raw lists and structured forms
RULES.setdefault("cards", {
    "iran":   {c["No"]: c for c in RULES.get("IRAN_CARDS", [])},
    "israel": {c["No"]: c for c in RULES.get("ISRAEL_CARDS", [])},
})
RULES.setdefault("cards_structured", getattr(_rb, "CARDS_STRUCTURED", RULES.get("cards_structured", {})))

# Tables: provide short aliases that your engine/mechanics already probe
RULES.setdefault("pgm_table", RULES.get("PGM_ATTACK_TABLE", {}))
RULES.setdefault("sam_table", RULES.get("SAM_COMBAT_TABLE", {}))
RULES.setdefault("aaa_table", RULES.get("AAA_COMBAT_TABLE", {}))

# Targets & OOB
RULES.setdefault("targets", RULES.get("TARGET_DEFENSES", {}))
RULES.setdefault("valid_targets", RULES.get("VALID_TARGETS", []))
RULES.setdefault("squadrons", RULES.get("SQUADRONS", {}))

# Knobs (river/turn/action/victory/opinion)
RULES.setdefault("river_rules", RULES.get("RIVER_RULES", {"slots":7,"discard_rightmost":True}))
RULES.setdefault("restrike_rules", RULES.get("RESTRIKE_RULES", {"plan_delay_turns":1,"execute_window_turns":1}))
RULES.setdefault("action_costs", RULES.get("ACTION_COSTS", {}))
RULES.setdefault("victory_thresholds", RULES.get("VICTORY_THRESHOLDS", {}))
RULES.setdefault("airspace_rules", RULES.get("AIRSPACE_RULES", {}))
RULES.setdefault("opinion_income_table", RULES.get("OPINION_INCOME_TABLE", {}))
RULES.setdefault("third_party_income_table", RULES.get("THIRD_PARTY_INCOME_TABLE", {}))

__all__ = ["RULES"]
