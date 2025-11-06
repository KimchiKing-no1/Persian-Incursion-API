# mechanics.py
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# ---- RNG ----
def d6(n=1):  return [random.randint(1, 6) for _ in range(n)]
def d10(n=1): return [random.randint(1,10) for _ in range(n)]

# ---- opinion helpers ----
def clamp_opinion(x: int) -> int:
    return max(-10, min(10, x))

def value_to_category(val: int) -> str:
    a = abs(val)
    if a >= 9: return "Ally"
    if a >= 5: return "Supporter"
    if a >= 1: return "Cordial"
    return "Neutral"

# These will be set by set_rules(...) so we can read tables from your RULES/rules_blob
_RULES: Dict[str, Any] = {}

def set_rules(rules: Dict[str, Any]) -> None:
    global _RULES
    _RULES = rules or {}

def _get_table(name: str) -> Dict[str, Any]:
    tbl = _RULES.get(name) or _RULES.get(name.upper()) or {}
    if not tbl:
        raise KeyError(f"rules table '{name}' not found; make sure rules_blob exposes it")
    return tbl

# --------- Opinion dice (roll vs target numbers) ----------
def opinion_roll(actor_side: str, targets: List[str], dice: int, current_opinion: Dict[str, int]) -> Dict[str, List[int]]:
    op_tnums = _get_table("OPINION_TARGET_NUMBERS")
    out = {t: [] for t in targets}
    delta = +1 if actor_side == "israel" else -1
    for t in targets:
        for _ in range(dice):
            curr = current_opinion.get(t, 0)
            cat  = value_to_category(curr)
            tn   = op_tnums[cat]["target_roll"]
            r    = random.randint(1,10)
            out[t].append(r)
            if r >= tn:
                current_opinion[t] = clamp_opinion(curr + delta)
    return out

# --------- PGM / SAM / AAA (lightweight stubs you can call) ----------
@dataclass
class PgmAttackContext:
    weapon: str
    target_size: str
    target_armor: int
    modifiers: Dict[str, int] = field(default_factory=dict)

@dataclass
class PgmAttackResult:
    rounds: int
    hits_on_target: int
    penetrations: int
    attack_rolls: List[float]
    pen_rolls: List[int]

def _pgm_table(): return _get_table("PGM_ATTACK_TABLE")
def _sam_table(): return _get_table("SAM_COMBAT_TABLE")
def _targets_table(): return _get_table("TARGET_DEFENSES")

def compute_hit_chance(weapon_name: str, size: str, modifiers: Dict[str,int]) -> float:
    w = _pgm_table()[weapon_name]
    base = w["Hit_Chance_Target_Size"][size]
    adj = sum(modifiers.values()) * 0.01
    return max(0.0, min(0.99, base + adj))

def resolve_pgm_attack(ctx: PgmAttackContext) -> PgmAttackResult:
    w = _pgm_table()[ctx.weapon]
    n = int(w["Hits"])
    hit_p = compute_hit_chance(ctx.weapon, ctx.target_size, ctx.modifiers)
    rolls = [random.random() for _ in range(n)]
    hits = sum(1 for r in rolls if r < hit_p)

    armor = ctx.target_armor or 0
    pen_rolls: List[int] = []
    penetrations = 0
    if w["Armor_Pen"] is not None:
        for _ in range(hits):
            pr = random.randint(1, 100)
            pen_rolls.append(pr)
            if w["Armor_Pen"] >= armor:
                penetrations += 1
    else:
        penetrations = hits  # ARMs etc.
    return PgmAttackResult(n, hits, penetrations, rolls, pen_rolls)

# Optional: AAA & SAM helpers (you can expand later)