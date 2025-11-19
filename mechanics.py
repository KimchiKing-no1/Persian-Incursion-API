# mechanics.py
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union

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
        # Graceful fallback if rules aren't loaded yet to prevent import crashes
        return {} 
    return tbl

# --------- Opinion dice (roll vs target numbers) ----------
def opinion_roll(actor_side: str, targets: List[str], dice: int, current_opinion: Dict[str, int]) -> Dict[str, List[int]]:
    op_tnums = _get_table("OPINION_TARGET_NUMBERS")
    if not op_tnums: return {t: [] for t in targets} # Safety check
    
    out = {t: [] for t in targets}
    delta = +1 if actor_side == "israel" else -1
    for t in targets:
        for _ in range(dice):
            curr = current_opinion.get(t, 0)
            cat  = value_to_category(curr)
            # Fallback to Neutral target if category missing
            tn_data = op_tnums.get(cat) or op_tnums.get("Neutral", {"target_roll": 6})
            tn = tn_data["target_roll"]
            
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
    # FIX: Armor can be int (e.g., 7) or str ("Heavy")
    target_armor: Union[int, str] 
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
    w = _pgm_table().get(weapon_name, {})
    if not w: return 0.0
    
    # Handle missing size class gracefully
    hit_chances = w.get("Hit_Chance_Target_Size", {})
    base = hit_chances.get(size, hit_chances.get("D", 0.08)) 
    
    adj = sum(modifiers.values()) * 0.01
    return max(0.0, min(0.99, base + adj))

def resolve_pgm_attack(ctx: PgmAttackContext) -> PgmAttackResult:
    w = _pgm_table().get(ctx.weapon, {})
    if not w: return PgmAttackResult(0,0,0,[],[])

    n = int(w.get("Hits", 1))
    hit_p = compute_hit_chance(ctx.weapon, ctx.target_size, ctx.modifiers)
    rolls = [random.random() for _ in range(n)]
    hits = sum(1 for r in rolls if r < hit_p)

    # FIX: Sanitize armor value (handle "Heavy")
    armor_val = ctx.target_armor
    if armor_val == "Heavy":
        armor_val = 100 # Treat Heavy as effectively invulnerable to non-pen weapons
    else:
        armor_val = int(armor_val or 0)

    pen_rolls: List[int] = []
    penetrations = 0
    
    weapon_pen = w.get("Armor_Pen")
    
    # Logic: If weapon has Pen, compare to Armor.
    # If Pen >= Armor, it's a "Penetration" (Full Damage).
    # If Pen < Armor, damage is usually reduced (quartered).
    # This helper just counts "Full Damage" hits.
    if weapon_pen is not None:
        for _ in range(hits):
            pr = random.randint(1, 100)
            pen_rolls.append(pr)
            if int(weapon_pen) >= armor_val:
                penetrations += 1
    else:
        # Weapons with no Pen stat (like ARMs or simple HE) might behave differently,
        # but default to treating hit as penetration if armor is 0.
        penetrations = hits if armor_val == 0 else 0
        
    return PgmAttackResult(n, hits, penetrations, rolls, pen_rolls)
