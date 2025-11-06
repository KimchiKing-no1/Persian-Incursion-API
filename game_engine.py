# game_engine.py
import copy, random, re
from typing import Any, Dict, List, Optional

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
}

THIRD_PARTY_OPINION_INCOME = {
    "prc": [
        ( 9, 10, {"israel": (2, 2, 0)}),
        ( 5,  8, {"israel": (1, 2, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"iran":   (1, 0, 0)}),
        (-4, -1, {"iran":   (1, 1, 2)}),
        (-8, -5, {"iran":   (2, 2, 2)}),
        (-10,-9, {"iran":   (4, 4, 4)}),
    ],
    "russia": [
        ( 9, 10, {"israel": (2, 2, 0)}),
        ( 5,  8, {"israel": (1, 2, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"iran":   (1, 0, 0)}),
        (-4, -1, {"iran":   (1, 2, 0)}),
        (-8, -5, {"iran":   (2, 3, 2)}),
        (-10,-9, {"iran":   (4, 4, 4)}),
    ],
    "saudi_gcc": [
        ( 9, 10, {"israel": (3, 2, 2)}),
        ( 5,  8, {"israel": (2, 0, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"israel": (0, 0, 0)}),
        (-4, -1, {"iran":   (1, 0, 0)}),
        (-8, -5, {"iran":   (2, 0, 0)}),
        (-10,-9, {"iran":   (3, 2, 2)}),
    ],
    "un": [
        ( 9, 10, {"israel": (4, 1, 0)}),
        ( 5,  8, {"israel": (2, 0, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"israel": (0, 0, 0)}),
        (-4, -1, {"iran":   (1, 0, 0)}),
        (-8, -5, {"iran":   (2, 0, 0)}),
        (-10,-9, {"iran":   (4, 1, 0)}),
    ],
    "jordan": [
        ( 9, 10, {"israel": (0, 0, 0)}),
        ( 5,  8, {"israel": (2, 0, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"israel": (0, 0, 0)}),
        (-4, -1, {"iran":   (1, 0, 0)}),
        (-8, -5, {"iran":   (2, 0, 0)}),
        (-10,-9, {"iran":   (3, 2, 2)}),
    ],
    "turkey": [
        ( 9, 10, {"israel": (3, 2, 2)}),
        ( 5,  8, {"israel": (2, 1, 0)}),
        ( 1,  4, {"israel": (1, 0, 0)}),
        ( 0,  0, {"israel": (0, 0, 0)}),
        (-4, -1, {"iran":   (1, 0, 0)}),
        (-8, -5, {"iran":   (2, 1, 0)}),
        (-10,-9, {"iran":   (3, 2, 0)}),
    ],
    "usa": [
        ( 9, 10, {"israel": (4, 4, 4)}),
        ( 5,  8, {"israel": (2, 3, 2)}),
        ( 1,  4, {"israel": (1, 2, 0)}),
        ( 0,  0, {"israel": (1, 0, 0)}),
        (-4, -1, {"iran":   (1, 0, 0)}),
        (-8, -5, {"iran":   (2, 1, 1)}),
        (-10,-9, {"iran":   (4, 2, 1)}),
    ],
}

_THIRD_PARTY_ALIASES = {
    "china": "prc", "prc": "prc",
    "russia": "russia",
    "saudi": "saudi_gcc", "saudi_arabia": "saudi_gcc", "gcc": "saudi_gcc",
    "un": "un", "united_nations": "un",
    "jordan": "jordan",
    "turkey": "turkey",
    "usa": "usa", "united_states": "usa", "us": "usa",
}

def _sum3(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def opinion_income_from_domestic(state, side):
    val = int(state.get("opinion", {}).get("domestic", {}).get(side, 0))
    for lo, hi, trip in DOMESTIC_OPINION_INCOME[side]:
        if lo <= val <= hi:
            return trip
    return (0,0,0)

def opinion_income_from_third_parties(state, side):
    total = (0,0,0)
    third = state.get("opinion", {}).get("third_party", {}) or \
            state.get("opinion", {}).get("third_parties", {}) or {}
    for raw_k, v in third.items():
        k = _THIRD_PARTY_ALIASES.get(str(raw_k).strip().lower())
        if not k: continue
        for lo, hi, payload in THIRD_PARTY_OPINION_INCOME.get(k, []):
            if lo <= int(v) <= hi:
                total = _sum3(total, payload.get(side, (0,0,0)))
                break
    return total

def apply_morning_opinion_income(state, carry_cap=None, log_fn=None):
    state.setdefault("players", {}).setdefault("israel", {}).setdefault("resources", {"pp":0,"ip":0,"mp":0})
    state.setdefault("players", {}).setdefault("iran",   {}).setdefault("resources", {"pp":0,"ip":0,"mp":0})
    for side in ("israel","iran"):
        add = _sum3(opinion_income_from_domestic(state, side),
                    opinion_income_from_third_parties(state, side))
        r = state["players"][side]["resources"]
        r["pp"] = r.get("pp",0)+add[0]; r["ip"]=r.get("ip",0)+add[1]; r["mp"]=r.get("mp",0)+add[2]
        if carry_cap is not None:
            for k in ("pp","ip","mp"):
                if r[k] > carry_cap: r[k] = carry_cap
        if callable(log_fn):
            log_fn(state, f"{side} gained from opinions: +{add[0]}PP, +{add[1]}IP, +{add[2]}MP")
# ==============================================================================


class GameEngine:
    # ---------------------------- RNG & LOGGING --------------------------------
    def __init__(self, rules=None):
        self.rules = rules or RULES
        mech_set_rules(self.rules)

        self.river_rules = self.rules.get("river_rules", {"slots": 7, "discard_rightmost": True})
        self.restrike_rules = self.rules.get("restrike_rules", {"plan_delay_turns": 1, "execute_window_turns": 1})
        self.action_costs = self.rules.get("action_costs", {})
        self.airspace_rules = self.rules.get("airspace_rules", {})
        self.victory_thresholds = self.rules.get("victory_thresholds", {})

        iran_map   = {c["No"]: c for c in self.rules.get("IRAN_CARDS", []) if "No" in c}
        israel_map = {c["No"]: c for c in self.rules.get("ISRAEL_CARDS", []) if "No" in c}
        self.rules["cards"] = {"iran": iran_map, "israel": israel_map}

        tgt_names = list(self.rules.get("TARGET_DEFENSES", {}).keys())
        self.rules["targets"] = {name: self.rules.get("TARGET_DEFENSES", {}).get(name, {}) for name in tgt_names}

        self._cards_index = {"iran": list(iran_map.keys()), "israel": list(israel_map.keys())}
    # ---- CARDS STATE: canonical location ----
    # We will ONLY use: state['players'][side]['deck'|'river'|'discard']
    # where `side` in {'israel','iran'}.
    
    def _ensure_player_cards_branch(self, state, side):
        players = state.setdefault('players', {})
        p = players.setdefault(side, {})
        p.setdefault('deck', [])
        p.setdefault('river', [None]*7)   # 7 slots (None = hole)
        p.setdefault('discard', [])
        return p
    
    def _normalize_cards_namespaces(self, state):
        """If older state keys exist (e.g. state['israel']['deck']), migrate them once."""
        for side in ('israel','iran'):
            p = self._ensure_player_cards_branch(state, side)
            legacy = state.get(side, {})
            if isinstance(legacy, dict):
                for key in ('deck','river','discard'):
                    if legacy.get(key) and not p.get(key):
                        p[key] = legacy[key]
                for key in ('deck','river','discard'):
                    if key in legacy:
                        del legacy[key]


    def _rng(self, state):
        r = state.setdefault('_rng', random.Random())
        seed = state.get("rng", {}).get("seed", None)
        if (seed is not None) and (not state.get("_rng_seeded", False)):
            r.seed(int(seed)); state["_rng_seeded"] = True
        return r

    def _roll(self, state, sides=6): return self._rng(state).randint(1, sides)
    def _choice(self, state, seq): return None if not seq else self._rng(state).choice(seq)

    def _log(self, state, msg):
        state.setdefault("log", []).append(str(msg))
    # ---- RIVER HELPERS ----

    def _remove_card_from_river(self, state, side, index):
        """Play/remove a card at `index` (0..6). Move to discard. Leave a None hole."""
        p = self._ensure_player_cards_branch(state, side)
        if not (0 <= index < 7):
            return False
        card = p['river'][index]
        if card is None:
            return False
        p['discard'].append(card)
        p['river'][index] = None
        return True
    def _on_card_removed_from_river(self, state, side, card_id, to_discard=True):
        """
        7-slot river variant:
        - Find the card in p['river'] (value equals card_id), set that slot to None.
        - If to_discard: push card_id into p['discard'].
        - Then compress to the RIGHT and refill from LEFT to 7, reshuffling if needed.
        """
        p = self._ensure_player_cards_branch(state, side)
        river = p['river']
        discard = p['discard']
        deck = p['deck']
    
        # Remove one instance from river (set the slot to None)
        for i in range(len(river)):
            if river[i] == card_id:
                river[i] = None
                break
        else:
            return  # card not found; nothing to do
    
        if to_discard:
            discard.append(card_id)
    
        # Compress RIGHT: non-Nones on right, Nones on left
        cards = [c for c in river if c is not None]
        holes = 7 - len(cards)
        river[:] = [None]*holes + cards
    
        # Draw helper
        def draw_one():
            if not deck:
                if discard:
                    self._rng(state).shuffle(discard)
                    deck[:], discard[:] = discard[:], []
                else:
                    return None
            return deck.pop(0) if deck else None
    
        # Refill from LEFT to 7
        for i in range(7):
            if river[i] is None:
                c = draw_one()
                if c is not None:
                    river[i] = c

    def _end_of_map_turn_river_step(self, state):
        """2.5 River: discard rightmost, compress right, refill from left to 7, reshuffle if needed."""
        for side in ('israel','iran'):
            p = self._ensure_player_cards_branch(state, side)
            river = p['river']
            deck = p['deck']
            discard = p['discard']
         

            if len(river) > 6 and river[6] is not None:
                discard.append(river[6])
                river[6] = None
           
            # 2) Compress RIGHT: non-None to the right, Nones on the left
            nones = [x for x in river if x is None]
            cards = [x for x in river if x is not None]
            river[:] = [None]*len(nones) + cards
    
            # 3) Refill from LEFT to size 7 (reshuffle discard -> deck as needed)
            def draw_one():
                if not deck:
                    if discard:
                        self._rng(state).shuffle(discard)
                        deck[:], discard[:] = discard[:], []
                    else:
                        return None
                return deck.pop(0) if deck else None
    
            for i in range(7):
                if river[i] is None:
                    nxt = draw_one()
                    if nxt is not None:
                        river[i] = nxt

    # --------------------------- STATE HELPERS ---------------------------------
    def _ensure_player(self, state, side):
        ps = state.setdefault('players', {}).setdefault(side, {})
        ps.setdefault('resources', {'pp': 0, 'ip': 0, 'mp': 0})
        ps.setdefault('river', [])
        ps.setdefault('deck', [])
        ps.setdefault('discard', [])
        return ps
    # === RIVER / TURN-LIMIT HELPERS ============================================
    def _shuffle_in_place(self, lst, state):
        """
        Deterministic shuffle helper. Replace with your RNG using state['rng'] if you have one.
        For now, simple Fisher-Yates using Python's random seeded by state.get('seed').
        """
        import random
        rnd = random.Random(state.get("seed", 0))
        for i in range(len(lst) - 1, 0, -1):
            j = rnd.randint(0, i)
            lst[i], lst[j] = lst[j], lst[i]
    
    def _on_impulse_start(self, state):
        """Reset the per-impulse card play flags for the current player/impulse if you want each side limited per impulse."""
        turn = state["turn"]
        # Reset for both or just reset current side? Rule intent: each side can play at most one card on its impulse.
        # We'll reset the flag for the side whose impulse is active.
        side = turn.get("current_player")
        turn["per_impulse_card_played"][side] = False
    
    # --- helpers used by ops / damage ------------------------------------------
    def _get_aircraft_count_for_squadron(self, state, side, squadron_name):
        oob = state.get('oob', {}).get(side, {}).get('squadrons', {})
        if squadron_name in oob:
            return int(oob[squadron_name].get('aircraft', 4))
        r = self.rules.get('squadrons', {}).get(side, {})
        if squadron_name in r:
            return int(r[squadron_name].get('aircraft', 4))
        return 4

    def _get_weapon_profile(self, weapon_name):
        table = self.rules.get('pgm_table', {}) or self.rules.get('PGM_ATTACK_TABLE', {})
        prof = table.get(weapon_name)
        if prof:
            return prof
        return {
            "vs": {
                "soft": {"p_hit": 0.5, "hits": [1]},
                "med":  {"p_hit": 0.35, "hits": [1]},
                "hard": {"p_hit": 0.2,  "hits": [1]}
            },
            "reliability": 0.9
        }

    def _armor_class_of_component(self, target_rules, comp_id):
        if comp_id in target_rules.get("Primary_Targets", {}):
            return target_rules["Primary_Targets"][comp_id].get("armor_class", "med")
        if comp_id in target_rules.get("Secondary_Targets", {}):
            return target_rules["Secondary_Targets"][comp_id].get("armor_class", "soft")
        return "med"

    def _apply_component_damage(self, state, target, comp, hits):
        td = state.setdefault("target_damage_status", {}).setdefault(target, {})
        entry = td.get(comp)
        if isinstance(entry, dict):
            entry["damage_boxes_hit"] = entry.get("damage_boxes_hit", 0) + int(hits)
        elif isinstance(entry, int):
            td[comp] = entry + int(hits)
        else:
            td[comp] = {"damage_boxes_hit": int(hits)}

    def _check_alt_victory_flags(self, state):
        # keep or extend with your custom victory logic
        return

    

    # ----------------------------- BOOTSTRAP -----------------------------------
    def bootstrap_rivers(self, state: dict, *, river_size: Optional[int] = None) -> dict:
        if "turn" not in state:
            state["turn"] = {"turn_number": 1, "current_player": "israel", "phase": "morning"}

        river_cap = int(self.river_rules.get("slots", 7)) if river_size is None else int(river_size)

        for side in ("israel", "iran"):
            p = self._ensure_player(state, side)
            if not p["deck"] and not p["river"]:
                deck_ids = list(self._cards_index[side])
                self._rng(state).shuffle(deck_ids)
                p["deck"] = deck_ids
                p["discard"] = []
                p["river"] = []
                while len(p["river"]) < river_cap and p["deck"]:
                    c = p["deck"].pop(0)  # deck was just shuffled; no need to reshuffle here
                    p["river"].insert(0, c)

                self._log(state, f"Initialized and dealt a new river for {side}.")
        return state

    # --------------------------- CARDS & COSTS ---------------------------------
    def _card_struct(self, side: str, cid: int):
        return self.rules.get("cards_structured", {}).get(
            "iran" if side.lower().startswith("ir") else "israel", {}
        ).get(cid, {})

    def _card_cost_map(self, side: str, cid: int):
        cm = self._card_struct(side, cid).get("cost_map")
        if cm: return cm
        cdef = self.rules.get("cards", {}).get(side, {}).get(cid, {})
        return self._parse_cost_to_map(cdef.get("Cost", ""))

    def _card_requires_text(self, side: str, cid: int):
        return self._card_struct(side, cid).get("requires", {}).get("text", "")

    def _card_flags(self, side: str, cid: int):
        return set(self._card_struct(side, cid).get("flags", []))

    @staticmethod
    def _parse_cost_to_map(cost_str: str):
        m = {"pp": 0, "ip": 0, "mp": 0}
        s = (cost_str or "").strip()
        if not s or s == "--": return m
        for amt, unit in re.findall(r'(\d+)\s*([PIM])', s.upper()):
            m[{"P": "pp", "I": "ip", "M": "mp"}[unit]] += int(amt)
        return m

    # ------------------------------ RULE GATES ---------------------------------
    def _requires_satisfied(self, state, side, req_text: str) -> bool:
        s = (req_text or "").lower()
        last = state.get("last_act", {})
        if "last act was dirty" in s and ("dirty" not in last.get("tags", [])):
            return False
        if "last act was covert" in s and ("covert" not in last.get("tags", [])):
            return False
        if "opponent overt last turn" in s and (state.get("opponent_overt_last_turn") is not True):
            return False
        return True

    def _mark_last_act(self, state, side, tags):
        state["last_act"] = {"side": side, "tags": list(set(tags))}

    # ------------------------------ PASS & CARDS --------------------------------
    def _resolve_pass(self, state):
        side = state.get("turn", {}).get("current_player", "israel")
        t = state.setdefault("turn", {})
        t["consecutive_passes"] = t.get("consecutive_passes", 0) + 1
        self._log(state, f"{side} passes.")
        return self._advance_turn(state)



    def _black_market_convert(self, state, side, spend_pp=0, spend_ip=0, spend_mp=0, receive="pp"):
        total = spend_pp + spend_ip + spend_mp
        if total < 3: return False
        sets = total // 3
        need = 3 * sets
        res = state['players'][side]['resources']
        take_pp = min(spend_pp, min(res.get("pp", 0), need)); need -= take_pp
        take_ip = min(spend_ip, min(res.get("ip", 0), need)); need -= take_ip
        take_mp = min(spend_mp, min(res.get("mp", 0), need)); need -= take_mp
        if need != 0: return False
        res["pp"] -= take_pp; res["ip"] -= take_ip; res["mp"] -= take_mp
        res[receive] = res.get(receive, 0) + sets
        return True

    def _resolve_play_card(self, state, action, do_advance=True):
        side = state['turn']['current_player']
    
        self._ensure_player(state, side)
        card_id = action.get('card_id')
        if card_id is None: return state

        cdef = self.rules.get('cards', {}).get(side, {}).get(card_id)
        if not cdef:
            self._log(state, f"{side} attempted to play unknown card {card_id}.")
            return state

        river = state['players'][side]['river']
        if card_id not in river:
            self._log(state, f"{side} tried to play card {card_id} not in river; no-op.")
            return state

        cm = self._card_cost_map(side, card_id)
        req_text = self._card_requires_text(side, card_id)
        if req_text and not self._requires_satisfied(state, side, req_text):
            self._log(state, f"{side} cannot play card {card_id}: requires not met.")
            return state

        res = state['players'][side]['resources']

        if (cdef.get("Cost") or "").strip() == "--":
            if (res.get("pp",0)+res.get("ip",0)+res.get("mp",0)) < 3:
                self._log(state, f"{side} cannot play {card_id}: needs ≥3 total points for conversion.")
                return state
            spend_pp = min(3, res.get("pp", 0))
            spend_ip = min(max(0, 3 - spend_pp), res.get("ip", 0))
            spend_mp = min(max(0, 3 - spend_pp - spend_ip), res.get("mp", 0))
            if not self._black_market_convert(state, side, spend_pp, spend_ip, spend_mp, receive="pp"):
                self._log(state, f"{side} failed Black Market conversion path.")
                return state
        else:
            for k in ("pp","ip","mp"):
                if res.get(k,0) < cm.get(k,0):
                    self._log(state, f"{side} cannot afford card {card_id}.")
                    return state
            for k in ("pp","ip","mp"):
                res[k] -= cm.get(k,0)
        state['turn'].setdefault('per_impulse_card_played', {})[side] = True
        self._on_card_removed_from_river(state, side, card_id, to_discard=True)

        effects_struct = self._card_struct(side, card_id).get('effects')
        if effects_struct:
            self._apply_structured_card_effects(state, side, effects_struct)
        else:
            self._apply_textual_card_effect(state, side, (cdef.get('Effect') or '').strip())

        self._mark_last_act(state, side, list(self._card_flags(side, card_id)))
        self._log(state, f"{side} played card {card_id}: {cdef.get('Name','?')}")
        state.setdefault("turn", {})["consecutive_passes"] = 0
        return self._advance_turn(state) if do_advance else state

    # ------------------------ EFFECTS (unchanged style) ------------------------
    def _apply_structured_card_effects(self, state, side, effects):
        for eff in effects:
            etype = eff.get('type')
            if etype == 'opinion':
                grp = eff.get('track', 'domestic')
                who = eff.get('who', side)
                delta = int(eff.get('delta', 0))
                state.setdefault('opinion', {}).setdefault(grp, {})
                state['opinion'][grp][who] = state['opinion'][grp].get(who, 0) + delta
                self._log(state, f"Opinion {grp}:{who} {delta:+d}")
            elif etype == 'resource':
                who = eff.get('who', side)
                self._grant_resources(
                    state, who,
                    delta_pp=int(eff.get('pp',0)),
                    delta_ip=int(eff.get('ip',0)),
                    delta_mp=int(eff.get('mp',0)),
                )
                self._log(state, f"Resources for {who}: +{int(eff.get('pp',0))}PP +{int(eff.get('ip',0))}IP +{int(eff.get('mp',0))}MP")

            elif etype == 'event' and eff.get('name','').lower() == 'cancel_strategic_event':
                state['active_strategic_effects'] = []
                self._log(state, "Active strategic effects cleared.")
            elif etype == 'flag':
                fname = eff.get('name')
                val = eff.get('value', True)
                if fname:
                    state[fname] = val
                    self._log(state, f"Flag {fname} set to {val}")

    def _apply_textual_card_effect(self, state, side, text):
        t = (text or "").lower()
        if "cancel strategic event" in t:
            state['active_strategic_effects'] = []
            self._log(state, "Active strategic effects cleared.")
            return
        if "●" in (text or ""):
            pip_count = text.count("●")
            if "is domestic" in t:
                state.setdefault('opinion', {}).setdefault('domestic', {})
                state['opinion']['domestic']['israel'] = state['opinion']['domestic'].get('israel', 0) + pip_count
                self._log(state, f"Israel domestic +{pip_count}")
            elif "ir domestic" in t:
                state.setdefault('opinion', {}).setdefault('domestic', {})
                state['opinion']['domestic']['iran'] = state['opinion']['domestic'].get('iran', 0) + pip_count
                self._log(state, f"Iran domestic +{pip_count}")
            elif "un" in t:
                state.setdefault('opinion', {}).setdefault('third_party', {})
                state['opinion']['third_party']['un'] = state['opinion']['third_party'].get('un', 0) + pip_count
                self._log(state, f"UN +{pip_count}")

    def _ensure_player_structs(self, state, side):
        players_dict = state.get('players')
        if players_dict is None:
            players_dict = {}
            state['players'] = players_dict
        ps = players_dict.setdefault(side, {})
        ps.setdefault('resources', {'pp': 0, 'ip': 0, 'mp': 0})
        return ps

    def _morning_reset_resources(self, state):
        """
        Binder step: 'Discard any points left over from the previous day.'
        Strict mode: zero all PP/IP/MP at Morning.
        Optional: if rules['night_carryover']=True, keep points explicitly earned during last Night.
        """
        carry_enabled = bool(self.rules.get('night_carryover', False))
        for side in ("israel","iran"):
            self._ensure_player_structs(state, side)
            res = state['players'][side]['resources']
            if carry_enabled:
                # keep only tracked Night carryover; everything else is discarded
                buf = state.get('_night_carryover', {}).get(side, {'pp':0,'ip':0,'mp':0})
                res['pp'], res['ip'], res['mp'] = buf.get('pp',0), buf.get('ip',0), buf.get('mp',0)
            else:
                res['pp'] = res['ip'] = res['mp'] = 0
        # clear carry buffer each Morning
        state['_night_carryover'] = {'israel': {'pp':0,'ip':0,'mp':0}, 'iran': {'pp':0,'ip':0,'mp':0}}

    def _grant_resources(self, state, who, delta_pp=0, delta_ip=0, delta_mp=0):
        """
        Centralized resource grant that also (optionally) tracks Night carryover.
        Replace ad-hoc res[...] += n with this.
        """
        self._ensure_player_structs(state, who)
        r = state['players'][who]['resources']
        r['pp'] = max(0, r.get('pp',0) + int(delta_pp))
        r['ip'] = max(0, r.get('ip',0) + int(delta_ip))
        r['mp'] = max(0, r.get('mp',0) + int(delta_mp))

        # If carryover is enabled and we’re in Night, track what was earned now
        if self.rules.get('night_carryover', False) and state.get('turn',{}).get('phase') == 'night':
            buf = state.setdefault('_night_carryover', {}).setdefault(who, {'pp':0,'ip':0,'mp':0})
            buf['pp'] += max(0, int(delta_pp))
            buf['ip'] += max(0, int(delta_ip))
            buf['mp'] += max(0, int(delta_mp))



    # ---------- Strategic Events (Morning) ----------------------------------------
    
    def _roll_strategic_event_morning(self, state):
        """
        Morning step: roll on Strategic Events.
        - Uses rules['strategic_events'] OR defaults.
        - Each entry: {"range": (lo, hi), "name": "X", "effects": [ ... ] }
          effect types: 'opinion', 'resource', 'flag', 'queue_event' (same schema you already use)
        """
        table = self.rules.get("strategic_events") or [
            {"range": (1,2), "name": "Quiet Day", "effects": []},
            {"range": (3,3), "name": "UN Condemnation", "effects":[{"type":"opinion","track":"third_party","who":"un","delta":-1}]},
            {"range": (4,4), "name": "US Aid", "effects":[{"type":"resource","who":"israel","pp":1,"ip":1}]},
            {"range": (5,5), "name": "Cyber Incident", "effects":[{"type":"resource","who":"iran","ip":-1}]},
            {"range": (6,6), "name": "Regional Escalation", "effects":[{"type":"opinion","track":"third_party","who":"saudi","delta":-1},{"type":"opinion","track":"third_party","who":"russia","delta":+1}]},
        ]
        # D6 by default; allow D10 via rules
        die = int(self.rules.get("strategic_events_die", 6))
        roll = self._roll(state, die)
        chosen = None
        for row in table:
            lo, hi = row.get("range",(0,0))
            if lo <= roll <= hi:
                chosen = row; break
        if not chosen:
            self._log(state, f"Strategic Events: roll {roll} (no effect)")
            return
        self._log(state, f"Strategic Events: roll {roll} -> {chosen.get('name','?')}")
        for eff in chosen.get("effects", []):
            et = eff.get("type")
            if et == "opinion":
                grp = eff.get("track","third_party")
                who = eff.get("who")
                delta = int(eff.get("delta",0))
                state.setdefault("opinion",{}).setdefault(grp,{})
                key = who if grp=="domestic" else str(who).lower()
                state["opinion"][grp][key] = state["opinion"][grp].get(key,0) + delta
            elif et == "resource":
                who = eff.get("who","israel")
                self._ensure_player_structs(state, who)
                res = state["players"][who]["resources"]
                for k in ("pp","ip","mp"):
                    if k in eff:
                        res[k] = max(0, res.get(k,0) + int(eff[k]))
            elif et == "flag":
                state[eff.get("name","flag")] = eff.get("value", True)
            elif et == "queue_event":
                ev = dict(eff); ev.pop("type",None)
                ev.setdefault("scheduled_for", state["turn"]["turn_number"]+1)
                state.setdefault("active_events_queue", []).append(ev)

    def _roll_player_assist_event(self, state, side):
        """
        Binder step: 'Each player should roll D6 for strategic events. If a player rolls a 6,
        he should then roll D10, check the player assist card and implement the results.'
        We provide a stub mapping rules['player_assist_d10'] = {1: [...effects...], ...}
        """
        d6 = self._roll(state, 6)
        if d6 != 6:
            self._log(state, f"Strategic Event (assist): {side} rolled {d6} (no effect).")
            return
        d10 = self._roll(state, 10)
        table = self.rules.get('player_assist_d10', {})
        effects = table.get(int(d10), [])
        self._log(state, f"Strategic Event (assist): {side} rolled 6 → d10={d10}.")
        for eff in effects:
            et = eff.get("type")
            if et == "opinion":
                grp = eff.get("track","third_party")
                who = eff.get("who", side)
                delta = int(eff.get("delta",0))
                state.setdefault("opinion",{}).setdefault(grp,{})
                key = who if grp=="domestic" else str(who).lower()
                state["opinion"][grp][key] = state["opinion"][grp].get(key,0) + delta
            elif et == "resource":
                who = eff.get("who", side)
                self._grant_resources(
                    state, who,
                    delta_pp=int(eff.get('pp',0)),
                    delta_ip=int(eff.get('ip',0)),
                    delta_mp=int(eff.get('mp',0)),
                )
            elif et == "flag":
                state[eff.get("name","flag")] = eff.get("value", True)
            elif et == "queue_event":
                ev = dict(eff); ev.pop("type",None)
                ev.setdefault("scheduled_for", state["turn"]["turn_number"])
                state.setdefault("active_events_queue", []).append(ev)

    # ---------- Red Breakdown (after each map turn) & Morning Repair --------------
    def _red_breakdown_after_map_turn(self, state):
        """
        Called after Iran completes their action (end of a map turn).
        Creates 'breakdown' counters for Red assets per simple chance model or rules table.
        Hook it into _advance_turn after Iran acts, BEFORE river aging (or after—your call).
        """
        rules = self.rules.get("red_breakdown", {"chance_per_asset": 0.1})
        p = float(rules.get("chance_per_asset", 0.1))
        # Assets: squadrons + SAMs + tankers (if you model tankers)
        red = state.setdefault("oob", {}).setdefault("iran", {})
        squadrons = list(red.get("squadrons", {}).keys())
        sams = list(state.get("air_defense_units", {}).keys())
        rng = self._rng(state)
        broken = []
        for sq in squadrons:
            if rng.random() < p:
                red["squadrons"].setdefault(sq, {})["status"] = "Broken"
                broken.append(("squadron", sq))
        for sam in sams:
            if rng.random() < p:
                state["air_defense_units"][sam]["status"] = "Broken"
                broken.append(("sam", sam))
        if broken:
            self._log(state, f"Iran breakdowns: {', '.join(f'{t}:{n}' for t,n in broken)}")

    def _morning_repair_rolls(self, state):
        """
        Morning: roll repairs for any 'Broken' units. Stand-down bonus if previous night pass.
        rules['repair'] = {'dc': 4, 'stand_down_bonus': +1, 'die': 6}
        """
        cfg = self.rules.get("repair", {"dc":4, "stand_down_bonus":1, "die":6})
        dc = int(cfg.get("dc",4))
        bonus = 0
        if state.get("last_night_was_stand_down"):
            bonus += int(cfg.get("stand_down_bonus",1))
        die = int(cfg.get("die",6))
        # Red & Blue squadrons + SAMs
        for side in ("israel","iran"):
            oob = state.get("oob",{}).get(side,{})
            for sq, meta in (oob.get("squadrons",{}) or {}).items():
                if isinstance(meta, dict) and meta.get("status")=="Broken":
                    r = self._roll(state, die) + bonus
                    if r >= dc:
                        meta["status"] = "Ready"
                        self._log(state, f"Repair success: {side} squadron {sq}.")
            # SAMs
            for uid, meta in (state.get("air_defense_units",{}) or {}).items():
                if meta.get("owner", "iran") != side: continue
                if meta.get("status")=="Broken":
                    r = self._roll(state, die) + bonus
                    if r >= dc:
                        meta["status"] = "Ready"
                        self._log(state, f"Repair success: {side} SAM {uid}.")
        # Clear stand-down marker every Morning (it only helps once)
        state["last_night_was_stand_down"] = False

    # ---------- Special Warfare (mission types, day/night penalty) ----------------
    def _sw_success_roll(self, state, pts, is_daytime=False):
        """
        Simplified SW success:
          base = d6 + floor(pts/2)
          day penalty: -1 (or rules['sw']['day_penalty'])
          critical fail on natural 1 before mods; backfire handler uses that.
        Returns tuple: (result_str, critical_fail:bool)
        """
        swr = self.rules.get("sw", {"day_penalty": 1})
        roll = self._roll(state, 6)
        crit_fail = (roll == 1)
        score = roll + (pts // 2) - (swr.get("day_penalty",1) if is_daytime else 0)
        # bands
        if score >= 9:
            return ("decisive", False if not crit_fail else True)
        if score >= 6:
            return ("tactical", False if not crit_fail else True)
        return ("fail", True if crit_fail else False)

    def _apply_sw_outcome(self, state, target, outcome):
        """
        Map SW outcome to target damage/opinions. You can refine with tables.
        """
        trules = self.rules.get("targets", {}).get(target, {})
        comps = list(trules.get("Secondary_Targets", {}).keys()) or list(trules.get("Primary_Targets", {}).keys())
        if not comps:
            return
        comp = self._choice(state, comps)
        if outcome == "decisive":
            self._apply_component_damage(state, target, comp, 2)
            self._log(state, f"SW decisive vs {target}:{comp} (2 boxes).")
        elif outcome == "tactical":
            self._apply_component_damage(state, target, comp, 1)
            self._log(state, f"SW tactical vs {target}:{comp} (1 box).")
        else:
            self._log(state, f"SW failed vs {target}.")

    # Modify your _resolve_sw_execution to use these helpers:
    # (Call from your existing method by replacing the core with this:)
    def _resolve_sw_execution_rules(self, state, event):
        target = event.get('target')
        pts = int(event.get('points_spent', 0))
        # Day/night: if phase when it executes is 'morning' or 'afternoon' -> daytime penalty
        is_daytime = state['turn']['phase'] in ("morning","afternoon")
        outcome, crit = self._sw_success_roll(state, pts, is_daytime=is_daytime)
        if crit and outcome == "fail":
            # backfire: simple UN -1
            state.setdefault("opinion",{}).setdefault("third_party",{})
            state["opinion"]["third_party"]["un"] = state["opinion"]["third_party"].get("un",0) - 1
            self._log(state, "SW critical failure backfires (UN -1).")
        self._apply_sw_outcome(state, target, outcome)

    # ---------- Terror: Iron Dome modifier ----------------------------------------
    def _terror_success_with_iron_dome(self, state, intensity):
        """
        Simple model: base d6 + intensity//2; Iron Dome gives -1 (or rules value)
        """
        dome_mod = int(self.rules.get("iron_dome_mod", 1))
        roll = self._roll(state, 6) + (intensity // 2) - (dome_mod if state.get("upgrades",{}).get("israel",set()) and "Iron Dome" in state["upgrades"]["israel"] else 0)
        return roll >= 6  # success threshold

    # ---------- Oil production tracking & victory checks --------------------------
    def _recompute_oil_production(self, state):
        """
        Compute crude and refinery outputs based on target damage.
        Expect target components to carry tags: {'type': 'crude'|'refinery'|'terminal', 'boxes': N}
        rules['oil'] = {'win_crude_pct':85, 'win_ref_pct':50}
        """
        totals = {"crude":0, "refinery":0, "terminal":0}
        hit = {"crude":0, "refinery":0, "terminal":0}
        # sweep target rules
        for tname, tr in (self.rules.get("targets",{}) or {}).items():
            for grp in ("Primary_Targets","Secondary_Targets"):
                for comp, meta in (tr.get(grp,{}) or {}).items():
                    typ = (meta.get("type") or "").lower()
                    boxes = int(meta.get("boxes",1))
                    if typ in totals:
                        totals[typ] += boxes
                        got = state.get("target_damage_status",{}).get(tname,{}).get(comp,0)
                        if isinstance(got, dict): got = got.get("damage_boxes_hit",0)
                        hit[typ] += min(int(got), boxes)
        state["oil_status"] = {"totals": totals, "hit": hit}
        # victory flags for quick checks
        crude_pct = 0 if totals["crude"]==0 else int(100 * hit["crude"]/totals["crude"])
        ref_pct   = 0 if totals["refinery"]==0 else int(100 * hit["refinery"]/totals["refinery"])
        win_req = self.rules.get("oil", {"win_crude_pct":85, "win_ref_pct":50})
        state.setdefault("victory_flags",{})
        if crude_pct >= int(win_req.get("win_crude_pct",85)) and ref_pct >= int(win_req.get("win_ref_pct",50)):
            state["victory_flags"]["israel_oil_strategy_success"] = True

    # Call this after any combat damage application (airstrike/BM/SW).

    # ---------- Nuclear victory roll (2d6 with modifiers) -------------------------
    def _check_nuclear_victory_roll(self, state):
        """
        Perform a nuclear victory attempt if rules say it’s allowed.
        rules['nuclear'] example:
          {'enabled': True, 'base_dc': 11,
          'mods': {'isr_domestic': +1 per 2 above 0, 'lost_aircraft': -1 per 3 lost, 'extra_decisive': +1 per decisive}}
        This is a *skeleton*; tune your real modifiers as per the binder.
        """
        cfg = self.rules.get("nuclear", {"enabled": False})
        if not cfg.get("enabled", False):
            return
        # You decide WHEN to call this (e.g., at Morning or after certain objectives)
        # Here we only attempt once per morning if not already won
        if state.get("victory_flags",{}).get("israel_nuclear_strategy_success"):
            return
        d = self._roll(state, 6) + self._roll(state, 6)
        mods = 0
        # Example modifiers
        isr_dom = int(state.get("opinion",{}).get("domestic",{}).get("israel",0))
        if isr_dom > 0:
            mods += isr_dom // 2
        lost = int(state.get("losses",{}).get("israel_aircraft",0))
        mods -= (lost // 3)
        extra_decisive = int(state.get("decisive_results",0))
        mods += extra_decisive
        total = d + mods
        dc = int(cfg.get("base_dc", 11))
        self._log(state, f"Nuclear victory roll: 2d6={d} mods={mods:+d} vs DC{dc} -> {total}")
        if total >= dc:
            state.setdefault("victory_flags",{})["israel_nuclear_strategy_success"] = True

    # ---------- Turn-1 freebies (pre-planned) -------------------------------------
    def _apply_turn1_freebies(self, state):
        """
        At Turn 1 Morning only: give one pre-planned Airstrike and/or SW without cost.
        Tweak via rules['turn1'] = {'free_airstrike': True, 'free_sw': True}
        We simply push events to the queue; you can make these player-configurable later.
        """
        if state.get("_turn1_freebies_done"):
            return
        r = self.rules.get("turn1", {"free_airstrike": True, "free_sw": True})
        if state["turn"]["turn_number"]==1 and state["turn"]["phase"]=="morning":
            if r.get("free_airstrike", True):
                ev = {"type":"airstrike_resolution", "scheduled_for":1, "side":"israel",
                      "target": self._choice(state, list(self.rules.get("targets",{}).keys())) or "Natanz",
                      "squadrons": ["69th","107th"], "loadout": {"PGMs": ["GBU-31 JDAM"]}}
                state.setdefault("active_events_queue", []).append(ev)
                self._log(state, "Turn 1: pre-planned Airstrike queued (free).")
            if r.get("free_sw", True):
                ev = {"type":"sw_execution", "scheduled_for":4, "side":"israel",
                      "target": self._choice(state, list(self.rules.get("targets",{}).keys())) or "Kharg Island",
                      "points_spent": 2}
                state.setdefault("active_events_queue", []).append(ev)
                self._log(state, "Turn 1: pre-planned SW queued (free).")
            state["_turn1_freebies_done"] = True

    # ---------- Reinforcements & redeploy (skeleton) ------------------------------
    def _apply_reinforcements_morning(self, state):
        """
        At Morning: check simple reinforcement table and add units.
        rules['reinforcements'] = [{'turn': 3, 'side': 'israel', 'type':'squadron','name':'122nd'}]
        """
        items = self.rules.get("reinforcements", [])
        turn = state["turn"]["turn_number"]
        for it in items:
            if int(it.get("turn",0)) == turn:
                side = it.get("side","israel")
                if it.get("type") == "squadron":
                    state.setdefault("oob",{}).setdefault(side,{}).setdefault("squadrons",{})[it.get("name")] = {"status":"Ready","aircraft":4}
                    self._log(state, f"Reinforcement: {side} squadron {it.get('name')} arrives.")

    # ---------- SAM placement/setup (very light bootstrap) ------------------------
    def _bootstrap_sams_if_missing(self, state):
        """
        If there are no SAM units yet, seed a minimal set for Iran with locations.
        This is only a convenience bootstrap; feel free to wire your full setup file.
        """
        if state.get("_sams_bootstrapped"): return
        sams = state.setdefault("air_defense_units", {})
        if not sams:
            sams.update({
                "S-200_1":{"owner":"iran","status":"Ready","location":"Tehran"},
                "S-300_1":{"owner":"iran","status":"Ready","location":"Bushehr"},
                "Tor_1":  {"owner":"iran","status":"Ready","location":"Natanz"},
            })
            self._log(state, "Initial SAMs placed for Iran.")
        state["_sams_bootstrapped"] = True

    # ---------- Aircraft loss outcomes & opinion effects --------------------------
    def _register_aircraft_loss(self, state, side, count=1):
        """
        Call whenever an aircraft is destroyed. We roll simple outcomes:
          POW/KIA/Neutral landing (each 1/3).
        Effects: KIA -> domestic -1; Neutral landing -> third-party -1 (US or UN).
        """
        rng = self._rng(state)
        st = state.setdefault("losses",{})
        st[f"{side}_aircraft"] = st.get(f"{side}_aircraft",0) + int(count)
        for _ in range(int(count)):
            r = rng.random()
            if r < 1/3:
                # POW: could return later; not modeled further here
                self._log(state, f"{side} aircraft down: pilot POW.")
            elif r < 2/3:
                # KIA -> domestic -1
                state.setdefault("opinion",{}).setdefault("domestic",{})
                state["opinion"]["domestic"][side] = state["opinion"]["domestic"].get(side,0) - 1
                self._log(state, f"{side} aircraft loss (KIA): {side} domestic -1.")
            else:
                # Neutral landing -> UN -1 (or US -1 if Israeli loss over US ally)
                state.setdefault("opinion",{}).setdefault("third_party",{})
                key = "un"
                state["opinion"]["third_party"][key] = state["opinion"]["third_party"].get(key,0) - 1
                self._log(state, f"{side} aircraft diverts to neutral: {key.upper()} -1.")

    # ======== END: add inside class GameEngine ====================================

    # ---------------------- ACTION LIST (minimal, incl. cards) -----------------
    def _corridor_ok(self, state, corridor_key: str) -> bool:
        rule = self.airspace_rules.get(corridor_key)
        if not rule: return True
        country = rule.get("country")
        min_op = rule.get("min_op", 0)
        third = state.get("opinion", {}).get("third_party", {}) or \
                state.get("opinion", {}).get("third_parties", {}) or {}
        op = third.get(str(country).lower(), 0)
        return int(op) >= int(min_op)


    def get_legal_actions(self, state, side=None):
        side = side or state.get("turn", {}).get("current_player", "israel")
        self._ensure_player_structs(state, side)
        res = state["players"][side]["resources"]
        river = list(state["players"][side].get("river", []))
        actions = [{"type": "Pass"}]

        # --- Cards from river ---
        card_map = self.rules.get("cards", {}).get(side, {})
        for cid in river:
            cdef = card_map.get(cid)
            if not cdef: 
                continue
            cm = self._card_cost_map(side, cid)
            afford = all(res.get(k, 0) >= cm.get(k, 0) for k in ("pp","ip","mp"))
            cost_str = (cdef.get("Cost") or "").strip()
            if not afford and not (cost_str == "--" and (res.get("pp",0)+res.get("ip",0)+res.get("mp",0)) >= 3):
                continue
            req_text = self._card_requires_text(side, cid)
            if req_text and not self._requires_satisfied(state, side, req_text):
                continue
            actions.append({"type": "Play Card", "card_id": cid})

        # --- Ops (simple defaults so the agent can choose) ---
        targets = list(self.rules.get("targets", {}).keys())
        if side == "israel":
            # Airstrike
            ac = self.action_costs.get("airstrike", {"mp": 3, "ip": 3})
            if res.get("mp",0) >= int(ac.get("mp",3)) and res.get("ip",0) >= int(ac.get("ip",3)):
                # pick a corridor that’s open
                for corridor in ("central","north","south"):
                    if self._corridor_ok(state, corridor):
                        for t in targets[:4]:
                            actions.append({
                                "type": "Order Airstrike",
                                "target": t,
                                "squadrons": ["69th","107th"],
                                "corridor": corridor,
                                "loadout": {"PGMs": ["GBU-31 JDAM","GBU-39 SDB"]}
                            })
                        break
            # Special Warfare
            swc = self.action_costs.get("special_warfare", {"mp": 1, "ip": 1})
            if res.get("mp",0) >= int(swc.get("mp",1)) and res.get("ip",0) >= int(swc.get("ip",1)):
                for t in targets[:3]:
                    actions.append({
                        "type": "Order Special Warfare",
                        "target": t,
                        "mp_cost": int(swc.get("mp",1)),
                        "ip_cost": int(swc.get("ip",1))
                    })

        if side == "iran":
            # Ballistic missile
            bmc = self.action_costs.get("ballistic_missile", {"mp": 1})
            if res.get("mp",0) >= int(bmc.get("mp",1)):
                for t in targets[:5]:
                    actions.append({
                        "type": "Order Ballistic Missile",
                        "target": t,
                        "battalions": 1,
                        "missile_type": "Shahab"
                    })
            # Terror attack
            ttc = self.action_costs.get("terror_attack", {"mp": 1, "ip": 1})
            if res.get("mp",0) >= int(ttc.get("mp",1)) and res.get("ip",0) >= int(ttc.get("ip",1)):
                actions.append({
                    "type": "Order Terror Attack",
                    "mp_cost": int(ttc.get("mp",1)),
                    "ip_cost": int(ttc.get("ip",1))
                })
        # Enforce “max one card per impulse” (house rule toggle you’re using)
        side_now = state.get('turn', {}).get('current_player', side)
        per_impulse = state.get('turn', {}).setdefault('per_impulse_card_played', {'israel': False, 'iran': False}).get(side_now, False)
        if per_impulse:
            actions = [a for a in actions if a.get('type') != 'Play Card']
        actions.append({"type": "End Impulse"})

        return actions if actions else [{"type": "Pass"}]



    # ------------------------------ EVENT RESOLVERS -----------------------------
    def _resolve_airstrike_combat(self, state, event):
        target = event.get('target')
        side = event.get('side', 'israel')
        squadrons_data = event.get('squadrons', [])
        loadout = event.get('loadout', {})
        trules = self.rules.get('targets', {}).get(target)
        self._check_alt_victory_flags(state)
        if not trules:
            self._log(state, f"Airstrike: target '{target}' not found. No effect.")
            return

        package = []
        squadron_names = []
                # helper: remove one aircraft and record the loss for the attacker
        def _kill_one():
            if not package:
                return
            package.pop(self._rng(state).randrange(len(package)))
            # Attacking side is 'side' (Israel by default); count the loss
            try:
                self._register_aircraft_loss(state, side=side, count=1)
            except Exception:
                # be defensive: if helper not present, just continue
                pass

        for sq_entry in squadrons_data:
            squadron_name = sq_entry.get("name") or sq_entry.get("id") if isinstance(sq_entry, dict) else sq_entry
            if not squadron_name:
                continue
            squadron_names.append(squadron_name)
            count = self._get_aircraft_count_for_squadron(state, side, squadron_name)
            for _ in range(count):
                package.append({"sq": squadron_name, "hp": 1, "weapons": list(loadout.get(squadron_name, []))})

        self._log(state, f"Package: {len(package)} a/c from {squadron_names} vs {target}.")

        def run_sam_layer(sam_names, layer_name):
            if not sam_names:
                return
            sam_table = self.rules.get('sam_table', {}) or self.rules.get('SAM_COMBAT_TABLE', {})
            rng = self._rng(state)
            engaged = 0
            for sam in sam_names:
                prof = sam_table.get(sam, {})
                shots = int(prof.get('shots', prof.get('attacks_per_turn', 2)))
                to_hit = float(prof.get('p_hit', prof.get('p_kill', 0.3)))
                ecm_mod = -0.05 if (side == 'israel' and 'EA-18G Support' in state.get('upgrades', {}).get('israel', set())) else 0.0
                for _ in range(max(0, shots)):
                    if not package:
                        break
                    engaged += 1
                    if rng.random() < max(0.0, min(1.0, to_hit + ecm_mod)):
                        _kill_one()
            self._log(state, f"{layer_name}: engaged {engaged} shots; survivors {len(package)}.")

        def run_aaa_layer(aaa_names):
            if not aaa_names:
                return
            aaa_table = self.rules.get('aaa_table', {}) or self.rules.get('AAA_COMBAT_TABLE', {})
            rng = self._rng(state)
            engaged = 0
            for aaa in aaa_names:
                prof = aaa_table.get(aaa, {})
                area = int(prof.get('area', 3))
                p_kill = float(prof.get('p_kill', 0.08))
                k = min(area, len(package))
                idxs = list(range(len(package)))
                rng.shuffle(idxs)
                for idx in sorted(idxs[:k], reverse=True):
                    engaged += 1
                    if rng.random() < p_kill:
                        _kill_one()
            self._log(state, f"AAA: engaged {engaged} a/c; survivors {len(package)}.")

        def run_gci_fighters(gci):
            if not gci:
                return
            rng = self._rng(state)
            engaged = 0
            for f in gci:
                count = int(f.get('count', 1))
                attacks = int(f.get('attacks_per', 1))
                p_kill = float(f.get('p_kill', 0.2))
                for _ in range(count * attacks):
                    if not package:
                        break
                    engaged += 1
                    if rng.random() < p_kill:
                        _kill_one()
            self._log(state, f"GCI: engagements {engaged}; survivors {len(package)}.")

        run_sam_layer((trules or {}).get('Long_Range_SAMs', []), "LR SAM")
        run_gci_fighters((trules or {}).get('GCI_Fighters', []))
        run_sam_layer((trules or {}).get('Medium_Range_SAMs', []), "MR SAM")
        run_aaa_layer((trules or {}).get('AAA', []))

        if not package:
            self._log(state, "All attackers lost before weapons release.")
            return

        prim = list((trules or {}).get('Primary_Targets', {}).keys())
        sec  = list((trules or {}).get('Secondary_Targets', {}).keys())
        comps_order = prim + sec if prim else sec

        for ac in package:
            wlist = ac.get('weapons', [])
            if not wlist and isinstance(event.get("loadout"), dict):
                global_pgms = event["loadout"].get("PGMs", [])
                if global_pgms:
                    wlist = [{"weapon": wname, "qty": 1} for wname in global_pgms]
            if not wlist:
                wlist = [{"weapon": "Mk-82", "qty": 2}]

            for w in wlist:
                wname = w.get('weapon', 'Mk-82')
                qty = int(w.get('qty', 1))
                prof = self._get_weapon_profile(wname)
                reliability = float(prof.get('reliability', 0.9))
                for _ in range(qty):
                    if self._rng(state).random() > reliability:
                        self._log(state, f"{wname} failed reliability.")
                        continue
                    comp = self._choice(state, comps_order)
                    if not comp:
                        continue
                    armor = self._armor_class_of_component(trules, comp)
                    vs = prof.get('vs', {}).get(armor, {"p_hit": 0.25, "hits": [1]})
                    p_hit = float(vs.get('p_hit', 0.25))
                    if self._rng(state).random() <= p_hit:
                        hits = self._choice(state, vs.get('hits', [1]))
                        self._apply_component_damage(state, target, comp, int(hits))
                        self._log(state, f"{wname} hit {target}:{comp} ({armor}) for {hits} box(es).")
                    else:
                        self._log(state, f"{wname} missed {target}:{comp} ({armor}).")

        for sq_name in squadron_names:
            state.setdefault('squadrons', {}).setdefault(side, {})
            state['squadrons'][side][sq_name] = 'Returning'
    def _resolve_order_airstrike(self, state, action, do_advance=True):
        res = state['players']['israel']['resources']
        ac = self.action_costs.get("airstrike", {"mp": 3, "ip": 3})
        need_mp, need_ip = int(ac.get("mp",3)), int(ac.get("ip",3))
        if res.get('mp',0) < need_mp or res.get('ip',0) < need_ip:
            self._log(state, "Israel cannot afford airstrike.")
            return state
        corridor = action.get("corridor","central")
        if not self._corridor_ok(state, corridor):
            self._log(state, f"Airstrike blocked: corridor '{corridor}' not available.")
            return state
        res['mp'] -= need_mp; res['ip'] -= need_ip
        ev = {
            "type": "airstrike_resolution",
            "scheduled_for": state['turn']['turn_number'] + 1,
            "side": "israel",
            "target": action.get("target"),
            "squadrons": list(action.get("squadrons", [])),
            "loadout": action.get("loadout", {})
        }
        state.setdefault('active_events_queue', []).append(ev)
        self._mark_last_act(state, "israel", ["overt"])
        self._log(state, f"Airstrike ordered vs {ev['target']} by {ev['squadrons']}.")
        state.setdefault("turn", {})["consecutive_passes"] = 0
        return self._advance_turn(state) if do_advance else state

    def _resolve_order_special_warfare(self, state, action, do_advance=True):
        res = state['players']['israel']['resources']
        swc = self.action_costs.get("special_warfare", {"mp": 1, "ip": 1})
        mp = int(action.get("mp_cost", swc.get("mp",1)))
        ip = int(action.get("ip_cost", swc.get("ip",1)))
        if res.get('mp',0) < mp or res.get('ip',0) < ip:
            self._log(state, "Israel cannot afford Special Warfare.")
            return state
        res['mp'] -= mp; res['ip'] -= ip
        ev = {
            "type": "sw_execution",
            "scheduled_for": state['turn']['turn_number'] + 3,
            "side": "israel",
            "target": action.get("target"),
            "points_spent": mp + ip
        }
        state.setdefault('active_events_queue', []).append(ev)
        self._mark_last_act(state, "israel", ["covert"])
        self._log(state, f"Special Warfare queued vs {ev['target']} (pts {mp+ip}).")
        state.setdefault("turn", {})["consecutive_passes"] = 0
        return self._advance_turn(state) if do_advance else state

    def _resolve_order_ballistic_missile(self, state, action, do_advance=True):
        res = state['players']['iran']['resources']
        bmc = self.action_costs.get("ballistic_missile", {"mp": 1})
        mp_cost = int(bmc.get("mp",1)) * max(1, int(action.get("battalions",1)))
        if res.get('mp',0) < mp_cost:
            self._log(state, "Iran cannot afford BM launch.")
            return state
        res['mp'] -= mp_cost
        ev = {
            "type": "ballistic_missile_impact",
            "scheduled_for": state['turn']['turn_number'] + 1,
            "side": "iran",
            "target": action.get("target"),
            "battalions": int(action.get("battalions",1)),
            "missile_type": action.get("missile_type","Shahab")
        }
        state.setdefault('active_events_queue', []).append(ev)
        self._mark_last_act(state, "iran", ["overt"])
        self._log(state, f"BM launch queued ({ev['missile_type']} x{ev['battalions']}) vs {ev['target']}.")
        state.setdefault("turn", {})["consecutive_passes"] = 0
        return self._advance_turn(state) if do_advance else state

    def _resolve_order_terror_attack(self, state, action, do_advance=True):
        res = state['players']['iran']['resources']
        ttc = self.action_costs.get("terror_attack", {"mp": 1, "ip": 1})
        mp = int(action.get("mp_cost", ttc.get("mp",1)))
        ip = int(action.get("ip_cost", ttc.get("ip",1)))
        if res.get('mp',0) < mp or res.get('ip',0) < ip:
            self._log(state, "Iran cannot afford Terror Attack.")
            return state
        res['mp'] -= mp; res['ip'] -= ip
        ev = {
            "type": "terror_attack_resolution",
            "scheduled_for": state['turn']['turn_number'] + 3,
            "side": "iran",
            "intensity": mp + ip
        }
        state.setdefault('active_events_queue', []).append(ev)
        self._mark_last_act(state, "iran", ["covert","dirty"])
        self._log(state, f"Terror Attack queued (intensity {mp+ip}).")
        state.setdefault("turn", {})["consecutive_passes"] = 0
        return self._advance_turn(state) if do_advance else state

    def _resolve_ballistic_missile_impact(self, state, event):
        target = event.get('target')
        trules = self.rules.get('targets', {}).get(target)
        self._check_alt_victory_flags(state)
        if not trules:
            self._log(state, f"BM: target '{target}' not found.")
            return
        mtype = event.get('missile_type', 'Shahab')
        battalions = int(event.get('battalions', 1))

        bm_table = self.rules.get('bm_table', {})
        prof = bm_table.get(mtype, {"p_hit": 0.33, "p_backlash": 1/6, "hits": 1})

        comps = list(trules.get('Primary_Targets', {}).keys()) + list(trules.get('Secondary_Targets', {}).keys())
        if not comps:
            return

        for _ in range(battalions):
            comp = self._choice(state, comps)
            r = self._rng(state).random()
            if r <= float(prof.get('p_hit', 0.33)):
                hits = int(prof.get('hits', 1))
                self._apply_component_damage(state, target, comp, hits)
                self._log(state, f"BM {mtype} hits {target}:{comp} for {hits} box.")
            else:
                self._log(state, f"BM {mtype} miss on {target}:{comp}.")
            if 'p_backlash' in prof:
                if self._rng(state).random() <= float(prof['p_backlash']):
                    state.setdefault('opinion', {}).setdefault('third_parties', {})
                    state['opinion']['third_parties']['un'] = state['opinion']['third_parties'].get('un', 0) - 1
                    self._log(state, "BM mishap/backlash: UN -1.")
            else:
                if self._roll(state, 6) == 1:
                    state.setdefault('opinion', {}).setdefault('third_parties', {})
                    state['opinion']['third_parties']['un'] = state['opinion']['third_parties'].get('un', 0) - 1
                    self._log(state, "BM mishap/backlash: UN -1.")

    def _resolve_sw_execution(self, state, event):
        target = event.get('target')
        trules = self.rules.get('targets', {}).get(target)
        self._check_alt_victory_flags(state)
        if not trules:
            return
        pts = int(event.get('points_spent', 0))
        roll = self._roll(state, 6) + (pts // 2)

        prim = list(trules.get('Primary_Targets', {}).keys())
        sec = list(trules.get('Secondary_Targets', {}).keys())
        pool = sec if roll < 9 else (prim or sec)
        if roll >= 6 and pool:
            comp = self._choice(state, pool)
            hits = 1 + (1 if self._roll(state, 6) >= 4 else 0)
            self._apply_component_damage(state, target, comp, hits)
            self._log(state, f"SW success (roll {roll}) vs {target}:{comp} for {hits} box(es).")
        else:
            self._log(state, f"SW failed (roll {roll}).")

    def _resolve_terror_attack(self, state, event):
        inten = int(event.get('intensity', 2))
        state.setdefault('opinion', {}).setdefault('domestic', {})
        state['opinion']['domestic']['israel'] = state['opinion']['domestic'].get('israel', 0) - 1
        if inten >= 4:
            state.setdefault('opinion', {}).setdefault('third_parties', {})
            state['opinion']['third_parties']['usa'] = state['opinion']['third_parties'].get('usa', 0) - 1
        self._log(state, f"Terror attack resolved (intensity {inten}).")

    def _resolve_rebase_complete(self, state, event):
        unit_id = event.get('unit_id')
        if not unit_id:
            return
        ad = state.setdefault('air_defense_units', {}).setdefault(unit_id, {})
        dest = ad.pop('destination', None)
        ad['status'] = 'Ready'
        if dest:
            ad['location'] = dest
        self._log(state, f"Air defense unit {unit_id} rebased to {dest} and is Ready.")

    # -------------------------- TURN ADVANCEMENT & UPKEEP ----------------------
    def _reset_impulse_flags(self, state):
        turn = state.setdefault('turn', {})
        turn['per_impulse_card_played'] = {'israel': False, 'iran': False}
    
    def _refill_to_seven_if_needed(self, state):
        # Ensure both rivers have exactly 7 slots and fill any Nones from deck (reshuffle discard if needed)
        for side in ('israel','iran'):
            p = self._ensure_player_cards_branch(state, side)
            river = p['river']
            if len(river) != 7:
                river[:] = (river + [None]*7)[:7]
            deck = p['deck']
            discard = p['discard']
            def draw_one():
                if not deck:
                    if discard:
                        self._rng(state).shuffle(discard)
                        deck[:], discard[:] = discard[:], []
                    else:
                        return None
                return deck.pop(0) if deck else None
            for i in range(7):
                if river[i] is None:
                    c = draw_one()
                    if c is not None:
                        river[i] = c

    def _advance_turn(self, state):
        turn  = state.setdefault('turn', {})
        phase = turn.get('phase', 'morning')
        cur   = turn.get('current_player', 'israel')
        other = 'iran' if cur == 'israel' else 'israel'
    
        # If both just passed, close this phase
        if turn.get('consecutive_passes', 0) >= 2:
            turn['consecutive_passes'] = 0
    
            if phase == 'morning':
                turn['phase'] = 'afternoon'
                turn['current_player'] = 'israel'
                self._reset_impulse_flags(state)
                return state
    
            if phase == 'afternoon':
                turn['phase'] = 'night'
                turn['current_player'] = 'israel'
                self._reset_impulse_flags(state)
                return state
    
            if phase == 'night':
               # 2.5 River aging at end of Map Turn
                self._end_of_map_turn_river_step(state)
                
                # Advance the turn number now that a full day passed
                turn['turn_number'] = int(turn.get('turn_number', 1)) + 1
                
                # Morning pipeline (binder order)
                self._morning_reset_resources(state)                  # discard leftover points
                self._morning_repair_rolls(state)                     # repair rolls
                carry_cap = self.rules.get("morning_carry_cap")       # or None
                apply_morning_opinion_income(state, carry_cap=carry_cap, log_fn=self._log)
                self._roll_strategic_event_morning(state)             # morning strategic events
                for s in ('israel','iran'):                           # player assist rolls
                    self._roll_player_assist_event(state, s)
                
                # Safety: top up rivers
                self._refill_to_seven_if_needed(state)
                
                # New Map Turn starts: Israel to act in Morning
                turn['phase'] = 'morning'
                turn['current_player'] = 'israel'
                self._reset_impulse_flags(state)
                return state

    
        # otherwise just alternate impulse
        turn['current_player'] = other
        self._reset_impulse_flags(state)  # reset per-impulse flags on handover
        return state

    def _handle_morning_upkeep(self, state):
        """
        Morning sequence (rules binder):
          1) Discard leftover points from previous day (we zero at morning).
          2) Repair rolls (aircraft & SAMs), with stand-down bonus if applicable.
          3) Check Opinion tracks and apply income (PP/IP/MP) for both sides.
          4) Roll Strategic Events (d6) and apply effects.
          5) Each player rolls d6 for assist; on a 6, roll d10 and apply the assist table.
          6) Turn-1 freebies (optional) and reinforcements (optional).
          7) Ensure SAMs are bootstrapped if your scenario didn’t add any.
        """
        turn_no = int(state.get("turn", {}).get("turn_number", 1))
        self._log(state, f"--- Morning Upkeep (Turn {turn_no}) ---")

        # (1) Discard leftover points (strict binder: zero at morning)
        self._morning_reset_resources(state)

        # (2) Repair rolls for Broken units (air, SAMs)
        self._morning_repair_rolls(state)

        # (3) Apply opinion -> income
        #    If you want a carry cap, pass a number; otherwise None means unlimited.
        carry_cap = self.rules.get("morning_carry_cap")  # or None
        apply_morning_opinion_income(state, carry_cap=carry_cap, log_fn=self._log)

        # (4) Strategic Events (main morning d6 table)
        self._roll_strategic_event_morning(state)

        # (5) Player Assist: each side rolls d6; on 6 roll d10 for assist table
        for side in ("israel", "iran"):
            self._roll_player_assist_event(state, side)

        # (6) Optional: freebies and reinforcements at morning
        self._apply_turn1_freebies(state)
        self._apply_reinforcements_morning(state)

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
     # ----------------------------- PUBLIC UTILITIES ----------------------------
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
    
        t = action.get("type")
    
        # ---------- End Impulse: hand turn over to opponent ----------
        if t == "End Impulse":
            turn = state.setdefault("turn", {})
            turn["consecutive_passes"] = turn.get("consecutive_passes", 0) + 1
            return self._advance_turn(state)

        
        if t == "Pass":
            turn = state.setdefault("turn", {})
            turn["consecutive_passes"] = turn.get("consecutive_passes", 0) + 1
            return self._advance_turn(state)

        # ---------- Play Card (one per impulse) ----------
        if t == "Play Card":
            # Enforce the one-card-per-impulse ceiling
            per_imp = state.setdefault("turn", {}).setdefault("per_impulse_card_played",
                                                              {"israel": False, "iran": False})
            if per_imp.get(side_now, False):
                self._log(state, f"{side_now} already played a card this impulse; card play blocked.")
                return state
    
            # Use your existing resolver; BUT we want to keep the impulse alive.
            # Create a copy of the action and run resolver with do_advance=False.
            # (See small edits to _resolve_play_card below.)
            state = self._resolve_play_card_keep_impulse(state, action)
            state.setdefault("turn", {})["consecutive_passes"] = 0
            per_imp[side_now] = True
            return state  # do NOT advance
    
        # ---------- Operational orders (multi allowed per impulse) ----------
# For these we call resolvers with do_advance=False so the impulse continues.
        if t in (
            "Order Airstrike",
            "Order Special Warfare",
            "Order Ballistic Missile",
            "Order Terror Attack",
        ):
            # Side gate
            if t in ("Order Airstrike", "Order Special Warfare") and side_now != "israel":
                self._log(state, f"Refused: {t} is Israel-only, current side is {side_now}")
                return state
            if t in ("Order Ballistic Missile", "Order Terror Attack") and side_now != "iran":
                self._log(state, f"Refused: {t} is Iran-only, current side is {side_now}")
                return state
        
            # Route to the appropriate resolver with do_advance=False (stay in same impulse)
            if t == "Order Airstrike":
                return self._resolve_order_airstrike(state, action, do_advance=False)
        
            if t == "Order Special Warfare":
                return self._resolve_order_special_warfare(state, action, do_advance=False)
        
            if t == "Order Ballistic Missile":
                return self._resolve_order_ballistic_missile(state, action, do_advance=False)
        
            if t == "Order Terror Attack":
                return self._resolve_order_terror_attack(state, action, do_advance=False)
        
        # ---------- Batch of operational orders in a single impulse ----------
        # Example action:
        # {"type":"MultiAction","actions":[{...op1...},{...op2...}, ...]}
        if t == "MultiAction":
            actions = action.get("actions", [])
            if not isinstance(actions, list) or not actions:
                self._log(state, "Refused MultiAction: missing or empty 'actions' list.")
                return state
        
            # Enforce that all sub-actions are operational and belong to current side
            op_names_isr = {"Order Airstrike", "Order Special Warfare"}
            op_names_irn = {"Order Ballistic Missile", "Order Terror Attack"}
            allowed_set = op_names_isr if side_now == "israel" else op_names_irn
        
            new_state = state
            for idx, sub in enumerate(actions):
                stype = sub.get("type")
                if stype not in allowed_set:
                    self._log(new_state, f"MultiAction[{idx}] refused: '{stype}' not allowed for {side_now}")
                    break
        
                # Check legality against *current* new_state (resources may drop as we go)
                legal_now = self.get_legal_actions(new_state)
                if not any(
                    la.get("type") == stype and all(la.get(k) == sub.get(k) for k in sub.keys())
                    for la in legal_now
                ):
                    self._log(new_state, f"MultiAction[{idx}] refused illegal/unauthorized sub-action: {sub}")
                    break
        
                # Dispatch with do_advance=False so we remain in the same impulse
                if stype == "Order Airstrike":
                    new_state = self._resolve_order_airstrike(new_state, sub, do_advance=False)
                elif stype == "Order Special Warfare":
                    new_state = self._resolve_order_special_warfare(new_state, sub, do_advance=False)
                elif stype == "Order Ballistic Missile":
                    new_state = self._resolve_order_ballistic_missile(new_state, sub, do_advance=False)
                elif stype == "Order Terror Attack":
                    new_state = self._resolve_order_terror_attack(new_state, sub, do_advance=False)
                else:
                    # Shouldn't happen, but be safe
                    self._log(new_state, f"MultiAction[{idx}] unknown type '{stype}', aborting batch.")
                    break
        
            # IMPORTANT: we *do not* advance here. Caller (or next action) should Pass or continue.
            return new_state
        
        # Fallback (shouldn’t happen often)
        self._log(state, f"Unknown or unhandled action '{t}'.")
        return state


    def _resolve_play_card_keep_impulse(self, state, action):
        """
        Same as _resolve_play_card but does NOT advance the turn; keeps the impulse alive.
        We reuse your existing logic by copying and trimming the final advance.
        """
        side = state['turn']['current_player']
        self._ensure_player(state, side)
        card_id = action.get('card_id')
        if card_id is None:
            return state
    
        cdef = self.rules.get('cards', {}).get(side, {}).get(card_id)
        if not cdef:
            self._log(state, f"{side} attempted to play unknown card {card_id}.")
            return state
    
        river = state['players'][side]['river']
        if card_id not in river:
            self._log(state, f"{side} tried to play card {card_id} not in river; no-op.")
            return state
    
        cm = self._card_cost_map(side, card_id)
        req_text = self._card_requires_text(side, card_id)
        if req_text and not self._requires_satisfied(state, side, req_text):
            self._log(state, f"{side} cannot play card {card_id}: requires not met.")
            return state
    
        res = state['players'][side]['resources']
    
        # Handle cost & optional black-market conversion (unchanged from your version)
        if (cdef.get("Cost") or "").strip() == "--":
            if (res.get("pp",0)+res.get("ip",0)+res.get("mp",0)) < 3:
                self._log(state, f"{side} cannot play {card_id}: needs ≥3 total points for conversion.")
                return state
            spend_pp = min(3, res.get("pp", 0))
            spend_ip = min(max(0, 3 - spend_pp), res.get("ip", 0))
            spend_mp = min(max(0, 3 - spend_pp - spend_ip), res.get("mp", 0))
            if not self._black_market_convert(state, side, spend_pp, spend_ip, spend_mp, receive="pp"):
                self._log(state, f"{side} failed Black Market conversion path.")
                return state
        else:
            for k in ("pp","ip","mp"):
                if res.get(k,0) < cm.get(k,0):
                    self._log(state, f"{side} cannot afford card {card_id}.")
                    return state
            for k in ("pp","ip","mp"):
                res[k] -= cm.get(k,0)
    
        # Remove from river → discard, then slide/right-align & refill from LEFT
        self._on_card_removed_from_river(state, side, card_id, to_discard=True)
    
        # Apply effects (your existing structured/textual paths)
        effects_struct = self._card_struct(side, card_id).get('effects')
        if effects_struct:
            self._apply_structured_card_effects(state, side, effects_struct)
        else:
            self._apply_textual_card_effect(state, side, (cdef.get('Effect') or '').strip())
    
        self._mark_last_act(state, side, list(self._card_flags(side, card_id)))
        self._log(state, f"{side} played card {card_id}: {cdef.get('Name','?')}")
        self._normalize_cards_namespaces(state)
        # IMPORTANT: do NOT advance the turn; keep the impulse alive for multi-actions
        return state
    

    # ----------------------------- VICTORY CONDITIONS ---------------------------
    def _domestic(self, state, side):
        return state.get("opinion", {}).get("domestic", {}).get(side, 0)

    def is_game_over(self, state) -> Optional[str]:
        if self._domestic(state, "iran") >= 10:
            return "israel"
        if self._domestic(state, "israel") <= -10:
            return "iran"

        v = state.get("victory_flags", {})
        if v.get("israel_nuclear_strategy_success"):
            return "israel"
        if v.get("israel_oil_strategy_success"):
            return "israel"

        cur_turn = state.get("turn", {}).get("turn_number", 1)
        scenario = (state.get("rules", {}) or {}).get("scenario")
        default_cap = 42 if str(scenario).lower() == "real_world" else 21
        max_turns = int(self.rules.get("max_turns", default_cap))
        if cur_turn > max_turns:
            return "iran"

        return None
