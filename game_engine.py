# game_engine.py
import copy
import random
import re
from typing import Any, Dict, List, Optional

# Import your rules and mixin
from mechanics import set_rules as mech_set_rules
from rules_global import RULES
from actions_ops import OpsLoggingMixin

class GameEngine(OpsLoggingMixin):
    # ---------------------------- INIT & RNG --------------------------------
    def __init__(self, rules=None):
        # Load rules from the blob
        self.rules = rules or RULES
        mech_set_rules(self.rules)

        # Initialize shortcuts from rules
        self.river_rules = self.rules.get("river_rules", {"slots": 7, "discard_rightmost": True})
        self.restrike_rules = self.rules.get("restrike_rules", {"plan_delay_turns": 1, "execute_window_turns": 1})
        self.action_costs = self.rules.get("action_costs", {})
        self.airspace_rules = self.rules.get("airspace_rules", {})
        self.victory_thresholds = self.rules.get("victory_thresholds", {})

        # Map cards for easy lookup
        iran_map = {c["No"]: c for c in self.rules.get("IRAN_CARDS", []) if "No" in c}
        israel_map = {c["No"]: c for c in self.rules.get("ISRAEL_CARDS", []) if "No" in c}
        self.rules["cards"] = {"iran": iran_map, "israel": israel_map}
        self._cards_index = {"iran": list(iran_map.keys()), "israel": list(israel_map.keys())}

        # Map targets
        tgt_names = list(self.rules.get("TARGET_DEFENSES", {}).keys())
        self.rules["targets"] = {name: self.rules.get("TARGET_DEFENSES", {}).get(name, {}) for name in tgt_names}

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

    # --------------------------- STATE HELPERS ---------------------------------
    def _ensure_player(self, state, side):
        # 1. Check if 'players' is missing OR strictly None (JSON null)
        if state.get('players') is None:
            state['players'] = {}
            
        # 2. Now safe to use setdefault because state['players'] is guaranteed to be a dict
        ps = state['players'].setdefault(side, {})
        
        # 3. Initialize sub-fields
        ps.setdefault('resources', {'pp': 0, 'ip': 0, 'mp': 0})
        ps.setdefault('river', [])
        ps.setdefault('deck', [])
        ps.setdefault('discard', [])
        return ps

    def _ensure_player_cards_branch(self, state, side):
        return self._ensure_player(state, side)

    def _normalize_cards_namespaces(self, state):
        for side in ('israel','iran'):
            p = self._ensure_player(state, side)
            legacy = state.get(side, {})
            if isinstance(legacy, dict):
                for key in ('deck','river','discard'):
                    if legacy.get(key) and not p.get(key):
                        p[key] = legacy[key]
                for key in ('deck','river','discard'):
                    if key in legacy:
                        del legacy[key]

    # --------------------------- INCOME & OPINION LOGIC -------------------------
    def _sum3(self, a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

    def _opinion_income_from_domestic(self, state, side):
        # Uses self.rules from rules_blob.py
        table = self.rules.get("OPINION_INCOME_TABLE", {})
        val = int(state.get("opinion", {}).get("domestic", {}).get(side, 0))
        
        if side not in table: return (0,0,0)
        
        for lo, hi, trip in table[side]:
            if lo <= val <= hi:
                return trip
        return (0,0,0)

    def _opinion_income_from_third_parties(self, state, side):
        total = (0,0,0)
        third = state.get("opinion", {}).get("third_party", {}) or \
                state.get("opinion", {}).get("third_parties", {}) or {}
        
        table = self.rules.get("THIRD_PARTY_INCOME_TABLE", {})
        
        # Aliases mapping
        aliases = {
            "china": "prc", "prc": "prc",
            "russia": "russia",
            "saudi": "saudi_gcc", "saudi_arabia": "saudi_gcc", "gcc": "saudi_gcc",
            "un": "un", "united_nations": "un",
            "jordan": "jordan",
            "turkey": "turkey",
            "usa": "usa", "united_states": "usa", "us": "usa",
        }

        for raw_k, v in third.items():
            k = aliases.get(str(raw_k).strip().lower())
            if not k: continue
            
            entry_list = table.get(k, [])
            for lo, hi, payload in entry_list:
                if lo <= int(v) <= hi:
                    inc = payload.get(side, (0,0,0))
                    total = self._sum3(total, inc)
                    break
        return total

    def apply_morning_opinion_income(self, state, carry_cap=None):
        self._ensure_player(state, "israel")
        self._ensure_player(state, "iran")
        
        for side in ("israel", "iran"):
            dom = self._opinion_income_from_domestic(state, side)
            intl = self._opinion_income_from_third_parties(state, side)
            add = self._sum3(dom, intl)
            
            r = state["players"][side]["resources"]
            r["pp"] = r.get("pp",0) + add[0]
            r["ip"] = r.get("ip",0) + add[1]
            r["mp"] = r.get("mp",0) + add[2]
            
            if carry_cap is not None:
                for k in ("pp","ip","mp"):
                    if r[k] > carry_cap: r[k] = carry_cap
            
            self._log(state, f"{side} gained from opinions: +{add[0]}PP, +{add[1]}IP, +{add[2]}MP")

    # --------------------------- CARDS & RIVER LOGIC ---------------------------
        def _apply_play_card(self, state, side: str, card_id: int, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core logic for playing a card:
          - check cost from rules
          - spend resources
          - remove from river (and refill)
          - mark per-impulse card usage
          - advance turn (unless 'keep_impulse' flag)
        """
        self._ensure_player(state, side)
        p = state["players"][side]
        res = p["resources"]

        # 1) 카드 정의 찾기
        card_map = self.rules.get("cards", {}).get(side, {})
        cdef = card_map.get(card_id)
        if not cdef:
            self._log(state, f"[WARN] {side} tried to play unknown card #{card_id}")
            return state

        # 2) 비용 계산 (rules_blob에서 가져오는 cost_map 사용)
        cost_map = self._card_cost_map(side, card_id)  # {'pp':..., 'ip':..., 'mp':...}
        need_pp = int(cost_map.get("pp", 0))
        need_ip = int(cost_map.get("ip", 0))
        need_mp = int(cost_map.get("mp", 0))

        if res.get("pp", 0) < need_pp or res.get("ip", 0) < need_ip or res.get("mp", 0) < need_mp:
            self._log(state, f"[CARD] {side} cannot afford card #{card_id} ({need_pp}P {need_ip}I {need_mp}M).")
            return state

        # 3) 자원 소모
        res["pp"] -= need_pp
        res["ip"] -= need_ip
        res["mp"] -= need_mp

        # 4) 리버에서 카드 제거 + 리필
        self._on_card_removed_from_river(state, side, card_id, to_discard=True)

        # 5) 카드 효과 처리 (정밀 구현은 TODO; 지금은 로그 + 플래그만)
        self._log(state, f"[CARD] {side} plays #{card_id}: {cdef.get('Name', '')}")

        # 여기서 실제 효과(여론 변경, 자원, 이벤트 등록 등)를
        # rules_blob.CARDS_STRUCTURED / Effect 필드를 읽어서 구현하면 완전 룰 준수 가능함.
        # 지금은 엔진 구조만 맞추고, 세부 효과는 추후 확장 포인트로 남겨둠.

        # 6) impulse 카드 사용 플래그
        turn = state.setdefault("turn", {})
        per_imp = turn.setdefault("per_impulse_card_played", {})
        per_imp[side] = True

        # 7) 'keep_impulse' 플래그 체크
        flags = set(self._card_struct(side, card_id).get("flags", []))
        if "keep_impulse" in flags:
            # 충격 카드 등: impulse 유지
            return self._resolve_play_card_keep_impulse(state, {"card_id": card_id})

        # 기본: 카드 플레이 후 상대 차례로 넘김 (Pass와 다르게 consecutive_passes는 리셋)
        turn["consecutive_passes"] = 0
        return self._advance_turn(state)

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

    def _remove_card_from_river(self, state, side, index):
        p = self._ensure_player(state, side)
        if not (0 <= index < 7): return False
        card = p['river'][index]
        if card is None: return False
        p['discard'].append(card)
        p['river'][index] = None
        return True

    def _on_card_removed_from_river(self, state, side, card_id, to_discard=True):
        p = self._ensure_player(state, side)
        river = p['river']
        discard = p['discard']
        deck = p['deck']
    
        # Remove
        for i in range(len(river)):
            if river[i] == card_id:
                river[i] = None
                break
        else:
            return 
    
        if to_discard:
            discard.append(card_id)
    
        # Compress Right
        cards = [c for c in river if c is not None]
        holes = 7 - len(cards)
        river[:] = [None]*holes + cards
    
        # Refill from Left
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

    def _end_of_map_turn_river_step(self, state):
        for side in ('israel','iran'):
            p = self._ensure_player(state, side)
            river = p['river']
            discard = p['discard']
            deck = p['deck']

            # Discard rightmost
            if len(river) > 6 and river[6] is not None:
                discard.append(river[6])
                river[6] = None
           
            # Compress Right
            cards = [x for x in river if x is not None]
            river[:] = [None]*(7-len(cards)) + cards
    
            # Refill Left
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

    def _refill_to_seven_if_needed(self, state):
        for side in ('israel','iran'):
            p = self._ensure_player(state, side)
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

    # ------------------------------ HELPER LOGIC --------------------------------
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

    def _reset_impulse_flags(self, state):
        turn = state.setdefault('turn', {})
        turn['per_impulse_card_played'] = {'israel': False, 'iran': False}

    def _corridor_ok(self, state, corridor_key: str) -> bool:
        rule = self.airspace_rules.get(corridor_key)
        if not rule: return True
        country = rule.get("country")
        min_op = rule.get("min_op", 0)
        third = state.get("opinion", {}).get("third_party", {}) or \
                state.get("opinion", {}).get("third_parties", {}) or {}
        op = third.get(str(country).lower(), 0)
        return int(op) >= int(min_op)
    
    # ------------------------ COMBAT / OPS HELPERS ------------------------

    def _get_aircraft_count_for_squadron(self, state, side, squadron_name):
        oob = state.get('oob', {}).get(side, {}).get('squadrons', {})
        if squadron_name in oob:
            return int(oob[squadron_name].get('aircraft', 4))
        r = self.rules.get('squadrons', {}).get(side, {})
        if squadron_name in r:
            return int(r[squadron_name].get('aircraft', 4))
        return 4
    
    def _get_pgm_hit_chance(self, weapon_name, target_size_class):
        pgm_table = self.rules.get("PGM_ATTACK_TABLE", {})
        w_data = pgm_table.get(weapon_name)
        if not w_data: 
            return 0.0 
        hit_chances = w_data.get("Hit_Chance_Target_Size", {})
        return float(hit_chances.get(target_size_class, 0.0))

    def _apply_component_damage(self, state, target, comp, hits):
        td = state.setdefault("target_damage_status", {}).setdefault(target, {})
        entry = td.get(comp)
        if isinstance(entry, dict):
            entry["damage_boxes_hit"] = entry.get("damage_boxes_hit", 0) + int(hits)
        elif isinstance(entry, int):
            td[comp] = entry + int(hits)
        else:
            td[comp] = {"damage_boxes_hit": int(hits)}

    def _register_aircraft_loss(self, state, side, count=1):
        rng = self._rng(state)
        st = state.setdefault("losses",{})
        st[f"{side}_aircraft"] = st.get(f"{side}_aircraft",0) + int(count)
        for _ in range(int(count)):
            r = rng.random()
            if r < 1/3:
                self._log(state, f"{side} aircraft down: pilot POW.")
            elif r < 2/3:
                state.setdefault("opinion",{}).setdefault("domestic",{})
                state["opinion"]["domestic"][side] = state["opinion"]["domestic"].get(side,0) - 1
                self._log(state, f"{side} aircraft loss (KIA): {side} domestic -1.")
            else:
                state.setdefault("opinion",{}).setdefault("third_party",{})
                key = "un"
                state["opinion"]["third_party"][key] = state["opinion"]["third_party"].get(key,0) - 1
                self._log(state, f"{side} aircraft diverts to neutral: {key.upper()} -1.")

    # ------------------------------ RESOLVERS -----------------------------------

    def _resolve_airstrike_combat(self, state, event):
        target = event.get('target')
        side = event.get('side', 'israel')
        squadrons_data = event.get('squadrons', [])
        loadout = event.get('loadout', {})
        trules = self.rules.get('targets', {}).get(target)
        
        if not trules:
            self._log(state, f"Airstrike: target '{target}' not found. No effect.")
            return

        package = []
        squadron_names = []
        
        def _kill_one():
            if not package: return
            package.pop(self._rng(state).randrange(len(package)))
            try:
                self._register_aircraft_loss(state, side=side, count=1)
            except Exception: pass

        # Build Package
        for sq_entry in squadrons_data:
            squadron_name = sq_entry.get("name") or sq_entry.get("id") if isinstance(sq_entry, dict) else sq_entry
            if not squadron_name: continue
            squadron_names.append(squadron_name)
            count = self._get_aircraft_count_for_squadron(state, side, squadron_name)
            for _ in range(count):
                package.append({"sq": squadron_name, "hp": 1, "weapons": list(loadout.get(squadron_name, []))})

        self._log(state, f"Package: {len(package)} a/c from {squadron_names} vs {target}.")

        # SAMs / AAA / GCI
        def run_defenses():
            rng = self._rng(state)
            # LR SAM
            for sam in (trules or {}).get('Long_Range_SAMs', []):
                prof = self.rules.get('SAM_COMBAT_TABLE', {}).get(sam, {})
                shots = int(prof.get('Attacks_Per_Battery', 1).split('/')[0]) if '/' in str(prof.get('Attacks_Per_Battery','1')) else int(prof.get('Attacks_Per_Battery', 1))
                for _ in range(shots):
                    if package and rng.random() < 0.3: _kill_one() # Simplified hit chance
            
            # GCI Fighters (simplified)
            # ... add GCI logic here if needed ...

            # AAA
            aaa_val = float(trules.get('AAA_Value', 0.0))
            if aaa_val > 0:
                 # Simple AAA check: check table or use simplified probability
                 for _ in range(len(package)):
                     if rng.random() < (aaa_val * 0.05): _kill_one()

        run_defenses()

        if not package:
            self._log(state, "All attackers lost before weapons release.")
            return

        # WEAPONS RELEASE
        prim = list((trules or {}).get('Primary_Targets', {}).keys())
        sec  = list((trules or {}).get('Secondary_Targets', {}).keys())
        comps_order = prim + sec if prim else sec

        for ac in package:
            wlist = ac.get('weapons', [])
            # Fallback loadout
            if not wlist and isinstance(event.get("loadout"), dict):
                global_pgms = event["loadout"].get("PGMs", [])
                if global_pgms:
                    wlist = [{"weapon": wname, "qty": 1} for wname in global_pgms]
            if not wlist:
                wlist = [{"weapon": "GBU-31 JDAM", "qty": 1}]

            for w in wlist:
                wname = w.get('weapon', 'GBU-31 JDAM')
                qty = int(w.get('qty', 1))
                
                # Weapon Stats
                w_stats = self.rules.get("PGM_ATTACK_TABLE", {}).get(wname, {})
                armor_pen = w_stats.get("Armor_Pen", 0)
                hits_per_shot = w_stats.get("Hits", 1)

                for _ in range(qty):
                    comp_id = self._choice(state, comps_order)
                    if not comp_id: continue

                    # Target Stats
                    comp_data = trules.get("Primary_Targets", {}).get(comp_id) or \
                                trules.get("Secondary_Targets", {}).get(comp_id)
                    if not comp_data: continue

                    size = comp_data.get("size_class", "D")
                    p_hit = self._get_pgm_hit_chance(wname, size)

                    if self._rng(state).random() <= p_hit:
                        t_armor_val = comp_data.get("armor_class", 0)
                        if t_armor_val == "Heavy": t_armor_val = 100
                        else: t_armor_val = int(t_armor_val)

                        damage_amount = hits_per_shot
                        if armor_pen is not None:
                            if int(armor_pen) < t_armor_val:
                                damage_amount = max(1, int(damage_amount / 4))
                        
                        self._apply_component_damage(state, target, comp_id, damage_amount)
                        self._log(state, f"{wname} HIT {target}:{comp_id} (Size {size}) for {damage_amount} box(es).")
                    else:
                        self._log(state, f"{wname} MISSED {target}:{comp_id} (Size {size}).")

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
    
    # ... [Specific Op resolvers mostly unchanged from mixin/default] ...
    # NOTE: For brevity, assuming _resolve_order_special_warfare, etc. follow the same pattern
    # utilizing action_ops mixin logic or local overrides as defined in your previous snippets.
    
    # ----------------------------- ACTION DISPATCH -----------------------------
    def get_legal_actions(self, state, side=None):
        if state is None: raise ValueError("state=None")
        side = side or state.get("turn", {}).get("current_player", "israel")
        self._ensure_player(state, side)
        res = state["players"][side]["resources"]
        river = list(state["players"][side].get("river", []))
        actions = [{"type": "Pass"}]

        # Cards
        card_map = self.rules.get("cards", {}).get(side, {})
        for cid in river:
            cdef = card_map.get(cid)
            if not cdef: continue
            cm = self._card_cost_map(side, cid)
            afford = all(res.get(k, 0) >= cm.get(k, 0) for k in ("pp","ip","mp"))
            cost_str = (cdef.get("Cost") or "").strip()
            # Allow Black Market override
            if not afford and not (cost_str == "--" and (sum(res.values()) >= 3)):
                continue
            req_text = self._card_requires_text(side, cid)
            if req_text and not self._requires_satisfied(state, side, req_text):
                continue
            actions.append({"type": "Play Card", "card_id": cid})

        # Ops
        targets = list(self.rules.get("targets", {}).keys())
        if side == "israel":
            ac = self.action_costs.get("airstrike", {"mp": 3, "ip": 3})
            if res.get("mp",0) >= int(ac.get("mp",3)) and res.get("ip",0) >= int(ac.get("ip",3)):
                for corridor in ("central","north","south"):
                    if self._corridor_ok(state, corridor):
                        for t in targets[:4]: # Limiting target search for performance
                            actions.append({
                                "type": "Order Airstrike",
                                "target": t,
                                "squadrons": ["69th","107th"],
                                "corridor": corridor,
                                "loadout": {"PGMs": ["GBU-31 JDAM"]}
                            })
                        break

        # ... Add other ops (SW, BM, Terror) similarly ...
        
        # Filter for Impulse limit
        side_now = state.get('turn', {}).get('current_player', side)
        per_impulse = state.get('turn', {}).setdefault('per_impulse_card_played', {}).get(side_now, False)
        if per_impulse:
            actions = [a for a in actions if a.get('type') != 'Play Card']
        
        actions.append({"type": "End Impulse"})
        return actions

        def apply_action(self, state: Dict[str, Any], action: Dict[str, Any], side: Optional[str] = None) -> Dict[str, Any]:
            """
            Single-step state transition for one high-level action.
            This is what MCTS/RL should call.
            """
            if state is None or not isinstance(state, dict):
                raise ValueError("state must be a dict")
            if not isinstance(action, dict):
                raise ValueError("action must be a dict")
    
            # We mutate in-place; callers (MCTS/RL) should deepcopy beforehand when needed.
            s = state
            turn = s.setdefault("turn", {})
            acting_side = side or turn.get("current_player", "israel")
            self._ensure_player(s, acting_side)
    
            a_type = action.get("type")
            if not a_type:
                raise ValueError(f"action is missing 'type': {action}")
    
            # -----------------------
            # 1) PASS / END IMPULSE
            # -----------------------
            if a_type in ("Pass", "End Impulse"):
                self._log(s, f"[ACTION] {acting_side}: {a_type}")
                turn["consecutive_passes"] = int(turn.get("consecutive_passes", 0)) + 1
                return self._advance_turn(s)
    
            # -----------------------
            # 2) PLAY CARD
            # -----------------------
            if a_type == "Play Card":
                card_id = action.get("card_id")
                if card_id is None:
                    raise ValueError("Play Card action must include 'card_id'")
                return self._apply_play_card(s, acting_side, card_id, action)
    
            # -----------------------
            # 3) OPS – AIRSTRIKE
            # -----------------------
            if a_type == "Order Airstrike":

                return self._resolve_order_airstrike(s, action, do_advance=True)
    
            # -----------------------
            # 4) OPS – SPECIAL WARFARE
            # -----------------------
            if a_type == "Order Special Warfare":
                return self._resolve_order_special_warfare(s, action, do_advance=True)
    
            # -----------------------
            # 5) OPS – BALLISTIC MISSILE
            # -----------------------
            if a_type == "Order Ballistic Missile":
                return self._resolve_order_ballistic_missile(s, action, do_advance=True)
    
            # -----------------------
            # 6) OPS – TERROR ATTACK
            # -----------------------
            if a_type == "Order Terror Attack":
                return self._resolve_order_terror_attack(s, action, do_advance=True)
    
            self._log(s, f"[WARN] Unknown action type {a_type}; treating as Pass.")
            turn["consecutive_passes"] = int(turn.get("consecutive_passes", 0)) + 1
            return self._advance_turn(s)

    # -------------------------- TURN MGMT --------------------------------------
    def _advance_turn(self, state):
        turn = state.setdefault('turn', {})
        phase = turn.get('phase', 'morning')
        cur = turn.get('current_player', 'israel')
        
        # If passes >= 2, advance phase
        if turn.get('consecutive_passes', 0) >= 2:
            turn['consecutive_passes'] = 0
            if phase == 'morning':
                turn['phase'] = 'afternoon'
                turn['current_player'] = 'israel'
            elif phase == 'afternoon':
                turn['phase'] = 'night'
                turn['current_player'] = 'israel'
            elif phase == 'night':
                self._end_of_map_turn_river_step(state)
                turn['turn_number'] = int(turn.get('turn_number', 1)) + 1
                
                # Morning Upkeep
                self._morning_reset_resources(state)
                self.apply_morning_opinion_income(state, carry_cap=None)
                self._refill_to_seven_if_needed(state)
                
                turn['phase'] = 'morning'
                turn['current_player'] = 'israel'
            
            self._reset_impulse_flags(state)
            return state
        
        # Toggle player
        turn['current_player'] = 'iran' if cur == 'israel' else 'israel'
        self._reset_impulse_flags(state)
        return state

        def _resolve_play_card_keep_impulse(self, state, action):
        
            side = state.get("turn", {}).get("current_player", "israel")
            
            self._log(state, f"[CARD] {side} keeps impulse after card play.")
            
            return state


    def _morning_reset_resources(self, state):
         for side in ("israel","iran"):
            self._ensure_player(state, side)
            res = state['players'][side]['resources']
            res['pp'] = res['ip'] = res['mp'] = 0
# ----------------------- MISSING OPERATIONAL RESOLVERS -----------------------

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

    # ----------------------- EVENT IMPACT RESOLVERS -----------------------

    def _resolve_ballistic_missile_impact(self, state, event):
        target = event.get('target')
        trules = self.rules.get('targets', {}).get(target)
        self._check_alt_victory_flags(state)
        
        if not trules:
            self._log(state, f"BM: target '{target}' not found.")
            return
            
        mtype = event.get('missile_type', 'Shahab')
        battalions = int(event.get('battalions', 1))

        # Default BM profile if not in rules
        bm_table = self.rules.get('bm_table', {})
        prof = bm_table.get(mtype, {"p_hit": 0.33, "p_backlash": 0.16, "hits": 1})

        comps = list(trules.get('Primary_Targets', {}).keys()) + list(trules.get('Secondary_Targets', {}).keys())
        if not comps: return

        for _ in range(battalions):
            comp = self._choice(state, comps)
            r = self._rng(state).random()
            if r <= float(prof.get('p_hit', 0.33)):
                hits = int(prof.get('hits', 1))
                self._apply_component_damage(state, target, comp, hits)
                self._log(state, f"BM {mtype} hits {target}:{comp} for {hits} box.")
            else:
                self._log(state, f"BM {mtype} miss on {target}:{comp}.")
            
            # Backlash Check
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
        if not trules: return
        
        pts = int(event.get('points_spent', 0))
        # Base roll logic
        roll = self._roll(state, 6) + (pts // 2)

        prim = list(trules.get('Primary_Targets', {}).keys())
        sec = list(trules.get('Secondary_Targets', {}).keys())
        
        # Higher roll allows hitting Primary targets
        pool = sec if roll < 9 else (prim or sec)
        
        if roll >= 6 and pool:
            comp = self._choice(state, pool)
            # Critical hit chance on follow-up roll
            hits = 1 + (1 if self._roll(state, 6) >= 4 else 0)
            self._apply_component_damage(state, target, comp, hits)
            self._log(state, f"SW success (roll {roll}) vs {target}:{comp} for {hits} box(es).")
        else:
            self._log(state, f"SW failed (roll {roll}).")

    def _resolve_terror_attack(self, state, event):
        inten = int(event.get('intensity', 2))
        state.setdefault('opinion', {}).setdefault('domestic', {})
        
        # Israel Domestic takes a hit
        state['opinion']['domestic']['israel'] = state['opinion']['domestic'].get('israel', 0) - 1
        
        # High intensity upsets USA
        if inten >= 4:
            state.setdefault('opinion', {}).setdefault('third_parties', {})
            state['opinion']['third_parties']['usa'] = state['opinion']['third_parties'].get('usa', 0) - 1
            
        self._log(state, f"Terror attack resolved (intensity {inten}).")

    def _resolve_rebase_complete(self, state, event):
        unit_id = event.get('unit_id')
        if not unit_id: return
        
        ad = state.setdefault('air_defense_units', {}).setdefault(unit_id, {})
        dest = ad.pop('destination', None)
        ad['status'] = 'Ready'
        if dest:
            ad['location'] = dest
        self._log(state, f"Air defense unit {unit_id} rebased to {dest} and is Ready.")

    # ----------------------------- RL & VICTORY HELPERS -----------------------------

    def rl_step(self, state: Dict[str, Any], action: Dict[str, Any], side: Optional[str] = None):
        """One RL-style environment step. Returns: (next_state, reward, done, info)"""
        if side is None:
            side = state.get("turn", {}).get("current_player", "israel").lower()

        before = copy.deepcopy(state)
        next_state = self.apply_action(before, action)

        winner = self.is_game_over(next_state)
        done = winner is not None
        reward = self._rl_reward(before, next_state, winner, side)

        info = {"winner": winner}
        return next_state, reward, done, info

    def _rl_reward(self, before: Dict[str, Any], after: Dict[str, Any], winner: Optional[str], side: str) -> float:
        side = side.lower()
        if winner is not None:
            return 1.0 if winner == side else -1.0

        def dom(st, s):
            return st.get("opinion", {}).get("domestic", {}).get(s, 0)

        def res_sum(st, s):
            r = ((st.get("players", {}) or {}).get(s, {}).get("resources", {}))
            return float(r.get("pp", 0) + r.get("ip", 0) + r.get("mp", 0))

        d_dom = dom(after, side) - dom(before, side)
        d_res = res_sum(after, side) - res_sum(before, side)

        # small weights so games don't blow up
        return 0.1 * d_dom + 0.01 * d_res

    def _domestic(self, state, side):
        return state.get("opinion", {}).get("domestic", {}).get(side, 0)

    def is_game_over(self, state) -> Optional[str]:
        # Political Victory Conditions
        if self._domestic(state, "iran") >= 10:
            return "israel"
        if self._domestic(state, "israel") <= -10:
            return "iran"

        v = state.get("victory_flags", {})
        if v.get("israel_nuclear_strategy_success"):
            return "israel"
        if v.get("israel_oil_strategy_success"):
            return "israel"

        # Turn Limit check
        cur_turn = state.get("turn", {}).get("turn_number", 1)
        scenario = (state.get("rules", {}) or {}).get("scenario")
        default_cap = 42 if str(scenario).lower() == "real_world" else 21
        max_turns = int(self.rules.get("max_turns", default_cap))
        
        if cur_turn > max_turns:
            return "iran"

        return None
    
    def _check_alt_victory_flags(self, state):
        # Placeholder for complex victory checks (e.g., calculating Oil damage %)
        pass
