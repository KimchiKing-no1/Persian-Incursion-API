import hashlib, json, uuid, importlib, copy
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from fastapi.openapi.utils import get_openapi

# ---------- Errors ----------
class StateError(HTTPException):
    def __init__(self, msg: str):
        super().__init__(status_code=422, detail=msg)

# ---------- External rules/engine ----------
rules_blob = importlib.import_module("rules_blob")   # your rules tables
try:
    ge = importlib.import_module("game_engine")      # engine helpers (legal actions, income, etc.)
except Exception:
    ge = None

# ---------- App ----------
app = FastAPI(
    title="Persian Incursion Strategy API",
    version="0.3.0",
    description="Authoritative rules + action enumerator to bound MyGPT",
    servers=[{"url": "https://persian-incursion-api.onrender.com"}]
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Persian Incursion Strategy API",
        version="0.3.0",
        description="Authoritative rules + action enumerator to bound MyGPT",
        routes=app.routes,
    )
    # IMPORTANT: add a public https URL here
    schema["servers"] = [
        {"url": "https://persian-incursion-api.onrender.com", "description": "prod"}
    ]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

PHASE_MAP = {"m": "morning", "a": "afternoon", "n": "night",
             "morning": "morning", "afternoon": "afternoon", "night": "night"}
SIDE_MAP  = {"I": "israel", "i": "israel", "R": "iran", "r": "iran",
             "israel": "israel", "iran": "iran"}

def _normalize_turn_and_resources(state: Dict[str, Any]) -> Dict[str, Any]:
    t = state.get("turn") or state.get("t")
    if not isinstance(t, dict):
        raise HTTPException(422, detail=(
            'The optimal plan generation couldn’t complete because the uploaded game state '
            'is missing the required "t/turn" object (turn number, side, time segment).\n\n'
            'Example compact:\n"t": { "n": 1, "s": "I", "ts": "m" }\n'
            'Example verbose:\n"turn": { "number": 1, "side": "Israel", "segment": "Morning" }'
        ))

    # accept n|turn_number|number
    n  = t.get("turn_number", t.get("n", t.get("number")))
    # accept s|side
    sd = t.get("side", t.get("s"))
    # accept ts|phase|segment
    ph = t.get("phase", t.get("ts", t.get("segment")))

    if n is None or sd is None or ph is None:
        raise HTTPException(422, detail=(
            'Turn object is incomplete. Required keys: number(n), side(s), segment(ts/phase).'
        ))

    side  = SIDE_MAP.get(str(sd), str(sd).lower())
    phase = PHASE_MAP.get(str(ph).lower(), str(ph).lower())

    resources = state.get("resources")
    if not resources:
        r = state.get("r")
        if isinstance(r, dict) and all(k in r for k in ("mp", "ip", "pp")):
            zero = {"mp": 0, "ip": 0, "pp": 0}
            resources = {"israel": zero.copy(), "iran": zero.copy()}
            resources[side] = {"mp": float(r["mp"]), "ip": float(r["ip"]), "pp": float(r["pp"])}
        else:
            raise HTTPException(422, detail='Missing resources (either "resources" or compact "r").')

    return {"turn_number": int(n), "side": side, "phase": phase, "resources": resources}


# ---------- Helpers ----------

def _checksum(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _alive_targets(state_dict: Dict[str, Any]) -> List[str]:
    ti = state_dict.get("ti") or {}
    return [name for name, meta in ti.items() if not meta.get("destroyed", False)]

def _day_phase(state_dict: Dict[str, Any]) -> bool:
    ph = (state_dict.get("turn") or {}).get("phase", "")
    return str(ph).lower() in ("morning", "afternoon")

def ctx_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    norm = _normalize_turn_and_resources(state)
    turn_number, phase, resources = norm["turn_number"], norm["phase"], norm["resources"]

    air = state.get("as") or state.get("air")
    if not air:
        raise HTTPException(422, detail="Missing air-side structure 'as'")

    isr_ready = [s["id"] for s in air.get("israel_squadrons", []) if str(s.get("st","")).lower() == "ready"]
    irn_ready = [s["id"] for s in air.get("iran_squadrons", [])   if str(s.get("st","")).lower() == "ready"]

    alive_targets = _alive_targets(state)
    meta  = state.get("meta") or {}
    overt = bool(meta.get("israel_overt_attack_done", False))

    return {
        "turn_number": turn_number,
        "phase": phase,
        "resources": resources,
        "ready": {"israel": isr_ready, "iran": irn_ready},
        "alive_targets": alive_targets,
        "flags": {"israel_overt_attack_done": overt},
        "checksum": _checksum(state)[:12],
        "raw": state,
    }

    
def _oil_victory_snapshot(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ge:
        return None
    try:
        eng = ge.GameEngine()
        s = copy.deepcopy(state_dict)
        # Use the engine's oil recompute helper (safe to call; updates s in-place)
        if hasattr(eng, "_recompute_oil_production"):
            eng._recompute_oil_production(s)
        os = s.get("oil_status", {})
        vf = s.get("victory_flags", {})
        totals = os.get("totals", {})
        hit = os.get("hit", {})
        # percentages
        crude_pct = 0 if not totals.get("crude") else int(100 * hit.get("crude", 0) / totals["crude"])
        ref_pct   = 0 if not totals.get("refinery") else int(100 * hit.get("refinery", 0) / totals["refinery"])
        return {
            "crude_pct": crude_pct,
            "refinery_pct": ref_pct,
            "win": bool(vf.get("israel_oil_strategy_success", False)),
        }
    except Exception:
        return None
        
def _nuclear_victory_snapshot(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Best-effort nuclear readiness advisory:
    1) Prefer engine API (e.g., eng.nuclear_status or victory flags)
    2) Else, use rules_blob.NUCLEAR_SITES if present
    3) Else, heuristic: count ti entries that look nuclear by name/tags
    Returns dict with counts and a conservative 'ready' boolean if an engine/victory flag exists.
    """
    if not state_dict:
        return None

    # 1) Prefer engine
    try:
        if ge:
            eng = ge.GameEngine()
            # If engine exposes a nuclear status/victory flag, use it
            # Try victory_flags first (common pattern)
            s = copy.deepcopy(state_dict)
            vf = (s.get("victory_flags") or {})
            if "israel_nuclear_program_neutralized" in vf:
                return {
                    "destroyed": None,
                    "total": None,
                    "ready": bool(vf["israel_nuclear_program_neutralized"]),
                    "source": "victory_flags"
                }
            # Or a method if provided
            if hasattr(eng, "nuclear_status"):
                try:
                    ns = eng.nuclear_status(s)  # should return something structured
                    return {
                        "destroyed": ns.get("destroyed"),
                        "total": ns.get("total"),
                        "ready": bool(ns.get("ready", False)),
                        "source": "engine"
                    }
                except Exception:
                    pass
    except Exception:
        pass

    # 2) Use rules table if present
    nuclear_sites = []
    try:
        nuclear_sites = getattr(rules_blob, "NUCLEAR_SITES", [])
    except Exception:
        nuclear_sites = []

    # 3) Fallback heuristic: find nuclear-looking targets in ti
    ti = state_dict.get("ti") or {}
    def looks_nuclear(name: str, meta: Dict[str, Any]) -> bool:
        lname = name.lower()
        tags = (meta.get("tags") or [])
        cat  = str(meta.get("category", "")).lower()
        # heuristic name matches
        name_hit = any(k in lname for k in ("nuclear", "uranium", "centrifuge", "natanz", "fordow", "isfahan", "arak", "bushehr"))
        tag_hit  = any(str(t).lower() in ("nuclear", "uranium", "centrifuge") for t in tags)
        cat_hit  = cat in ("nuclear", "uranium")
        list_hit = (name in nuclear_sites) if nuclear_sites else False
        return name_hit or tag_hit or cat_hit or list_hit

    nuc_targets = [(n, m) for n, m in ti.items() if looks_nuclear(n, m)]
    total = len(nuc_targets)
    destroyed = sum(1 for _, m in nuc_targets if m.get("destroyed", False))

    # Without engine thresholds, don't claim victory—just report status
    return {
        "destroyed": destroyed,
        "total": total,
        "ready": False,       # advisory only (no hard claim)
        "source": "heuristic" if not nuclear_sites else "rules_blob"
    }
        
def _sum_plan_costs(steps: List["PlanStep"], uni: Dict[str, "EnumeratedAction"]) -> Dict[str, float]:
    need = {"pp": 0.0, "ip": 0.0, "mp": 0.0}
    for st in steps:
        a = uni.get(st.action_id)
        if not a:
            continue
        for k, v in (a.cost or {}).items():
            if k in need:
                need[k] += float(v)
    return need

def _side_code(side_str: str) -> str:
    s = (side_str or "").strip().lower()
    if s in ("israel", "i"): return "is"
    if s in ("iran", "r"):   return "ir"
    return s  # fallback

# ---------- Models ----------
class Squadron(BaseModel):
    id: str
    ty: str
    st: str
    cu: Optional[int] = None
    b: Optional[str] = None
    mission_status: Optional[str] = None

class AirSide(BaseModel):
    israel_squadrons: List[Squadron] = Field(default_factory=list)
    iran_squadrons: List[Squadron] = Field(default_factory=list)

class GameState(BaseModel):
    t: Optional[dict] = None
    r: Optional[dict] = None
    o: Optional[dict] = None
    as_: AirSide = Field(..., alias="as")
    u: Optional[dict] = None
    bm: Optional[dict] = None
    swm: Optional[dict] = None
    ti: Optional[dict] = None
    opinion: Optional[dict] = None
    players: Optional[dict] = None
    meta: Optional[dict] = None
    turn: Optional[dict] = None
    resources: Optional[dict] = None  # ensure present for ctx checks

class EnumerateActionsRequest(BaseModel):
    state: GameState
    side_to_move: Optional[str] = "Israel"
    max_actions: Optional[int] = 60

class EnumeratedAction(BaseModel):
    action_id: str
    kind: str
    description: str
    preconditions: List[str] = Field(default_factory=list)
    cost: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    timing: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    objective: Optional[str] = None
    steps: List["PlanStep"] = Field(default_factory=list)
    state: Dict[str, Any]


class PlanStep(BaseModel):
    action_id: str
    rationale: Optional[str] = None

class NoncedPlan(BaseModel):
    nonce: str
    plan: Plan

# ---------- Universes (nonce → action_id → EnumeratedAction) ----------
_UNIVERSES: Dict[str, Dict[str, EnumeratedAction]] = {}

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "service": "PI-API", "version": "0.3.0"}

# ---------- State validate / canonicalize ----------
@app.post("/state/validate")
def state_validate(payload: GameState):
    warn = []
    if not payload.as_.israel_squadrons:
        warn.append("No israel_squadrons found")
    if not payload.ti:
        warn.append("No target intelligence (ti) found")
    note = None
    if ge and hasattr(ge, "apply_morning_opinion_income"):
        s = payload.dict(by_alias=True)
        ge.apply_morning_opinion_income(s, carry_cap=None, log_fn=None)
        note = "Opinion income computed (not persisted)"
    return {"ok": True, "warnings": warn, "note": note}

@app.post("/state/canonicalize")
def state_canonicalize(payload: GameState):
    sdict = payload.dict(by_alias=True)
    ctx = ctx_from_state(sdict)
    return {
        "ok": True,
        "checksum": ctx["checksum"],
        "turn": ctx["turn_number"],
        "phase": ctx["phase"],
        "ready_counts": {k: len(v) for k, v in ctx["ready"].items()},
        "targets_alive": len(ctx["alive_targets"]),
    }


@app.post("/advisory/victory")
def advisory_victory(payload: GameState):
    sdict = payload.dict(by_alias=True)
    ctx = ctx_from_state(sdict)
    oil = _oil_victory_snapshot(sdict)
    nuke = _nuclear_victory_snapshot(sdict)
    return {
        "ok": True,
        "turn": ctx["turn_number"],
        "phase": ctx["phase"],
        "oil": oil,
        "nuclear": nuke
    }



# ---------- Action enumeration (engine-backed + dynamic gates) ----------
def _engine_actions(sdict: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    if not ge:
        return []
    eng = ge.GameEngine()
    return eng.get_legal_actions(sdict, side=side)  # engine respects costs/corridors/cards  # noqa

def _derive_actions(state_dict: Dict[str, Any], side_to_move: str) -> List[EnumeratedAction]:
    ctx = ctx_from_state(state_dict)
    side = side_to_move.lower()

    # Day-only gate for launching air ops (per rules)
    if side == "israel" and ctx["phase"] not in ("morning", "afternoon"):
        raise HTTPException(422, detail="Air operations must be launched in Day phases.")

    engine_out = _engine_actions(state_dict, side)
    alive = set(ctx["alive_targets"])
    ready = {
        "israel": set(ctx["ready"]["israel"]),
        "iran": set(ctx["ready"]["iran"]),
    }

    # Pull restrike/exec window hints from engine (best-effort)
    plan_delay = 1
    exec_window = 1
    try:
        if ge and hasattr(ge.GameEngine(), "restrike_rules"):
            rr = ge.GameEngine().restrike_rules
            plan_delay = int(rr.get("plan_delay_turns", plan_delay))
            exec_window = int(rr.get("execute_window_turns", exec_window))
    except Exception:
        pass

    pruned: List[EnumeratedAction] = []
    for a in engine_out:
        atype = a.get("type", "")

        # ---------- SAM legality filters ----------
        # Relocate SAM: only after overt Israeli attack
        if atype in ("Relocate SAM", "RelocateSAM"):
            if not ctx["flags"]["israel_overt_attack_done"]:
                continue

        # Pre-overt: block special long/med SAM engagement types if your engine emits them
        # (examples: "SAM Engagement", "LongRangeSAM", "MediumRangeSAM")
        if atype in ("SAM Engagement", "LongRangeSAM", "MediumRangeSAM"):
            sysname = (a.get("system") or a.get("sam") or "").upper()
            if not ctx["flags"]["israel_overt_attack_done"]:
                if sysname in ("S-300", "HQ-9", "TOR", "S300", "HQ9"):
                    continue

        # Airstrike target sanity and Israeli day-only gate
        if atype in ("Order Airstrike", "AirStrike"):
            if not ctx["ready"]["israel"]:
                continue
            tgt = a.get("target")
            if tgt not in alive:
                continue

        # ----- timing metadata (helps the model schedule correctly) -----
        # Defaults per type
        if atype in ("Plan Strike", "PLAN_STRIKE"):
            timing = {
                "earliest_phase": "next_morning",  # plans execute next eligible window
                "duration_phases": 0,
                "carry_over": True,
                "window": {
                    "start_turn": ctx["turn_number"] + plan_delay,
                    "end_turn": ctx["turn_number"] + plan_delay + exec_window
                }
            }
        elif atype in ("Order Airstrike", "AirStrike", "Recon"):
            timing = {
                "earliest_phase": ctx["phase"],     # can start now if legal
                "duration_phases": 1,
                "carry_over": False
            }
        elif atype in ("Relocate SAM", "RelocateSAM", "InterceptCAP"):
            timing = {
                "earliest_phase": ctx["phase"],
                "duration_phases": 1,
                "carry_over": False
            }
        else:
            # Safe default for unknown engine verbs
            timing = {
                "earliest_phase": ctx["phase"],
                "duration_phases": 1,
                "carry_over": False
            }

        ea = EnumeratedAction(
            action_id=str(uuid.uuid4()),
            kind=atype,
            description=(f"{atype} → {a.get('target')}" if a.get("target") else atype),
            preconditions=a.get("preconditions", []),
            cost=a.get("cost", {}),
            tags=a.get("tags", []),
            timing=timing,
        )
        pruned.append(ea)

    # If engine provides nothing (early scaffolding), offer minimal legal shells
    if not pruned:
        if side == "israel" and ready["israel"] and ctx["phase"] in ("morning", "afternoon"):
            for tgt in list(alive)[:8]:
                pruned.append(EnumeratedAction(
                    action_id=str(uuid.uuid4()),
                    kind="AirStrike",
                    description=f"Airstrike → {tgt}",
                    preconditions=["Day phase", "≥1 Ready squadron"],
                    cost={"mp": 1.0},
                    tags=["offense"],
                    timing={
                        "earliest_phase": ctx["phase"],
                        "duration_phases": 1,
                        "carry_over": False
                    }
                ))
            pruned.append(EnumeratedAction(
                action_id=str(uuid.uuid4()),
                kind="Recon",
                description="Daytime recon in front sectors",
                preconditions=["Day phase", "Weather OK"],
                cost={"mp": 0.5},
                tags=["intel"],
                timing={
                    "earliest_phase": ctx["phase"],
                    "duration_phases": 1,
                    "carry_over": False
                }
            ))
        if side == "iran" and ready["iran"] and ctx["phase"] in ("morning", "afternoon"):
            pruned.append(EnumeratedAction(
                action_id=str(uuid.uuid4()),
                kind="InterceptCAP",
                description="Establish CAP over critical SAM zones",
                preconditions=["Day phase", "GCI available"],
                cost={"mp": 0.7},
                tags=["defense"],
                timing={
                    "earliest_phase": ctx["phase"],
                    "duration_phases": 1,
                    "carry_over": False
                }
            ))
            if ctx["flags"]["israel_overt_attack_done"]:
                pruned.append(EnumeratedAction(
                    action_id=str(uuid.uuid4()),
                    kind="Relocate SAM",
                    description="Move one SAM battery to a new node (post-overt only)",
                    preconditions=["Post-overt-attack only"],
                    cost={"ip": 1.0},
                    tags=["air-defense"],
                    timing={
                        "earliest_phase": ctx["phase"],
                        "duration_phases": 1,
                        "carry_over": False
                    }
                ))
    return pruned

@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)
    actions = _derive_actions(sdict, req.side_to_move)
    actions = actions[: (req.max_actions or 60)]

    code = _side_code(req.side_to_move)           
    cs = _checksum(sdict)[:12]
    nonce = f"N-{cs}-{code}"
    _UNIVERSES[nonce] = {a.action_id: a for a in actions}

    return {
        "nonce": nonce,
        "side_to_move": req.side_to_move,
        "turn_phase": sdict.get("turn") or sdict.get("t"),
        "actions": [a.dict() for a in actions],
    }



# ---------- Plan validate ----------
@app.post("/plan/validate")
def plan_validate(req: NoncedPlan):
    uni = _UNIVERSES.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")

    # bind to state checksum encoded in nonce
    plan_state = req.plan.state if isinstance(req.plan.state, dict) else None
    if not plan_state:
        return {"ok": False, "illegal_steps": [{"index": 0, "reason": "plan.state missing"}]}

    cs = _checksum(plan_state)[:12]
    if not req.nonce.startswith(f"N-{cs}-"):
        return {"ok": False, "illegal_steps": [{"index": 0, "reason": "state/nonce mismatch"}]}

    illegal: List[Dict[str, Any]] = []
    for i, st in enumerate(req.plan.steps):
        if st.action_id not in uni:
            illegal.append({"index": i, "action_id": st.action_id, "reason": "Not in legal action set"})

    # Parse state context once (we'll use it for timing & resources)
    try:
        ctx = ctx_from_state(plan_state)
    except HTTPException as e:
        return {"ok": False, "illegal_steps": illegal + [{"index": 0, "reason": f"state invalid: {e.detail}"}]}

    # -------- Timing window enforcement --------
    # Current turn number from the provided state
    turn = ctx["turn_number"]
    illegal_timing: List[Dict[str, Any]] = []

    for i, st in enumerate(req.plan.steps):
        a = uni.get(st.action_id)
        if not a:
            continue  # already captured as illegal above
        # 'uni' holds EnumeratedAction objects; timing is a dict (possibly empty)
        tmeta = getattr(a, "timing", None) or {}
        win = tmeta.get("window")
        if isinstance(win, dict):
            start_turn = win.get("start_turn")
            end_turn = win.get("end_turn")
            # Enforce only when both bounds exist and are ints
            if isinstance(start_turn, int) and isinstance(end_turn, int):
                if not (start_turn <= turn <= end_turn):
                    illegal_timing.append({
                        "index": i,
                        "action_id": st.action_id,
                        "reason": f"timing window {win} not satisfied at turn {turn}"
                    })

    if illegal_timing:
        return {"ok": False, "illegal_steps": illegal + illegal_timing}

    # -------- Resource feasibility check --------
    # Side extraction from nonce suffix: ...-is or ...-ir
    side_code = req.nonce.rsplit("-", 1)[-1]
    if side_code == "is":
        side = "israel"
    elif side_code == "ir":
        side = "iran"
    else:
        raise HTTPException(422, f"Bad nonce side code: {side_code}")


    have = {k: float(ctx["resources"][side].get(k, 0)) for k in ("pp", "ip", "mp")}
    need = _sum_plan_costs(req.plan.steps, uni)

    deficits = []
    for k in ("pp", "ip", "mp"):
        if need[k] > have[k]:
            deficits.append({"resource": k, "need": round(need[k], 3), "have": round(have[k], 3)})

    ok = (len(illegal) == 0) and (len(deficits) == 0)
    return {
        "ok": ok,
        "illegal_steps": illegal,
        "resource_check": {
            "side": side,
            "need": {k: round(v, 3) for k, v in need.items()},
            "have": {k: round(v, 3) for k, v in have.items()},
            "deficits": deficits
        }
    }

# ---------- Plan score (simple, deterministic) ----------
@app.post("/plan/score")
def plan_score(req: NoncedPlan):
    uni = _UNIVERSES.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")

    dmg = intel = 0.0
    cost = 0.0
    for st in req.plan.steps:
        a = uni.get(st.action_id)
        if not a:
            continue
        if a.kind == "AirStrike":
            dmg += 4.0
            cost -= float(a.cost.get("mp", 1.5))
        elif a.kind == "Recon":
            intel += 1.0
            cost -= float(a.cost.get("mp", 0.5))
        elif a.kind == "InterceptCAP":
            intel += 0.2
            cost -= float(a.cost.get("mp", 0.5))
        elif a.kind in ("Relocate SAM", "RelocateSAM"):
            cost -= 0.25 * float(a.cost.get("ip", 1.0))

    total = round(dmg + intel + cost, 3)

    # --- advisories based on current state snapshot (not simulated effects)
    plan_state = req.plan.state if isinstance(req.plan.state, dict) else None
    oil_hint = _oil_victory_snapshot(plan_state) if plan_state else None
    nuke_hint = _nuclear_victory_snapshot(plan_state) if plan_state else None

    return {
        "total": total,
        "breakdown": {
            "damage": round(dmg, 3),
            "intel": round(intel, 3),
            "resource_cost": round(cost, 3)
        },
        "advisory": {
            "oil": oil_hint,
            "nuclear": nuke_hint
        }
    }
    
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "PI-API",
        "version": "0.3.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

# ---------- Baseline suggest (optional A/B for the model) ----------
@app.post("/plan/suggest")
def plan_suggest(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)
    actions = _derive_actions(sdict, req.side_to_move)

    code = _side_code(req.side_to_move)         # ✅ use helper
    cs = _checksum(sdict)[:12]
    nonce = f"N-{cs}-{code}"
    _UNIVERSES[nonce] = {a.action_id: a for a in actions}

    picks = [a for a in actions if a.kind == "AirStrike"][:2]
    recon = next((a for a in actions if a.kind == "Recon"), None)
    if recon:
        picks.append(recon)

    plan = {
        "objective": "Disrupt high-value facilities while gaining intel",
        "steps": [{"action_id": a.action_id, "rationale": a.description} for a in picks],
        "state": sdict,
    }
    return {"nonce": nonce, "plan": plan}


@app.get("/privacy")
def privacy():
    return HTMLResponse("""
    <h2>Privacy Policy</h2>
    <p>This API processes only game state JSON for simulation. No personal data is collected or stored.</p>
    """)

@app.get("/terms")
def terms():
    return HTMLResponse("""
    <h2>Terms of Use</h2>
    <p>This API is for research and entertainment. Use at your own risk.</p>
    """)



@app.post("/plan/execute")
def plan_execute(req: NoncedPlan):
    """
    Apply a validated plan to the provided state and return the NEW state
    in the SAME format the user provided (no invented fields).
    """
    uni = _UNIVERSES.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")

    # 1) Bind to exact state via checksum-in-nonce (same as /plan/validate)
    plan_state = req.plan.state if isinstance(req.plan.state, dict) else None
    if not plan_state:
        raise HTTPException(422, "plan.state missing")
    cs = _checksum(plan_state)[:12]
    if not req.nonce.startswith(f"N-{cs}-"):
        raise HTTPException(422, "state/nonce mismatch")

    # 2) Steps must be from legal set
    illegal = [i for i, st in enumerate(req.plan.steps) if st.action_id not in uni]
    if illegal:
        raise HTTPException(422, f"Illegal steps at indexes: {illegal}")

    # 3) Resource feasibility (reuse your helper)
    side_code = req.nonce.rsplit("-", 1)[-1]
    if side_code == "is":
        side = "israel"
    elif side_code == "ir":
        side = "iran"
    else:
        raise HTTPException(422, f"Bad nonce side code: {side_code}")

    ctx = ctx_from_state(plan_state)  # will raise 422 if bad
    have = {k: float(ctx["resources"][side].get(k, 0)) for k in ("pp", "ip", "mp")}
    need = _sum_plan_costs(req.plan.steps, uni)
    for k in ("pp","ip","mp"):
        if need[k] > have[k]:
            raise HTTPException(422, f"Insufficient {k}: need {need[k]}, have {have[k]}")

    # 4) Apply with engine if available
    s0 = copy.deepcopy(plan_state)
    log: List[str] = []
    if ge and hasattr(ge, "apply_actions"):
        # Expect engine to accept a state and a list of abstract actions
        # Convert action_ids back to the engine action dicts we stored
        engine_actions = []
        for st in req.plan.steps:
            a = uni.get(st.action_id)
            if not a:
                continue
            # minimal back-projection for engine; include 'type' and common fields
            engine_actions.append({
                "type": a.kind,
                "target": (a.description.split("→",1)[-1].strip() if "→" in a.description else None),
                "cost": a.cost,
                "tags": a.tags,
            })
        try:
            s1, engine_log = ge.apply_actions(s0, engine_actions, side=side)
            if isinstance(engine_log, list):
                log.extend([str(x) for x in engine_log])
            new_state = s1
        except Exception as e:
            raise HTTPException(500, f"Engine apply_actions failed: {e}")
    else:
        # Strict fallback: if there is no engine, refuse to "invent" effects.
        raise HTTPException(501, "Engine not available: cannot execute plan without changing state")

    # 5) Return NEW state in the SAME format (no reformatting)
    return {
        "ok": True,
        "applied_steps": [st.action_id for st in req.plan.steps],
        "side": side,
        "resource_spend": {k: round(need[k], 3) for k in need},
        "log": log,
        "state": new_state  # <-- feed this straight back into the human->MyGPT loop
    }

@app.post("/turn/ai_move")
def turn_ai_move(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)
    actions = _derive_actions(sdict, req.side_to_move)
    if not actions:
        raise HTTPException(422, "No legal actions for this side/phase")

    picks = [a for a in actions if a.kind == "AirStrike"][:2]
    recon = next((a for a in actions if a.kind == "Recon"), None)
    if recon: picks.append(recon)
    if not picks:
        picks = actions[:min(3, len(actions))]

    code = _side_code(req.side_to_move)           
    cs = _checksum(sdict)[:12]
    nonce = f"N-{cs}-{code}"
    _UNIVERSES[nonce] = {a.action_id: a for a in actions}

    plan = {"objective": "AI best guess",
            "steps": [{"action_id": a.action_id} for a in picks],
            "state": sdict}
    plan_req = NoncedPlan(nonce=nonce, plan=Plan(**plan))

    v = plan_validate(plan_req)
    if not v["ok"]:
        raise HTTPException(422, f"Plan invalid: {v}")

    exec_out = plan_execute(plan_req)
    return {"nonce": nonce, "chosen_steps": [s["action_id"] for s in plan["steps"]], "result": exec_out}

