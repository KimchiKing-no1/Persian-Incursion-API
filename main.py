import hashlib, json, uuid, importlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    turn = state.get("turn", {})
    turn_number = turn.get("turn_number")
    phase = str(turn.get("phase", "")).lower()
    if not turn_number or phase not in ("morning", "afternoon", "night"):
        raise HTTPException(422, detail="turn.turn_number or turn.phase missing/invalid")

    resources = state.get("resources")
    if not resources or "israel" not in resources or "iran" not in resources:
        raise HTTPException(422, detail="Missing resources.israel or resources.iran")

    air = state.get("as")
    if not air:
        raise HTTPException(422, detail="Missing air-side structure 'as'")
    isr_ready = [s["id"] for s in air.get("israel_squadrons", []) if str(s.get("st", "")).lower() == "ready"]
    irn_ready = [s["id"] for s in air.get("iran_squadrons", []) if str(s.get("st", "")).lower() == "ready"]

    alive_targets = _alive_targets(state)
    meta = state.get("meta") or {}
    overt = bool(meta.get("israel_overt_attack_done", False))

    return {
        "turn_number": int(turn_number),
        "phase": phase,
        "resources": resources,
        "ready": {"israel": isr_ready, "iran": irn_ready},
        "alive_targets": alive_targets,
        "flags": {"israel_overt_attack_done": overt},
        "checksum": _checksum(state)[:12],
        "raw": state,
    }

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
    preconditions: List[str] = []
    cost: Dict[str, float] = {}
    tags: List[str] = []

class PlanStep(BaseModel):
    action_id: str
    rationale: Optional[str] = None

class Plan(BaseModel):
    objective: Optional[str] = None
    steps: List[PlanStep] = []
    # required: API must bind the plan to the exact state used for enumeration
    state: Dict[str, Any]

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
def state_canonicalize(payload: Dict[str, Any]):
    ctx = ctx_from_state(payload)
    return {
        "ok": True,
        "checksum": ctx["checksum"],
        "turn": ctx["turn_number"],
        "phase": ctx["phase"],
        "ready_counts": {k: len(v) for k, v in ctx["ready"].items()},
        "targets_alive": len(ctx["alive_targets"]),
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

    pruned: List[EnumeratedAction] = []
    for a in engine_out:
        atype = a.get("type", "")
        # Filter based on state reality
        if atype == "Order Airstrike":
            if not ready["israel"]:
                continue
            tgt = a.get("target")
            if tgt not in alive:
                continue
        if atype == "Relocate SAM":
            if not ctx["flags"]["israel_overt_attack_done"]:
                continue

        ea = EnumeratedAction(
            action_id=str(uuid.uuid4()),
            kind=atype,
            description=(f"{atype} → {a.get('target')}" if a.get("target") else atype),
            preconditions=a.get("preconditions", []),
            cost=a.get("cost", {}),
            tags=a.get("tags", []),
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
                ))
            pruned.append(EnumeratedAction(
                action_id=str(uuid.uuid4()),
                kind="Recon",
                description="Daytime recon in front sectors",
                preconditions=["Day phase", "Weather OK"],
                cost={"mp": 0.5},
                tags=["intel"],
            ))
        if side == "iran" and ready["iran"] and ctx["phase"] in ("morning", "afternoon"):
            pruned.append(EnumeratedAction(
                action_id=str(uuid.uuid4()),
                kind="InterceptCAP",
                description="Establish CAP over critical SAM zones",
                preconditions=["Day phase", "GCI available"],
                cost={"mp": 0.7},
                tags=["defense"],
            ))
            if ctx["flags"]["israel_overt_attack_done"]:
                pruned.append(EnumeratedAction(
                    action_id=str(uuid.uuid4()),
                    kind="Relocate SAM",
                    description="Move one SAM battery to a new node (post-overt only)",
                    preconditions=["Post-overt-attack only"],
                    cost={"ip": 1.0},
                    tags=["air-defense"],
                ))
    return pruned

@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)
    actions = _derive_actions(sdict, req.side_to_move)
    actions = actions[: (req.max_actions or 60)]

    cs = _checksum(sdict)[:12]
    nonce = f"N-{cs}-{req.side_to_move[:2].lower()}"
    _UNIVERSES[nonce] = {a.action_id: a for a in actions}

    return {
        "nonce": nonce,
        "side_to_move": req.side_to_move,
        "turn_phase": sdict.get("turn"),
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

    illegal = []
    for i, st in enumerate(req.plan.steps):
        if st.action_id not in uni:
            illegal.append({"index": i, "action_id": st.action_id, "reason": "Not in legal action set"})
    return {"ok": len(illegal) == 0, "illegal_steps": illegal}

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
        elif a.kind == "Relocate SAM":
            cost -= 0.25 * float(a.cost.get("ip", 1.0))

    total = round(dmg + intel + cost, 3)
    return {"total": total, "breakdown": {"damage": round(dmg, 3), "intel": round(intel, 3), "resource_cost": round(cost, 3)}}

# ---------- Baseline suggest (optional A/B for the model) ----------
@app.post("/plan/suggest")
def plan_suggest(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)
    actions = _derive_actions(sdict, req.side_to_move)
    nonce = f"N-{_checksum(sdict)[:12]}-{req.side_to_move[:2].lower()}"
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
