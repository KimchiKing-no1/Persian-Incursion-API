import hashlib, json 
import uuid
import importlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple


class StateError(HTTPException):
    def __init__(self, msg: str):
        super().__init__(status_code=422, detail=msg)

ge = importlib.import_module("game_engine")  # you already ship this

def _alive_targets(state_dict):
    ti = state_dict.get("ti") or {}
    return [name for name, meta in ti.items() if not meta.get("destroyed", False)]

def _day_phase(state_dict):
    ph = (state_dict.get("turn") or {}).get("phase","").lower()
    return ph in ("morning","afternoon")

@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    sdict = req.state.dict(by_alias=True)  # exact user state, no invention
    side = req.side_to_move.lower()

    # hard gates the rules expect
    if side == "israel" and not _day_phase(sdict):
        raise HTTPException(422, detail="Air operations must be launched in Day phases per rules.")

    # generate with your engine (respects costs/corridors/cards)
    eng = ge.GameEngine()
    legal = eng.get_legal_actions(sdict, side=side)

    # filter engine’s output to current alive targets & readiness
    alive = set(_alive_targets(sdict))
    ready = {
      "israel": {sq["id"] for sq in (sdict.get("as") or {}).get("israel_squadrons",[]) if (sq.get("st","").lower()=="ready")},
      "iran":   {sq["id"] for sq in (sdict.get("as") or {}).get("iran_squadrons",[])   if (sq.get("st","").lower()=="ready")}
    }
    pruned = []
    for a in legal:
        if a["type"] == "Order Airstrike":
            if a.get("target") not in alive: 
                continue
            # no night frag; require at least one Ready squadron present
            if not ready["israel"]:
                continue
        if a["type"] == "Relocate SAM":
            # allowed only AFTER overt attack per rules
            if not (sdict.get("meta") or {}).get("israel_overt_attack_done", False):
                continue
        pruned.append(a)

    # convert to your EnumeratedAction shape
    out = []
    for i, a in enumerate(pruned[: (req.max_actions or 60)]):
        out.append({
          "action_id": f"{uuid.uuid4()}",
          "kind": a["type"],
          "description": a.get("target") and f"{a['type']} → {a['target']}" or a["type"],
          "preconditions": a.get("preconditions", []),
          "cost": a.get("cost", {}),
          "tags": a.get("tags", [])
        })

    nonce = f"N-{hashlib.sha256(json.dumps(sdict, sort_keys=True).encode()).hexdigest()[:12]}-{side[:2]}"
    _universes[nonce] = {x["action_id"]: x for x in out}
    return {"nonce": nonce, "side_to_move": req.side_to_move, "actions": out}

def _get(state: Dict[str, Any], path: List[str], *, required=True):
    cur = state
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            if required:
                raise StateError(f"Missing required key: {'.'.join(path)}")
            return None
    return cur

def _checksum(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":")).encode()).hexdigest()[:16]

def ctx_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strictly extract everything from the user's JSON file, no defaults."""
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
    isr_ready = [s["id"] for s in air.get("israel_squadrons", []) if str(s.get("st","")).lower()=="ready"]
    irn_ready = [s["id"] for s in air.get("iran_squadrons", []) if str(s.get("st","")).lower()=="ready"]

    ti = state.get("ti") or {}
    alive_targets = [k for k,v in ti.items() if not v.get("destroyed", False)]

    meta = state.get("meta") or {}
    overt = bool(meta.get("israel_overt_attack_done", False))
    real_world = bool(meta.get("real_world", False))

    return {
        "turn_number": int(turn_number),
        "phase": phase,
        "resources": resources,
        "ready": {"israel": isr_ready, "iran": irn_ready},
        "alive_targets": alive_targets,
        "flags": {"israel_overt_attack_done": overt, "real_world": real_world},
        "checksum": _checksum(state),
        "raw": state,
    }


# ---- bring in your rules & (optionally) engine helpers ----
rules_blob = importlib.import_module("rules_blob")   # your file
try:
    ge = importlib.import_module("game_engine")      # optional use
except Exception:
    ge = None

app = FastAPI(
    title="Persian Incursion Strategy API",
    version="0.2.0",
    description="Authoritative rules + action enumerator to bound MyGPT",
    servers=[{"url": "https://persian-incursion-api.onrender.com"}]  # <-- add this
)

# -------- Models aligned with your JSON samples --------
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

class NoncedPlan(BaseModel):
    nonce: str
    plan: Plan

# -------- in-memory nonce→universe --------
_universes: Dict[str, Dict[str, EnumeratedAction]] = {}

@app.get("/health")
def health():
    return {"ok": True, "service": "PI-API", "version": "0.2.0"}

@app.post("/state/validate")
def state_validate(payload: GameState):
    # Basic shape checks: do we see expected keys from your samples?
    warn = []
    if not payload.as_.israel_squadrons:
        warn.append("No israel_squadrons found")
    if not payload.ti:
        warn.append("No target intelligence (ti) found")
    # Optional: apply opinion income using your GameEngine
    income_note = None
    if ge and hasattr(ge, "apply_morning_opinion_income"):
        s = payload.dict(by_alias=True)
        ge.apply_morning_opinion_income(s, carry_cap=None, log_fn=None)
        income_note = "Opinion income applied (PP/IP/MP updated)"
    return {"ok": True, "warnings": warn, "note": income_note}

# ---- Simple, safe enumerator (replace with your full rules later) ----
def _ready(sq: Squadron): return (sq.st or "").lower() == "ready"

def _derive_actions(state: Dict[str, Any], side_to_move: str) -> List[EnumeratedAction]:
    ctx = ctx_from_state(state)
    side = side_to_move.lower()
    acts: List[EnumeratedAction] = []

    # Phase restriction: only day operations for airstrikes / recon
    def day_only() -> bool: return ctx["phase"] in ("morning", "afternoon")

    res = ctx["resources"][side]
    ready = ctx["ready"][side]
    targets = ctx["alive_targets"]

    if side == "israel":
        if res["mp"] > 0 and ready and day_only():
            for tgt in targets[:10]:
                aid = str(uuid.uuid4())
                acts.append(EnumeratedAction(
                    action_id=aid,
                    kind="AirStrike",
                    description=f"Airstrike on {tgt} using available Israeli squadrons",
                    preconditions=["Day phase", "At least one Ready squadron"],
                    cost={"mp": 1},
                    tags=["offense"]
                ))
        if ready and day_only():
            aid = str(uuid.uuid4())
            acts.append(EnumeratedAction(
                action_id=aid,
                kind="Recon",
                description="Daytime recon mission near front sectors",
                preconditions=["Day phase", "Weather OK"],
                cost={"mp": 0.5},
                tags=["intel"]
            ))

    elif side == "iran":
        if res["mp"] > 0 and ready and day_only():
            aid = str(uuid.uuid4())
            acts.append(EnumeratedAction(
                action_id=aid,
                kind="InterceptCAP",
                description="Establish CAP over Tehran or major SAM zones",
                preconditions=["Day phase", "GCI available"],
                cost={"mp": 0.7},
                tags=["defense"]
            ))
        if ctx["flags"]["israel_overt_attack_done"] and res["ip"] > 0:
            aid = str(uuid.uuid4())
            acts.append(EnumeratedAction(
                action_id=aid,
                kind="RelocateSAM",
                description="Move one SAM battery to a new node",
                preconditions=["Post-overt-attack only"],
                cost={"ip": 1},
                tags=["air-defense"]
            ))
    return acts


@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    actions = _derive_actions(req.state.dict(by_alias=True), req.side_to_move)
    actions = actions[: (req.max_actions or 60)]
    nonce = f"N-{_checksum(req.state.dict(by_alias=True))}-{req.side_to_move[:2].lower()}"
    _universes[nonce] = {a.action_id: a for a in actions}
    return {
        "nonce": nonce,
        "side_to_move": req.side_to_move,
        "turn_phase": req.state.dict().get("turn"),
        "actions": [a.dict() for a in actions]
    }


@app.post("/plan/validate")
def plan_validate(req: NoncedPlan):
    uni = _universes.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")

    # bind to state checksum hidden in nonce
    plan_state = getattr(req.plan, "state", None)
    if not plan_state:
        return {"ok": False, "illegal_steps":[{"index":0,"reason":"plan.state missing"}]}
    cs = hashlib.sha256(json.dumps(plan_state, sort_keys=True).encode()).hexdigest()[:12]
    if f"N-{cs}" not in req.nonce:
        return {"ok": False, "illegal_steps":[{"index":0,"reason":"state/nonce mismatch"}]}

    illegal = []
    for i, st in enumerate(req.plan.steps):
        if st.action_id not in uni:
            illegal.append({"index": i, "action_id": st.action_id, "reason": "Not in legal action set"})
    # optional: oil/nuclear flags sanity (advisory)
    return {"ok": len(illegal)==0, "illegal_steps": illegal}


@app.post("/plan/score")
def plan_score(req: NoncedPlan):
    uni = _universes.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")
    score = 0.0
    breakdown = {"damage": 0.0, "intel": 0.0, "resource_cost": 0.0}
    for s in req.plan.steps:
        a = uni.get(s.action_id)
        if not a: 
            continue
        if a.kind == "AirStrike":
            breakdown["damage"] += 4.0
            breakdown["resource_cost"] -= 1.5
        elif a.kind in ("Recon","InterceptCAP"):
            breakdown["intel"] += 1.0 if a.kind == "Recon" else 0.2
            breakdown["resource_cost"] -= 0.5
    total = breakdown["damage"] + breakdown["intel"] + breakdown["resource_cost"]
    return {"total": round(total, 3), "breakdown": {k: round(v, 3) for k, v in breakdown.items()}}

# Optional: server baseline plan for A/B
@app.post("/plan/suggest")
def plan_suggest(req: EnumerateActionsRequest):
    actions = _derive_actions(req.state, req.side_to_move)
    nonce = str(uuid.uuid4())
    _universes[nonce] = {a.action_id: a for a in actions}
    picks = [a for a in actions if a.kind == "AirStrike"][:2] + [a for a in actions if a.kind == "Recon"][:1]
    plan = {"objective": "Disrupt high-value facilities while gaining intel",
            "steps": [{"action_id": a.action_id, "rationale": a.description} for a in picks]}
    return {"nonce": nonce, "plan": plan}

@app.post("/state/canonicalize")
def state_canonicalize(payload: Dict[str, Any]):
    ctx = ctx_from_state(payload)
    return {
        "ok": True,
        "checksum": ctx["checksum"],
        "turn": ctx["turn_number"],
        "phase": ctx["phase"],
        "ready_counts": {k: len(v) for k, v in ctx["ready"].items()},
        "targets_alive": len(ctx["alive_targets"])
    }

