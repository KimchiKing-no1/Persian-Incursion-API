from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid, importlib

# ---- bring in your rules & (optionally) engine helpers ----
rules_blob = importlib.import_module("rules_blob")   # your file
try:
    ge = importlib.import_module("game_engine")      # optional use
except Exception:
    ge = None

app = FastAPI(
    title="Persian Incursion Strategy API",
    version="0.2.0",
    description="Authoritative rules + action enumerator to bound MyGPT"
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

def _derive_actions(state: GameState, side_to_move: str) -> List[EnumeratedAction]:
    acts: List[EnumeratedAction] = []
    if side_to_move.lower() == "israel":
        ready = [s for s in state.as_.israel_squadrons if _ready(s)]
        enemy_targets = list((state.ti or {}).keys())
        # Naive examples (replace with your real legality):
        for s in ready[:12]:
            # Strike the first few known targets
            for tgt in enemy_targets[:3]:
                aid = str(uuid.uuid4())
                acts.append(EnumeratedAction(
                    action_id=aid,
                    kind="AirStrike",
                    description=f"{s.id} ({s.ty}) strike against {tgt}",
                    preconditions=["Aircraft ready", "Payload configured", "Range/SAM corridor satisfied"],
                    cost={"sorties": 1, "fuel": 1},
                    tags=["offense"]
                ))
            # Recon option
            rid = str(uuid.uuid4())
            acts.append(EnumeratedAction(
                action_id=rid,
                kind="Recon",
                description=f"{s.id} ({s.ty}) sector recon near {s.b or 'base'}",
                preconditions=["Weather OK"],
                cost={"sorties": 0.5},
                tags=["intel","low-risk"]
            ))
    else:
        ready = [s for s in state.as_.iran_squadrons if _ready(s)]
        for s in ready[:12]:
            rid = str(uuid.uuid4())
            acts.append(EnumeratedAction(
                action_id=rid,
                kind="InterceptCAP",
                description=f"{s.id} ({s.ty}) establish CAP from {s.b or 'base'}",
                preconditions=["Fuel window", "GCI available if using"],
                cost={"sorties": 0.7},
                tags=["defense","air-superiority"]
            ))
    return acts

@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    actions = _derive_actions(req.state, req.side_to_move)
    actions = actions[: (req.max_actions or 60)]
    nonce = str(uuid.uuid4())
    _universes[nonce] = {a.action_id: a for a in actions}
    return {
        "nonce": nonce,
        "side_to_move": req.side_to_move,
        "ruleset_version": (req.state.meta or {}).get("ruleset_version", "0"),
        "actions": [a.dict() for a in actions],
        "hints": {
            "cards_known_examples": {
                "iran_cards": getattr(rules_blob, "IRAN_CARDS", [])[:2],
                "israel_cards": getattr(rules_blob, "ISRAEL_CARDS", [])[:2]
            }
        }
    }

@app.post("/plan/validate")
def plan_validate(req: NoncedPlan):
    uni = _universes.get(req.nonce)
    if not uni:
        raise HTTPException(400, "Unknown/expired nonce. Re-enumerate.")
    illegal = []
    for i, step in enumerate(req.plan.steps):
        if step.action_id not in uni:
            illegal.append({"index": i, "action_id": step.action_id, "reason": "Not in enumerated legal set"})
    return {"ok": len(illegal) == 0, "illegal_steps": illegal}

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
