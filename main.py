import hashlib, json, uuid, importlib, copy, os
from typing import Any, Dict, List, Optional
from google.cloud import firestore
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from features import load_action_map, save_action_map
import time
from game_engine import GameEngine
from mcts import MCTSAgent
from fastapi import Query
EPISODES: Dict[str, List[dict]] = {}
_UNIVERSES: Dict[str, Dict[str, Any]] = {}


# ---------- Firestore client (optional) ----------
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")

firestore_client = None
if FIREBASE_PROJECT_ID and FIREBASE_CREDENTIALS_JSON:
    try:
        creds_info = json.loads(FIREBASE_CREDENTIALS_JSON)
        creds = service_account.Credentials.from_service_account_info(creds_info)
        firestore_client = firestore.Client(
            project=FIREBASE_PROJECT_ID,
            credentials=creds,
        )
        print("✅ Firestore client initialized")
    except Exception as e:
        print(f"⚠ Firestore init failed: {e}")
        firestore_client = None
else:
    print("ℹ Firestore env vars not set; logs only kept in memory.")

def log_debug_input(game_id: str, side: str, state: dict):
    """Step B: Logs RAW input immediately (dict + pretty JSON text)."""
    if firestore_client is None:
        return
    try:
        timestamp = str(int(time.time() * 1000))
        payload = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "side_requested": side,
            
            "state": state,
           
            "state_json": json.dumps(state, ensure_ascii=False, indent=2),
        }
        (
            firestore_client.collection("debug_logs")
            .document(game_id)
            .collection("inputs")
            .document(timestamp)
            .set(payload)
        )
    except Exception as e:
        print(f"⚠ Debug Input Log Failed: {e}")


def log_debug_output(
    game_id: str,
    action: dict,
    gpt_context: dict,
    state_before: dict | None = None,
    state_after: dict | None = None,
    error: str | None = None,
):
    """Logs output (action + states) and errors into Firestore."""
    if firestore_client is None:
        return
    try:
        timestamp = str(int(time.time() * 1000))
        payload: Dict[str, Any] = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "action": action or {},
            "gpt_context": gpt_context or {},
        }

        if state_before is not None:
            payload["state_before"] = state_before
            payload["state_before_json"] = json.dumps(
                state_before, ensure_ascii=False, indent=2
            )
        if state_after is not None:
            payload["state_after"] = state_after
            payload["state_after_json"] = json.dumps(
                state_after, ensure_ascii=False, indent=2
            )

        if error:
            payload["error"] = error

        (
            firestore_client.collection("debug_logs")
            .document(game_id)
            .collection("outputs")
            .document(timestamp)
            .set(payload)
        )
    except Exception as e:
        print(f"⚠ Debug Output Log Failed: {e}")

# -------------------------------------

def log_transition(game_id, state, side, action, reward, done, info, policy=None):
    """
    Save one (s, a, r, done, info, policy) transition.

    - Always push into in-memory EPISODES[game_id]
    - If Firestore is configured, also write to:
      episodes/{game_id}/steps/{index}
    """
    if game_id not in EPISODES:
        EPISODES[game_id] = []

    record = {
        "index": len(EPISODES[game_id]),
        "side": side,
        "state": state,
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "info": info or {},
        "policy": policy or {},
    }

    # 1) 
    EPISODES[game_id].append(record)

    # 2) 
    if firestore_client is not None:
        try:
            doc_ref = (
                firestore_client
                .collection("episodes")
                .document(game_id)
                .collection("steps")
                .document(str(record["index"]))
            )
            doc_ref.set(record)
        except Exception as e:
            print(f"⚠ Firestore log_transition failed for {game_id}: {e}")


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


ENGINE = GameEngine()

AGENTS: Dict[str, MCTSAgent] = {
    "israel": MCTSAgent(ENGINE, "israel", simulations=1000, strict=True, reuse_tree=True),
    "iran":   MCTSAgent(ENGINE, "iran",   simulations=1000, strict=True, reuse_tree=True),
}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("Loading AI Memory...")
    load_action_map()  # Restores the meaning of actions
    
    yield  # App runs here
    
    # --- Shutdown ---
    print("Saving AI Memory...")
    save_action_map()  # Saves new actions learned
    if AGENTS.get("israel").pv_model:
        AGENTS["israel"].pv_model.save("israel_model.pth")
    if AGENTS.get("iran").pv_model:
        AGENTS["iran"].pv_model.save("iran_model.pth")
# ---------- App ----------
app = FastAPI(
    title="Persian Incursion Strategy API",
    version="0.3.0",
    description="Authoritative rules + action enumerator to bound MyGPT",
    lifespan=lifespan,
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

from typing import Any, Dict  

_SIDE_NORMALIZE = {
    "I": "israel", "i": "israel", "Israel": "israel",
    "R": "iran",   "r": "iran",   "Iran": "iran",
    "israel": "israel", "iran": "iran",
}

def _normalize_turn_and_resources(state: Dict[str, Any]) -> Dict[str, Any]:
    # ---- TURN ----
    t = state.get("turn") or {}
    if not isinstance(t, dict):
        t = {}

    # Legacy (compact) vs engine fields
    legacy_num  = t.get("number")
    legacy_side = t.get("side")
    legacy_seg  = t.get("segment")

    eng_num   = t.get("turn_number")
    eng_side  = t.get("current_player")
    eng_phase = t.get("phase")

    # Number
    if legacy_num is not None:
        number = legacy_num
    elif eng_num is not None:
        number = eng_num
    else:
        number = 1

    # Side: accept anything that *looks like* Israel / Iran
    side_raw = legacy_side if legacy_side is not None else (eng_side if eng_side is not None else "Israel")
    s_norm = str(side_raw).strip().lower()
    if s_norm.startswith("i"):
        side_compact = "Israel"
        side_engine = "israel"
    else:
        side_compact = "Iran"
        side_engine = "iran"

    # Segment / phase
    seg_raw = legacy_seg if legacy_seg is not None else (eng_phase if eng_phase is not None else "morning")
    seg = str(seg_raw).strip().lower()
    if seg.startswith("m"):
        segment = "Morning"
    elif seg.startswith("a"):
        segment = "Afternoon"
    elif seg.startswith("n"):
        segment = "Night"
    else:
        segment = "Morning"

    phase = segment.lower()  # engine-style

    norm_turn = {
        "number": int(number),
        "side": side_compact,    # external (“Israel” / “Iran”)
        "segment": segment,      # external (“Morning”…)
        "phase": phase,          # internal (“morning”…)
        "engine_side": side_engine,
    }

    # ---- RESOURCES ----
    resources: Dict[str, Dict[str, float]] = {}

    # 1) Try r-block or generic resources-block
    r_block = state.get("r") or state.get("resources") or {}

    def canonical_side_name(k: str) -> Optional[str]:
        lk = k.lower()
        if lk.startswith("is"):   # israel / Israel / ISRAEL / etc.
            return "israel"
        if lk.startswith("ir"):   # iran / Iran / IRAN / etc.
            return "iran"
        return None

    # Collect from r-block, no hard-coded “Israel” keys
    if isinstance(r_block, dict):
        for k, v in r_block.items():
            canon = canonical_side_name(str(k))
            if not canon:
                continue
            if not isinstance(v, dict):
                continue
            pp = v.get("pp", v.get("PP", 0.0))
            ip = v.get("ip", v.get("IP", 0.0))
            mp = v.get("mp", v.get("MP", 0.0))
            resources[canon] = {
                "pp": float(pp),
                "ip": float(ip),
                "mp": float(mp),
            }

    # 2) Fill or override from players[*].resources if present
    players = state.get("players")
    if isinstance(players, dict):
        for k, pdata in players.items():
            canon = canonical_side_name(str(k))
            if not canon:
                continue
            res = (pdata or {}).get("resources") or {}
            if isinstance(res, dict):
                resources[canon] = {
                    "pp": float(res.get("pp", 0.0)),
                    "ip": float(res.get("ip", 0.0)),
                    "mp": float(res.get("mp", 0.0)),
                }

    # 3) Ensure both sides exist
    for s in ("israel", "iran"):
        resources.setdefault(s, {"pp": 0.0, "ip": 0.0, "mp": 0.0})

    return {"turn": norm_turn, "resources": resources}


def _ensure_players_block(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure state has the engine-style players block,
    using normalized turn + resources from compact or engine formats.
    """
    norm = _normalize_turn_and_resources(state)

    # Keep original turn structure for compact format, but also
    # make sure engine can read it.
    state["turn"] = state.get("turn", {})
    if not isinstance(state["turn"], dict):
        state["turn"] = {}

    # Compact fields (for your 11.json logs)
    state["turn"]["number"] = norm["turn"]["number"]
    state["turn"]["side"] = norm["turn"]["side"]
    state["turn"]["segment"] = norm["turn"]["segment"]

    # Engine fields (for GameEngine)
    state["turn"].setdefault("turn_number", norm["turn"]["number"])
    # current_player must be lowercase for engine
    cur_side = norm["turn"]["side"]
    cur_side_engine = "israel" if str(cur_side).lower().startswith("i") else "iran"
    state["turn"].setdefault("current_player", cur_side_engine)
    # simple mapping of segment -> phase
    seg = norm["turn"]["segment"].lower()
    if seg.startswith("m"):
        phase = "morning"
    elif seg.startswith("a"):
        phase = "afternoon"
    elif seg.startswith("n"):
        phase = "night"
    else:
        phase = "morning"
    state["turn"].setdefault("phase", phase)

    # Resources
    resources = norm["resources"]
    state["r"] = resources  # keep compact r-block always in sync

    players = state.setdefault("players", {})
    if not isinstance(players, dict):
        players = {}
        state["players"] = players

    for side in ("israel", "iran"):
        p = players.setdefault(side, {})
        res = p.setdefault("resources", {})
        base = resources.get(side, {"pp": 0.0, "ip": 0.0, "mp": 0.0})
        res["pp"] = float(base.get("pp", 0.0))
        res["ip"] = float(base.get("ip", 0.0))
        res["mp"] = float(base.get("mp", 0.0))

        p.setdefault("river", [])
        p.setdefault("deck", [])
        p.setdefault("discard", [])

    return state





# ---------- Helpers ----------

def _checksum(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _alive_targets(state_dict: Dict[str, Any]) -> List[str]:
    ti = state_dict.get("ti") or {}
    return [name for name, meta in ti.items() if not meta.get("destroyed", False)]

def _day_phase(state_dict: Dict[str, Any]) -> bool:
    ph = (state_dict.get("turn") or {}).get("phase", "")
    return str(ph).lower() in ("morning", "afternoon")

def ctx_from_state(state: Dict[str, Any], side: Optional[str] = None) -> Dict[str, Any]:
    norm = _normalize_turn_and_resources(state)
    turn = norm["turn"]
    resources = norm["resources"]

    turn_number = int(turn["number"])
    phase = str(turn.get("phase", turn.get("segment", "morning"))).lower()

    # Ready squadrons from 'as' block (AirSide)
    ready = {"israel": [], "iran": []}
    as_block = state.get("as") or state.get("as_", {})
    if isinstance(as_block, dict):
        for side_key, field in (("israel", "israel_squadrons"), ("iran", "iran_squadrons")):
            sq_list = as_block.get(field, [])
            if isinstance(sq_list, list):
                for sq in sq_list:
                    if not isinstance(sq, dict):
                        continue
                    st = str(sq.get("st", "")).lower()
                    if st.startswith("r"):  # e.g. "Ready"
                        sq_id = sq.get("id")
                        if sq_id:
                            ready[side_key].append(sq_id)

    # Alive targets from ti
    alive_targets = _alive_targets(state)

    # Simple flags (can grow later)
    flags = {
        "israel_overt_attack_done": bool(
            (state.get("flags") or {}).get("israel_overt_attack_done", False)
        )
    }

    ctx = {
        "checksum": _checksum(state),
        "turn_number": turn_number,
        "phase": phase,
        "resources": resources,
        "ready": ready,
        "alive_targets": alive_targets,
        "flags": flags,
    }

    # Optionally expose the logical side, but no one indexes it now
    if side:
        ctx["side"] = side
    else:
        ctx["side"] = state.get("side", "Israel")

    return ctx

    
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
class AllowExtraModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    
class Squadron(BaseModel):
    id: str
    ty: str
    st: str
    cu: Optional[int] = None
    b: Optional[str] = None
    mission_status: Optional[str] = None

class AirSide(AllowExtraModel):
    israel_squadrons: List[Squadron] = Field(default_factory=list)
    iran_squadrons: List[Squadron] = Field(default_factory=list)

class GameState(AllowExtraModel):
    t: Optional[Any] = None
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
    turn: Optional[Any] = None
    resources: Optional[dict] = None  # ensure present for ctx checks

class EnumerateActionsRequest(AllowExtraModel):
    state: GameState
    side_to_move: Optional[str] = "Israel"
    max_actions: Optional[int] = 60

class EnumeratedAction(AllowExtraModel):
    action_id: str
    kind: str
    description: str
    preconditions: List[str] = Field(default_factory=list)
    cost: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    timing: Dict[str, Any] = Field(default_factory=dict)
    engine_payload: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

class Plan(AllowExtraModel):
    objective: Optional[str] = None
    steps: List["PlanStep"] = Field(default_factory=list)
    state: Dict[str, Any]


class PlanStep(AllowExtraModel):
    action_id: str
    rationale: Optional[str] = None

class NoncedPlan(AllowExtraModel):
    nonce: str
    plan: Plan

class HumanMoveRequest(BaseModel):
    game_id: str
    side: str              # "israel" or "iran"
    state: Dict[str, Any]
    action: Dict[str, Any]


@app.post("/human_move")
def human_move(req: HumanMoveRequest):
    if not ge:
        raise HTTPException(500, "Engine not available.")
    eng = ge.GameEngine()

    next_state, reward, done, info = eng.rl_step(
        copy.deepcopy(req.state),
        req.action,
        side=req.side,
    )

    log_transition(
        req.game_id,
        req.state,
        req.side,
        req.action,
        reward,
        done,
        info,
        policy=None,
    )

    return {
        "new_state": next_state,
        "reward": reward,
        "done": done,
        "info": info,
    }

def _project_engine_state_back_to_compact(
    base_state: Dict[str, Any],
    full_state: Dict[str, Any],
) -> Dict[str, Any]:
    compact = copy.deepcopy(base_state)

    # ---- TURN ---- (same as before, just updating values)
    base_turn = compact.get("turn") or {}
    eng_turn = (full_state.get("turn") or {}).copy()
    number = eng_turn.get("turn_number", base_turn.get("number", 1))
    current_player = eng_turn.get("current_player", base_turn.get("side", "Israel"))
    cp_norm = str(current_player).lower()
    side = "Israel" if cp_norm.startswith("i") else "Iran"

    phase = eng_turn.get("phase", base_turn.get("segment", "Morning"))
    ph = str(phase).lower()
    if ph.startswith("m"):
        segment = "Morning"
    elif ph.startswith("a"):
        segment = "Afternoon"
    elif ph.startswith("n"):
        segment = "Night"
    else:
        segment = base_turn.get("segment", "Morning")

    compact["turn"] = {
        "number": int(number),
        "side": side,
        "segment": segment,
    }

    # ---- VICTORY FLAGS ----
    if isinstance(full_state.get("victory_flags"), dict):
        compact["victory_flags"] = copy.deepcopy(full_state["victory_flags"])

    # ---- PLAYERS ----
    players = full_state.get("players") or {}
    compact["players"] = copy.deepcopy(players)

    # ---- RESOURCES r-block (mirror original shape if any) ----
    r_base = compact.get("r")
    r_new: Dict[str, Dict[str, float]] = {}

    def from_engine(side_name: str) -> Dict[str, float]:
        p = players.get(side_name) or {}
        res = p.get("resources") or {}
        return {
            "pp": float(res.get("pp", 0.0)),
            "ip": float(res.get("ip", 0.0)),
            "mp": float(res.get("mp", 0.0)),
        }

    if isinstance(r_base, dict) and r_base:
        # preserve whatever keys the user used originally
        for k in r_base.keys():
            lk = str(k).lower()
            if lk.startswith("is"):
                r_new[k] = from_engine("israel")
            elif lk.startswith("ir"):
                r_new[k] = from_engine("iran")
            else:
                # unknown key – just pass through unchanged
                r_new[k] = r_base[k]
    else:
        # no existing r-block → create a sane default
        r_new = {
            "israel": from_engine("israel"),
            "iran": from_engine("iran"),
        }

    compact["r"] = r_new

    # ---- LOG ----
    if isinstance(full_state.get("log"), list):
        compact["log"] = list(full_state["log"])

    return compact

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
        s = payload.model_dump(by_alias=True)
        ge.apply_morning_opinion_income(s, carry_cap=None, log_fn=None)
        note = "Opinion income computed (not persisted)"
    return {"ok": True, "warnings": warn, "note": note}

@app.post("/state/canonicalize")
def state_canonicalize(payload: GameState):
    sdict = payload.model_dump(by_alias=True)
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
    sdict = payload.model_dump(by_alias=True)
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
    
    # --- FIX: Sanitize state for engine ---
    # The engine expects dicts, but Pydantic provides None for missing optional fields.
    state_for_engine = sdict.copy()
    
    # 1. Fix 'turn'
    if state_for_engine.get("turn") is None:
        state_for_engine["turn"] = state_for_engine.get("t") or {}

    # 2. Fix 'opinion' (This is what caused your specific error)
    if state_for_engine.get("opinion") is None:
        state_for_engine["opinion"] = {}

    # 3. Fix 'ti' (Target Intelligence) - Good practice to prevent next crash
    if state_for_engine.get("ti") is None:
        state_for_engine["ti"] = {}
    # --------------------------------------

    eng = ge.GameEngine()
    return eng.get_legal_actions(state_for_engine, side=side)

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
            engine_payload=copy.deepcopy(a),
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

class RLSimpleRequest(BaseModel):
    state: Dict[str, Any]
    side: str

@app.post("/rl/legal_actions")
def rl_legal_actions(req: RLSimpleRequest):
    if not ge:
        raise HTTPException(500, "Engine not available.")
    eng = ge.GameEngine()
    acts = eng.get_legal_actions(copy.deepcopy(req.state), side=req.side)
    return {"actions": acts}
class RLStepRequest(BaseModel):
    state: Dict[str, Any]
    side: str
    action: Dict[str, Any]

@app.post("/rl/step")
def rl_step(req: RLStepRequest):
    if not ge:
        raise HTTPException(500, "Engine not available.")
    eng = ge.GameEngine()
    next_state, reward, done, info = eng.rl_step(
        copy.deepcopy(req.state),
        req.action,
        side=req.side,
    )
    return {
        "next_state": next_state,
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.post("/actions/enumerate")
def actions_enumerate(req: EnumerateActionsRequest):
    sdict = req.state.model_dump(by_alias=True)
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
        "actions": [a.model_dump() for a in actions],
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

# main.py
@app.post("/plan/suggest", include_in_schema=False)
def plan_suggest(req: EnumerateActionsRequest):
    if not req.state:
        raise HTTPException(status_code=422, detail="Missing 'state' object in request body.")
    
    sdict = req.state.model_dump(by_alias=True)

    # 2. Normalize
    norm = _normalize_turn_and_resources(sdict)
    norm_turn = norm["turn"]
    norm_res = norm["resources"]

    # 3. Apply normalized values back to sdict (engine-safe)
    sdict["turn"] = {
        "turn_number": norm_turn["number"],
        "current_player": norm_turn["engine_side"],  # "israel"/"iran"
        "phase": norm_turn["phase"],                 # "morning"/"afternoon"/"night"
    }

    sdict["players"] = sdict.get("players") or {}
    for side in ("israel", "iran"):
        sdict["players"].setdefault(side, {})["resources"] = norm_res.get(
            side, {"mp": 0.0, "ip": 0.0, "pp": 0.0}
        )

    # 4. derive actions
    actions = _derive_actions(sdict, req.side_to_move)

    code = _side_code(req.side_to_move)           
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

@app.post("/plan/execute", include_in_schema=False)
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
    
    # FIX: Check class method existence, then instantiate
    if ge and hasattr(ge.GameEngine, "apply_actions"):
        eng = ge.GameEngine()
        
        # Expect engine to accept a state and a list of abstract actions
        # Convert action_ids back to the engine action dicts we stored
        engine_actions = []
        for st in req.plan.steps:
            a = uni.get(st.action_id)
            if not a:
                continue
            payload = copy.deepcopy(getattr(a, "engine_payload", None) or {})
            payload.setdefault("type", a.kind)

            # Fill target hint if engine payload omitted it
            if payload.get("target") is None:
                if "→" in a.description:
                    payload["target"] = a.description.split("→", 1)[-1].strip()

            cost_map = {}
            for ck, cv in (a.cost or {}).items():
                if cv is None:
                    continue
                try:
                    cost_map[str(ck).lower()] = float(cv)
                except (TypeError, ValueError):
                    continue

            t = payload.get("type", "")
            if t in ("Order Special Warfare", "Order Terror Attack"):
                if "mp_cost" not in payload and "mp" in cost_map:
                    payload["mp_cost"] = int(round(cost_map["mp"]))
                if "ip_cost" not in payload and "ip" in cost_map:
                    payload["ip_cost"] = int(round(cost_map["ip"]))
            if t == "Order Ballistic Missile":
                payload.setdefault("battalions", 1)
                payload.setdefault("missile_type", "Shahab")
            if t == "Order Airstrike":
                payload.setdefault("corridor", "central")
                if not payload.get("squadrons"):
                    ready_isr = ctx["ready"].get("israel", [])
                    payload["squadrons"] = [{"id": sq} for sq in ready_isr[:2]]
                payload.setdefault("loadout", {})

            engine_actions.append(payload)
        try:
            # FIX: Call instance method on 'eng'
            s1, engine_log = eng.apply_actions(s0, engine_actions, side=side)
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

@app.get("/episodes/{game_id}")
def get_episode_logs(game_id: str, limit: int = Query(50, ge=1, le=500)):
   
    # 1) Firestore 
    if firestore_client is not None:
        try:
            steps_ref = (
                firestore_client
                .collection("episodes")
                .document(game_id)
                .collection("steps")
                .order_by("index")
            )
            docs = list(steps_ref.stream())
            steps = [d.to_dict() for d in docs]
            if limit and len(steps) > limit:
                steps = steps[-limit:]
            return {"game_id": game_id, "steps": steps}
        except Exception as e:
            print(f"⚠ Firestore get_episode_logs failed for {game_id}: {e}")
            #  fallback

    # 2)  fallback
    steps = EPISODES.get(game_id, [])
    if limit and len(steps) > limit:
        steps = steps[-limit:]
    return {"game_id": game_id, "steps": steps}

def run_ai_move_core(game_id: str, side: Optional[str], state: Dict[str, Any]):
    if not ge:
        raise HTTPException(500, "Engine not available.")


        if not isinstance(d, dict):
            return False
        # Compact / engine formats usually have at least one of these:
        for key in ("turn", "t", "as", "r", "u", "bm", "swm", "ti"):
            if key in d:
                return True
        return False

    while (
        isinstance(state, dict)
        and "state" in state
        and isinstance(state["state"], dict)
        and not looks_like_game_state(state)
    ):
        # Peel one wrapper layer
        state = state["state"]

    # 0) Preserve the original (possibly still compact) state
    base_state = copy.deepcopy(state)

    # 1) Internal working state: add players/resources, etc.
    work_state = copy.deepcopy(base_state)
    work_state = _ensure_players_block(work_state))
    # 2) Detect side
    target_side = side
    if not target_side:
        target_side = work_state.get("turn", {}).get("current_player", "israel").lower()
    target_side = str(target_side).lower()
    if target_side.startswith("i"):
        if "s" in target_side[0:2]:
            target_side = "israel"
        elif "r" in target_side[0:2]:
            target_side = "iran"

    log_debug_input(game_id, target_side, work_state)

    agent = AGENTS.get(target_side)
    if not agent:
        raise HTTPException(400, f"Invalid side detected: {target_side}")

    try:
        best_action, policy = agent.choose_action(copy.deepcopy(work_state))

        eng = ge.GameEngine()
        # Apply action on a full copy of the working state
        full_state_after = eng.apply_action(copy.deepcopy(work_state), best_action, side=target_side)

        reward = 0.0
        done = bool(eng.is_game_over(full_state_after)) if hasattr(eng, "is_game_over") else False
        info = {}

    except Exception as e:
        error_msg = f"AI Crash detected: {str(e)}"
        print(error_msg)
        log_debug_output(game_id, {}, {}, error=error_msg)
        raise HTTPException(500, detail=error_msg)

    # 3) Project back to your compact 11.json-style state
    next_state_compact = _project_engine_state_back_to_compact(base_state, full_state_after)

    # 4) RL-style logging using the full engine state
    log_transition(
        game_id,
        work_state,
        target_side,
        best_action,
        reward,
        done,
        info,
        policy,
    )

    # 5) Build GPT context from full_state_after (for narrative)
    threats = []
    dmg_map = full_state_after.get("target_damage_status", {})
    for target_name, damage_data in dmg_map.items():
        if isinstance(damage_data, dict):
            if any(
                v.get("damage_boxes_hit", 0) > 0
                for v in damage_data.values()
                if isinstance(v, dict)
            ):
                threats.append(target_name)

    role_title = "IAF Commander" if target_side == "israel" else "IRGC Commander"

    gpt_context = {
        "active_side": target_side,
        "narrative_role": role_title,
        "move_description": f"Executing {best_action.get('type')} against {best_action.get('target', 'unknown target')}.",
        "strategic_reasoning": "MCTS simulations indicate this move maximizes long-term strategic advantage.",
        "critical_alerts": threats[:3],
        "game_over": done,
    }

    log_debug_output(
        game_id,
        best_action,
        gpt_context,
        state_before=work_state,
        state_after=full_state_after,
    )

    # IMPORTANT: return the compact state, not the full engine blob
    return best_action, next_state_compact, gpt_context, done


class AIStateOnlyResponse(BaseModel):
    action: Dict[str, Any]
    state: Dict[str, Any]      
    gpt_context: Dict[str, Any]
    done: bool


@app.post(
    "/ai_move",
    response_model=AIStateOnlyResponse,
    summary="RL/MCTS move with raw game-state JSON",
    description=(
        "Request body = full Persian Incursion game state JSON (same format as 11.json). "
        "Response returns the chosen action and the UPDATED game state JSON under 'state', "
        "plus explanation context for MyGPT."
    ),
)
def ai_move(
    state: Dict[str, Any] = Body(..., embed=False),
    game_id: str = Query("default_game"),
    side: Optional[str] = Query(None),
):
    best_action, next_state, gpt_context, done = run_ai_move_core(
        game_id, side, state
    )
    return {
        "action": best_action,
        "state": next_state,
        "gpt_context": gpt_context,
        "done": done,
    }

def _merge_engine_state_into_base(
    base_state: Dict[str, Any],
    engine_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a new state that:
    - preserves the *shape* of base_state
    - but updates key dynamic fields from engine_state
    Works for many schema variants (11.json compact, verbose, future versions).
    """
    out = copy.deepcopy(base_state)

    # 1) Always-safe global fields – just overwrite
    for key in (
        "victory_flags",
        "target_damage_status",
        "losses",
        "opinion",
        "active_events_queue",
        "log",
    ):
        if key in engine_state:
            out[key] = copy.deepcopy(engine_state[key])

    # 2) Players/resources copied back (engine always has canonical players[*].resources)
    if "players" in engine_state:
        # preserve any extra per-player fields from base_state
        base_players = out.get("players", {})
        eng_players = engine_state["players"]
        merged_players = copy.deepcopy(base_players)

        for side, pdata in eng_players.items():
            bp = merged_players.setdefault(side, {})
            # overwrite resources from engine, keep other keys
            if "resources" in pdata:
                bp["resources"] = copy.deepcopy(pdata["resources"])
        out["players"] = merged_players

    # 3) If base used compact "r" resources, rebuild it from engine players
    if "r" in out and "players" in engine_state:
        r = out["r"]
        for side_key, side_name in (("Israel", "israel"), ("Iran", "iran"),
                                    ("israel", "israel"), ("iran", "iran")):
            if side_key in r and side_name in engine_state["players"]:
                eng_res = engine_state["players"][side_name].get("resources", {})
                r[side_key]["pp"] = float(eng_res.get("pp", 0.0))
                r[side_key]["ip"] = float(eng_res.get("ip", 0.0))
                r[side_key]["mp"] = float(eng_res.get("mp", 0.0))

    # 4) Turn mapping – update values but keep original style
    if "turn" in engine_state and "turn" in out:
        eng_t = engine_state["turn"]
        base_t = out["turn"]

        if isinstance(base_t, dict) and isinstance(eng_t, dict):
            # number
            num = eng_t.get("turn_number") or eng_t.get("n") or eng_t.get("number")
            if num is not None:
                for k in ("turn_number", "number", "n"):
                    if k in base_t:
                        base_t[k] = num
            # current_player / side
            cp = eng_t.get("current_player") or eng_t.get("side")
            if cp is not None:
                if "current_player" in base_t:
                    base_t["current_player"] = cp
                if "side" in base_t:
                    # keep capitalisation if base used "Israel"/"Iran"
                    base_t["side"] = cp.capitalize() if isinstance(base_t["side"], str) else cp
            # phase / segment
            phase = eng_t.get("phase")
            if phase is not None:
                if "phase" in base_t:
                    base_t["phase"] = phase
                if "segment" in base_t:
                    base_t["segment"] = phase.capitalize()
            out["turn"] = base_t
        else:
            # if base used a simple int turn, keep it or bump if engine advanced
            eng_num = eng_t.get("turn_number") if isinstance(eng_t, dict) else None
            if isinstance(base_t, int) and isinstance(eng_num, int) and eng_num > base_t:
                out["turn"] = eng_num

    return out



