# app/core/router.py
import os, re, time
from typing import Literal, Dict

FAST_MODEL   = os.getenv("FAST_MODEL", "phi3:mini")
MAIN_MODEL   = os.getenv("MAIN_MODEL", "qwen2.5:7b-instruct")
CODER_MODEL  = os.getenv("CODER_MODEL") or ""  # optional

SIMPLE_WORDS = int(os.getenv("ROUTER_SIMPLE_WORDS", "25"))
HYST_TURNS   = int(os.getenv("ROUTER_HYSTERESIS_TURNS", "2"))

Label = Literal["simple","code","reasoning"]
_CODE_HINTS = ["stack trace","traceback","exception","error","bug","function","class","def ",
               "compile","python","js","typescript","java","c++","c#","code","snippet","script"]

# in-memory hysteresis per session
_state: Dict[str, Dict[str, int]] = {}  # {session: {"model": str, "age": int}}

def classify(text: str) -> Label:
    t = (text or "").lower()
    if any(k in t for k in _CODE_HINTS):
        return "code"
    if len(t.split()) < SIMPLE_WORDS:
        return "simple"
    return "reasoning"

def pick_model(session: str, user_text: str) -> str:
    label = classify(user_text)
    # base choice
    if label == "code" and CODER_MODEL:
        choice = CODER_MODEL
    elif label == "simple":
        choice = FAST_MODEL or MAIN_MODEL
    else:
        choice = MAIN_MODEL

    # hysteresis: avoid ping-pong; stick with prior model for HYST_TURNS turns
    st = _state.get(session)
    if st and st.get("age", 0) < HYST_TURNS:
        # if the previous model is "good enough" for this label, keep it
        prev = st["model"]
        if label in ("simple","reasoning") and prev in (MAIN_MODEL, FAST_MODEL):
            _state[session]["age"] += 1
            return prev
        if label == "code" and CODER_MODEL and prev == CODER_MODEL:
            _state[session]["age"] += 1
            return prev

    # commit new choice
    _state[session] = {"model": choice, "age": 0}
    return choice

def note_turn_end(session: str):
    if session in _state:
        _state[session]["age"] = 0  # reset age after a completed answer
