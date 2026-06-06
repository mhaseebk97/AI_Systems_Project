# app/api/server.py
import os, json, asyncio, logging, re
from typing import Dict, Any, List, Tuple
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import httpx

# ---- logging ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mygpt")

# ---- settings ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("DEFAULT_MODEL", "llama3:latest")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful offline assistant. Be concise, use Markdown. "
    "When writing code, use fenced code blocks."
)

# ---- style & grounding rules ----
CHATGPT_STYLE = """
You are MyGPT, a helpful, expert assistant.

Style:
- Start with a 1–2 sentence direct answer.
- Then add short bullet points or tight paragraphs.
- Use Markdown: **bold** for key terms, `code` inline, fenced code blocks when relevant.
- Be concise. No greetings, no meta-chatter, no rhetorical questions.

Grounding & citations:
- If CONTEXT is provided, base the answer on it and add inline citations like [1], [2] exactly where those facts appear.
- If a requested fact is not in CONTEXT, say briefly that it is not found in the documents (and answer from general knowledge only if the user didn’t ask specifically for doc-based facts).
- Do not invent facts or sources. Never cite standards/values not present in CONTEXT.

Language & length:
- Reply in the user’s language unless they request otherwise.
- Prefer ≤ 8 bullets/lines total unless the user asks for more detail.

Rules:
- Never invent or fabricate verses, rituals, or sources.
- Don’t mention “RAG”, “snippets”, or “context” explicitly in the reply.
"""

# ---- signage fallback rules ----
SIGNAGE_RULES = """
For construction safety signage, always use these standards (fallback if docs are vague):
- **Segnali di Divieto (Prohibition)**: Circular, red border, white background, black symbol with red diagonal.
- **Segnali di Avvertimento (Warning)**: Triangular, yellow background, black symbol.
- **Segnali di Prescrizione / Obbligo (Mandatory)**: Circular, blue background, white symbol.
- **Segnali di Salvataggio / Soccorso (Emergency/Rescue)**: Square or rectangular, green background, white symbol.
- **Segnali di Antincendio (Fire Safety)**: Square or rectangular, red background, white symbol.

Important:
- If you rely on these fallback standards, DO NOT add citations.
- Only cite [n] when the detail is explicitly present in the CONTEXT.
"""

# ---- strict examples mode ----
STRICT_EXAMPLES_MODE = """
When the user asks for examples of safety signs (e.g., 'esempio', 'esempi', 'cartello tipico', 'typical sign', 'example'),
follow these rules strictly:
- **Only** list examples that appear verbatim or clearly in the CONTEXT.
- After each example, add a citation [n] tied to the specific line where that example is shown.
- If the CONTEXT has no explicit examples, answer with: "Nei documenti forniti non sono presenti esempi espliciti di cartelli." and stop.
- Do **not** supply examples from general knowledge in this mode.
"""

EXAMPLE_QUERY_RX = re.compile(
    r"\b(esempio|esempi|cartello tipico|cartelli tipici|example|examples)\b", re.IGNORECASE
)

# ---- app ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# ---- basic pings ----
@app.get("/")
def root():
    return PlainTextResponse("OK")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "ollama": OLLAMA_URL}

@app.get("/chat_test")
async def chat_test():
    async def gen():
        for w in ["Streaming", " works", " from", " FastAPI", " ✅\n"]:
            yield w
            await asyncio.sleep(0.25)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)

# ---- memory & tools ----
from app.core.memory import add_msg, load_window
from app.core.tools import maybe_calc

# ---- helpers ----
_HEADING_RX = re.compile(
    r"^(#{1,6}\s*|[-–•]\s*)?(?P<h>[\wÀ-ÖØ-öø-ÿ0-9 .,/()%-]{3,80})(:)?$"
)

def _guess_heading(text: str) -> str | None:
    if not text:
        return None
    head = text[:200]
    for line in re.split(r"[\n\r]+", head):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") and len(line) <= 100:
            return line.lstrip("# ").strip(" :")
        m = _HEADING_RX.match(line)
        if m:
            cand = m.group("h").strip()
            if 3 <= len(cand) <= 80 and not cand.lower().startswith(
                ("collana prevenzione", "manuale", "immagine", "analisi dell", "esempi di domande")
            ):
                return cand
        break
    return None

def format_rag_blocks(top: List[Tuple[str,int,str,float]]) -> Tuple[str, List[Tuple[int,str,int]]]:
    if not top:
        return "", []
    blocks, labels = [], []
    for i, (path, chunk_ix, text, _score) in enumerate(top, start=1):
        base = f"{Path(path).name}#{chunk_ix}"
        heading = _guess_heading(text) or ""
        title = f"{base} — {heading}" if heading else base
        snippet = " ".join(text.split())
        blocks.append(f"[{i}] {title}: {snippet}")
        labels.append((i, path, chunk_ix))
    return "\n".join(blocks), labels

# ---- debug endpoint ----
@app.post("/rag_debug")
async def rag_debug(req: Request):
    body: Dict[str, Any] = await req.json()
    q = (body.get("query") or "").strip()
    if not q:
        return JSONResponse({"error": "query required"}, status_code=400)
    try:
        from app.core.rag import search as rag_search
        top = await rag_search(q, k=8)
        ctx_text, labels = format_rag_blocks(top)
        return {"query": q, "hits": len(labels), "context": ctx_text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---- chat endpoint ----
@app.post("/chat")
async def chat(req: Request):
    body: Dict[str, Any] = await req.json()
    session = body.get("session", "default")
    user_msg = (body.get("message") or "").strip()

    log.info("POST /chat session=%s len=%s", session, len(user_msg))
    if not user_msg:
        return JSONResponse({"error": "message required"}, status_code=400)

    # calculator fast-path
    calc = maybe_calc(user_msg)
    if calc:
        add_msg(session, "user", user_msg)
        add_msg(session, "assistant", calc)
        return StreamingResponse(iter([calc]), media_type="text/event-stream", headers=STREAM_HEADERS)

    add_msg(session, "user", user_msg)
    history = load_window(session, limit=10)

    # optional RAG
    ctx_text = ""
    try:
        from app.core.rag import search as rag_search
        top = await rag_search(user_msg, k=6)
        ctx_text, _labels = format_rag_blocks(top)
        if ctx_text:
            log.info("RAG: using %d context blocks", len(_labels))
        else:
            log.info("RAG: 0 blocks")
    except Exception as e:
        log.warning("RAG disabled: %s", e)

    # explicit warning if user asks about docs but none loaded
    lower_q = user_msg.lower()
    if not ctx_text and any(k in lower_q for k in ["my data","my doc","docs","document","provided","rag","file","files","documenti","allegati"]):
        msg = ("I don’t have any loaded documents to reference yet. "
               "Add files to the `docs/` folder and run the ingester, then ask again.")
        add_msg(session, "assistant", msg)
        return StreamingResponse(iter([msg]), media_type="text/event-stream", headers=STREAM_HEADERS)

    # Build system prompt
    base = SYSTEM_PROMPT + "\n" + CHATGPT_STYLE + "\n" + SIGNAGE_RULES
    if EXAMPLE_QUERY_RX.search(user_msg):
        base += "\n" + STRICT_EXAMPLES_MODE

    if ctx_text:
        base += (
            "\nCONTEXT (numbered sources):\n"
            f"{ctx_text}\n"
            "Instructions: Only use facts explicitly present in the CONTEXT when you add citations. "
            "If the user asks for a fact not contained in the CONTEXT, say briefly that it is not found "
            "in the provided documents. If you add information from general knowledge, give it without citations. "
            "If you rely on SIGNAGE_RULES for fallback (forms/colors), do not add citations to those lines."
        )
    system_msg = base

    messages = [{"role": "system", "content": system_msg}, *history, {"role": "user", "content": user_msg}]

    async def stream_ollama():
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.15,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "num_predict": 512
            }
        }
        log.info("→ Ollama %s /api/chat model=%s", OLLAMA_URL, MODEL)
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                    log.info("← Ollama %s", r.status_code)
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if data.get("done"):
                            break
        except Exception as e:
            log.exception("Ollama error: %s", e)
            yield f"\n[server error] {e}\n"

    async def wrap_and_store():
        buf = []
        async for chunk in stream_ollama():
            buf.append(chunk)
            yield chunk
        full = "".join(buf).strip()
        if full:
            add_msg(session, "assistant", full)
            log.info("saved assistant reply len=%s", len(full))

    return StreamingResponse(wrap_and_store(), media_type="text/event-stream", headers=STREAM_HEADERS)
