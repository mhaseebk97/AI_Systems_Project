# app/core/rag.py
import os, re, sqlite3, logging
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import httpx, faiss

log = logging.getLogger("rag")

# -------------------
# Paths / settings
# -------------------
DB = os.getenv("RAG_DB", "app/data/db/rag.db")
INDEX_PATH = os.getenv("RAG_INDEX", "app/data/indexes/rag.faiss")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

os.makedirs(os.path.dirname(DB), exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# -------------------
# SQLite setup
# -------------------
conn = sqlite3.connect(DB, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  doc_path TEXT NOT NULL,
  chunk_ix INTEGER NOT NULL,
  text TEXT NOT NULL,
  heading TEXT,
  tags TEXT
)""")

def _add_col(name: str):
    try:
        cur.execute(f"ALTER TABLE chunks ADD COLUMN {name} TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

_add_col("heading")
_add_col("tags")

HAS_FTS = True
try:
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, heading, tags, content='chunks', content_rowid='id');
    """)
    conn.commit()
except sqlite3.OperationalError:
    HAS_FTS = False
    log.warning("FTS not available, fallback to FAISS only")

# -------------------
# Utilities
# -------------------
_WS = re.compile(r"\s+")
def _norm_spaces(t: str) -> str:
    return _WS.sub(" ", (t or "")).strip()

def _looks_like_heading(line: str) -> bool:
    line = (line or "").strip()
    if not line or len(line) > 120:
        return False
    if line.startswith("#"): return True
    if line.endswith(":"): return True
    if re.fullmatch(r"[A-Z0-9 \-_/]{3,}", line): return True
    return False

def _attach_heading_to_chunk(raw_text: str, start_idx: int) -> Optional[str]:
    scan_from = max(0, start_idx - 500)
    window = raw_text[scan_from:start_idx]
    lines = [l.strip() for l in window.splitlines() if l.strip()]
    for line in reversed(lines[-6:]):
        if _looks_like_heading(line):
            if line.startswith("#"):
                line = line.lstrip("# ").strip(" :")
            return line
    return None

def _chunk_text_with_headings(text: str, chunk_size=900, overlap=150):
    chunks = []
    i, n = 0, len(text or "")
    while i < n:
        j = min(n, i + chunk_size)
        raw = text[i:j]
        ch = _norm_spaces(raw)
        if ch:
            heading = _attach_heading_to_chunk(text, i)
            chunks.append((ch, heading))
        i += max(1, chunk_size - overlap)
    return chunks

# ---- tag extraction ----
STOP = set("""a ad ai al alla alle agli all' con col dei del della delle dello degli di da dal dalla dalle dallo dagli in nel nella nelle nello negli
per tra fra su sul sulla sulle sullo sugli e ed o oppure che cui non il lo la le i gli un uno una l' d' n° art art. cap. sez. ecc
the of and or to for from on at as is are be by with into over under about this that those these it its their his her your our an
""".split())

CANON_MAP: Dict[str, str] = {
    "divieto": "segnale_divieto",
    "avvertimento": "segnale_avvertimento",
    "prescrizione": "segnale_prescrizione",
    "obbligo": "segnale_prescrizione",
    "salvataggio": "segnale_salvataggio",
    "soccorso": "segnale_salvataggio",
    "antincendio": "segnale_antincendio",
    "dpi": "dpi",
    "casco": "dpi_casco",
    "guanti": "dpi_guanti",
    "occhiali": "dpi_occhi",
    "respiratore": "dpi_respiratore",
    "maschera": "dpi_respiratore",
    "imbracatura": "dpi_imbracatura",
    "udito": "dpi_udito",
    "rumore": "rischio_rumore",
    "vibrazioni": "rischio_vibrazioni",
    "polveri": "rischio_polveri",
    "silice": "rischio_silice",
    "microclima": "microclima",
    "aerazione": "aerazione",
    "ventilazione": "aerazione",
    "illuminazione": "illuminazione",
    "lux": "illuminazione_lux",
    "uscita": "uscita_emergenza",
    "emergenza": "emergenza",
    "porta": "porte_portoni",
    "portone": "porte_portoni",
    "datore": "datore_lavoro",
    "preposto": "preposto",
    "rspp": "rspp",
    "medico": "medico_competente",
    "rls": "rls",
    "sotterranei": "locali_sotterranei",
    "semisotterranei": "locali_sotterranei",
}

TOKEN_RX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}")

def _extract_tags(text: str, heading: Optional[str]) -> str:
    raw = f"{heading or ''} {text or ''}".lower()
    toks = TOKEN_RX.findall(raw)
    tags: Set[str] = set()
    for t in toks:
        if t in STOP:
            continue
        if t in CANON_MAP:
            tags.add(CANON_MAP[t])
        elif len(t) <= 20 and not t.isdigit():
            tags.add(t)
    return ",".join(sorted(tags)[:20])

# -------------------
# Embeddings
# -------------------
async def _embed(texts: List[str]) -> np.ndarray:
    vecs = []
    async with httpx.AsyncClient(timeout=None) as client:
        for t in texts:
            r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                  json={"model": EMBED_MODEL, "prompt": t})
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
            vecs.append(np.array(emb, dtype=np.float32))
    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def _open_index(dim: int) -> faiss.Index:
    if Path(INDEX_PATH).exists():
        return faiss.read_index(INDEX_PATH)
    idx = faiss.IndexFlatIP(dim)
    faiss.write_index(idx, INDEX_PATH)
    return idx

# -------------------
# Ingest
# -------------------
async def ingest_doc(path: str, chunk_size=900, overlap=150) -> int:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    parts = _chunk_text_with_headings(raw, chunk_size=chunk_size, overlap=overlap)
    if not parts:
        return 0

    chunks = [c for c, _h in parts]
    heads  = [h for _c, h in parts]
    tagstr = [_extract_tags(c, h) for c, h in parts]

    X = await _embed(chunks)
    idx = _open_index(X.shape[1])
    idx.add(X)
    faiss.write_index(idx, INDEX_PATH)

    for ix, (ch, hd, tg) in enumerate(zip(chunks, heads, tagstr)):
        cur.execute(
            "INSERT INTO chunks(doc_path,chunk_ix,text,heading,tags) VALUES(?,?,?,?,?)",
            (path, ix, ch, hd, tg)
        )
        rowid = cur.lastrowid
        if HAS_FTS:
            cur.execute(
                "INSERT INTO chunks_fts(rowid,text,heading,tags) VALUES (?,?,?,?)",
                (rowid, ch, hd or "", tg or "")
            )
    conn.commit()
    log.info("[ingest] %s -> %s chunks", path, len(chunks))
    return len(chunks)

# -------------------
# Search
# -------------------
def _fts_search(query: str, limit: int = 20) -> List[Tuple[int, float]]:
    if not HAS_FTS:
        return []
    try:
        sql = """
        SELECT rowid, bm25(chunks_fts) AS score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """
        rows = cur.execute(sql, (query, limit)).fetchall()
        return [(int(rid), 1.0/(1.0+float(bm))) for rid,bm in rows]
    except sqlite3.OperationalError:
        rows = cur.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                           (query, limit)).fetchall()
        return [(int(r[0]), 0.5) for r in rows]

async def search(query: str, k=5) -> List[Tuple[str,int,str,float]]:
    q = await _embed([query])
    qv = q[0]
    idx = _open_index(q.shape[1])
    if idx.ntotal == 0:
        return []

    k_pre = min(idx.ntotal, max(k*4, k))
    D, I = idx.search(q, k_pre)
    faiss_ids = [rid for rid in I[0] if rid >= 0]

    fts_rows = _fts_search(query, limit=20)
    fts_ids = [rid for rid,_ in fts_rows]

    cand_ids, seen = [], set()
    for rid in faiss_ids + fts_ids:
        if rid not in seen:
            cand_ids.append(rid); seen.add(rid)

    if not cand_ids:
        log.info("[RAG search] %s -> no candidates", query)
        return []

    cands = []
    for rid in cand_ids:
        r = cur.execute("SELECT id,doc_path,chunk_ix,text,heading,tags FROM chunks WHERE id=?",(rid,)).fetchone()
        if not r: continue
        _id,path,ix,text,heading,tags = r
        prefix = ""
        if heading: prefix += f"{heading}: "
        if tags:    prefix += f"[{tags}] "
        cands.append((_id,path,ix,text,prefix+(text or "")))

    cand_texts = [c[4] for c in cands]
    C = await _embed(cand_texts)
    sims = (C @ qv)

    fts_dict = {rid:s for rid,s in fts_rows}
    scored = []
    for (rid,path,ix,text,_aug),cos in zip(cands,sims.tolist()):
        fts_s = fts_dict.get(rid,0.0)
        final = 0.8*float(cos)+0.2*float(fts_s)
        scored.append((path,ix,text,final))

    scored.sort(key=lambda x:x[3],reverse=True)

    out, seen_keys = [], set()
    for path,ix,text,s in scored:
        key = (path,(text[:80] if text else "")+(str(ix)))
        if key in seen_keys: continue
        seen_keys.add(key)
        out.append((path,ix,text,float(s)))
        if len(out) >= k: break

    log.info("[RAG search] query='%s' -> %d hits", query, len(out))
    return out
