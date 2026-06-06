# app/core/memory.py
import sqlite3, time, os
DB_PATH = os.getenv("CHAT_DB", "app/data/db/chat.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY,
  session TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  ts REAL NOT NULL
)""")
conn.commit()

def add_msg(session: str, role: str, content: str):
    cur.execute("INSERT INTO messages(session,role,content,ts) VALUES(?,?,?,?)",
                (session, role, content, time.time()))
    conn.commit()

def load_window(session: str, limit: int = 12):
    rows = cur.execute(
        "SELECT role, content FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
        (session, limit)
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]
