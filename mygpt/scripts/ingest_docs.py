# scripts/ingest_docs.py
import asyncio, os, glob, sys
from pathlib import Path
from app.core.rag import ingest_doc

ALLOWED = (".txt", ".md", ".markdown", ".html", ".htm", ".py", ".json", ".csv")

async def main(paths):
    total = 0
    for p in paths:
        p = str(p)
        files = []
        if os.path.isdir(p):
            files = [f for f in glob.glob(os.path.join(p, "**/*.*"), recursive=True)
                     if Path(f).suffix.lower() in ALLOWED]
        else:
            if Path(p).suffix.lower() in ALLOWED:
                files = [p]
        for f in files:
            try:
                n = await ingest_doc(f)
                print(f"[ingested] {f} -> {n} chunks")
                total += n
            except Exception as e:
                print(f"[skip] {f}: {e}")
    print(f"done. total chunks: {total}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["./docs"]))
