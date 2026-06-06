# app/core/llm_engine.py
import json, httpx, os, logging
from typing import AsyncGenerator, List, Dict, Any

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
log = logging.getLogger("llm_engine")

async def chat_stream_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    num_ctx: int = 4096,
    num_predict: int = 512,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama /api/chat."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
        },
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if obj.get("done"):
                    break
