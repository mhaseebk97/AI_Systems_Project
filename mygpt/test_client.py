import httpx, asyncio

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            "http://127.0.0.1:11434/api/chat",
            json={"model":"llama3:latest","messages":[{"role":"user","content":"hi"}]}
        )
        print(r.status_code, r.text)

asyncio.run(main())
