"""Start configs/api.yaml first; requires OFLOW_ALPHA_API_KEY and OFLOW_BETA_API_KEY."""

import asyncio
import os

import httpx


async def main():
    async with httpx.AsyncClient(
        base_url=os.environ.get("OFLOW_API_URL", "http://127.0.0.1:8000"),
        timeout=70,
        trust_env=False,
    ) as client:

        async def tenant(name):
            headers = {"Authorization": f"Bearer {os.environ[f'OFLOW_{name}_API_KEY']}"}
            results = await asyncio.gather(
                *[
                    client.post(
                        "/v1/tasks/execute",
                        headers=headers,
                        json={
                            "task": "text.score",
                            "input": {"text": f"{name} document {i}"},
                            "profile": "latency_first",
                        },
                    )
                    for i in range(8)
                ]
            )
            assert all(r.status_code in {200, 429} for r in results)
            print(name, [r.status_code for r in results])

        await asyncio.gather(tenant("ALPHA"), tenant("BETA"))


if __name__ == "__main__":
    asyncio.run(main())
