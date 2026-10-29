"""Interactive latency and bounded bulk scoring on the same engine."""

import asyncio

from oflow import OFlowEngine, TaskRequest


async def main():
    async with OFlowEngine.from_config("configs/cpu.yaml") as engine:
        interactive = await engine.execute(
            TaskRequest(
                task="text.score", input={"text": "interactive request"}, profile="latency_first"
            )
        )
        bulk = await engine.execute_many(
            [
                TaskRequest(
                    task="text.score", input={"text": f"document {i}"}, profile="throughput_first"
                )
                for i in range(100)
            ],
            concurrency=16,
        )
        assert not interactive.error and all(not r.error for r in bulk)
        print(f"interactive={interactive.status}; bulk completed={len(bulk)}")


if __name__ == "__main__":
    asyncio.run(main())
