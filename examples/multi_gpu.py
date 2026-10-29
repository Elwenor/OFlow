"""Optional: install .[torch], configure visible devices, then run this example."""

import asyncio

from oflow import OFlowEngine, TaskRequest


async def main():
    async with OFlowEngine.from_config("configs/multi_gpu.yaml") as engine:
        for index in range(2):
            result = await engine.execute(
                TaskRequest(
                    task="text.embed",
                    input={"text": "byte embeddings"},
                    preferred_backend=f"embeddings-gpu{index}",
                )
            )
            print(result.model_dump_json(indent=2))
        print(engine.resources.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
