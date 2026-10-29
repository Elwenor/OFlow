"""Single-GPU execution; exits with an error if the configured GPU is unavailable."""

import asyncio

from oflow import OFlowEngine, TaskRequest


async def main():
    async with OFlowEngine.from_config("configs/gpu.yaml") as engine:
        result = await engine.execute(
            TaskRequest(
                task="text.embed",
                input={"text": "GPU execution"},
                preferred_backend="embeddings-gpu0",
            )
        )
        if result.error:
            raise RuntimeError(f"GPU execution failed: {result.error.category.value}")
        print(result.model_dump_json(indent=2))
        print(engine.resources.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
