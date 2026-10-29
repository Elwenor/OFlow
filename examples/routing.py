"""Change only YAML data to route the same task to a different compatible backend."""

import asyncio

import yaml

from oflow import OFlowEngine, TaskRequest
from oflow.config import EngineConfig


async def main():
    with open("configs/cpu.yaml", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for target in ["cpu", "mock"]:
        config["routing"]["rules"]["text.score"] = [target]
        async with OFlowEngine(EngineConfig.model_validate(config)) as engine:
            result = await engine.execute(
                TaskRequest(task="text.score", input={"text": "same workload"})
            )
            assert result.backend == target
            print(f"selected={result.backend}; output={result.output}")


if __name__ == "__main__":
    asyncio.run(main())
