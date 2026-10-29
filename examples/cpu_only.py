"""python examples/cpu_only.py"""

import asyncio

from oflow import OFlowEngine, TaskRequest


async def main():
    async with OFlowEngine.from_config("configs/cpu.yaml") as engine:
        result = await engine.execute(
            TaskRequest(task="text.score", input={"text": "Ala ma kota. Ala ma psa."})
        )
        assert result.output["method"] == "lexical_diversity"
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
