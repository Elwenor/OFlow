"""A complete trusted plugin example. Run from the project root."""

import asyncio

from oflow import OFlowEngine, TaskRequest
from oflow.contracts import BackendDescriptor, BackendHealth, Contract

OFLOW_PLUGIN_API_VERSION = "1"
OFLOW_PLUGIN_VERSION = "0.1"


class Input(Contract):
    text: str


class Output(Contract):
    text: str


class UppercaseBackend:
    def __init__(self, config, tasks):
        if config.options or config.device != "cpu":
            raise ValueError("uppercase example accepts no options and runs on cpu")
        self.descriptor = BackendDescriptor(
            id=config.id,
            version="0.1",
            max_concurrency=config.concurrency,
            capabilities=(tasks.capability("demo.uppercase"),),
        )

    def initialize(self):
        pass

    def shutdown(self):
        pass

    def health(self):
        return BackendHealth(healthy=True)

    def execute(self, request, context):
        context.check_cancelled()
        return {"text": request.input["text"].upper()}


def register(tasks):
    tasks.register("demo.uppercase", Input, Output)
    return {"uppercase": UppercaseBackend}


async def main():
    async with OFlowEngine.from_config("examples/custom.yaml") as engine:
        result = await engine.execute(
            TaskRequest(task="demo.uppercase", input={"text": "hello OFlow"})
        )
        assert result.output == {"text": "HELLO OFLOW"}
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
