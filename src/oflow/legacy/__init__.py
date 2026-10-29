"""Optional trusted ORIS/mmBERT adapter. Never imported by CPU-only startup."""

OFLOW_PLUGIN_API_VERSION = "1"
OFLOW_PLUGIN_VERSION = "1.0.0b1"


def register(tasks):
    from .adapter import LegacyBackend, LegacyInput, LegacyOutput

    tasks.register("legacy.semantic", LegacyInput, LegacyOutput)
    return {"legacy": LegacyBackend}
