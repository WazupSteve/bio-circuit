"""BioCircuit null harness using Verifiers' MCP 1.x client and timeout config."""

from verifiers.v1.harnesses.null.harness import NullHarness

__all__ = ["CircuitNullHarness"]


class CircuitNullHarness(NullHarness):
    """Use the upstream launch protocol; configs allow hour-long circuit calls."""
