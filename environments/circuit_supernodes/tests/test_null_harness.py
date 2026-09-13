from circuit_null import PROGRAM_SOURCE, CircuitNullHarness
from verifiers.v1.loaders import harness_class, harness_config_type


def test_circuit_null_harness_pins_mcp_one():
    assert harness_class("circuit-null") is CircuitNullHarness
    assert harness_config_type("circuit-null")(id="circuit-null").id == "circuit-null"
    assert '# dependencies = ["openai", "mcp>=1.24,<2"' in PROGRAM_SOURCE
    assert "MCP_TIMEOUT = httpx.Timeout(3600.0, connect=5.0)" in PROGRAM_SOURCE
