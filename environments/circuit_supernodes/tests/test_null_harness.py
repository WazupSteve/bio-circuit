import tomllib
from pathlib import Path

import pytest
from circuit_null import CircuitNullHarness
from verifiers.v1.utils.loaders import harness_class, harness_config_type

CONFIGS = Path(__file__).parents[3] / "configs"


def test_circuit_null_harness_resolves_with_hour_long_tool_timeout():
    assert harness_class("circuit-null") is CircuitNullHarness
    config = harness_config_type("circuit-null")(id="circuit-null", tool_timeout=3600)
    assert config.tool_timeout == 3600


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*/*.toml")), ids=lambda p: p.name)
def test_configs_allow_hour_long_circuit_tool_calls(path):
    harnesses = list(_harness_tables(tomllib.loads(path.read_text())))
    assert harnesses
    assert all(
        h["id"] == "circuit-null" and h["tool_timeout"] == 3600 for h in harnesses
    )


def _harness_tables(value):
    """Every ``harness`` table in an eval or RL config, at any nesting depth."""

    if isinstance(value, list):
        for item in value:
            yield from _harness_tables(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "harness":
                yield item
            else:
                yield from _harness_tables(item)
