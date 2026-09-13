import json

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.null.harness import (
    PROGRAM_SOURCE as UPSTREAM_PROGRAM_SOURCE,
)
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

__all__ = ["CircuitNullHarness"]

_UPSTREAM_DEPENDENCIES = '# dependencies = ["openai", "mcp", "httpx", "tenacity"]'
_PINNED_DEPENDENCIES = (
    '# dependencies = ["openai", "mcp>=1.24,<2", "httpx", "tenacity"]'
)
_UPSTREAM_MCP_TIMEOUT = "MCP_TIMEOUT = httpx.Timeout(600.0, connect=5.0)"
_EXTENDED_MCP_TIMEOUT = "MCP_TIMEOUT = httpx.Timeout(3600.0, connect=5.0)"

if UPSTREAM_PROGRAM_SOURCE.count(_UPSTREAM_DEPENDENCIES) != 1:
    raise RuntimeError(
        "Verifiers null-harness dependencies changed; update circuit-null"
    )
if UPSTREAM_PROGRAM_SOURCE.count(_UPSTREAM_MCP_TIMEOUT) != 1:
    raise RuntimeError("Verifiers null-harness timeout changed; update circuit-null")

PROGRAM_SOURCE = UPSTREAM_PROGRAM_SOURCE.replace(
    _UPSTREAM_DEPENDENCIES,
    _PINNED_DEPENDENCIES,
).replace(_UPSTREAM_MCP_TIMEOUT, _EXTENDED_MCP_TIMEOUT)


class CircuitNullHarness(NullHarness):
    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
        ]
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        if mcp_urls:
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path,
                json.dumps([message_to_wire(message) for message in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], self.config.resolved_env)
