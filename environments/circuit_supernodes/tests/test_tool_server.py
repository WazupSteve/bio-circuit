import asyncio
import os
import subprocess
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def _list_tools(url: str):
    async with (
        streamable_http_client(url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.list_tools()


def test_ablation_tool_module_starts_mcp_server(tmp_path):
    port_file = tmp_path / "mcp-port"
    env = {
        **os.environ,
        "VF_CONFIG": "{}",
        "MCP_PORT_FILE": str(port_file),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "circuit_supernodes.experiments"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        for _ in range(100):
            if port_file.exists() and port_file.read_text().strip().isdigit():
                break
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"tool server exited before startup:\n{output}")
            time.sleep(0.05)
        else:
            raise AssertionError("tool server did not publish a port")

        url = f"http://127.0.0.1:{port_file.read_text().strip()}/mcp"
        last_error = None
        for _ in range(100):
            try:
                tools = asyncio.run(_list_tools(url))
                break
            except Exception as exc:
                last_error = exc
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    raise AssertionError(
                        f"tool server exited during startup:\n{output}"
                    ) from exc
                time.sleep(0.05)
        else:
            raise AssertionError(
                f"tool server published a port but did not serve MCP: {last_error}"
            )
        assert [tool.name for tool in tools.tools] == ["run_ablation"]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
