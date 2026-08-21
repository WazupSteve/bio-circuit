"""Modal deployment entrypoint for persistent-corruption Circuit Tracer."""

from modal_app.restoration_backend import RestorationAttributor, app

__all__ = ["RestorationAttributor", "app"]


@app.local_entrypoint()
def main(prompt: str) -> None:
    graph = RestorationAttributor().attribute_prompt.remote(prompt)
    if graph is None:
        raise SystemExit("attribution failed")
    nodes = graph.get("nodes", [])
    print(f"{len(nodes)} nodes, {len(graph.get('links', []))} edges")
