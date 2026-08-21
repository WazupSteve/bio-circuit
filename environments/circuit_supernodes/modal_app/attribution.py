import json

from modal_app.backend import Attributor, app

__all__ = ["Attributor", "app"]


@app.local_entrypoint()
def main(prompt: str, answers: str = "", max_new_tokens: int = 50):
    if answers:
        result = Attributor().sample_until_answer.remote(
            {
                "prompt": prompt,
                "possible_answers": answers.split(","),
                "max_new_tokens": max_new_tokens,
            }
        )
        print(json.dumps(result, indent=2))
        return
    graph = Attributor().attribute_prompt.remote(prompt)
    if graph is None:
        raise SystemExit("attribution failed")
    print(f"{len(graph.get('nodes', []))} nodes, {len(graph.get('links', []))} edges")
