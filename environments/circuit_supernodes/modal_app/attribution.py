import json

from circuit_supernodes.runtime.modal import SubjectConfig
from modal_app.backend import Attributor, app

__all__ = ["Attributor", "app"]


@app.local_entrypoint()
def main(
    prompt: str,
    answers: str = "",
    max_new_tokens: int = 50,
    model_name: str = SubjectConfig.model_fields["model_name"].default,
    transcoder_set: str = SubjectConfig.model_fields["transcoder_set"].default,
    node_threshold: float = SubjectConfig.model_fields["node_threshold"].default,
    edge_threshold: float = SubjectConfig.model_fields["edge_threshold"].default,
):
    subject = Attributor(
        model_name=model_name,
        transcoder_set=transcoder_set,
        node_threshold=str(node_threshold),
        edge_threshold=str(edge_threshold),
    )
    if answers:
        result = subject.sample_until_answer.remote(
            {
                "prompt": prompt,
                "possible_answers": answers.split(","),
                "max_new_tokens": max_new_tokens,
            }
        )
        print(json.dumps(result, indent=2))
        return
    graph = subject.attribute_prompt.remote(prompt)
    if graph is None:
        raise SystemExit("attribution failed")
    print(f"{len(graph.get('nodes', []))} nodes, {len(graph.get('links', []))} edges")
