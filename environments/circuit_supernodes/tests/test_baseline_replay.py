from types import SimpleNamespace
from typing import Any

import pytest
from scripts.dataset_pipeline import clean_attributions, grader_baseline_failures

from scripts import dataset_pipeline


def graph(top: str) -> dict:
    return {
        "metadata": {"prompt_tokens": ["x"]},
        "nodes": [
            {"node_id": "1_2_0", "feature_type": "feature", "influence": 0.5},
            {
                "node_id": "logit",
                "feature_type": "logit",
                "clerp": f'"{top}"',
                "token_prob": 0.9,
                "influence": 1.0,
            },
        ],
        "links": [],
    }


class Call:
    def __init__(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class Subject:
    """Replays a scripted baseline outcome per prompt."""

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self.outcomes = outcomes
        self.replayed: list[dict] = []

    def attribution_graphs(self, prompts: list[str]) -> list[dict]:
        return [graph(f" {prompt}") for prompt in prompts]

    def spawn(self, method: str, positives: list[dict], controls: list[dict]) -> Call:
        assert method == "evaluate_ablation_cohort"
        assert controls == []
        payload = positives[0]
        self.replayed.append(payload)
        outcome = self.outcomes[payload["prompt"]]
        if isinstance(outcome, Exception):
            return Call(outcome)
        return Call(
            {
                "positives": [
                    {
                        "expected_token": payload["expected_token"],
                        "baseline_expected_is_top": outcome,
                        "changed_top_token": False,
                    }
                ],
                "controls": [],
            }
        )


def prepared(prompt: str) -> SimpleNamespace:
    return SimpleNamespace(
        sample=SimpleNamespace(prompt=prompt), target_token=f" {prompt}"
    )


def test_builders_drop_samples_the_grader_would_flag_as_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        dataset_pipeline, "graph_quality", lambda graph, limit: {"error": 0.0}
    )
    subject = Subject({"a": True, "b": False, "c": ValueError("no round trip")})
    kept = list(
        clean_attributions(
            subject,
            [prepared("a"), prepared("b"), prepared("c")],
            max_error_share=0.5,
            batch_size=8,
            concurrent_calls=False,
        )
    )
    assert [item.prepared.sample.prompt for item in kept] == ["a"]
    # The replay is the grader's own unintervened request.
    assert subject.replayed[0] == {
        "prompt": "a",
        "expected_token": " a",
        "interventions": [],
        "baseline_interventions": [],
    }


def test_baseline_replay_classifies_failures_and_detects_outages() -> None:
    from circuit_supernodes.models import DatasetExample

    examples = [
        DatasetExample(
            prompt=prompt, target_token=f" {prompt}", graph=graph(f" {prompt}")
        )
        for prompt in ("a", "b", "c")
    ]
    subject = Subject({"a": True, "b": False, "c": RuntimeError("boom")})
    assert grader_baseline_failures(subject, examples) == [None, "drift", "error"]

    down = Subject({"a": RuntimeError("down")})
    with pytest.raises(RuntimeError, match="every grader baseline replay"):
        grader_baseline_failures(down, examples[:1])
    assert grader_baseline_failures(down, []) == []
