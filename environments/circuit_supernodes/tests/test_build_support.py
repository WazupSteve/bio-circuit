import json

from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
    TracerGraphPayload,
    TracerNode,
)
from scripts.build_support import (
    file_sha256,
    manifest_path,
    open_text,
    write_jsonl_atomic,
)
from scripts.dataset_pipeline import mix_family_rows, write_manifest


def test_streaming_gzip_jsonl_round_trip(tmp_path):
    path = tmp_path / "cohorts.jsonl.gz"
    expected = [{"index": 1, "payload": "x" * 1000}, {"index": 2}]

    write_jsonl_atomic(path, iter(expected))

    with open_text(path) as stream:
        actual = [json.loads(line) for line in stream]
    assert actual == expected
    assert file_sha256(path) == file_sha256(path)
    assert manifest_path(path) == tmp_path / "cohorts.manifest.json"


def test_streaming_plain_jsonl_round_trip(tmp_path):
    path = tmp_path / "cohorts.jsonl"
    expected = [{"index": 1}]

    write_jsonl_atomic(path, expected)

    with open_text(path) as stream:
        assert [json.loads(line) for line in stream] == expected
    assert manifest_path(path) == tmp_path / "cohorts.manifest.json"


def test_model_rows_are_compacted_recursively_at_jsonl_boundary(tmp_path) -> None:
    path = tmp_path / "cohorts.jsonl"
    graph = TracerGraphPayload(
        nodes=[TracerNode(node_id="1_2_0", feature_type="feature")]
    )
    row = CircuitDatasetRow(
        family="test",
        example_prompts=[],
        heldout_prompts=[
            DatasetExample(prompt="prompt", target_token=" token", graph=graph)
        ],
        control_prompts=[],
    )

    write_jsonl_atomic(path, [row])

    payload = json.loads(path.read_text())
    assert set(payload) == {
        "family",
        "example_prompts",
        "heldout_prompts",
        "control_prompts",
    }
    assert set(payload["heldout_prompts"][0]) == {"prompt", "target_token", "graph"}
    assert payload["heldout_prompts"][0]["graph"] == {
        "nodes": [{"node_id": "1_2_0", "feature_type": "feature"}]
    }


def test_manifest_stores_schema_and_generation_provenance_once(tmp_path) -> None:
    dataset = tmp_path / "cohorts.jsonl"
    write_jsonl_atomic(dataset, [{"family": "test"}])
    triage = {"keep_correct": 4, "keep_incorrect": 1, "reject": 0}
    sidecar = write_manifest(
        dataset,
        rows=1,
        subject_backend="test/Subject",
        families=["test"],
        triage=triage,
        max_error_share=0.35,
        quality=[],
        generator={
            "seed": 1,
            "candidates": 10,
            "tasks": 1,
            "holdout_size": 2,
            "null_size": 2,
            "max_records_per_goal": 4,
        },
    )
    manifest = json.loads(sidecar.read_text())
    assert manifest["subject"] == {
        "model": "Qwen/Qwen3-4B",
        "transcoder_set": "mwhanna/qwen3-4b-transcoders",
        "node_threshold": 0.8,
        "edge_threshold": 0.98,
        "backend": "test/Subject",
    }
    assert manifest["dataset_schema_version"] == "runtime-minimal-v2"
    assert manifest["generation_policy"] == "maximal-circuit-tracer-export-v1"
    assert manifest["maximal_graphs"] is True
    assert manifest["max_feature_nodes"] == 4096
    assert "corruption_version" not in manifest
    assert "reward_version" not in manifest
    assert manifest["sampler"]["triage"] == triage
    assert "answer_sampling" not in manifest["generator"]


def test_family_rows_are_randomly_mixed_and_seed_deterministic() -> None:
    rows = [
        CircuitDatasetRow(
            family=family,
            example_prompts=[],
            heldout_prompts=[],
            control_prompts=[],
        )
        for family in ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    ]

    first = mix_family_rows(rows, seed=7)
    second = mix_family_rows(rows, seed=7)

    first_families = [row.family for row in first]
    assert first_families == [row.family for row in second]
    assert first_families != [row.family for row in rows]
    assert sorted(first_families) == sorted(row.family for row in rows)
