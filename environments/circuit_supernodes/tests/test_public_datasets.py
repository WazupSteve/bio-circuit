from pathlib import Path

import pytest
from circuit_supernodes.runtime import datasets


def test_explicit_local_dataset_does_not_download(tmp_path, monkeypatch):
    def unexpected_download(**kwargs):
        pytest.fail("local data must not trigger a network request")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_download)
    local = tmp_path / "custom.jsonl"
    local.write_text("{}\n")
    assert datasets.resolve_dataset(tmp_path, str(local)) == local


def test_release_download_is_anonymous_and_revision_pinned(tmp_path, monkeypatch):
    calls = []
    cached = tmp_path / "cached.jsonl.gz"

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    resolved = datasets.resolve_dataset(
        tmp_path / "visible_dataset", "heldout-cohorts.jsonl.gz"
    )
    assert resolved == cached
    assert calls == [
        {
            "repo_id": "amitprakash2005/bio-circuit",
            "filename": "data/visible_dataset/heldout-cohorts.jsonl.gz",
            "repo_type": "dataset",
            "revision": datasets.HF_DATASET_REVISION,
            "token": False,
        }
    ]
    assert len(datasets.HF_DATASET_REVISION) == 40
    int(datasets.HF_DATASET_REVISION, 16)


def test_missing_training_data_fails_without_network(tmp_path, monkeypatch):
    def unexpected_download(**kwargs):
        pytest.fail("unpublished training data must not be fetched")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_download)
    with pytest.raises(FileNotFoundError, match="Only evaluation cohorts"):
        datasets.resolve_dataset(tmp_path / "visible_dataset", "train-cohorts.jsonl.gz")


def test_all_default_datasets_are_published():
    from circuit_supernodes.tasksets import rediscovery, restoration, visible

    for module, config in (
        (visible, visible.CircuitTasksetConfig()),
        (rediscovery, rediscovery.DiscoveryTasksetConfig()),
        (restoration, restoration.CircuitTasksetConfig()),
    ):
        assert (
            f"{Path(module.DATA_DIR).name}/{config.dataset_file}"
            in datasets.PUBLIC_DATASETS
        )
