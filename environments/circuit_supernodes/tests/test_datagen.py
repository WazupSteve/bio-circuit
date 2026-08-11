import pytest
from circuit_supernodes.datagen import (
    EVAL_FAMILIES,
    FAMILIES,
    PROPERTIES,
    SYMBOLS,
    TRAIN_FAMILIES,
    generate,
)


def test_default_generation_is_training_only() -> None:
    samples = generate(len(TRAIN_FAMILIES) * 3, seed=7)
    assert {sample.family for sample in samples} == set(TRAIN_FAMILIES)
    assert not ({sample.family for sample in samples} & set(EVAL_FAMILIES))


def test_eval_registry_is_disjoint_and_drops_easy_addition() -> None:
    assert set(TRAIN_FAMILIES).isdisjoint(EVAL_FAMILIES)
    assert set(TRAIN_FAMILIES) | set(EVAL_FAMILIES) == set(FAMILIES)
    assert "addition" not in FAMILIES
    assert {
        "path_composition",
        "modular_arithmetic",
        "rule_composition",
        "spatial_composition",
        "boolean_composition",
    } <= set(EVAL_FAMILIES)


@pytest.mark.parametrize("family", EVAL_FAMILIES)
def test_eval_families_generate_unique_nonempty_examples(family: str) -> None:
    samples = generate(20, seed=19, families=[family])
    assert len(samples) == 20
    assert {sample.family for sample in samples} == {family}
    assert len({sample.prompt for sample in samples}) == len(samples)
    assert all(sample.prompt and sample.target for sample in samples)


def test_generation_is_seed_deterministic() -> None:
    first = generate(18, seed=123, families=list(EVAL_FAMILIES))
    second = generate(18, seed=123, families=list(EVAL_FAMILIES))
    assert first == second


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_declares_its_answer_vocabulary(family: str) -> None:
    sample = generate(1, seed=31, families=[family])[0]
    answers = [answer.strip().casefold() for answer in sample.possible_answers]
    assert answers
    assert len(answers) == len(set(answers))
    assert sample.target.strip().casefold() in answers


@pytest.mark.parametrize(
    ("family", "expected_answers"),
    [
        ("path_composition", SYMBOLS),
        ("rule_composition", PROPERTIES),
    ],
)
def test_symbolic_families_declare_the_complete_keyword_domain(
    family: str, expected_answers: tuple[str, ...]
) -> None:
    sample = generate(1, seed=31, families=[family])[0]
    assert sample.possible_answers == expected_answers
