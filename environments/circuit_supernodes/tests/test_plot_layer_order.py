"""The blog charts are read straight off archive/, so the only thing worth unit
testing is the one piece of real math in them (the best-of-k estimator) and
that every chart function still produces well-formed SVG against the actual,
current archive files — a silently broken chart is a worse failure mode than
a crashing one, since nobody looks at a media/ asset in code review.
"""

import itertools
import xml.dom.minidom as minidom

import pytest
from plot_layer_order import (
    chart1,
    chart2,
    chart3,
    chart4,
    chart5,
    expected_best_of_k,
    load_data,
    load_gate_pass,
    load_search_curve,
    load_v1_components,
)


def brute_force_best_of_k(samples: list[float], k: int) -> float:
    """Exact E[max of k draws without replacement] by enumerating every
    combination — what expected_best_of_k must agree with for any n, k."""
    combos = list(itertools.combinations(samples, k))
    return sum(max(c) for c in combos) / len(combos)


@pytest.mark.parametrize(
    ("samples", "k"),
    [
        ([0.1, 0.2, 0.3, 0.4], 1),
        ([0.1, 0.2, 0.3, 0.4], 2),
        ([0.1, 0.2, 0.3, 0.4], 4),
        ([0.5, 0.5, 0.1, 0.9, 0.3, 0.2], 3),
    ],
)
def test_expected_best_of_k_matches_brute_force(samples, k):
    assert expected_best_of_k(samples, k) == pytest.approx(
        brute_force_best_of_k(samples, k)
    )


def test_expected_best_of_k_at_k_equals_n_is_the_max():
    samples = [0.1, 0.7, 0.3]
    assert expected_best_of_k(samples, len(samples)) == max(samples)


def test_expected_best_of_k_at_k_equals_1_is_the_mean():
    samples = [0.1, 0.7, 0.3, 0.2]
    assert expected_best_of_k(samples, 1) == pytest.approx(sum(samples) / len(samples))


def assert_valid_svg(svg: str) -> None:
    doc = minidom.parseString(svg)
    root = doc.documentElement
    assert root.tagName == "svg"
    assert root.getAttribute("viewBox")
    # Every <g><title>...</title></g> must close, every rect/circle/line/text
    # tag balanced — minidom.parseString already raised if not.


def test_every_chart_renders_valid_svg_from_the_real_archive():
    models, attacks = load_data()
    assert_valid_svg(chart1(models, attacks))
    assert_valid_svg(chart2(models, attacks))
    assert_valid_svg(chart3(load_v1_components()))
    assert_valid_svg(chart4(load_search_curve(), models))
    assert_valid_svg(chart5(load_gate_pass(models)))


def test_load_data_gate_counts_match_reward_components():
    # scored + gated must equal every rollout in the file, and "sub" must
    # describe exactly what scored/total say — a chart that quotes a
    # human-readable caption disagreeing with its own bars is worse than one
    # with no caption at all.
    models, _ = load_data()
    for stats in models.values():
        assert stats["scored"] <= stats["total"]
        gated = stats["total"] - stats["scored"]
        assert (
            stats["sub"]
            == f"{stats['scored']}/{stats['total']} rollouts, {gated} gated"
        )
