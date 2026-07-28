"""The write-up is the deliverable, so its numbers must match the archived data.

A results document that quietly drifts from the runs behind it is worse than no
document. This parses the tables out of FINDINGS.md and checks each figure
against the JSON the audit wrote, so editing either side without the other
fails here rather than in review.
"""

import json
import re
from pathlib import Path
from statistics import mean

ENV = Path(__file__).parents[1]
FINDINGS = ENV / "FINDINGS.md"

# Which archived run backs each table, and which column of it.
TABLES = {
    "archive/oracle_layers.json": {"reward": 1, "separation": 2},
    "archive/oracle_layers_full.json": {
        "reward": 1,
        "separation": 1,
        "separation_layer_null": 2,
        "separation_over_null": 3,
    },
    "archive/oracle_null.json": {
        "separation": 1,
        "separation_layer_null": 2,
        "separation_over_null": 3,
    },
}
ROWS = {
    "layer-bands": "v2/layer-bands",
    "layer-spanning": "v2/layer-spanning",
    "community partition": "v2/community",
    "coherence-matched": "v2/coherence-matched",
    "budget-feasible": "v2/budget-feasible",
    "blind-balanced": "v2/blind-balanced",
    "random, best of 2": "v2/random-multistart",
    "random, best of 3": "v2/random-multistart",
}


def measured(path: str, key: str, field: str) -> float | None:
    rows = json.loads((ENV / path).read_text())
    values = [r["scores"][key].get(field) for r in rows]
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def claimed() -> dict[tuple[str, int], float]:
    """Every markdown table row of the form `| label | ... | number |`."""
    out = {}
    for line in FINDINGS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        label = cells[0]
        if label not in ROWS:
            continue
        for i, cell in enumerate(cells):
            if re.fullmatch(r"−?-?\d\.\d\d\d", cell):
                out[(label, i)] = float(cell.replace("−", "-"))
    return out


def archived_values(key: str) -> list[float]:
    """Every statistic the audit recorded for this row, across runs and fields."""
    values = []
    for path, fields in TABLES.items():
        if not (ENV / path).exists():
            continue
        for field in fields:
            got = measured(path, key, field)
            if got is not None:
                values.append(got)
    return values


def test_every_figure_in_findings_matches_the_archived_runs():
    """Each number must correspond to something actually measured.

    Counting how many figures match and requiring "enough" of them is not a
    test — an earlier version did that and passed with a deliberately corrupted
    table. Every parsed figure has to find its value in the data.
    """
    figures = claimed()
    assert figures, "no parsable result rows found in FINDINGS.md"
    unmatched = [
        (label, stated)
        for (label, _), stated in figures.items()
        if not any(abs(v - stated) < 0.002 for v in archived_values(ROWS[label]))
    ]
    assert not unmatched, f"figures with no archived value: {unmatched}"


def test_every_script_findings_cites_exists():
    text = FINDINGS.read_text()
    for script in sorted(set(re.findall(r"scripts/[a-z_]+\.py", text))):
        assert (ENV / script).exists(), f"FINDINGS.md cites a missing {script}"
