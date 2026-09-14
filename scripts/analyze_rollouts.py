# /// script
# requires-python = ">=3.13"
# dependencies = ["matplotlib>=3.10,<4", "numpy>=2,<3"]
# ///

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
VIEWER_ROOT = MEDIA / "viewer" / "evals"

BACKGROUND = "#F7F4EE"
TEXT = "#17202A"
MUTED = "#667085"
GRID = "#D8D2C4"
SUCCESS = "#168C7A"
FAILURE = "#D95D4F"
INVALID = "#AAA59B"


@dataclass(frozen=True)
class RunSpec:
    label: str
    slug: str
    color: str
    trace_path: Path


RUNS = (
    RunSpec(
        label="Kimi K3",
        slug="moonshotai--kimi-k3",
        color="#159D91",
        trace_path=ROOT
        / "outputs/circuit-supernodes--moonshotai--kimi-k3--circuit-null"
        / "ec9ab273-1f45-4f7a-b86e-8e09bd803a16-min2/traces.jsonl",
    ),
    RunSpec(
        label="Fable 5.1",
        slug="anthropic--claude-fable-5.1",
        color="#7655C5",
        trace_path=ROOT
        / "outputs/circuit-supernodes--anthropic--claude-fable-5.1--circuit-null"
        / "cf7c924e-e5b9-4a30-b1be-bf1f9d6502b0-min2/traces.jsonl",
    ),
    RunSpec(
        label="GPT-6 Astra",
        slug="openai--gpt-6-astra",
        color="#E8873A",
        trace_path=ROOT
        / "outputs/circuit-supernodes--openai--gpt-6-astra--circuit-null"
        / "95e91976-68da-4a90-a0ef-a715236b5d73-min2/traces.jsonl",
    ),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout trace: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "unknown")
    return str(call.get("name") or "unknown")


def timing_duration(trace: dict[str, Any]) -> float:
    timing = trace.get("timing") or {}
    start = float(timing.get("start") or 0.0)
    scoring = timing.get("scoring") or {}
    generation = timing.get("generation") or {}
    end = float(scoring.get("end") or generation.get("end") or start)
    return max(0.0, end - start)


def proposal_nodes(trace: dict[str, Any]) -> list[str]:
    score = (trace.get("info") or {}).get("cohort_score") or {}
    proposal = score.get("proposal") or {}
    if not isinstance(proposal, dict):
        return []
    nodes: list[str] = []
    for value in proposal.values():
        if isinstance(value, dict | list):
            nodes.extend(str(node) for node in value)
    return nodes


def flatten_rollouts(
    traces_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        task_occurrences: Counter[int] = Counter()
        for trace in traces_by_model[spec.label]:
            task_data = (trace.get("task") or {}).get("data") or {}
            task_index = int(task_data["idx"])
            task_occurrences[task_index] += 1
            reward = float(
                (trace.get("rewards") or {}).get("circuit_faithfulness", 0.0)
            )
            metrics = trace.get("metrics") or {}
            usage: Counter[str] = Counter()
            tool_counts: Counter[str] = Counter()
            for call in trace.get("calls") or []:
                for key, value in (call.get("usage") or {}).items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        usage[key] += value
            for node in trace.get("nodes") or []:
                message = node.get("message") or {}
                for call in message.get("tool_calls") or []:
                    tool_counts[tool_call_name(call).removeprefix("circuit_")] += 1
            nodes = proposal_nodes(trace)
            score = (trace.get("info") or {}).get("cohort_score") or {}
            row = {
                "model": spec.label,
                "model_id": (trace.get("agent") or {}).get("model", ""),
                "task": task_index,
                "rollout": task_occurrences[task_index],
                "family": str(task_data.get("family") or "unknown"),
                "trace_id": trace.get("id", ""),
                "success": int(reward > 0.0),
                "reward": reward,
                "baseline_drift": int(float(metrics.get("baseline_drift", 0.0)) > 0),
                "grading_validation_error": int(
                    float(metrics.get("grading_validation_error", 0.0)) > 0
                ),
                "evidence_success_rate": float(
                    metrics.get("evidence_success_rate", 0.0)
                ),
                "holdout_success_rate": float(
                    metrics.get("holdout_success_rate", 0.0)
                ),
                "null_failure_rate": float(metrics.get("null_failure_rate", 0.0)),
                "evidence_pass": int(float(metrics.get("evidence_pass", 0.0)) > 0),
                "holdout_pass": int(float(metrics.get("holdout_pass", 0.0)) > 0),
                "null_pass": int(float(metrics.get("null_pass", 0.0)) > 0),
                "duration_minutes": timing_duration(trace) / 60.0,
                "api_calls": len(trace.get("calls") or []),
                "tool_attribute": tool_counts["attribute"],
                "tool_query_graph": tool_counts["query_graph"],
                "tool_intervene": tool_counts["intervene"],
                "tool_calls": sum(tool_counts.values()),
                "prompt_tokens": int(usage["prompt_tokens"]),
                "cached_input_tokens": int(usage["cached_input_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "reasoning_tokens": int(usage["reasoning_tokens"]),
                "reported_api_cost": float(usage["cost"]),
                "proposal_valid": int(bool(score.get("valid"))),
                "proposal_nodes": len(nodes),
                "stop_condition": trace.get("stop_condition", ""),
            }
            rows.append(row)
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def title(
    fig: plt.Figure,
    heading: str,
    subtitle: str,
    *,
    top: float = 0.95,
) -> None:
    fig.text(0.06, top, heading, fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.06, top - 0.045, subtitle, fontsize=11, color=MUTED)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(MEDIA / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(MEDIA / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def model_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["model"] == label]


def plot_outcomes(rows: list[dict[str, Any]]) -> None:
    labels = [spec.label for spec in RUNS]
    counts = []
    gradeable_rates = []
    for label in labels:
        selected = model_rows(rows, label)
        invalid = sum(row["baseline_drift"] for row in selected)
        successful = sum(row["success"] for row in selected)
        gradeable = len(selected) - invalid
        counts.append((successful, gradeable - successful, invalid))
        gradeable_rates.append(successful / gradeable)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.18, right=0.94, top=0.79, bottom=0.18)
    y_positions = np.arange(len(labels))
    segments = (
        ("Successful", SUCCESS, [value[0] for value in counts]),
        ("Unsuccessful · gradeable", FAILURE, [value[1] for value in counts]),
        ("Subject baseline drift", INVALID, [value[2] for value in counts]),
    )
    left = np.zeros(len(labels))
    for segment_label, color, values in segments:
        values_array = np.array(values)
        bars = ax.barh(
            y_positions,
            values_array,
            left=left,
            height=0.58,
            color=color,
            label=segment_label,
        )
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if color != INVALID else TEXT,
                    fontweight="bold",
                )
        left += values_array

    for index, rate in enumerate(gradeable_rates):
        ax.text(
            32.6,
            index,
            f"{rate:.1%} gradeable pass",
            ha="left",
            va="center",
            fontweight="bold",
            color=RUNS[index].color,
        )
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xlabel("Rollouts per model")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    title(
        fig,
        "Three models, one difficult circuit benchmark",
        "Each model completed 32 rollouts; five shared tasks drifted at the "
        "subject baseline.",
    )
    fig.text(
        0.06,
        0.045,
        "Success requires evidence ≥60%, holdout ≥40%, and zero null-control "
        "failures. n=2 per task/model.",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(fig, "01_outcome_overview")


def task_order(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    tasks = {(int(row["task"]), str(row["family"])) for row in rows}
    return sorted(tasks)


def plot_task_matrix(rows: list[dict[str, Any]]) -> None:
    tasks = task_order(rows)
    labels = [spec.label for spec in RUNS]
    matrix = np.full((len(tasks), len(labels)), np.nan)
    annotations: list[list[str]] = [["" for _ in labels] for _ in tasks]
    for task_row, (task_index, _) in enumerate(tasks):
        for model_column, label in enumerate(labels):
            selected = [
                row
                for row in rows
                if row["task"] == task_index and row["model"] == label
            ]
            if selected and all(row["baseline_drift"] for row in selected):
                annotations[task_row][model_column] = "drift"
                continue
            success_count = sum(row["success"] for row in selected)
            matrix[task_row, model_column] = success_count
            annotations[task_row][model_column] = f"{success_count}/{len(selected)}"

    cmap = ListedColormap(["#E9B6AF", "#F4D88B", "#64B8A7"])
    cmap.set_bad(INVALID)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10.5, 11.5))
    fig.subplots_adjust(left=0.33, right=0.94, top=0.83, bottom=0.11)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=12)
    y_labels = [
        f"{task_index:02d}  {family.replace('_', ' ').title()}"
        for task_index, family in tasks
    ]
    ax.set_yticks(np.arange(len(tasks)), y_labels)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for task_row in range(len(tasks)):
        for model_column in range(len(labels)):
            invalid = np.isnan(matrix[task_row, model_column])
            if invalid:
                ax.add_patch(
                    Rectangle(
                        (model_column - 0.5, task_row - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="////",
                        edgecolor="#817D75",
                        linewidth=0,
                    )
                )
            ax.text(
                model_column,
                task_row,
                annotations[task_row][model_column],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if matrix[task_row, model_column] == 2 else TEXT,
            )
    ax.spines[:].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#E9B6AF", label="0/2"),
            Patch(facecolor="#F4D88B", label="1/2"),
            Patch(facecolor="#64B8A7", label="2/2"),
            Patch(facecolor=INVALID, hatch="////", label="Subject baseline drift"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncols=4,
    )
    title(
        fig,
        "Success clustered in four tasks",
        "Two-hop task 11 was the only unanimous 2/2 result; five tasks were "
        "invalidated before model comparison.",
        top=0.96,
    )
    save_figure(fig, "02_task_model_matrix")


def family_order(rows: list[dict[str, Any]]) -> list[str]:
    first_task: dict[str, int] = {}
    for row in rows:
        family = str(row["family"])
        first_task[family] = min(first_task.get(family, 10_000), int(row["task"]))
    return sorted(first_task, key=first_task.get)


def plot_family_performance(rows: list[dict[str, Any]]) -> None:
    families = family_order(rows)
    labels = [spec.label for spec in RUNS]
    matrix = np.zeros((len(families), len(labels)))
    annotations: list[list[str]] = [["" for _ in labels] for _ in families]
    for family_row, family in enumerate(families):
        for model_column, label in enumerate(labels):
            selected = [
                row
                for row in rows
                if row["family"] == family
                and row["model"] == label
                and not row["baseline_drift"]
            ]
            successful = sum(row["success"] for row in selected)
            rate = successful / len(selected) if selected else 0.0
            matrix[family_row, model_column] = rate
            annotations[family_row][model_column] = (
                f"{rate:.0%}\n{successful}/{len(selected)}"
            )

    cmap = LinearSegmentedColormap.from_list(
        "circuit_success", ["#E8B0A9", "#F6E39E", "#55AE9D"]
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.30, right=0.94, top=0.78, bottom=0.13)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_yticks(
        np.arange(len(families)),
        [family.replace("_", " ").title() for family in families],
    )
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for family_row in range(len(families)):
        for model_column in range(len(labels)):
            ax.text(
                model_column,
                family_row,
                annotations[family_row][model_column],
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if matrix[family_row, model_column] > 0.72 else TEXT,
            )
    ax.spines[:].set_visible(False)
    title(
        fig,
        "Reasoning family mattered more than model choice",
        "Gradeable success rate. Denominators shrink from four to two when one "
        "task in a family had baseline drift.",
    )
    fig.text(
        0.06,
        0.045,
        "Cells show success percentage and successful/gradeable rollouts.",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(fig, "03_family_performance")


def plot_scoring_gates(rows: list[dict[str, Any]]) -> None:
    gates = (
        ("Evidence ≥60%", "evidence_pass"),
        ("Holdout ≥40%", "holdout_pass"),
        ("Null failures = 0", "null_pass"),
        ("All gates", "success"),
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.20)
    x_positions = np.arange(len(gates))
    width = 0.23
    for model_index, spec in enumerate(RUNS):
        selected = [
            row
            for row in model_rows(rows, spec.label)
            if not row["baseline_drift"]
        ]
        counts = [sum(row[key] for row in selected) for _, key in gates]
        rates = np.array(counts) / len(selected)
        bars = ax.bar(
            x_positions + (model_index - 1) * width,
            rates,
            width,
            label=spec.label,
            color=spec.color,
        )
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{count}/22",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT,
            )
    ax.set_xticks(x_positions, [label for label, _ in gates])
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of gradeable rollouts")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    title(
        fig,
        "Different models failed at different scoring gates",
        "Fable and Astra usually protected the null control; Kimi reached evidence "
        "more often but leaked specificity.",
    )
    fig.text(
        0.06,
        0.05,
        "Gate rates are independent, not cumulative. Shared baseline-drift "
        "rollouts are excluded.",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(fig, "04_scoring_gates")


def plot_tool_strategy(rows: list[dict[str, Any]]) -> None:
    labels = [spec.label for spec in RUNS]
    tool_fields = (
        ("Attribute", "tool_attribute", "#376996"),
        ("Query graph", "tool_query_graph", "#76A5AF"),
        ("Intervene", "tool_intervene", "#E07A5F"),
    )
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.18, right=0.89, top=0.79, bottom=0.18)
    y_positions = np.arange(len(labels))
    left = np.zeros(len(labels))
    for tool_label, field, color in tool_fields:
        averages = np.array(
            [
                sum(row[field] for row in model_rows(rows, label))
                / len(model_rows(rows, label))
                for label in labels
            ]
        )
        ax.barh(
            y_positions,
            averages,
            left=left,
            height=0.58,
            color=color,
            label=tool_label,
        )
        left += averages
    for index, total in enumerate(left):
        selected = [
            row
            for row in model_rows(rows, labels[index])
            if not row["baseline_drift"]
        ]
        success_rate = sum(row["success"] for row in selected) / len(selected)
        ax.text(
            total + 1.0,
            index,
            f"{total:.1f} tools · {success_rate:.1%} pass",
            va="center",
            fontweight="bold",
            color=RUNS[index].color,
        )
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.34)
    ax.set_xlabel("Average MCP tool calls per rollout")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3)
    title(
        fig,
        "Tool volume was not the bottleneck",
        "Astra made 2.6× as many circuit calls as Kimi, but had the lowest "
        "gradeable success rate.",
    )
    save_figure(fig, "05_tool_strategy")


def plot_cost_efficiency(rows: list[dict[str, Any]]) -> None:
    labels = [spec.label for spec in RUNS]
    colors = [spec.color for spec in RUNS]
    totals = [
        sum(row["reported_api_cost"] for row in model_rows(rows, label))
        for label in labels
    ]
    cost_per_success = [
        total / sum(row["success"] for row in model_rows(rows, label))
        for total, label in zip(totals, labels, strict=True)
    ]
    median_durations = [
        median(row["duration_minutes"] for row in model_rows(rows, label))
        for label in labels
    ]
    panels = (
        ("Reported API cost", totals, "$", ""),
        ("Cost per success", cost_per_success, "$", ""),
        ("Median wall time", median_durations, "", " min"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.73, bottom=0.24, wspace=0.34)
    for ax, (panel_title, values, prefix, suffix) in zip(axes, panels, strict=True):
        bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.66)
        ax.set_title(panel_title, fontsize=13, pad=12)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.grid(axis="y")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        upper = max(values) * 1.18
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                f"{prefix}{value:,.1f}{suffix}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    title(
        fig,
        "Kimi was the cost-efficiency outlier",
        "Fable won one more rollout, but its reported policy-model spend was "
        "nearly 11× Kimi's.",
    )
    fig.text(
        0.06,
        0.05,
        "Costs are summed from trace usage.cost and exclude Modal GPU/subject-model "
        "compute.",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(fig, "06_cost_efficiency")


def plot_feature_layers(
    traces_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    layers = np.arange(36)
    matrix = np.zeros((len(RUNS), len(layers)))
    late_shares: list[float] = []
    proposal_counts: list[int] = []
    for model_index, spec in enumerate(RUNS):
        counts: Counter[int] = Counter()
        proposals = 0
        for trace in traces_by_model[spec.label]:
            nodes = proposal_nodes(trace)
            if nodes:
                proposals += 1
            for node in nodes:
                match = re.match(r"^(\d+)_(\d+)$", node)
                if match:
                    counts[int(match.group(1))] += 1
        total = sum(counts.values())
        if total:
            matrix[model_index] = [counts[layer] / total for layer in layers]
            late_shares.append(sum(counts[layer] for layer in range(28, 36)) / total)
        else:
            late_shares.append(0.0)
        proposal_counts.append(proposals)

    cmap = LinearSegmentedColormap.from_list(
        "layer_density", [BACKGROUND, "#8CC3B8", "#145E68"]
    )
    fig, ax = plt.subplots(figsize=(14, 5.8))
    fig.subplots_adjust(left=0.18, right=0.94, top=0.72, bottom=0.23)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(matrix.max(), 0.01), aspect="auto")
    ax.set_xticks(np.arange(0, 36, 2), [str(layer) for layer in range(0, 36, 2)])
    ax.set_xlabel("Qwen3-4B layer")
    y_labels = [
        f"{spec.label}  ·  {late_shares[index]:.0%} late-layer"
        for index, spec in enumerate(RUNS)
    ]
    ax.set_yticks(np.arange(len(RUNS)), y_labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks(np.arange(-0.5, 36, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RUNS), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(27.5, color="#E8873A", linewidth=2, linestyle="--")
    for model_index in range(len(RUNS)):
        for layer in range(36):
            share = matrix[model_index, layer]
            if share >= 0.055:
                ax.text(
                    layer,
                    model_index,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if share > 0.09 else TEXT,
                )
    ax.spines[:].set_visible(False)
    title(
        fig,
        "Models searched different parts of the circuit",
        "Share of submitted layer_feature nodes by layer. Kimi and Fable favored "
        "late layers; Astra explored broadly.",
    )
    fig.text(
        0.06,
        0.05,
        "Dashed line marks layers 28–35. Parseable proposals: "
        + ", ".join(
            f"{spec.label} {proposal_counts[index]}/32"
            for index, spec in enumerate(RUNS)
        )
        + ".",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(fig, "07_feature_layer_selection")


def plot_rollout_efficiency(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.16)
    for spec in RUNS:
        selected = model_rows(rows, spec.label)
        failed = [row for row in selected if not row["success"]]
        successful = [row for row in selected if row["success"]]
        ax.scatter(
            [row["tool_calls"] for row in failed],
            [row["duration_minutes"] for row in failed],
            color=spec.color,
            alpha=0.30,
            s=48,
            linewidth=0,
        )
        ax.scatter(
            [row["tool_calls"] for row in successful],
            [row["duration_minutes"] for row in successful],
            color=spec.color,
            marker="*",
            edgecolor=TEXT,
            linewidth=0.6,
            s=150,
            zorder=4,
        )
    ax.set_xlabel("MCP tool calls")
    ax.set_ylabel("Wall-clock minutes")
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=spec.color,
            markersize=8,
            label=spec.label,
        )
        for spec in RUNS
    ]
    result_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="white",
            markeredgecolor=TEXT,
            markersize=12,
            label="Successful",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=INVALID,
            alpha=0.5,
            markersize=8,
            label="Unsuccessful / invalid",
        ),
    ]
    ax.legend(
        handles=model_handles + result_handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncols=5,
    )
    title(
        fig,
        "More experimentation often meant longer—not better—rollouts",
        "Each point is one rollout. Stars mark the 18 successful circuit discoveries.",
    )
    save_figure(fig, "08_rollout_efficiency")


def write_csvs(rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with (MEDIA / "rollouts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for spec in RUNS:
        selected = model_rows(rows, spec.label)
        gradeable = [row for row in selected if not row["baseline_drift"]]
        successes = sum(row["success"] for row in selected)
        total_cost = sum(row["reported_api_cost"] for row in selected)
        summary_rows.append(
            {
                "model": spec.label,
                "rollouts": len(selected),
                "gradeable_rollouts": len(gradeable),
                "successful_rollouts": successes,
                "raw_success_rate": successes / len(selected),
                "gradeable_success_rate": successes / len(gradeable),
                "evidence_passes": sum(row["evidence_pass"] for row in gradeable),
                "holdout_passes": sum(row["holdout_pass"] for row in gradeable),
                "null_passes": sum(row["null_pass"] for row in gradeable),
                "reported_api_cost": total_cost,
                "reported_cost_per_success": total_cost / successes,
                "median_duration_minutes": median(
                    row["duration_minutes"] for row in selected
                ),
                "average_tool_calls": sum(row["tool_calls"] for row in selected)
                / len(selected),
            }
        )
    with (MEDIA / "model_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    task_rows: list[dict[str, Any]] = []
    for task_index, family in task_order(rows):
        task_row: dict[str, Any] = {"task": task_index, "family": family}
        for spec in RUNS:
            selected = [
                row
                for row in rows
                if row["task"] == task_index and row["model"] == spec.label
            ]
            key = spec.label.lower().replace(" ", "_").replace("-", "_")
            task_row[f"{key}_successes"] = sum(row["success"] for row in selected)
            task_row[f"{key}_baseline_drift"] = int(
                bool(selected) and all(row["baseline_drift"] for row in selected)
            )
        task_rows.append(task_row)
    with (MEDIA / "task_model_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)


def split_messages(trace: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    prompt: list[Any] = []
    completion: list[Any] = []
    generation_started = False
    for node in trace.get("nodes") or []:
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            generation_started = True
        if generation_started:
            completion.append(message)
        else:
            prompt.append(message)
    return prompt, completion


def write_viewer_export(
    traces_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    if VIEWER_ROOT.parent.exists():
        shutil.rmtree(VIEWER_ROOT.parent)
    for spec in RUNS:
        traces = traces_by_model[spec.label]
        source_dir = spec.trace_path.parent
        run_dir = VIEWER_ROOT / f"circuit-supernodes--{spec.slug}" / source_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        rewards = [
            float((trace.get("rewards") or {}).get("circuit_faithfulness", 0.0))
            for trace in traces
        ]
        metadata = {
            "env_id": "circuit-supernodes",
            "model": (traces[0].get("agent") or {}).get("model", spec.label),
            "num_examples": 16,
            "rollouts_per_example": 2,
            "avg_reward": sum(rewards) / len(rewards),
            "source_format": "verifiers.v1 traces.jsonl",
            "source_path": str(source_dir.relative_to(ROOT)),
            "tools": traces[0].get("tools") or [],
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        task_occurrences: Counter[int] = Counter()
        with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for trace in traces:
                task_data = (trace.get("task") or {}).get("data") or {}
                task_index = int(task_data["idx"])
                task_occurrences[task_index] += 1
                prompt, completion = split_messages(trace)
                rewards_by_name = trace.get("rewards") or {}
                record = {
                    "env_id": "circuit-supernodes",
                    "model": (trace.get("agent") or {}).get("model", spec.label),
                    "example_id": (
                        f"task-{task_index}-rollout-{task_occurrences[task_index]}"
                    ),
                    "task": task_data,
                    "prompt": prompt,
                    "completion": completion,
                    "reward": float(rewards_by_name.get("circuit_faithfulness", 0.0)),
                    "reward_per_func": rewards_by_name,
                    "metrics": trace.get("metrics") or {},
                    "info": {
                        "trace_id": trace.get("id"),
                        "stop_condition": trace.get("stop_condition"),
                        **(trace.get("info") or {}),
                    },
                    "error": (trace.get("errors") or [None])[0],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        source_log = source_dir / "eval.log"
        if source_log.exists():
            shutil.copy2(source_log, run_dir / "eval.log")


def main() -> None:
    MEDIA.mkdir(exist_ok=True)
    traces_by_model = {spec.label: load_jsonl(spec.trace_path) for spec in RUNS}
    for label, traces in traces_by_model.items():
        if len(traces) != 32:
            raise ValueError(
                f"Expected 32 final traces for {label}, found {len(traces)}"
            )
    rows = flatten_rollouts(traces_by_model)
    setup_style()
    plot_outcomes(rows)
    plot_task_matrix(rows)
    plot_family_performance(rows)
    plot_scoring_gates(rows)
    plot_tool_strategy(rows)
    plot_cost_efficiency(rows)
    plot_feature_layers(traces_by_model)
    plot_rollout_efficiency(rows)
    write_csvs(rows)
    write_viewer_export(traces_by_model)
    print(f"Wrote rollout analysis to {MEDIA}")


if __name__ == "__main__":
    main()
