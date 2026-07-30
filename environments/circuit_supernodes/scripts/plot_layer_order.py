"""Regenerate the FINDINGS.md charts in media/ directly from archived data.

Each figure makes one claim from FINDINGS.md concrete: sec.2 (the v1 reward's
ranking came from two degenerate terms, not the causal one), sec.4-5 (most of
the causal term is a transformer's forward flow, which needs no experiments to
predict), and sec.3 (a GRPO group turns a blind partition into a search). This
keeps them reproducible in the way the *old* media charts were not — pure
Python, no plotting library, and every number traced to an archived run rather
than hand-copied into a script.

    uv run python scripts/plot_layer_order.py

Output is SVG: it renders natively in GitHub-flavoured markdown and any
browser with no added dependency, and (unlike this repo's other scripts) nothing
here touches Modal — every input is already on disk under archive/.
"""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward_components import COMPONENTS, read_traces  # noqa: E402

ARCHIVE = Path(__file__).parent.parent / "archive"
MEDIA = Path(__file__).parent.parent.parent.parent / "media"

MODEL_RUNS = {
    "Claude Opus 5": "opus5",
    "GPT-5.6 Sol": "sol56",
    "Kimi K3": "kimik3",
}
ATTACK_KEYS = {
    "Community partition": "v2/community",
    "Layer-bands heuristic": "v2/layer-bands",
    "Random, best of 2": "v2/random-multistart",
    "Coherence-matched": "v2/coherence-matched",
    "Blind-balanced": "v2/blind-balanced",
    "Layer-spanning (control)": "v2/layer-spanning",
    "Budget-feasible": "v2/budget-feasible",
}

# Literal light-mode hex, not CSS custom properties: this is a static asset
# for the repo's media/ folder, consumed by GitHub's markdown renderer and
# plain image viewers — not a toggleable Artifact page. var() support is
# inconsistent across SVG rasterizers (confirmed: one resolves it to black),
# so literal values from the validated light palette are the correct choice
# here, not a workaround.
STYLE = """
  .viz-root { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .bg { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font-size: 17px; font-weight: 600; }
  .subtitle { fill: #52514e; font-size: 12.5px; }
  .panel-title { fill: #0b0b0b; font-size: 12.5px; font-weight: 600; }
  .axis-label { fill: #898781; font-size: 10.5px; }
  .row-label { fill: #0b0b0b; font-size: 12px; }
  .row-sub { fill: #898781; font-size: 9.5px; }
  .value-label { fill: #0b0b0b; font-size: 11.5px; font-weight: 600; }
  .legend-text { fill: #52514e; font-size: 11px; }
  .grid-line { stroke: #e1e0d9; stroke-width: 1; }
  .baseline { stroke: #c3c2b7; stroke-width: 1; }
  .diag { stroke: #898781; stroke-width: 1.5; stroke-dasharray: 4 4; }
  .leader { stroke: #c3c2b7; stroke-width: 1; }
  .seg-null { fill: #2a78d6; }
  .seg-over { fill: #eb6834; }
  .seg-third { fill: #1baf7a; }
  .footnote { fill: #898781; font-size: 10px; }
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def load_data() -> tuple[dict, dict]:
    models = {}
    for name, slug in MODEL_RUNS.items():
        text = (ARCHIVE / "final" / slug / "traces.jsonl").read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        all_comps = [(t.get("info") or {}).get("reward_components") for t in rows]
        comps = [c for c in all_comps if c and "separation" in c]
        gated = sum(1 for c in all_comps if c and "gated" in c)
        models[name] = {
            "separation": st.mean(c["separation"] for c in comps),
            "null": st.mean(c["separation_layer_null"] for c in comps),
            "over": st.mean(c["separation_over_null"] for c in comps),
            "reward": st.mean(sum(t["rewards"].values()) for t in rows),
            "scored": len(rows) - gated,
            "total": len(rows),
            "sub": f"{len(rows) - gated}/{len(rows)} rollouts, {gated} gated",
        }

    audit = json.loads((ARCHIVE / "oracle_layers_full.json").read_text())
    attacks = {}
    for name, key in ATTACK_KEYS.items():
        scored = [
            row["scores"][key] for row in audit if "separation" in row["scores"][key]
        ]
        attacks[name] = {
            "separation": st.mean(s["separation"] for s in scored),
            "null": st.mean(s["separation_layer_null"] for s in scored),
            "over": st.mean(s["separation_over_null"] for s in scored),
        }
    return models, attacks


# ---------------------------------------------------------------------------
# Chart 1 — stacked composition: layer-order null vs. the residual over it
# ---------------------------------------------------------------------------


def chart1(models: dict, attacks: dict) -> str:
    model_rows = sorted(models.items(), key=lambda kv: -kv[1]["separation"])
    attack_rows = sorted(attacks.items(), key=lambda kv: -kv[1]["separation"])

    domain_max = 0.65
    plot_x0, plot_w = 250, 430
    row_h, bar_h = 30, 16

    def sx(v):
        return plot_x0 + v / domain_max * plot_w

    ticks = [0.0, 0.15, 0.30, 0.45, 0.60]
    parts: list[str] = []

    def row(y: float, name: str, sub: str | None, stats: dict) -> None:
        w1 = stats["null"] / domain_max * plot_w
        w2 = stats["over"] / domain_max * plot_w
        by = y - bar_h / 2
        title = (
            f"{esc(name)}: separation {stats['separation']:.3f} "
            f"(layer-order {stats['null']:.3f}, beyond it {stats['over']:.3f})"
        )
        parts.append(f"<g><title>{title}</title>")
        parts.append(
            f'<rect class="seg-null" x="{plot_x0:.1f}" y="{by:.1f}" '
            f'width="{w1:.1f}" height="{bar_h}" rx="2"/>'
        )
        if w2 > 0.5:
            parts.append(
                f'<rect class="seg-over" x="{plot_x0 + w1:.1f}" y="{by:.1f}" '
                f'width="{w2:.1f}" height="{bar_h}" rx="2"/>'
            )
        parts.append("</g>")
        parts.append(
            f'<text class="row-label" x="{plot_x0 - 12}" y="{y - 2:.1f}" '
            f'text-anchor="end">{esc(name)}</text>'
        )
        if sub:
            parts.append(
                f'<text class="row-sub" x="{plot_x0 - 12}" y="{y + 10:.1f}" '
                f'text-anchor="end">{esc(sub)}</text>'
            )
        parts.append(
            f'<text class="value-label" x="{plot_x0 + w1 + w2 + 8:.1f}" '
            f'y="{y + 4:.1f}">{stats["separation"]:.3f}</text>'
        )

    y = 132
    grid_top = 108
    parts.append(
        f'<text class="panel-title" x="24" y="{y - 20}">'
        "Frontier models (xhigh reasoning, 150 features)</text>"
    )
    for name, stats in model_rows:
        row(y, name, stats["sub"], stats)
        y += row_h
    y += 26

    parts.append(
        f'<text class="panel-title" x="24" y="{y - 20}">'
        "Scripted attacks &amp; heuristics &#8212; zero experiments run</text>"
    )
    for name, stats in attack_rows:
        row(y, name, None, stats)
        y += row_h

    grid_bottom = y - row_h / 2 + 14
    grid_lines = []
    for t in ticks:
        x = sx(t)
        grid_lines.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{grid_top:.1f}" '
            f'x2="{x:.1f}" y2="{grid_bottom:.1f}"/>'
        )
        grid_lines.append(
            f'<text class="axis-label" x="{x:.1f}" y="{grid_top - 8:.1f}" '
            f'text-anchor="middle">{t:.2f}</text>'
        )
    axis_label = (
        f'<text class="axis-label" x="{plot_x0 + plot_w / 2:.1f}" '
        f'y="{grid_top - 24:.1f}" text-anchor="middle" style="font-size:11px">'
        "separation (causal term of the reward)</text>"
    )

    svg_h = grid_bottom + 56
    legend_y = 66
    body = "\n".join(
        [
            f'<rect class="bg" x="0" y="0" width="900" height="{svg_h:.0f}"/>',
            '<text class="title" x="24" y="30">'
            "Almost all of the reward's causal credit is free</text>",
            '<text class="subtitle" x="24" y="50">separation, split into what a '
            "layer-ordering already predicts and what is left over &#8212; twenty "
            "held-out graphs</text>",
            f'<rect x="24" y="{legend_y - 11}" width="14" height="14" '
            'rx="2" class="seg-null"/>',
            f'<text class="legend-text" x="44" y="{legend_y}">'
            "layer-order null (0 experiments needed)</text>",
            f'<rect x="330" y="{legend_y - 11}" width="14" height="14" '
            'rx="2" class="seg-over"/>',
            f'<text class="legend-text" x="350" y="{legend_y}">'
            "beyond the layer order (the real signal)</text>",
            *grid_lines,
            axis_label,
            *parts,
            f'<line class="baseline" x1="{plot_x0}" y1="{grid_top:.1f}" '
            f'x2="{plot_x0}" y2="{grid_bottom:.1f}"/>',
            f'<text class="footnote" x="24" y="{svg_h - 14:.0f}">'
            "circuit-supernodes reward v2.1 &#183; scripts/plot_layer_order.py "
            "from archive/final/*, archive/oracle_layers_full.json</text>",
        ]
    )
    return (
        f'<svg viewBox="0 0 900 {svg_h:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'class="viz-root" role="img" '
        'aria-label="Bar chart splitting causal separation into a layer-order '
        "null and the residual beyond it, for three frontier models and seven "
        'scripted attacks">'
        f"<style>{STYLE}</style>{body}</svg>"
    )


# ---------------------------------------------------------------------------
# Chart 2 — separation vs. the layer-order null, with the y=x reference
# ---------------------------------------------------------------------------


def chart2(models: dict, attacks: dict) -> str:
    domain_max = 0.65
    margin_l, margin_b, margin_t, margin_r = 66, 54, 96, 40
    plot_w = plot_h = 460
    x0, y0 = margin_l, margin_t
    # Wider than the plot needs, so the header text has somewhere to go — a
    # viewBox sized only to the plot clips any title/subtitle wider than it.
    width = max(margin_l + plot_w + margin_r, 860)
    height = margin_t + plot_h + margin_b

    def sx(v):
        return x0 + v / domain_max * plot_w

    def sy(v):
        return y0 + plot_h - v / domain_max * plot_h

    ticks = [0.0, 0.15, 0.30, 0.45, 0.60]
    parts: list[str] = []
    for t in ticks:
        gx, gy = sx(t), sy(t)
        parts.append(
            f'<line class="grid-line" x1="{gx:.1f}" y1="{y0:.1f}" '
            f'x2="{gx:.1f}" y2="{y0 + plot_h:.1f}"/>'
        )
        parts.append(
            f'<line class="grid-line" x1="{x0:.1f}" y1="{gy:.1f}" '
            f'x2="{x0 + plot_w:.1f}" y2="{gy:.1f}"/>'
        )
        parts.append(
            f'<text class="axis-label" x="{gx:.1f}" y="{y0 + plot_h + 18:.1f}" '
            f'text-anchor="middle">{t:.2f}</text>'
        )
        parts.append(
            f'<text class="axis-label" x="{x0 - 8:.1f}" y="{gy + 3:.1f}" '
            f'text-anchor="end">{t:.2f}</text>'
        )

    parts.append(
        f'<line class="diag" x1="{sx(0):.1f}" y1="{sy(0):.1f}" '
        f'x2="{sx(domain_max):.1f}" y2="{sy(domain_max):.1f}"/>'
    )
    # A caption placed ON the diagonal cuts through the point labels near it;
    # the top-left plot corner (high separation, low layer-order null) has no
    # data in it, so the explanation goes there instead, horizontal and clear.
    caption_lines = [
        "On the dashed line, separation",
        "equals the layer-order null &#8212;",
        "zero credit for anything else.",
    ]
    for i, line in enumerate(caption_lines):
        cap_y = y0 + 22 + 14 * i
        parts.append(
            f'<text class="footnote" x="{x0 + 14:.1f}" y="{cap_y:.1f}">{line}</text>'
        )

    # Four points (Community, Layer-bands, Opus 5, Sol) sit inside a 50x40px
    # box, and two more (Layer-spanning, Budget-feasible) sit 8px apart — no
    # per-point offset avoids collisions there. Those get a callout: a fixed
    # label position with a thin leader line back to the real dot. Everything
    # else is far enough apart for a simple inline offset.
    label_offsets = {
        "Kimi K3": (10, 16),
        "Random, best of 2": (10, -10),
        "Coherence-matched": (8, -16),
    }
    callouts = {
        "Community partition": (562, 148),
        "Layer-bands heuristic": (562, 170),
        "Claude Opus 5": (562, 192),
        "GPT-5.6 Sol": (562, 214),
        "Blind-balanced": (150, 434),
        "Layer-spanning (control)": (150, 456),
        "Budget-feasible": (150, 478),
    }

    points = [(name, stats, "model") for name, stats in models.items()]
    points += [(name, stats, "attack") for name, stats in attacks.items()]

    for name, stats, kind in points:
        cx, cy = sx(stats["null"]), sy(stats["separation"])
        cls = "seg-over" if kind == "model" else "seg-null"
        title = (
            f"{esc(name)}: separation {stats['separation']:.3f}, layer-order null "
            f"{stats['null']:.3f}, beyond it {stats['over']:.3f}"
        )
        parts.append(f"<g><title>{title}</title>")
        parts.append(
            f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="6" '
            'stroke="#fcfcfb" stroke-width="1.5"/>'
        )
        parts.append("</g>")
        if name in callouts:
            lx, ly = callouts[name]
            parts.append(
                f'<line class="leader" x1="{cx:.1f}" y1="{cy:.1f}" '
                f'x2="{lx - 6:.1f}" y2="{ly - 3:.1f}"/>'
            )
            parts.append(
                f'<text class="row-label" x="{lx:.1f}" y="{ly:.1f}">{esc(name)}</text>'
            )
            continue
        dx, dy = label_offsets.get(name, (10, -8))
        parts.append(
            f'<text class="row-label" x="{cx + dx:.1f}" y="{cy + dy:.1f}" '
            f'text-anchor="start">{esc(name)}</text>'
        )

    legend_y = 66
    body = "\n".join(
        [
            f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
            '<text class="title" x="24" y="30">'
            "Frontier models land on the same line as the attacks</text>",
            '<text class="subtitle" x="24" y="50">separation vs. what a layer '
            "ordering predicts with zero experiments &#8212; height above the "
            "line is the real signal</text>",
            f'<circle cx="30" cy="{legend_y}" r="6" class="seg-over"/>',
            f'<text class="legend-text" x="42" y="{legend_y + 4}">'
            "Frontier model, xhigh</text>",
            f'<circle cx="220" cy="{legend_y}" r="6" class="seg-null"/>',
            f'<text class="legend-text" x="232" y="{legend_y + 4}">'
            "Scripted attack / heuristic</text>",
            *parts,
            f'<text class="axis-label" x="{x0 + plot_w / 2:.1f}" y="{height - 14}" '
            'text-anchor="middle" style="font-size:11px">'
            "layer-order null (predictable with 0 experiments)</text>",
            f'<text class="axis-label" x="16" y="{y0 + plot_h / 2:.1f}" '
            'text-anchor="middle" '
            f'transform="rotate(-90 16 {y0 + plot_h / 2:.1f})" '
            'style="font-size:11px">measured separation</text>',
        ]
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'class="viz-root" role="img" '
        'aria-label="Scatter plot of separation against the layer-order null '
        "for three frontier models and seven scripted attacks, with a y=x "
        'reference line">'
        f"<style>{STYLE}</style>{body}</svg>"
    )


# ---------------------------------------------------------------------------
# Chart 3 — the v1 reward's three terms, one archived rollout per model
# ---------------------------------------------------------------------------

V1_RUNS = {
    "Claude Fable 5": "fable",
    "GPT-5.6 Sol": "sol",
    "Claude Opus 4.8": "opus",
}
COMPONENT_CLASS = {
    "separation": "seg-null",
    "coverage": "seg-over",
    "completeness": "seg-third",
}
COMPONENT_LABEL = {
    "separation": "separation (the only causal term)",
    "coverage": "coverage",
    "completeness": "completeness",
}


def load_v1_components() -> dict:
    out = {}
    for name, slug in V1_RUNS.items():
        rewards, comps = read_traces(ARCHIVE / "eval15" / slug / "traces.jsonl")
        out[name] = {
            "reward": st.mean(rewards),
            **{k: st.mean(c[k] for c in comps) for k in COMPONENTS},
        }
    return out


def chart3(models: dict) -> str:
    rows = sorted(models.items(), key=lambda kv: -kv[1]["reward"])

    plot_x0, plot_w = 190, 480
    bar_h, bar_gap, group_gap = 15, 3, 22

    def sx(v):
        return plot_x0 + v * plot_w

    parts: list[str] = []
    y = 118
    group_top = {}
    for name, stats in rows:
        group_top[name] = y
        for comp in COMPONENTS:
            w = stats[comp] * plot_w
            parts.append(
                f'<rect class="{COMPONENT_CLASS[comp]}" x="{plot_x0}" y="{y:.1f}" '
                f'width="{w:.1f}" height="{bar_h}" rx="2"/>'
            )
            parts.append(
                f'<text class="value-label" x="{plot_x0 + w + 8:.1f}" '
                f'y="{y + bar_h - 3:.1f}" style="font-size:10.5px;font-weight:400">'
                f"{stats[comp]:.3f}</text>"
            )
            y += bar_h + bar_gap
        y += group_gap

    labels = []
    for name, stats in rows:
        top = group_top[name]
        mid = top + 1.5 * (bar_h + bar_gap)
        labels.append(
            f'<text class="row-label" x="{plot_x0 - 12}" y="{mid + 4:.1f}" '
            f'text-anchor="end" style="font-weight:600">{esc(name)}</text>'
        )
        labels.append(
            f'<text class="row-sub" x="{plot_x0 - 12}" y="{mid + 18:.1f}" '
            f'text-anchor="end">reward {stats["reward"]:.3f}</text>'
        )

    grid_top, grid_bottom = 108, y - group_gap + 10
    ticks = [0.0, 0.25, 0.50, 0.75, 1.00]
    grid = []
    for t in ticks:
        x = sx(t)
        grid.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{grid_top:.1f}" '
            f'x2="{x:.1f}" y2="{grid_bottom:.1f}"/>'
        )
        grid.append(
            f'<text class="axis-label" x="{x:.1f}" y="{grid_top - 8:.1f}" '
            f'text-anchor="middle">{t:.2f}</text>'
        )

    svg_h = grid_bottom + 46
    legend_y = 66
    legend = []
    lx = 24
    for comp in COMPONENTS:
        legend.append(
            f'<rect x="{lx}" y="{legend_y - 11}" width="14" height="14" rx="2" '
            f'class="{COMPONENT_CLASS[comp]}"/>'
        )
        label = COMPONENT_LABEL[comp]
        legend.append(
            f'<text class="legend-text" x="{lx + 20}" y="{legend_y}">{label}</text>'
        )
        lx += 26 + 7.2 * len(label)

    body = "\n".join(
        [
            f'<rect class="bg" x="0" y="0" width="820" height="{svg_h:.0f}"/>',
            '<text class="title" x="24" y="30">'
            "The old reward's ranking was not about circuit quality</text>",
            '<text class="subtitle" x="24" y="50">'
            "separation barely moves between models; coverage and completeness "
            "did the ranking, and both hit 1.0 by grouping everything shown"
            "</text>",
            *legend,
            *grid,
            *parts,
            *labels,
            f'<line class="baseline" x1="{plot_x0}" y1="{grid_top:.1f}" '
            f'x2="{plot_x0}" y2="{grid_bottom:.1f}"/>',
            f'<text class="footnote" x="24" y="{svg_h - 14:.0f}">'
            "circuit-supernodes reward v1, archive/eval15 (one rollout per "
            "model) &#8212; scripts/plot_layer_order.py from "
            "scripts/reward_components.py</text>",
        ]
    )
    return (
        f'<svg viewBox="0 0 820 {svg_h:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'class="viz-root" role="img" '
        'aria-label="Bar chart of the v1 reward'
        "'"
        "s three terms per model, showing separation nearly flat while "
        'coverage and completeness vary and drive the ranking">'
        f"<style>{STYLE}</style>{body}</svg>"
    )


# ---------------------------------------------------------------------------
# Chart 4 — best-of-k: a GRPO group turns a blind partition into a search
# ---------------------------------------------------------------------------


def expected_best_of_k(samples: list[float], k: int) -> float:
    """E[max of k draws without replacement] from a finite sample, exactly —
    not simulated. Same estimator oracle_audit.py's own summary uses."""
    from math import comb

    n = len(samples)
    ordered = sorted(samples)
    return sum(ordered[i] * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)


def load_search_curve() -> dict:
    audit = json.loads((ARCHIVE / "oracle_bestof8.json").read_text())
    per_graph = [row["scores"]["v2/random-multistart"]["all_rewards"] for row in audit]
    n = len(per_graph[0])
    ks = [k for k in (1, 2, 4, 8) if k <= n]
    return {k: st.mean(expected_best_of_k(s, k) for s in per_graph) for k in ks}


def declutter(items: list[tuple[str, float]], min_gap: float) -> dict[str, float]:
    """Push labels apart along one axis, in true-value order, so two nearly
    equal values (e.g. 0.296 and 0.303) don't render on top of each other.

    A per-item offset keyed only to its own rank (as chart4's first version
    used) does not do this: two labels a few pixels apart start a few pixels
    apart no matter how they're each nudged individually. This walks the
    sorted list and enforces a minimum gap against the *previous* placed
    label, which is the actual definition of "not overlapping."
    """
    ordered = sorted(items, key=lambda kv: kv[1])
    placed: dict[str, float] = {}
    prev = None
    for name, y in ordered:
        y = max(y, prev + min_gap) if prev is not None else y
        placed[name] = y
        prev = y
    return placed


def chart4(curve: dict, models: dict) -> str:
    margin_l, margin_b, margin_t, margin_r = 60, 70, 96, 210
    plot_w, plot_h = 420, 380
    x0, y0 = margin_l, margin_t
    width = max(margin_l + plot_w + margin_r, 860)
    height = margin_t + plot_h + margin_b
    y_max = 0.65

    ks = sorted(curve)  # equally spaced by rank, not by raw k — 1,2,4,8 are
    # not equal steps, and placing them proportionally crowds the low end.

    def sx(k):
        return x0 + ks.index(k) / (len(ks) - 1) * plot_w

    def sy(v):
        return y0 + plot_h - v / y_max * plot_h

    parts: list[str] = []
    for t in (0.0, 0.15, 0.30, 0.45, 0.60):
        gy = sy(t)
        parts.append(
            f'<line class="grid-line" x1="{x0:.1f}" y1="{gy:.1f}" '
            f'x2="{x0 + plot_w:.1f}" y2="{gy:.1f}"/>'
        )
        parts.append(
            f'<text class="axis-label" x="{x0 - 8:.1f}" y="{gy + 3:.1f}" '
            f'text-anchor="end">{t:.2f}</text>'
        )
    for k in ks:
        gx = sx(k)
        parts.append(
            f'<text class="axis-label" x="{gx:.1f}" y="{y0 + plot_h + 18:.1f}" '
            f'text-anchor="middle">{k}</text>'
        )

    points = " ".join(f"{sx(k):.1f},{sy(curve[k]):.1f}" for k in ks)
    parts.append(
        f'<polyline points="{points}" fill="none" class="seg-null" '
        'style="stroke-width:2.5"/>'
    )
    for k in ks:
        parts.append(
            f'<circle cx="{sx(k):.1f}" cy="{sy(curve[k]):.1f}" r="5" class="seg-null" '
            f'stroke="#fcfcfb" stroke-width="1.5"/>'
        )
        if k == ks[0]:
            # Centering above the leftmost point puts the label astride the
            # y-axis, where it collides with that row's gridline value — pull
            # it to the right of the point instead.
            parts.append(
                f'<text class="value-label" x="{sx(k) + 10:.1f}" '
                f'y="{sy(curve[k]) - 8:.1f}" text-anchor="start" '
                f'style="font-size:11px">{curve[k]:.3f}</text>'
            )
        else:
            parts.append(
                f'<text class="value-label" x="{sx(k):.1f}" '
                f'y="{sy(curve[k]) - 12:.1f}" text-anchor="middle" '
                f'style="font-size:11px">{curve[k]:.3f}</text>'
            )

    g8_x = sx(8) if 8 in curve else None
    if g8_x is not None:
        parts.append(
            f'<line x1="{g8_x:.1f}" y1="{y0:.1f}" x2="{g8_x:.1f}" '
            f'y2="{y0 + plot_h:.1f}" class="leader" stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<text class="row-sub" x="{g8_x:.1f}" y="{y0 - 8:.1f}" '
            'text-anchor="middle">planned GRPO group size</text>'
        )

    model_lines = []
    true_y = {name: sy(stats["reward"]) for name, stats in models.items()}
    label_y = declutter(list(true_y.items()), min_gap=16)
    for name, y in true_y.items():
        model_lines.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + plot_w:.1f}" y2="{y:.1f}" '
            'class="baseline" stroke-dasharray="2 4"/>'
        )
        ly = label_y[name]
        model_lines.append(
            f'<line class="leader" x1="{x0 + plot_w:.1f}" y1="{y:.1f}" '
            f'x2="{x0 + plot_w + 10:.1f}" y2="{ly:.1f}"/>'
        )
        model_lines.append(
            f'<text class="row-sub" x="{x0 + plot_w + 14:.1f}" y="{ly + 3:.1f}">'
            f"{esc(name)} {models[name]['reward']:.3f}</text>"
        )

    legend_y = 66
    body = "\n".join(
        [
            f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
            '<text class="title" x="24" y="30">'
            "A GRPO group turns a blind partition into a search</text>",
            '<text class="subtitle" x="24" y="50">expected reward of the best '
            "of k random partitions &#8212; dashed lines are what each model "
            "actually scored, on the same twenty graphs</text>",
            f'<circle cx="30" cy="{legend_y}" r="5" class="seg-null"/>',
            f'<text class="legend-text" x="42" y="{legend_y + 4}">'
            "E[best of k blind partitions]</text>",
            *model_lines,
            *parts,
            f'<line class="baseline" x1="{x0}" y1="{y0:.1f}" x2="{x0}" '
            f'y2="{y0 + plot_h:.1f}"/>',
            f'<text class="axis-label" x="{x0 + plot_w / 2:.1f}" '
            f'y="{y0 + plot_h + 40:.1f}" text-anchor="middle" style="font-size:11px">'
            "k (rollouts sampled per graph)</text>",
            f'<text class="footnote" x="24" y="{height - 12}">'
            "circuit-supernodes reward v2.1 &#8212; scripts/plot_layer_order.py "
            "from archive/oracle_bestof8.json, archive/final/*</text>",
        ]
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'class="viz-root" role="img" '
        'aria-label="Line chart of expected best-of-k reward for blind '
        "partitions against each rollout budget, with model rewards as "
        'reference lines">'
        f"<style>{STYLE}</style>{body}</svg>"
    )


# ---------------------------------------------------------------------------
# Chart 5 — gate pass rate, old prompt vs. corrected prompt + strict gates
# ---------------------------------------------------------------------------

GATE_LABEL = {
    "fable": "Claude Fable 5",
    "sol": "GPT-5.6 Sol",
    "opus": "Claude Opus 4.8",
}


def load_gate_pass(models: dict) -> dict:
    before = json.loads((ARCHIVE / "gate_pass_v21.json").read_text())
    return {
        "before": {
            GATE_LABEL[r["run"]]: (r["scored"], r["total"])
            for r in before
            if r["run"] in GATE_LABEL
        },
        "after": {
            name: (stats["scored"], stats["total"]) for name, stats in models.items()
        },
    }


def chart5(gates: dict) -> str:
    plot_x0, plot_w = 250, 400
    row_h, bar_h = 30, 16
    domain_max = 100

    def sx(pct):
        return plot_x0 + pct / domain_max * plot_w

    parts: list[str] = []
    y = 132
    grid_top = 108

    def panel(title: str, rows: dict) -> None:
        nonlocal y
        parts.append(
            f'<text class="panel-title" x="24" y="{y - 20}">{esc(title)}</text>'
        )
        for name, (scored, total) in rows.items():
            pct = 100 * scored / total
            by = y - bar_h / 2
            parts.append(
                f'<rect class="seg-null" x="{plot_x0}" y="{by:.1f}" '
                f'width="{sx(pct) - plot_x0:.1f}" height="{bar_h}" rx="2"/>'
            )
            parts.append(
                f'<text class="row-label" x="{plot_x0 - 12}" y="{y - 2:.1f}" '
                f'text-anchor="end">{esc(name)}</text>'
            )
            parts.append(
                f'<text class="row-sub" x="{plot_x0 - 12}" y="{y + 10:.1f}" '
                f'text-anchor="end">{scored}/{total} passed the gates</text>'
            )
            parts.append(
                f'<text class="value-label" x="{sx(pct) + 8:.1f}" y="{y + 4:.1f}">'
                f"{pct:.0f}%</text>"
            )
            y += row_h
        y += 26

    panel(
        "Old prompt, lenient gates (archive/eval15, rescored under v2.1)",
        gates["before"],
    )
    panel(
        "Corrected prompt, strict gates (this project's final xhigh runs)",
        gates["after"],
    )

    grid_bottom = y - row_h - 4
    ticks = [0, 25, 50, 75, 100]
    grid = []
    for t in ticks:
        x = sx(t)
        grid.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{grid_top:.1f}" '
            f'x2="{x:.1f}" y2="{grid_bottom:.1f}"/>'
        )
        grid.append(
            f'<text class="axis-label" x="{x:.1f}" y="{grid_top - 8:.1f}" '
            f'text-anchor="middle">{t}%</text>'
        )

    svg_h = grid_bottom + 46
    body = "\n".join(
        [
            f'<rect class="bg" x="0" y="0" width="820" height="{svg_h:.0f}"/>',
            '<text class="title" x="24" y="30">'
            "Strict gates plus a stated contract, and models clear it</text>",
            '<text class="subtitle" x="24" y="50">share of rollouts whose '
            "proposal was not rejected by the reward's membership, topology "
            "or dominance gates</text>",
            *grid,
            *parts,
            f'<line class="baseline" x1="{plot_x0}" y1="{grid_top:.1f}" '
            f'x2="{plot_x0}" y2="{grid_bottom:.1f}"/>',
            f'<text class="footnote" x="24" y="{svg_h - 14:.0f}">'
            "circuit-supernodes reward v2.1 &#8212; scripts/plot_layer_order.py "
            "from archive/gate_pass_v21.json, archive/final/*</text>",
        ]
    )
    return (
        f'<svg viewBox="0 0 820 {svg_h:.0f}" xmlns="http://www.w3.org/2000/svg" '
        'class="viz-root" role="img" '
        'aria-label="Bar chart of gate pass rate before and after the '
        'topology-gate fix and corrected prompt">'
        f"<style>{STYLE}</style>{body}</svg>"
    )


def main() -> None:
    models, attacks = load_data()
    charts = {
        "03_layer_order_composition.svg": chart1(models, attacks),
        "04_separation_vs_layer_null.svg": chart2(models, attacks),
        "05_v1_reward_was_not_causal.svg": chart3(load_v1_components()),
        "06_best_of_k_search.svg": chart4(load_search_curve(), models),
        "07_gate_pass_rate.svg": chart5(load_gate_pass(models)),
    }
    for name, svg in charts.items():
        (MEDIA / name).write_text(svg)
        print(f"wrote {MEDIA / name}")


if __name__ == "__main__":
    main()
