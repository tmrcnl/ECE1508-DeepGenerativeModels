"""Turn ``results.json`` into the tables and figures for the report and slides.

Emits:

* ``results.md`` / ``results.tex`` -- the main comparison table
* ``fig_perplexity.png``           -- quality by variant
* ``fig_capacity.png``             -- virtual experts vs. per-token compute
* ``fig_scaling.png``              -- AAG scaling curve (two panels, shared x)
* ``fig_routing.png``              -- routing entropy by layer

Figure conventions: one measure per axis (never a second y-scale), categorical
colour assigned by model family in fixed order, direct value labels so nothing
depends on colour alone, recessive grid and axes.

Author: Mohammad Al Dridi
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light surface #fcfcfb): blue / orange / aqua.
# All-pairs CVD dE 9.2, normal-vision dE 24.0. Aqua sits below 3:1 contrast, so
# every chart below carries direct labels.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcd8"
FAMILY_COLOR = {"dense": "#2a78d6", "moe": "#eb6834", "aag": "#1baf7a"}
FAMILY_LABEL = {"dense": "Dense GPT-2", "moe": "Mixture of Experts", "aag": "AAG (chunked)"}


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=1.0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)


def _figure(width=8.0, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def load(results_path):
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    ok = [r for r in payload["results"] if "error" not in r]
    return payload, ok


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

COLUMNS = [
    ("Variant", lambda r: r["label"]),
    ("Perplexity", lambda r: f"{r['quality']['perplexity']:.2f}"),
    ("Stored MLP", lambda r: f"{r['capacity']['mlp_stored_params'] / 1e6:.1f}M"),
    ("Active/token", lambda r: f"{r['capacity']['mlp_active_params_per_token'] / 1e6:.1f}M"),
    ("Compute", lambda r: f"{r['_flops_mult']:.2f}x"),
    ("Virtual experts", lambda r: _pow10(r["capacity"]["log10_virtual_experts"])),
    ("Routing entropy", lambda r: _ratio(r["routing"]["entropy_ratio"])),
    ("tok/s", lambda r: f"{r['speed']['throughput_tok_s']:.0f}"),
]


def _pow10(log10_value):
    if log10_value <= 0:
        return "1"
    if log10_value < 6:
        return f"{10 ** log10_value:,.0f}"
    return f"1e{log10_value:.0f}"


def _ratio(value):
    return "n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def annotate(results):
    """Add per-token compute relative to the dense baseline."""
    dense = next(
        (r for r in results if r["family"] == "dense"),
        None,
    )
    reference = (
        dense["capacity"]["mlp_active_params_per_token"] if dense else None
    )
    for r in results:
        active = r["capacity"]["mlp_active_params_per_token"]
        r["_flops_mult"] = active / reference if reference else float("nan")
    return results


def markdown_table(results):
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    rows = [
        "| " + " | ".join(fn(r) for _, fn in COLUMNS) + " |"
        for r in results
    ]
    return "\n".join([header, rule, *rows])


def latex_table(results, caption="Model comparison on the frozen Alpaca split.",
                label="tab:results"):
    spec = "l" + "r" * (len(COLUMNS) - 1)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
        " & ".join(name for name, _ in COLUMNS) + r" \\", r"\midrule",
    ]
    for r in results:
        cells = [fn(r).replace("x", r"$\times$").replace("%", r"\%") for _, fn in COLUMNS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_perplexity(results, out_path):
    ordered = sorted(results, key=lambda r: r["quality"]["perplexity"])
    labels = [r["label"] for r in ordered]
    values = [r["quality"]["perplexity"] for r in ordered]
    colors = [FAMILY_COLOR[r["family"]] for r in ordered]

    fig, ax = _figure(height=0.55 * len(ordered) + 2.0)
    positions = range(len(ordered))
    ax.barh(list(positions), values, height=0.62, color=colors, zorder=3)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    span = max(values) if values else 1.0
    for y, value in zip(positions, values):
        ax.text(value + span * 0.012, y, f"{value:.2f}", va="center",
                color=INK, fontsize=9, fontweight="bold")
    ax.set_xlim(0, span * 1.15)

    _style(ax, xlabel="Validation perplexity (lower is better)",
           title="Response-token perplexity on the frozen Alpaca split")
    _family_legend(ax, ordered)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def fig_capacity(results, out_path):
    """The proposal's central claim, as one picture.

    x: arithmetic actually performed per token, relative to vanilla GPT-2.
    y: how many distinct weight configurations the model can assemble.
    Up and to the left is better -- more expressive routing for less compute.
    """
    fig, ax = _figure(height=5.2)

    xs = [r["_flops_mult"] for r in results]
    ys = [max(r["capacity"]["log10_virtual_experts"], 0.0) for r in results]
    x_lo, x_hi = min(xs), max(xs)
    x_pad = max((x_hi - x_lo) * 0.18, 0.25)
    ax.set_xlim(x_lo - x_pad * 0.5, x_hi + x_pad)
    y_hi = max(ys) if ys else 1.0
    ax.set_ylim(-y_hi * 0.10, y_hi * 1.22)

    ax.axvline(1.0, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("vanilla GPT-2 compute", (1.0, y_hi * 1.19),
                textcoords="offset points", xytext=(6, 0),
                color=INK_SOFT, fontsize=8, va="top")

    midpoint = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    for r, x, y in zip(results, xs, ys):
        ax.scatter([x], [y], s=140, color=FAMILY_COLOR[r["family"]],
                   edgecolors=SURFACE, linewidths=2.0, zorder=3)
        # Points past the midline label leftwards so nothing runs off the edge.
        right_half = x > midpoint
        ax.annotate(
            r["label"], (x, y), textcoords="offset points",
            xytext=(-12 if right_half else 12, 6),
            ha="right" if right_half else "left",
            color=INK, fontsize=9, zorder=4,
        )

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    _style(ax,
           xlabel="MLP arithmetic per token, relative to vanilla GPT-2",
           ylabel="Virtual experts per model (log$_{10}$)",
           title="Routing capacity bought per unit of compute")
    _family_legend(ax, results, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def fig_scaling(results, out_path):
    """AAG scaling curve. Two stacked panels rather than a second y-axis."""
    aag = sorted(
        (r for r in results
         if r["family"] == "aag" and r["meta"].get("pretrained_init", True)),
        key=lambda r: r["meta"]["num_chunks"],
    )
    if len(aag) < 2:
        return False

    chunks = [r["meta"]["num_chunks"] for r in aag]
    perplexity = [r["quality"]["perplexity"] for r in aag]
    capacity = [r["capacity"]["log10_virtual_experts"] for r in aag]
    color = FAMILY_COLOR["aag"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6.4), dpi=200, sharex=True,
        gridspec_kw={"hspace": 0.22}, layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)

    top.plot(chunks, perplexity, color=color, linewidth=2.0,
             marker="o", markersize=8, markeredgecolor=SURFACE,
             markeredgewidth=2.0, zorder=3)
    for x, y in zip(chunks, perplexity):
        top.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 11), ha="center", color=INK, fontsize=9)
    top.grid(True, color=GRID, linewidth=0.8, zorder=0)
    top.set_axisbelow(True)
    _style(top, ylabel="Validation perplexity",
           title="AAG scaling: capacity grows exponentially, cost does not")

    bottom.plot(chunks, capacity, color=color, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=3)
    for x, y in zip(chunks, capacity):
        bottom.annotate(f"1e{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", color=INK, fontsize=9)
    bottom.grid(True, color=GRID, linewidth=0.8, zorder=0)
    bottom.set_axisbelow(True)
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(chunks)
    bottom.set_xticklabels([str(c) for c in chunks])
    _style(bottom, xlabel="Chunks per projection",
           ylabel="Virtual experts (log$_{10}$)")

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_routing(results, out_path, max_series=4):
    """Per-layer routing entropy. Flat and high means experts stay in use."""
    series = [r for r in results if r["routing"]["per_layer"]][:max_series]
    if not series:
        return False

    fig, ax = _figure(height=4.6)
    for r in series:
        per_layer = r["routing"]["per_layer"]
        layers = sorted(int(k) for k in per_layer)
        max_entropy = r["routing"]["max_entropy"] or 1.0
        values = [per_layer[str(i)]["entropy"] / max_entropy
                  if str(i) in per_layer else per_layer[i]["entropy"] / max_entropy
                  for i in layers]
        ax.plot(layers, values, color=FAMILY_COLOR[r["family"]], linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=r["label"], zorder=3)
        ax.annotate(r["label"], (layers[-1], values[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    color=INK, fontsize=8, va="center")

    ax.axhline(1.0, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(0, 1.01, "perfectly balanced", color=INK_SOFT, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style(ax, xlabel="Transformer block", ylabel="Routing entropy / maximum",
           title="Are the experts staying in use, or collapsing?")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return True


def _family_legend(ax, results, loc="lower right"):
    seen = []
    for r in results:
        if r["family"] not in seen:
            seen.append(r["family"])
    if len(seen) < 2:
        return
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
        for f in seen
    ]
    legend = ax.legend(handles=handles, frameon=False, fontsize=9, loc=loc)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)


# --------------------------------------------------------------------------

def build_all(results_path, out_dir=None):
    results_path = Path(results_path)
    out_dir = Path(out_dir or results_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, results = load(results_path)
    if not results:
        raise SystemExit(f"no successful results in {results_path}")
    results = annotate(results)

    (out_dir / "results.md").write_text(markdown_table(results), encoding="utf-8")
    (out_dir / "results.tex").write_text(latex_table(results), encoding="utf-8")

    written = ["results.md", "results.tex"]
    fig_perplexity(results, out_dir / "fig_perplexity.png")
    written.append("fig_perplexity.png")
    fig_capacity(results, out_dir / "fig_capacity.png")
    written.append("fig_capacity.png")
    if fig_scaling(results, out_dir / "fig_scaling.png"):
        written.append("fig_scaling.png")
    if fig_routing(results, out_dir / "fig_routing.png"):
        written.append("fig_routing.png")

    print(markdown_table(results))
    print("\nwrote: " + ", ".join(written) + f"  (in {out_dir})")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="results/results.json")
    parser.add_argument("--out", default=None)
    opts = parser.parse_args()
    build_all(opts.results, opts.out)
