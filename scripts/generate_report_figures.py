"""Generates the figures used in docs/final_report.md / the final PDF.

Two categories, and they must never be visually confused:
  1. Real-data figures, computed from this project's own generated result
     files (results/metrics/data_split_validation.json,
     dataset/dcase_part1/splits/build_manifests_summary.json).
  2. Reference/context figures (architecture, pipeline, and the DCASE
     published-benchmark comparison), which are conceptual or externally
     sourced and are labeled as such directly in the figure.

Training-loss curves are produced by `src/training/plots.py` from
`logs/training_log.jsonl` after training; this script only builds
dataset and reference figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz skill reference palette, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Question-type distribution (real data)
# ---------------------------------------------------------------------------
def plot_question_type_distribution() -> Path:
    report = load_json("results/metrics/data_split_validation.json")
    splits = ["train", "validation", "test"]
    colors = {"train": BLUE, "validation": ORANGE, "test": AQUA}

    all_types = sorted({t for s in splits for t in report["splits"][s]["question_type_distribution"]})
    fig, ax = plt.subplots(figsize=(9, 6))
    y = range(len(all_types))
    bar_h = 0.25

    for i, split in enumerate(splits):
        dist = report["splits"][split]["question_type_distribution"]
        pcts = [dist.get(t, {}).get("percentage", 0.0) for t in all_types]
        offsets = [pos + (i - 1) * bar_h for pos in y]
        ax.barh(offsets, pcts, height=bar_h, color=colors[split], label=f"{split} (n={report['splits'][split]['records']})")

    ax.set_yticks(list(y))
    ax.set_yticklabels(all_types)
    ax.set_xlabel("Share of split (%)")
    ax.set_title("Question-Type Distribution by Split\nAudio Context Layer — Prepared DCASE Part 1 Manifests (measured)", fontsize=12, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / "question_type_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. DCASE 2025 published reference comparison (external numbers, NOT ours)
# ---------------------------------------------------------------------------
def plot_dcase_reference_comparison() -> Path:
    systems = ["Qwen2-Audio-7B\n(baseline)", "AudioFlamingo 2\n(baseline)", "Gemini-2.0-Flash\n(baseline)", "Qwen-Omni-2.5\n(Chen_SRCN GRPO)"]
    scores = [45.0, 45.7, 52.5, 81.26]
    colors = [BLUE, ORANGE, AQUA, YELLOW]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(systems, scores, color=colors, width=0.55, zorder=3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.5, f"{score:.1f}%", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=10)

    ax.set_ylim(0, 95)
    ax.set_ylabel("Overall average accuracy (%)")
    ax.set_title(
        "DCASE 2025 Published Reference Results — Not Audio Context Layer Measurements\n"
        "Source: Yang et al. 2025; official DCASE 2025 Task 5 results page",
        fontsize=11, loc="left"
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02, "External benchmark context only — no result from this project appears in this figure.", ha="center", fontsize=9, color=TEXT_SECONDARY, style="italic")
    fig.tight_layout()
    out = PLOTS_DIR / "dcase_benchmark_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Architecture diagram (conceptual)
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, facecolor, textcolor="#ffffff", fontsize=10.5):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=textcolor, fontsize=fontsize, zorder=3, wrap=True)
    return box


def _arrow(ax, start, end, color=TEXT_SECONDARY):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=color, linewidth=1.6, zorder=1)
    ax.add_patch(arrow)


def plot_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, (0.3, 4.2), 2.0, 1.0, "Audio\nWaveform", AQUA)
    _box(ax, (0.3, 2.6), 2.0, 1.0, "Question +\nChoices (A–D)", ORANGE)

    _box(ax, (3.0, 3.9), 2.2, 1.0, "Audio Tower\n(Whisper-style encoder)\nfrozen", VIOLET)
    _box(ax, (3.0, 2.6), 2.2, 1.0, "Text Tokenizer", TEXT_SECONDARY)

    _box(ax, (5.7, 3.9), 1.6, 1.0, "Multi-modal\nProjector\nfrozen", VIOLET)

    _box(ax, (7.7, 3.25), 2.0, 1.4, "Qwen2 Language\nModel Decoder\n+ LoRA adapters\n(trainable)", BLUE, fontsize=10)

    _box(ax, (7.7, 0.9), 2.0, 1.0, "Generated Answer\n\"C. dog barking\"", AQUA)

    _box(ax, (3.0, 0.9), 3.5, 1.0, "Answer Extraction & Matching\n(option-label + normalized text)", ORANGE, textcolor="#ffffff", fontsize=9.5)

    _arrow(ax, (2.3, 4.7), (3.0, 4.4))
    _arrow(ax, (2.3, 3.1), (3.0, 3.1))
    _arrow(ax, (5.2, 4.4), (5.7, 4.4))
    _arrow(ax, (5.2, 3.1), (7.7, 3.7))
    _arrow(ax, (7.3, 4.4), (7.7, 4.1))
    _arrow(ax, (8.7, 3.25), (8.7, 1.9))
    _arrow(ax, (7.7, 1.4), (6.5, 1.4))

    ax.text(5.0, 5.6, "Audio Context Layer — System Architecture", ha="center", fontsize=13, color=TEXT_PRIMARY, weight="bold")
    ax.text(5.0, 0.15, "LoRA (rank 16, scoped to language_model.*) is the only trainable component; audio tower and projector remain frozen.", ha="center", fontsize=8.5, color=TEXT_SECONDARY)

    out = PLOTS_DIR / "architecture_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Data pipeline diagram (conceptual, matches scripts/build_manifests.py)
# ---------------------------------------------------------------------------
def plot_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    canvas_w = 13.0
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages = [
        ("Raw DCASE\nmetadata\n(8,221 + 2,466\nJSON)", TEXT_SECONDARY),
        ("Normalize\nrecords", BLUE),
        ("Resolve to\nlocal audio", BLUE),
        ("Audio-level\n90/10 split", ORANGE),
        ("Remove test-\noverlapping\naudio from\ntrain/val", ORANGE),
        ("Validate:\nleakage, missing\naudio, answer/\nchoice", AQUA),
        ("train /\nvalidation /\ntest .jsonl", VIOLET),
    ]
    n = len(stages)
    box_w, gap = 1.5, 0.3
    total_w = n * box_w + (n - 1) * gap
    x0 = (canvas_w - total_w) / 2
    xs = [x0 + i * (box_w + gap) for i in range(n)]

    for x, (label, color) in zip(xs, stages):
        _box(ax, (x, 1.1), box_w, 1.4, label, color, fontsize=8.5)
    for i in range(n - 1):
        _arrow(ax, (xs[i] + box_w, 1.8), (xs[i + 1], 1.8))

    ax.text(canvas_w / 2, 2.95, "Reproducible Manifest-Generation Pipeline (scripts/build_manifests.py)", ha="center", fontsize=12.5, color=TEXT_PRIMARY, weight="bold")
    ax.text(canvas_w / 2, 0.5, "Deterministic given the same raw metadata: re-running the script reproduces the same manifests.", ha="center", fontsize=9, color=TEXT_SECONDARY)

    out = PLOTS_DIR / "data_pipeline_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    outputs = [
        plot_question_type_distribution(),
        plot_dcase_reference_comparison(),
        plot_architecture_diagram(),
        plot_pipeline_diagram(),
    ]
    for path in outputs:
        print("wrote", path)
