#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float(v):
    try:
        x = float(v)
        return np.nan if x != x else x
    except Exception:
        return np.nan


def _adapter_game_from_run_name(run_name: str) -> str:
    # e.g. "llama3-8b-sft-game_codenames bbh eksik" -> "codenames"
    if "game_" not in run_name:
        return run_name
    tail = run_name.split("game_", 1)[1]
    tail = re.split(r"\s+(bbh|stat|eksik|missing)\b", tail, maxsplit=1)[0]
    return tail.strip()


def _read_clem_row(results_csv: Path):
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[0]


def _quality_cols(row: dict[str, str]) -> list[str]:
    cols = []
    for c in row.keys():
        if c.endswith(", Quality Score"):
            cols.append(c)
    return cols


def _game_from_quality_col(c: str) -> str:
    return c.rsplit(", Quality Score", 1)[0]


def _pairwise_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) <= 1e-12 or np.nanstd(bb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def main():
    parser = argparse.ArgumentParser(
        description="Plot game-to-game correlations from per-game adapter CLEM results."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("playpen-eval/per-game adapters"),
        help="Root folder containing per-game adapter run dirs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("playpen-eval/per-game adapters/correlation"),
        help="Output folder for plots and CSVs.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    rows = []
    all_games = []
    for run in run_dirs:
        clem_csv = run / "clem" / "results.csv"
        if not clem_csv.exists():
            continue
        row = _read_clem_row(clem_csv)
        if row is None:
            continue
        adapter_game = _adapter_game_from_run_name(run.name)
        qcols = _quality_cols(row)
        if not qcols:
            continue
        scores = {}
        for c in qcols:
            g = _game_from_quality_col(c)
            scores[g] = _to_float(row.get(c))
            all_games.append(g)
        rows.append((run.name, adapter_game, scores))

    games = sorted(set(all_games))
    if not rows or not games:
        raise SystemExit("No usable per-game adapter CLEM results found.")

    # Build adapter x game score matrix.
    mat = np.full((len(rows), len(games)), np.nan, dtype=float)
    for i, (_run_name, _adapter_game, scores) in enumerate(rows):
        for j, g in enumerate(games):
            mat[i, j] = scores.get(g, np.nan)

    # Correlation among target games across adapters.
    corr = np.full((len(games), len(games)), np.nan, dtype=float)
    for i in range(len(games)):
        for j in range(len(games)):
            if i == j:
                corr[i, j] = 1.0
            else:
                corr[i, j] = _pairwise_corr(mat[:, i], mat[:, j])

    # Save numeric outputs.
    np.savetxt(out / "game_correlation_matrix.csv", corr, delimiter=",", fmt="%.6f")
    with (out / "game_index.txt").open("w", encoding="utf-8") as f:
        for idx, g in enumerate(games):
            f.write(f"{idx},{g}\n")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Game-to-Game Correlation (per-game adapters)")
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games, rotation=90, fontsize=8)
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    # Per-game bar plots
    per_game_dir = out / "per_game"
    per_game_dir.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(games):
        pairs = []
        for j, g2 in enumerate(games):
            if i == j:
                continue
            pairs.append((g2, corr[i, j]))
        pairs = [x for x in pairs if np.isfinite(x[1])]
        pairs.sort(key=lambda x: x[1], reverse=True)
        if not pairs:
            continue
        labels = [x[0] for x in pairs]
        vals = [x[1] for x in pairs]

        h = max(4.0, 0.3 * len(labels))
        fig, ax = plt.subplots(figsize=(10, h))
        y = np.arange(len(labels))
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-1, 1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Pearson r")
        ax.set_title(f"Correlations with '{g}'")
        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", g)
        fig.savefig(per_game_dir / f"{safe}.png", dpi=170)
        plt.close(fig)

    # Adapter->game score table
    with (out / "adapter_game_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "adapter_game", *games])
        for run_name, adapter_game, scores in rows:
            w.writerow([run_name, adapter_game, *[scores.get(g, np.nan) for g in games]])

    print(f"Saved outputs to: {out}")
    print(f"Adapters used: {len(rows)}")
    print(f"Games used: {len(games)}")


if __name__ == "__main__":
    main()
