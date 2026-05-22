#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLUSTER_MAP = {
    "wordguessing": {"codenames", "taboo", "guesswhat", "wordle", "wordle_withclue", "wordle_withcritic"},
    "explorationnavigation": {"adventuregame", "textmapworld", "textmapworld_graphreasoning", "textmapworld_specificroom"},
    "cooperation": {"imagegame", "matchit_ascii", "referencegame", "privateshared"},
}
NEGOTIATION_GAMES = {"clean_up", "dond", "hot_air_balloon"}


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _gold_cluster(game: str) -> str | None:
    for c, games in CLUSTER_MAP.items():
        if game in games:
            return c
    return None


def _norm_loss_picker(name: str) -> str:
    n = (name or "").strip()
    if n.endswith("cluster_wordguessing"):
        return "wordguessing"
    if n.endswith("cluster_explorationnavigation"):
        return "explorationnavigation"
    if n.endswith("cluster_cooperation"):
        return "cooperation"
    return n


def _key(r: dict) -> Tuple[str, str, int] | None:
    g = str(r.get("game", "")).strip()
    e = str(r.get("experiment", "")).strip()
    t = r.get("task_id")
    try:
        tid = int(t)
    except Exception:
        return None
    if not g or not e:
        return None
    return (g, e, tid)


def _plot_confusion(path: Path, counts: Dict[Tuple[str, str], int]):
    labels = ["wordguessing", "explorationnavigation", "cooperation"]
    mat = np.zeros((3, 3), dtype=float)
    for i, g in enumerate(labels):
        for j, p in enumerate(labels):
            mat[i, j] = counts.get((g, p), 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Confusion (excluding negotiation)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_per_game_accuracy(path: Path, rows: List[Tuple[str, bool, bool]]):
    # rows: (game, adapter_ok, loss_ok)
    by_game = defaultdict(lambda: [0, 0, 0])  # n, adapter_ok, loss_ok
    for g, aok, lok in rows:
        by_game[g][0] += 1
        by_game[g][1] += 1 if aok else 0
        by_game[g][2] += 1 if lok else 0
    games = sorted(by_game.keys())
    if not games:
        return
    adapter_acc = [by_game[g][1] / max(1, by_game[g][0]) for g in games]
    loss_acc = [by_game[g][2] / max(1, by_game[g][0]) for g in games]
    x = np.arange(len(games))
    w = 0.38
    fig_w = max(8, 0.7 * len(games))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(x - w / 2, adapter_acc, width=w, color="#2563eb", label="adapter-bar")
    ax.bar(x + w / 2, loss_acc, width=w, color="#f59e0b", label="loss-router")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("accuracy vs gold cluster")
    ax.set_title("Per-game accuracy (excluding negotiation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare Adapter-BAR vs Loss-router selections.")
    ap.add_argument("--adapter-log", required=True, type=Path)
    ap.add_argument("--loss-log", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    adapter_rows = _read_jsonl(args.adapter_log.expanduser())
    loss_rows = _read_jsonl(args.loss_log.expanduser())
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_map = {}
    for r in adapter_rows:
        k = _key(r)
        if not k:
            continue
        adapter_map[k] = str(r.get("predicted_adapter", "")).strip()

    loss_map = {}
    for r in loss_rows:
        k = _key(r)
        if not k:
            continue
        loss_map[k] = _norm_loss_picker(str(r.get("selected_expert", "")).strip())

    common = sorted(set(adapter_map.keys()) & set(loss_map.keys()))
    kept = []
    for g, e, t in common:
        if g in NEGOTIATION_GAMES:
            continue
        gold = _gold_cluster(g)
        if gold is None:
            continue
        apick = adapter_map[(g, e, t)]
        lpick = loss_map[(g, e, t)]
        kept.append((g, e, t, gold, apick, lpick))

    adapter_ok = 0
    loss_ok = 0
    agree = 0
    confusion_adapter = Counter()
    confusion_loss = Counter()
    per_game_rows = []
    for g, _e, _t, gold, apick, lpick in kept:
        aok = apick == gold
        lok = lpick == gold
        adapter_ok += 1 if aok else 0
        loss_ok += 1 if lok else 0
        agree += 1 if apick == lpick else 0
        confusion_adapter[(gold, apick)] += 1
        confusion_loss[(gold, lpick)] += 1
        per_game_rows.append((g, aok, lok))

    n = len(kept)
    summary = {
        "matched_rows_all_games": len(common),
        "matched_rows_excluding_negotiation": n,
        "adapter_accuracy_vs_gold_excl_negotiation": (adapter_ok / n) if n else None,
        "loss_accuracy_vs_gold_excl_negotiation": (loss_ok / n) if n else None,
        "adapter_vs_loss_agreement_excl_negotiation": (agree / n) if n else None,
        "confusion_adapter_vs_gold": {f"{g}->{p}": c for (g, p), c in sorted(confusion_adapter.items())},
        "confusion_loss_vs_gold": {f"{g}->{p}": c for (g, p), c in sorted(confusion_loss.items())},
    }
    (out_dir / "router_vs_loss.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_confusion(out_dir / "adapter_vs_gold_confusion.png", confusion_adapter)
    _plot_confusion(out_dir / "loss_vs_gold_confusion.png", confusion_loss)
    _plot_per_game_accuracy(out_dir / "per_game_accuracy.png", per_game_rows)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>Adapter vs Loss Comparison</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:24px;}}img{{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 20px;}}pre{{background:#f8fafc;padding:12px;border:1px solid #ddd;}}</style>
</head><body>
<h1>Adapter-BAR vs Loss-router</h1>
<p>Excluding negotiation games: clean_up, dond, hot_air_balloon.</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>Adapter-BAR vs Gold</h2><img src="adapter_vs_gold_confusion.png" />
<h2>Loss-router vs Gold</h2><img src="loss_vs_gold_confusion.png" />
<h2>Per-game Accuracy</h2><img src="per_game_accuracy.png" />
</body></html>"""
    (out_dir / "router_vs_loss.report.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
