#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "cluster_wordguessing": [
        "codenames",
        "taboo",
        "guesswhat",
        "wordle",
        "wordle_withclue",
        "wordle_withcritic",
    ],
    "cluster_explorationnavigation": [
        "adventuregame",
        "textmapworld",
        "textmapworld_graphreasoning",
        "textmapworld_specificroom",
    ],
    "cluster_cooperation": [
        "imagegame",
        "matchit_ascii",
        "referencegame",
        "privateshared",
    ],
    "ood_negotiation": ["clean_up", "dond", "hot_air_balloon"],
}

GAME_ORDER = (
    GROUPS["cluster_wordguessing"]
    + GROUPS["cluster_explorationnavigation"]
    + GROUPS["cluster_cooperation"]
    + GROUPS["ood_negotiation"]
)

# Stable preferred picker order (covers both Adapter-BAR short names and loss-router full model names).
PICKER_PREFERRED_ORDER = [
    "wordguessing",
    "explorationnavigation",
    "cooperation",
    "llama3-8b-sft-cluster_wordguessing",
    "llama3-8b-sft-cluster_explorationnavigation",
    "llama3-8b-sft-cluster_cooperation",
]

PICKER_COLOR_MAP = {
    "wordguessing": "#2563eb",
    "llama3-8b-sft-cluster_wordguessing": "#2563eb",
    "explorationnavigation": "#f59e0b",
    "llama3-8b-sft-cluster_explorationnavigation": "#f59e0b",
    "cooperation": "#10b981",
    "llama3-8b-sft-cluster_cooperation": "#10b981",
}
FALLBACK_COLORS = ["#ef4444", "#8b5cf6", "#06b6d4", "#14b8a6", "#e11d48", "#64748b"]


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_float(v):
    try:
        x = float(v)
        return None if x != x else x
    except Exception:
        return None


def _ordered_games(games: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for g in GAME_ORDER:
        if g in games and g not in seen:
            ordered.append(g)
            seen.add(g)
    for g in sorted(games):
        if g not in seen:
            ordered.append(g)
            seen.add(g)
    return ordered


def _group_tag(game: str) -> str:
    if game in set(GROUPS["cluster_wordguessing"]):
        return "WG"
    if game in set(GROUPS["cluster_explorationnavigation"]):
        return "EN"
    if game in set(GROUPS["cluster_cooperation"]):
        return "CO"
    if game in set(GROUPS["ood_negotiation"]):
        return "OOD"
    return "OTH"


def _labeled_games(games: List[str]) -> List[str]:
    return [f"{_group_tag(g)}:{g}" for g in games]


def _ordered_pickers(pickers: List[str]) -> List[str]:
    pset = set(pickers)
    out = [p for p in PICKER_PREFERRED_ORDER if p in pset]
    out.extend(sorted([p for p in pickers if p not in set(out)]))
    return out


def _picker_color(name: str) -> str:
    if name in PICKER_COLOR_MAP:
        return PICKER_COLOR_MAP[name]
    idx = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def _picker_label(name: str) -> str:
    s = str(name)
    for prefix in ("llama3-8b-sft-", "llama3-8b-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _rows_from_root(root: Path, suite: str) -> Tuple[str, List[dict]]:
    # 1) Prefer loss-router per-suite logs when present.
    loss_candidates = sorted(root.glob(f"*.{suite}.loss_router.jsonl"))
    if loss_candidates:
        rows = _read_jsonl(loss_candidates[0])
        for r in rows:
            r["_picker"] = str(r.get("selected_expert", "")).strip()
            r["_margin"] = _safe_float(r.get("top1_minus_top2_margin"))
            r["_loss"] = _safe_float(r.get("selected_mean_nll"))
        return "loss_router", rows

    # 2) Prefer suite-specific Adapter-BAR materialized rows when present.
    # These are produced by materialize_adapter_bar_replay.py and avoid mixing clem/static rows.
    mat_candidates = sorted(root.glob(f"*.{suite}.materialize.rows.jsonl"))
    if mat_candidates:
        rows = _read_jsonl(mat_candidates[0])
        for r in rows:
            r["_picker"] = str(r.get("predicted_adapter", "")).strip()
            r["_margin"] = None
            r["_loss"] = None
        return "adapter_bar_materialized", rows

    # 3) Fallback: Adapter-BAR shared routing log.
    adapter_path = root / "adapter_bar_routing_log.jsonl"
    if adapter_path.exists():
        rows = _read_jsonl(adapter_path)
        # If a suite field exists, keep only matching suite rows.
        suite_rows = [r for r in rows if str(r.get("suite", "")).strip().lower() == suite.lower()]
        if suite_rows:
            rows = suite_rows
        for r in rows:
            r["_picker"] = str(r.get("predicted_adapter", "")).strip()
            r["_margin"] = None
            r["_loss"] = None
        return "adapter_bar", rows

    return "none", []


def _usage_by_game(rows: List[dict]) -> Tuple[Dict[str, Counter], Counter]:
    by_game = defaultdict(Counter)
    total = Counter()
    for r in rows:
        g = str(r.get("game", "")).strip()
        p = str(r.get("_picker", "")).strip()
        if not g or not p:
            continue
        by_game[g][p] += 1
        total[p] += 1
    return by_game, total


def _plot_picker_totals(path: Path, suite: str, totals: Counter) -> str | None:
    if not totals:
        return None
    labels = _ordered_pickers(list(totals.keys()))
    vals = [totals[k] for k in labels]
    x = np.arange(len(labels))
    fig_w = max(7, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    ax.bar(x, vals, color=[_picker_color(k) for k in labels])
    ax.set_xticks(x)
    ax.set_xticklabels([_picker_label(x) for x in labels], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"{suite}: Picker usage totals")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _plot_grouped_games(path: Path, suite: str, by_game: Dict[str, Counter], pickers: List[str]) -> str | None:
    if not by_game or not pickers:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.flatten()
    pickers = _ordered_pickers(pickers)

    for ax_idx, (group, games) in enumerate(GROUPS.items()):
        ax = axes[ax_idx]
        present_games = [g for g in games if g in by_game]
        if not present_games:
            ax.set_title(f"{group} (no rows)")
            ax.axis("off")
            continue
        x = np.arange(len(present_games))
        width = 0.78 / max(1, len(pickers))
        for i, p in enumerate(pickers):
            vals = [by_game[g].get(p, 0) for g in present_games]
            offset = (i - (len(pickers) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=_picker_label(p), color=_picker_color(p))
        ax.set_xticks(x)
        ax.set_xticklabels(_labeled_games(present_games), rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("count")
        ax.set_title(group)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), fontsize=8)
    fig.suptitle(f"{suite}: Per-game picker counts grouped by domain clusters", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _plot_contribution_stacked(path: Path, suite: str, by_game: Dict[str, Counter], pickers: List[str]) -> str | None:
    games = _ordered_games(list(by_game.keys()))
    if not games or not pickers:
        return None
    pickers = _ordered_pickers(pickers)
    data = np.array([[float(by_game[g].get(p, 0)) for g in games] for p in pickers], dtype=float)
    totals = data.sum(axis=0)
    totals[totals == 0] = 1.0
    props = data / totals

    game_labels = _labeled_games(games)
    fig_w = max(9, min(24, 0.8 * len(games)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bottom = np.zeros(len(games))
    for i, p in enumerate(pickers):
        vals = props[i]
        ax.bar(game_labels, vals, bottom=bottom, label=_picker_label(p), color=_picker_color(p))
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("pick share")
    ax.set_title(f"{suite}: Adapter contribution by game")
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _plot_margin_by_game(path: Path, suite: str, rows: List[dict]) -> str | None:
    vals = defaultdict(list)
    for r in rows:
        m = r.get("_margin")
        g = str(r.get("game", "")).strip()
        if g and m is not None:
            vals[g].append(float(m))
    if not vals:
        return None
    games = sorted(vals.keys())
    means = [float(np.mean(vals[g])) for g in games]
    x = np.arange(len(games))
    fig_w = max(8, min(20, 0.65 * len(games)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    ax.bar(x, means, color="#0ea5e9")
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("avg top1-top2 margin")
    ax.set_title(f"{suite}: Average routing margin by game")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _plot_loss_hist(path: Path, suite: str, rows: List[dict]) -> str | None:
    losses = [float(r["_loss"]) for r in rows if r.get("_loss") is not None]
    if not losses:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(losses, bins=24, color="#f59e0b", edgecolor="black", alpha=0.85)
    ax.set_title(f"{suite}: Selected mean NLL distribution")
    ax.set_xlabel("selected_mean_nll")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _table(headers: List[str], rows: List[List[object]]) -> str:
    out = ["<table><thead><tr>"]
    out.extend([f"<th>{h}</th>" for h in headers])
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        out.extend([f"<td>{c}</td>" for c in r])
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def build_report(root: Path) -> Path:
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    suite_entries = []

    for suite in ("clem", "static"):
        mode, rows = _rows_from_root(root, suite)
        if not rows:
            continue
        by_game, totals = _usage_by_game(rows)
        pickers = _ordered_pickers(list(totals.keys()))
        total_rows = len(rows)

        refs = []
        fn = _plot_picker_totals(plots_dir / f"{suite}_picker_totals.png", suite, totals)
        if fn:
            refs.append((f"{suite} totals", f"plots/{fn}"))
        fn = _plot_grouped_games(plots_dir / f"{suite}_grouped_games.png", suite, by_game, pickers)
        if fn:
            refs.append((f"{suite} grouped games", f"plots/{fn}"))
        fn = _plot_contribution_stacked(plots_dir / f"{suite}_contribution_by_game.png", suite, by_game, pickers)
        if fn:
            refs.append((f"{suite} contribution by game", f"plots/{fn}"))
        fn = _plot_margin_by_game(plots_dir / f"{suite}_margin_by_game.png", suite, rows)
        if fn:
            refs.append((f"{suite} margin by game", f"plots/{fn}"))
        fn = _plot_loss_hist(plots_dir / f"{suite}_selected_nll_hist.png", suite, rows)
        if fn:
            refs.append((f"{suite} selected NLL hist", f"plots/{fn}"))

        game_rows = []
        for g in _ordered_games(list(by_game.keys())):
            total_g = sum(by_game[g].values())
            top_picker, top_count = by_game[g].most_common(1)[0]
            game_rows.append([g, total_g, top_picker, top_count])

        suite_entries.append(
            {
                "suite": suite,
                "mode": mode,
                "rows": total_rows,
                "pickers": pickers,
                "totals": totals,
                "by_game": by_game,
                "game_rows": game_rows,
                "refs": refs,
            }
        )

    val_files = sorted(root.glob("*.val.json"))
    val_payload = _read_json(val_files[0]) if val_files else {}
    clemscore = _safe_float((val_payload or {}).get("clemscore"))
    statscore = _safe_float((val_payload or {}).get("statscore"))

    summary_rows = []
    for e in suite_entries:
        summary_rows.append([e["suite"], e["mode"], e["rows"], len(e["pickers"])])

    parts = [
        "<!doctype html><html><head><meta charset='utf-8' />",
        "<title>Cluster Routing Report</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;margin-bottom:18px;}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px;}thead th{background:#f3f4f6;}h1,h2,h3{margin:8px 0 10px;}.muted{color:#555;} img{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 20px;}</style>",
        "</head><body>",
        "<h1>Cluster Routing Report</h1>",
        f"<div class='muted'>{root}</div>",
        f"<p><b>clemscore:</b> {clemscore if clemscore is not None else 'NaN'} &nbsp; <b>statscore:</b> {statscore if statscore is not None else 'NaN'}</p>",
    ]
    if summary_rows:
        parts.append("<h2>Suites</h2>")
        parts.append(_table(["suite", "routing_log_type", "rows", "unique_pickers"], summary_rows))

    for e in suite_entries:
        parts.append(f"<h2>{e['suite'].upper()}</h2>")
        parts.append(_table(["game", "rows", "top_picker", "top_picker_count"], e["game_rows"]))
        for label, href in e["refs"]:
            parts.append(f"<h3>{label}</h3><img src='{href}' alt='{label}' />")

    parts.append("</body></html>")
    out_html = root / "routing_report.html"
    out_html.write_text("\n".join(parts), encoding="utf-8")
    return out_html


def main():
    parser = argparse.ArgumentParser(description="Build routing reports for cluster-router style folders.")
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more folders (e.g., playpen-eval/cluster-router playpen-eval/cluster-loss-router).",
    )
    args = parser.parse_args()

    for r in args.roots:
        root = r.expanduser().resolve()
        out = build_report(root)
        print(f"[ok] {root} -> {out}")


if __name__ == "__main__":
    main()
