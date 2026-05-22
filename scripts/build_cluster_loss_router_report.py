#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

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

PICKER_COLOR_MAP = {
    "wordguessing": "#2563eb",
    "llama3-8b-sft-cluster_wordguessing": "#2563eb",
    "explorationnavigation": "#f59e0b",
    "llama3-8b-sft-cluster_explorationnavigation": "#f59e0b",
    "cooperation": "#10b981",
    "llama3-8b-sft-cluster_cooperation": "#10b981",
}
FALLBACK_COLOR = "#64748b"


def _picker_color(name: str) -> str:
    return PICKER_COLOR_MAP.get(str(name), FALLBACK_COLOR)


def _picker_label(name: str) -> str:
    s = str(name)
    for prefix in ("llama3-8b-sft-", "llama3-8b-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s

def _ordered_games(games):
    seen = set()
    out = []
    for g in GAME_ORDER:
        if g in games and g not in seen:
            out.append(g)
            seen.add(g)
    for g in sorted(games):
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out

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

def _label_game(game: str) -> str:
    return f"{_group_tag(game)}:{game}"


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x):
    try:
        v = float(x)
        return None if v != v else v
    except Exception:
        return None


def _read_score_csv(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[0]
    out = {}
    for k, v in row.items():
        out[k] = v
    return out


def _build_routing_stats(assignments):
    by_game = defaultdict(Counter)
    by_expert = Counter()
    by_game_total = Counter()
    for a in assignments:
        g = str(a.get("game"))
        e = str(a.get("expert_model"))
        by_game[g][e] += 1
        by_expert[e] += 1
        by_game_total[g] += 1
    return by_game, by_expert, by_game_total


def _table(headers, rows):
    parts = ["<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{h}</th>")
    parts.append("</tr></thead><tbody>")
    for r in rows:
        parts.append("<tr>")
        for c in r:
            parts.append(f"<td>{c}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _plot_bar(path: Path, title: str, labels, values, ylabel: str):
    if not labels:
        return None
    x = np.arange(len(labels))
    fig_w = max(8, min(18, 0.55 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(x, values, color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _plot_stacked_usage(path: Path, title: str, usage_counts_by_game: dict, experts: list[str]):
    games = _ordered_games(list(usage_counts_by_game.keys()))
    if not games or not experts:
        return None
    game_labels = [_label_game(g) for g in games]
    data = np.array([[float((usage_counts_by_game.get(g) or {}).get(e, 0)) for g in games] for e in experts], dtype=float)
    totals = data.sum(axis=0)
    totals[totals == 0] = 1.0
    props = data / totals

    fig_w = max(9, min(24, 0.8 * len(games)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bottom = np.zeros(len(games))
    for i, e in enumerate(experts):
        vals = props[i]
        ax.bar(game_labels, vals, bottom=bottom, label=_picker_label(e), color=_picker_color(e))
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("proportion")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description="Build combined cluster-loss-router report.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("playpen-eval/cluster-loss-router"),
        help="Cluster loss router results folder.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    clem_moe = _load_json(root / "llama3-8b-cluster-lossrouter.clem.moe.json") or {}
    static_moe = _load_json(root / "llama3-8b-cluster-lossrouter.static.moe.json") or {}
    val = _load_json(root / "llama3-8b-cluster-lossrouter.val.json") or {}

    clem_assign = list(clem_moe.get("assignments") or [])
    static_assign = list(static_moe.get("assignments") or [])

    clem_by_game, clem_by_expert, clem_game_total = _build_routing_stats(clem_assign)
    static_by_game, static_by_expert, static_game_total = _build_routing_stats(static_assign)

    all_experts = sorted(set(clem_by_expert) | set(static_by_expert))

    clem_scores = _read_score_csv(root / "clem" / "results.csv")
    static_scores = _read_score_csv(root / "static" / "results.csv")
    clem_analysis = _load_json(root / "loss_router_analysis.clem.json") or {}
    static_analysis = _load_json(root / "loss_router_analysis.static.json") or {}

    summary_rows = [[
        "clem",
        len(clem_assign),
        f"{_safe_float(val.get('clemscore')):.2f}" if _safe_float(val.get("clemscore")) is not None else "NaN",
    ], [
        "static",
        len(static_assign),
        f"{_safe_float(val.get('statscore')):.2f}" if _safe_float(val.get("statscore")) is not None else "NaN",
    ]]

    def build_game_rows(by_game, game_total):
        rows = []
        for g in sorted(game_total.keys()):
            total = game_total[g]
            picks = by_game[g]
            top = picks.most_common(1)[0] if picks else ("-", 0)
            rows.append([g, total, top[0], top[1]])
        return rows

    def build_pivot_rows(by_game, game_total):
        rows = []
        for g in sorted(game_total.keys()):
            row = [g, game_total[g]]
            for e in all_experts:
                row.append(by_game[g].get(e, 0))
            rows.append(row)
        return rows

    def score_preview_rows(score_map, label):
        rows = []
        for k in sorted(score_map.keys()):
            if "Quality Score" in k or "clemscore" in k.lower() or "Average" in k:
                rows.append([label, k, score_map[k]])
        return rows[:40]

    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_refs = []
    for suite_name, analysis in [("clem", clem_analysis), ("static", static_analysis)]:
        if not analysis:
            continue
        avg_margin = analysis.get("average_routing_margin_by_game") or {}
        entropy = analysis.get("expert_selection_entropy_by_game") or {}
        usage = analysis.get("usage_counts_by_game") or {}
        experts = list(analysis.get("experts") or [])

        if isinstance(avg_margin, dict) and avg_margin:
            labels = _ordered_games(list(avg_margin.keys()))
            values = [float(avg_margin[k]) for k in labels]
            fn = _plot_bar(
                plots_dir / f"{suite_name}_avg_margin_by_game.png",
                f"{suite_name}: Avg Routing Margin by Game",
                [_label_game(x) for x in labels],
                values,
                "avg margin",
            )
            if fn:
                plot_refs.append((f"{suite_name} avg margin", f"plots/{fn}"))

        if isinstance(entropy, dict) and entropy:
            labels = _ordered_games(list(entropy.keys()))
            values = [float(entropy[k]) for k in labels]
            fn = _plot_bar(
                plots_dir / f"{suite_name}_entropy_by_game.png",
                f"{suite_name}: Expert Selection Entropy by Game",
                [_label_game(x) for x in labels],
                values,
                "entropy",
            )
            if fn:
                plot_refs.append((f"{suite_name} entropy", f"plots/{fn}"))

        if isinstance(usage, dict) and usage and experts:
            fn = _plot_stacked_usage(
                plots_dir / f"{suite_name}_expert_usage_by_game.png",
                f"{suite_name}: Expert Usage Proportions by Game",
                usage,
                experts,
            )
            if fn:
                plot_refs.append((f"{suite_name} expert usage", f"plots/{fn}"))

    analysis_rows = []
    for suite_name, analysis in [("clem", clem_analysis), ("static", static_analysis)]:
        if not analysis:
            continue
        analysis_rows.append([
            suite_name,
            analysis.get("rows_total"),
            analysis.get("rows_id"),
            analysis.get("routing_regret_vs_oracle_id_only"),
            analysis.get("semantic_cluster_accuracy_id_only"),
            analysis.get("semantic_cluster_regret_id_only"),
        ])

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cluster Loss Router Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
    thead th {{ background: #f3f4f6; position: sticky; top: 0; }}
    h1, h2, h3 {{ margin: 8px 0 12px; }}
    .muted {{ color: #555; }}
    .links a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>Cluster Loss Router Combined Report</h1>
  <div class="muted">{root}</div>
  <div class="links">
    <a href="./clem/results.html">clem/results.html</a>
    <a href="./static/results.html">static/results.html</a>
  </div>

  <h2>Suite Summary</h2>
  {_table(["suite", "assignment_rows", "suite_score"], summary_rows)}

  <h2>CLEM Routing Stats</h2>
  <h3>By Game (top expert)</h3>
  {_table(["game", "tasks", "top_expert", "top_expert_count"], build_game_rows(clem_by_game, clem_game_total))}
  <h3>By Game x Expert</h3>
  {_table(["game", "tasks", *all_experts], build_pivot_rows(clem_by_game, clem_game_total))}

  <h2>STATIC Routing Stats</h2>
  <h3>By Game (top expert)</h3>
  {_table(["game", "tasks", "top_expert", "top_expert_count"], build_game_rows(static_by_game, static_game_total))}
  <h3>By Game x Expert</h3>
  {_table(["game", "tasks", *all_experts], build_pivot_rows(static_by_game, static_game_total))}

  <h2>Score Column Preview</h2>
  {_table(["suite", "column", "value"], score_preview_rows(clem_scores, "clem") + score_preview_rows(static_scores, "static"))}

  <h2>Loss Router Analysis</h2>
  {_table(
      ["suite", "rows_total", "rows_id", "routing_regret_vs_oracle_id_only", "semantic_cluster_accuracy_id_only", "semantic_cluster_regret_id_only"],
      analysis_rows
  )}

  <h2>Loss/Margin Plots</h2>
  <div class="muted">Plots are built from local loss_router_analysis JSON summaries (margins, entropy, usage proportions).</div>
  {"".join([f"<h3>{name}</h3><img src='{src}' style='max-width:100%; border:1px solid #ddd; margin-bottom:16px;'/>" for name, src in plot_refs])}
</body>
</html>
"""
    out = root / "combined_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
