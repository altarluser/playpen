#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _entropy(counts: Iterable[float]) -> float:
    vals = [float(x) for x in counts if float(x) > 0.0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in vals:
        p = c / total
        h -= p * math.log(p)
    return float(h)


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze latent residual MoE eval usage logs.")
    parser.add_argument("eval_usage_csv", type=str, help="Path to latent_eval_usage.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: eval_usage_csv directory)")
    parser.add_argument(
        "--training-diagnostics",
        type=str,
        default=None,
        help="Optional path to latent_training_diagnostics.csv (copied to out-dir).",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_usage_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else eval_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(eval_path)
    if not rows:
        raise SystemExit(f"No rows in {eval_path}")

    by_game = defaultdict(float)  # (game, split, regime, expert_id) -> top1_count
    game_totals = defaultdict(float)  # (game, split, regime) -> top1_count
    by_layer = defaultdict(float)  # (layer, expert_id) -> top1_count
    layer_totals = defaultdict(float)  # layer -> top1_count
    id_vs_ood = defaultdict(lambda: {"id": 0.0, "ood": 0.0, "other": 0.0})  # expert -> counts

    for row in rows:
        game = str(row.get("game", "unknown") or "unknown")
        split = str(row.get("split", "unknown") or "unknown")
        regime = str(row.get("regime", "unknown") or "unknown").lower()
        layer = int(float(row.get("layer", 0) or 0))
        expert_id = int(float(row.get("expert_id", 0) or 0))
        top1_count = float(row.get("top1_count", 0.0) or 0.0)

        by_game[(game, split, regime, expert_id)] += top1_count
        game_totals[(game, split, regime)] += top1_count
        by_layer[(layer, expert_id)] += top1_count
        layer_totals[layer] += top1_count

        if regime == "id":
            id_vs_ood[expert_id]["id"] += top1_count
        elif regime == "ood":
            id_vs_ood[expert_id]["ood"] += top1_count
        else:
            id_vs_ood[expert_id]["other"] += top1_count

    game_rows = []
    for (game, split, regime, expert_id), count in sorted(by_game.items()):
        total = game_totals[(game, split, regime)]
        proportion = count / max(1.0, total)
        game_rows.append([game, split, regime, expert_id, count, proportion])
    _write_csv(
        out_dir / "latent_usage_by_game.csv",
        ["game", "split", "regime", "expert_id", "top1_count", "usage_proportion"],
        game_rows,
    )

    layer_rows = []
    for (layer, expert_id), count in sorted(by_layer.items()):
        total = layer_totals[layer]
        proportion = count / max(1.0, total)
        layer_rows.append([layer, expert_id, count, proportion])
    _write_csv(
        out_dir / "latent_usage_by_layer.csv",
        ["layer", "expert_id", "top1_count", "usage_proportion"],
        layer_rows,
    )

    entropy_game = {}
    for (game, split, regime), total in sorted(game_totals.items()):
        counts = [c for (g, s, r, _), c in by_game.items() if g == game and s == split and r == regime]
        entropy_game[f"{game}|{split}|{regime}"] = _entropy(counts)

    entropy_layer = {}
    for layer in sorted(layer_totals):
        counts = [c for (l, _), c in by_layer.items() if l == layer]
        entropy_layer[str(layer)] = _entropy(counts)

    id_ood_comparison = {}
    for expert_id, counts in sorted(id_vs_ood.items()):
        id_count = counts["id"]
        ood_count = counts["ood"]
        total = id_count + ood_count
        id_ood_comparison[str(expert_id)] = {
            "id_count": id_count,
            "ood_count": ood_count,
            "id_share_over_id_plus_ood": (id_count / total) if total > 0 else None,
        }

    summary = {
        "source_eval_usage_csv": str(eval_path),
        "latent_usage_by_game_csv": str(out_dir / "latent_usage_by_game.csv"),
        "latent_usage_by_layer_csv": str(out_dir / "latent_usage_by_layer.csv"),
        "entropy_by_game_split_regime": entropy_game,
        "entropy_by_layer": entropy_layer,
        "id_vs_ood_expert_usage": id_ood_comparison,
    }
    (out_dir / "latent_usage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.training_diagnostics:
        src = Path(args.training_diagnostics).expanduser()
        if src.exists():
            dst = out_dir / "latent_training_diagnostics.csv"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
