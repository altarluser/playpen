#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def _read_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(vals: Iterable[float]) -> float:
    data = [float(x) for x in vals]
    return float(sum(data) / len(data)) if data else 0.0


def _entropy(labels: List[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return float(h)


def _extract_experts_from_row(row: Mapping[str, object]) -> List[str]:
    experts = []
    for key in row.keys():
        if key.startswith("mean_nll_"):
            experts.append(key[len("mean_nll_") :])
    return sorted(experts)


def _oracle_from_nll(row: Mapping[str, object], experts: List[str]) -> Tuple[str, float]:
    best = None
    best_val = float("inf")
    for expert in experts:
        v = row.get(f"mean_nll_{expert}")
        if v is None:
            continue
        fv = float(v)
        if fv < best_val:
            best_val = fv
            best = expert
    return best, best_val


def main():
    parser = argparse.ArgumentParser(description="Analyze whole-game top-1 loss-router logs.")
    parser.add_argument("router_log", type=str, help="Path to JSONL router log.")
    parser.add_argument("--id-regimes", type=str, default="id", help="Comma-separated list, default: id")
    args = parser.parse_args()

    path = Path(args.router_log).expanduser()
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in: {path}")

    id_regimes = {x.strip().lower() for x in args.id_regimes.split(",") if x.strip()}
    experts = _extract_experts_from_row(rows[0])

    usage_counts_by_game = defaultdict(Counter)
    usage_props_by_game: Dict[str, Dict[str, float]] = {}
    margins_by_game = defaultdict(list)
    selected_by_game = defaultdict(list)

    id_rows = []
    for row in rows:
        game = str(row.get("game"))
        selected = str(row.get("selected_expert"))
        usage_counts_by_game[game][selected] += 1
        selected_by_game[game].append(selected)
        margin = row.get("top1_minus_top2_margin")
        if margin is not None:
            try:
                margins_by_game[game].append(float(margin))
            except Exception:
                pass
        regime = str(row.get("regime", "")).lower()
        if regime in id_regimes:
            id_rows.append(row)

    for game, cnt in usage_counts_by_game.items():
        total = sum(cnt.values()) or 1
        usage_props_by_game[game] = {expert: c / total for expert, c in cnt.items()}

    id_with_oracle = [r for r in id_rows if r.get("oracle_expert")]
    acc = None
    if id_with_oracle:
        acc = _mean([1.0 if str(r.get("selected_expert")) == str(r.get("oracle_expert")) else 0.0 for r in id_with_oracle])

    regrets = []
    for r in id_rows:
        sel = str(r.get("selected_expert"))
        sel_nll = r.get(f"mean_nll_{sel}")
        if sel_nll is None:
            continue
        _, best_nll = _oracle_from_nll(r, experts)
        if math.isfinite(best_nll):
            regrets.append(float(sel_nll) - float(best_nll))
    routing_regret = _mean(regrets) if regrets else None

    avg_margin_by_game = {g: _mean(v) for g, v in margins_by_game.items() if v}
    entropy_by_game = {g: _entropy(v) for g, v in selected_by_game.items() if v}

    out = {
        "rows_total": len(rows),
        "rows_id": len(id_rows),
        "experts": experts,
        "usage_counts_by_game": {g: dict(c) for g, c in usage_counts_by_game.items()},
        "usage_proportions_by_game": usage_props_by_game,
        "routing_accuracy_vs_oracle_id_only": acc,
        "routing_regret_vs_oracle_id_only": routing_regret,
        "average_routing_margin_by_game": avg_margin_by_game,
        "expert_selection_entropy_by_game": entropy_by_game,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
