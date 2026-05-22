#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TaskKey:
    game: str
    experiment: str
    task_id: int


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
    return rows


def _load_adapter_name_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"--adapter-name-map must be a JSON object, got: {type(obj).__name__}")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("adapter-name-map must be string->string")
        out[k] = v
    return out


def _iter_scores_files(eval_root: Path, model_name: str, suite: str, game: str, experiment: str) -> Iterable[Path]:
    # Most specific expected layouts first, then recursive fallbacks for mixed run layouts.
    patterns = [
        f"{model_name}/{suite}/{game}/{experiment}/instance_*/scores.json",
        f"{model_name}/{game}/{experiment}/instance_*/scores.json",
        f"{suite}/{model_name}/{game}/{experiment}/instance_*/scores.json",
        f"clem/{model_name}/{game}/{experiment}/instance_*/scores.json",
        f"*/{suite}/{model_name}/{game}/{experiment}/instance_*/scores.json",
        f"**/{suite}/{model_name}/{game}/{experiment}/instance_*/scores.json",
        f"**/{model_name}/{game}/{experiment}/instance_*/scores.json",
    ]
    seen = set()
    for pat in patterns:
        for p in sorted(eval_root.glob(pat)):
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                yield p


def _score_task_lookup(
    eval_root: Path,
    model_name: str,
    suite: str,
    game: str,
    experiment: str,
    task_id: int,
) -> Tuple[Optional[dict], Optional[Path]]:
    for score_path in _iter_scores_files(eval_root, model_name, suite, game, experiment):
        try:
            payload = json.loads(score_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        meta_game = str(meta.get("game_name", ""))
        meta_exp = str(meta.get("experiment_name", ""))
        meta_task = meta.get("game_id")
        try:
            meta_task_int = int(meta_task) if meta_task is not None else None
        except Exception:
            meta_task_int = None
        if meta_game != game or meta_exp != experiment or meta_task_int != task_id:
            continue
        episode_scores = payload.get("episode scores", {}) if isinstance(payload, dict) else {}
        if isinstance(episode_scores, dict):
            return episode_scores, score_path
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline replay for Adapter-BAR routing logs using existing per-adapter eval outputs."
    )
    ap.add_argument("--routing-log", required=True, help="Path to adapter_bar_routing_log.jsonl")
    ap.add_argument("--eval-root", required=True, help="Root directory that contains per-adapter eval results")
    ap.add_argument("--suite", default="clem", help="Suite folder name used in eval-root layout (default: clem)")
    ap.add_argument(
        "--adapter-prefix",
        default="",
        help="Optional model prefix, e.g. 'llama3-8b-sft-cluster_' to map predicted_adapter->model_name",
    )
    ap.add_argument(
        "--adapter-name-map",
        default="",
        help="Optional JSON file mapping predicted_adapter -> exact model_name",
    )
    ap.add_argument("--out-json", required=True, help="Output JSON summary path")
    ap.add_argument("--out-jsonl", required=True, help="Output task-level replay JSONL path")
    args = ap.parse_args()

    routing_log = Path(args.routing_log).expanduser()
    eval_root = Path(args.eval_root).expanduser()
    out_json = Path(args.out_json).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()

    rows = _read_jsonl(routing_log)
    name_map = _load_adapter_name_map(Path(args.adapter_name_map).expanduser() if args.adapter_name_map else None)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    replay_rows: List[dict] = []
    missing: List[dict] = []
    routed_count = Counter()
    found_count = Counter()
    aborted_count = Counter()

    for row in rows:
        game = str(row.get("game", "")).strip()
        experiment = str(row.get("experiment", "")).strip()
        pred = str(row.get("predicted_adapter", "")).strip()
        task_raw = row.get("task_id")
        try:
            task_id = int(task_raw)
        except Exception:
            missing.append(
                {
                    "reason": "invalid_task_id",
                    "row": row,
                }
            )
            continue
        if not game or not experiment or not pred:
            missing.append(
                {
                    "reason": "missing_fields",
                    "row": row,
                }
            )
            continue

        model_name = name_map.get(pred) or (f"{args.adapter_prefix}{pred}" if args.adapter_prefix else pred)
        routed_count[model_name] += 1

        episode_scores, score_path = _score_task_lookup(
            eval_root=eval_root,
            model_name=model_name,
            suite=args.suite,
            game=game,
            experiment=experiment,
            task_id=task_id,
        )
        if episode_scores is None or score_path is None:
            missing.append(
                {
                    "reason": "score_not_found",
                    "game": game,
                    "experiment": experiment,
                    "task_id": task_id,
                    "predicted_adapter": pred,
                    "resolved_model_name": model_name,
                }
            )
            continue

        found_count[model_name] += 1
        aborted = int(episode_scores.get("Aborted", 0)) if isinstance(episode_scores, dict) else 0
        if aborted:
            aborted_count[model_name] += 1

        replay_row = {
            "game": game,
            "experiment": experiment,
            "task_id": task_id,
            "predicted_adapter": pred,
            "resolved_model_name": model_name,
            "score_path": str(score_path),
            "episode_scores": episode_scores,
        }
        replay_rows.append(replay_row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in replay_rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary = {
        "routing_log": str(routing_log),
        "eval_root": str(eval_root),
        "suite": args.suite,
        "total_routing_rows": len(rows),
        "replayed_rows": len(replay_rows),
        "missing_rows": len(missing),
        "routed_count_by_model": dict(routed_count),
        "found_count_by_model": dict(found_count),
        "aborted_count_by_model": dict(aborted_count),
        "missing_examples": missing[:200],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
