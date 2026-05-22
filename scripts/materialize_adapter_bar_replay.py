#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def _iter_candidate_instances(
    eval_root: Path, model_name: str, suite: str, game: str, experiment: str, task_id: int
) -> Iterable[Path]:
    inst = f"instance_{int(task_id):05d}"
    patterns = [
        f"{model_name}/{suite}/{game}/{experiment}/{inst}",
        f"{suite}/{model_name}/{game}/{experiment}/{inst}",
        f"*/{suite}/{model_name}/{game}/{experiment}/{inst}",
        f"**/{suite}/{model_name}/{game}/{experiment}/{inst}",
        f"**/{model_name}/{game}/{experiment}/{inst}",
    ]
    seen = set()
    for pat in patterns:
        for p in eval_root.glob(pat):
            if p.is_dir():
                sp = str(p.resolve())
                if sp in seen:
                    continue
                seen.add(sp)
                yield p


def _first_instance_path(
    eval_root: Path, model_name: str, suite: str, game: str, experiment: str, task_id: int
) -> Optional[Path]:
    for p in _iter_candidate_instances(eval_root, model_name, suite, game, experiment, task_id):
        return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a merged replay results tree from Adapter-BAR routing selections."
    )
    ap.add_argument("--routing-log", required=True, help="Path to adapter_bar_routing_log.jsonl")
    ap.add_argument("--eval-root", required=True, help="Root containing per-adapter eval outputs")
    ap.add_argument("--out-root", required=True, help="Output root (e.g., playpen-eval/cluster-router)")
    ap.add_argument("--suite", default="clem", help="Suite to replay (default: clem)")
    ap.add_argument("--virtual-model", default="adapter_bar_replay", help="Target merged model name")
    ap.add_argument(
        "--adapter-prefix",
        default="",
        help="Optional model prefix for predicted_adapter, e.g. llama3-8b-sft-cluster_",
    )
    ap.add_argument("--adapter-name-map", default="", help="JSON path: predicted_adapter -> model_name")
    ap.add_argument("--clean-output", action="store_true", help="Delete existing output model folder first")
    args = ap.parse_args()

    routing_log = Path(args.routing_log).expanduser()
    eval_root = Path(args.eval_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    name_map = _load_adapter_name_map(Path(args.adapter_name_map).expanduser() if args.adapter_name_map else None)

    out_model_root = out_root / args.suite / args.virtual_model
    if args.clean_output and out_model_root.exists():
        shutil.rmtree(out_model_root)
    out_model_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(routing_log)
    copied = 0
    missing = 0
    by_adapter = Counter()
    missing_rows: List[dict] = []
    copied_rows: List[dict] = []

    seen_target = set()
    for row in rows:
        game = str(row.get("game", "")).strip()
        experiment = str(row.get("experiment", "")).strip()
        pred = str(row.get("predicted_adapter", "")).strip()
        task_raw = row.get("task_id")
        try:
            task_id = int(task_raw)
        except Exception:
            missing += 1
            missing_rows.append({"reason": "invalid_task_id", "row": row})
            continue
        if not (game and experiment and pred):
            missing += 1
            missing_rows.append({"reason": "missing_fields", "row": row})
            continue

        src_model = name_map.get(pred) or (f"{args.adapter_prefix}{pred}" if args.adapter_prefix else pred)
        src_inst = _first_instance_path(eval_root, src_model, args.suite, game, experiment, task_id)
        if src_inst is None:
            missing += 1
            missing_rows.append(
                {
                    "reason": "instance_not_found",
                    "game": game,
                    "experiment": experiment,
                    "task_id": task_id,
                    "predicted_adapter": pred,
                    "resolved_model_name": src_model,
                }
            )
            continue

        inst_name = f"instance_{task_id:05d}"
        dst_inst = out_model_root / game / experiment / inst_name
        dst_key = str(dst_inst)
        if dst_key in seen_target:
            continue
        seen_target.add(dst_key)

        dst_inst.parent.mkdir(parents=True, exist_ok=True)
        if dst_inst.exists():
            shutil.rmtree(dst_inst)
        shutil.copytree(src_inst, dst_inst)
        copied += 1
        by_adapter[src_model] += 1
        copied_rows.append(
            {
                "game": game,
                "experiment": experiment,
                "task_id": task_id,
                "predicted_adapter": pred,
                "resolved_model_name": src_model,
                "source_instance_path": str(src_inst),
                "target_instance_path": str(dst_inst),
            }
        )

    manifest = {
        "routing_log": str(routing_log),
        "eval_root": str(eval_root),
        "out_root": str(out_root),
        "suite": args.suite,
        "virtual_model": args.virtual_model,
        "total_rows": len(rows),
        "copied_rows": copied,
        "missing_rows": missing,
        "copied_by_source_model": dict(by_adapter),
        "missing_examples": missing_rows[:200],
    }
    (out_root / f"{args.virtual_model}.{args.suite}.materialize.summary.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    with (out_root / f"{args.virtual_model}.{args.suite}.materialize.rows.jsonl").open("w", encoding="utf-8") as f:
        for r in copied_rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
