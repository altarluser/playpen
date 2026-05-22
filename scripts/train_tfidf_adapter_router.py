#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from playpen.adapter_bar_sequence import (
    _input_text_from_example,
    _load_eval_dataset,
    build_router_splits,
    get_game_name,
    load_adapter_bar_mode_config,
    normalize_game_name,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _row_get(row: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return default


def _extract_eval_row(row: Mapping[str, Any], router_type: str, cfg: Mapping[str, Any]) -> Tuple[str, str, int, str]:
    game = normalize_game_name(get_game_name(row, training=False) or "")
    if not game:
        raise ValueError("missing game name")
    exp = str(_row_get(row, "experiment", "experiment_name", default="") or "").strip()
    if not exp:
        exp = "unknown_experiment"
    task_raw = _row_get(row, "task_id", "game_id", "instance_id", default=None)
    if task_raw is None:
        raise ValueError("missing task id")
    task_id = int(task_raw)
    text = _input_text_from_example(row)
    _ = (router_type, cfg)  # inference routing does not require a known gold label
    return game, exp, task_id, text if isinstance(text, str) else str(text)


def _train_and_route(
    *,
    config_path: str,
    router_type: str,
    output_root: Path,
    max_eval_rows: int | None = None,
) -> Dict[str, Any]:
    cfg = load_adapter_bar_mode_config(config_path, router_type)
    # Build the exact split format used by Adapter-BAR.
    split_summary = build_router_splits(config_path, router_type)
    out_cfg_dir = Path(str(cfg.get("output_dir", f"outputs/adapter_bar_sequence/{router_type}"))).expanduser()
    split_dir = out_cfg_dir / "splits"
    train_rows = _read_jsonl(split_dir / "router_train.jsonl")
    val_rows = _read_jsonl(split_dir / "router_validation.jsonl")

    x_train = [str(r.get("input_text") or "") for r in train_rows]
    y_train = [str(r.get("label_name") or "") for r in train_rows]
    x_val = [str(r.get("input_text") or "") for r in val_rows]
    y_val = [str(r.get("label_name") or "") for r in val_rows]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train_t = vec.fit_transform(x_train)
    x_val_t = vec.transform(x_val) if x_val else None

    clf = LogisticRegression(max_iter=5000)
    clf.fit(x_train_t, y_train)

    val_acc = None
    if x_val_t is not None and len(y_val) > 0:
        y_hat = clf.predict(x_val_t)
        val_acc = float(accuracy_score(y_val, y_hat))

    model_dir = output_root / router_type / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)
    with (model_dir / "logreg_router.pkl").open("wb") as f:
        pickle.dump(clf, f)

    suite_specs = [("clem", "instances"), ("static", "instances-static")]
    routing_paths: Dict[str, str] = {}
    suite_stats: Dict[str, Dict[str, Any]] = {}

    for suite_name, dataset_name in suite_specs:
        ds = _load_eval_dataset(dataset_name)
        rows = [dict(ds[i]) for i in range(len(ds))]
        if max_eval_rows is not None and max_eval_rows > 0:
            rows = rows[: int(max_eval_rows)]

        out_rows = []
        skipped = 0
        for idx, row in enumerate(rows):
            try:
                game, exp, task_id, text = _extract_eval_row(row, router_type, cfg)
            except Exception:
                skipped += 1
                continue
            x = vec.transform([text or ""])
            probs = clf.predict_proba(x)[0]
            pred_idx = int(probs.argmax())
            pred_label = str(clf.classes_[pred_idx])
            out_rows.append(
                {
                    "suite": suite_name,
                    "instance_id": int(idx),
                    "example_id": row.get("example_id") or f"{game}:{exp}:{task_id}",
                    "game": game,
                    "experiment": exp,
                    "task_id": int(task_id),
                    "predicted_adapter": pred_label,
                    "predicted_adapter_id": pred_idx,
                    "router_confidence": float(probs[pred_idx]),
                    "router_probs": [float(p) for p in probs],
                    "routing_mode": f"tfidf_logreg_{router_type}",
                }
            )

        out_dir = output_root / router_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"adapter_bar_routing_log.{suite_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        routing_paths[suite_name] = str(out_path)
        suite_stats[suite_name] = {
            "dataset": dataset_name,
            "total_rows": len(rows),
            "routed_rows": len(out_rows),
            "skipped_rows": skipped,
        }

    summary = {
        "router_type": router_type,
        "config_path": str(config_path),
        "split_summary": split_summary,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_accuracy": val_acc,
        "routing_paths": routing_paths,
        "suite_stats": suite_stats,
    }
    summary_path = output_root / router_type / "tfidf_router_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TF-IDF + logistic adapter router and emit routing-only JSONLs.")
    ap.add_argument("--config", type=str, default="configs/adapter_bar_sequence.yaml")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--router-types", type=str, default="cluster,game")
    ap.add_argument("--max-eval-rows", type=int, default=None)
    args = ap.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    modes = [x.strip() for x in str(args.router_types).split(",") if x.strip()]
    all_summaries = {}
    for mode in modes:
        all_summaries[mode] = _train_and_route(
            config_path=args.config,
            router_type=mode,
            output_root=output_root,
            max_eval_rows=args.max_eval_rows,
        )
    (output_root / "tfidf_router_all_summary.json").write_text(
        json.dumps(all_summaries, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
