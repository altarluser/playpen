#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_registry(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError(f"Invalid registry JSON in {path}")


def _latest_checkpoint(adapter_dir: Path) -> Optional[Path]:
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        return None
    cks = [p for p in adapter_dir.glob("checkpoint-*") if p.is_dir()]
    if not cks:
        return adapter_dir

    def key(p: Path) -> Tuple[int, str]:
        m = re.search(r"checkpoint-(\d+)$", p.name)
        return (int(m.group(1)) if m else -1, p.name)

    cks.sort(key=key)
    return cks[-1]


def _entry_by_name(entries: List[dict]) -> Dict[str, dict]:
    out = {}
    for e in entries:
        n = e.get("model_name")
        if isinstance(n, str) and n:
            out[n] = e
    return out


def _ensure_base_entry(entries: List[dict], learner_model: str) -> dict:
    by_name = _entry_by_name(entries)
    if learner_model not in by_name:
        raise ValueError(f"Base learner model '{learner_model}' not found in registry.")
    return by_name[learner_model]


def _upsert(entries: List[dict], new_entry: dict) -> None:
    name = str(new_entry.get("model_name"))
    for i, e in enumerate(entries):
        if str(e.get("model_name")) == name:
            entries[i] = new_entry
            return
    entries.append(new_entry)


def _make_sft_entry(base_entry: dict, model_name: str, peft_model: str) -> dict:
    e = deepcopy(base_entry)
    e["model_name"] = model_name
    cfg = dict(e.get("model_config") or {})
    cfg["peft_model"] = peft_model
    cfg["requires_api_key"] = False
    e["model_config"] = cfg
    e.pop("lookup_source", None)
    return e


def _make_merge_entry(base_entry: dict, model_name: str, method: str, peft_models: List[str]) -> dict:
    e = deepcopy(base_entry)
    e["model_name"] = model_name
    cfg = dict(e.get("model_config") or {})
    cfg.pop("peft_model", None)
    cfg["peft_models"] = list(peft_models)
    cfg["merge"] = str(method)
    cfg["requires_api_key"] = False
    e["model_config"] = cfg
    e.pop("lookup_source", None)
    return e


def _make_moe_entry(base_entry: dict, model_name: str, adapter_path: str, moe_state_path: Optional[str]) -> dict:
    e = deepcopy(base_entry)
    e["model_name"] = model_name
    cfg = dict(e.get("model_config") or {})
    cfg["peft_model"] = adapter_path
    cfg["moe_lora_adapter_path"] = adapter_path
    if moe_state_path:
        cfg["moe_state_path"] = moe_state_path
    cfg["moe_enabled"] = True
    cfg["moe_mode"] = "residual_skill"
    cfg["requires_api_key"] = False
    e["model_config"] = cfg
    e.pop("lookup_source", None)
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description="Build consolidated manual eval model registry.")
    ap.add_argument("--base-registry", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--learner-model", type=str, default="llama3-8b")
    ap.add_argument("--adapter-root", type=Path, required=True,
                    help="Path like .../llama3-8b-adapters with game_*/cluster_* dirs.")
    ap.add_argument("--moe-parent", type=Path, required=True,
                    help="Parent path containing moe-residual-sweep6-* dirs.")
    ap.add_argument("--moe-sweep-prefix", type=str, default="moe-residual-sweep6-")
    ap.add_argument("--hf-snapshot-path", type=Path, default=None)
    args = ap.parse_args()

    entries = _load_registry(args.base_registry.expanduser())
    base_entry = _ensure_base_entry(entries, args.learner_model)

    # Optionally pin base snapshot path.
    if args.hf_snapshot_path:
        snap = args.hf_snapshot_path.expanduser()
        if not snap.exists():
            raise FileNotFoundError(f"--hf-snapshot-path not found: {snap}")
        for e in entries:
            if str(e.get("model_name")) == args.learner_model:
                e["huggingface_id"] = str(snap)
                cfg = dict(e.get("model_config") or {})
                cfg["requires_api_key"] = False
                e["model_config"] = cfg

    adapter_root = args.adapter_root.expanduser()
    if not adapter_root.exists():
        raise FileNotFoundError(f"--adapter-root not found: {adapter_root}")

    # Cluster adapters.
    cluster_paths: Dict[str, str] = {}
    for p in sorted(adapter_root.glob("cluster_*")):
        if not p.is_dir():
            continue
        ck = _latest_checkpoint(p)
        if ck is None:
            continue
        suffix = p.name
        model_name = f"{args.learner_model}-sft-{suffix}"
        _upsert(entries, _make_sft_entry(base_entry, model_name, str(ck)))
        cluster_paths[suffix] = str(ck)

    # Per-game adapters.
    game_paths: Dict[str, str] = {}
    for p in sorted(adapter_root.glob("game_*")):
        if not p.is_dir():
            continue
        ck = _latest_checkpoint(p)
        if ck is None:
            continue
        suffix = p.name
        model_name = f"{args.learner_model}-sft-{suffix}"
        _upsert(entries, _make_sft_entry(base_entry, model_name, str(ck)))
        game_paths[suffix] = str(ck)

    # Merge entries: clusters WA/TA.
    required_clusters = [
        "cluster_explorationnavigation",
        "cluster_wordguessing",
        "cluster_cooperation",
    ]
    if all(k in cluster_paths for k in required_clusters):
        peft_models = [cluster_paths[k] for k in required_clusters]
        _upsert(
            entries,
            _make_merge_entry(
                base_entry,
                f"{args.learner_model}-merge-clusters-selected-wa",
                "weight_averaging",
                peft_models,
            ),
        )
        _upsert(
            entries,
            _make_merge_entry(
                base_entry,
                f"{args.learner_model}-merge-clusters-selected-ta",
                "task_arithmetic",
                peft_models,
            ),
        )

    # Merge entries: per-game WA/TA.
    if game_paths:
        peft_models = [game_paths[k] for k in sorted(game_paths.keys())]
        _upsert(
            entries,
            _make_merge_entry(
                base_entry,
                f"{args.learner_model}-merge-per-game-wa",
                "weight_averaging",
                peft_models,
            ),
        )
        _upsert(
            entries,
            _make_merge_entry(
                base_entry,
                f"{args.learner_model}-merge-per-game-ta",
                "task_arithmetic",
                peft_models,
            ),
        )

    # MoE residual runs from sweep folders.
    moe_parent = args.moe_parent.expanduser()
    moe_dirs = []
    for sweep_dir in sorted(moe_parent.glob(f"{args.moe_sweep_prefix}*")):
        cand = sweep_dir / f"{args.learner_model}-adapters"
        if not cand.exists() or not cand.is_dir():
            continue
        for run_dir in sorted(cand.glob("moe-residual-*")):
            if run_dir.is_dir():
                moe_dirs.append(run_dir)

    for run_dir in moe_dirs:
        ck = _latest_checkpoint(run_dir)
        if ck is None:
            continue
        model_name = f"{args.learner_model}-sft-{run_dir.name}"
        moe_state = run_dir / "moe" / "moe_state.pt"
        moe_state_path = str(moe_state) if moe_state.exists() else None
        _upsert(entries, _make_moe_entry(base_entry, model_name, str(ck), moe_state_path))

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(entries, indent=2), encoding="utf-8")

    print(f"Wrote registry: {args.output.expanduser()}")
    print(f"Detected clusters: {len(cluster_paths)}")
    print(f"Detected game adapters: {len(game_paths)}")
    print(f"Detected MoE runs: {len(moe_dirs)}")


if __name__ == "__main__":
    main()
