#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CLUSTER_MAP = {
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
    "cluster_negotiation": ["clean_up", "dond", "hot_air_balloon"],
}


def _iter_scores(source_root: Path, suite: str, model_name: str) -> Iterable[Path]:
    pats = [
        f"**/{suite}/{model_name}/*/*/instance_*/scores.json",
        f"**/{suite}/{suite}/{model_name}/*/*/instance_*/scores.json",
    ]
    seen = set()
    for pat in pats:
        for p in source_root.glob(pat):
            sp = str(p.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            yield p


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _instance_key_from_scores(path: Path) -> Optional[Tuple[str, str, int]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        meta = payload.get("meta") or {}
        try:
            game = str(meta.get("game_name", "")).strip()
            exp = str(meta.get("experiment_name", "")).strip()
            gid = int(meta.get("game_id"))
            if game and exp:
                return (game, exp, gid)
        except Exception:
            pass

    # Fallback: parse from path .../<game>/<experiment>/instance_XXXXX/scores.json
    try:
        inst_dir = path.parent
        exp_dir = inst_dir.parent
        game_dir = exp_dir.parent
        game = game_dir.name
        exp = exp_dir.name
        inst_name = inst_dir.name
        if not inst_name.startswith("instance_"):
            return None
        gid = int(inst_name.split("instance_", 1)[1])
        if game and exp:
            return (game, exp, gid)
    except Exception:
        return None
    return None


def _build_model_index(source_root: Path, suite: str, model_name: str) -> Dict[Tuple[str, str, int], Path]:
    out: Dict[Tuple[str, str, int], Path] = {}
    for sp in _iter_scores(source_root, suite, model_name):
        key = _instance_key_from_scores(sp)
        if key is None:
            continue
        out[key] = sp.parent
    return out


def _cluster_for_game(game: str) -> Optional[str]:
    for cluster, games in CLUSTER_MAP.items():
        if game in games:
            return cluster
    return None


def _all_games_from_cluster_map() -> List[str]:
    out = []
    for games in CLUSTER_MAP.values():
        out.extend(games)
    return out


def _copy_instance_dir(src_inst_dir: Path, dst_inst_dir: Path):
    dst_inst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_inst_dir.exists():
        shutil.rmtree(dst_inst_dir)
    shutil.copytree(src_inst_dir, dst_inst_dir)


def _discover_existing_models(source_root: Path, suite: str) -> List[str]:
    # Best-effort discovery from filesystem layout.
    names = set()
    for p in source_root.glob(f"**/{suite}/*"):
        if p.is_dir():
            # likely .../<suite>/<model_name>
            names.add(p.name)
    return sorted(names)


def _materialize_cluster_oracle(
    source_root: Path,
    out_run_dir: Path,
    suite: str,
    cluster_model_prefix: str,
    negotiation_fallback_model: Optional[str] = None,
):
    game_to_model = {}
    for game in _all_games_from_cluster_map():
        c = _cluster_for_game(game)
        if c == "cluster_negotiation":
            if negotiation_fallback_model:
                game_to_model[game] = negotiation_fallback_model
            continue
        if c is None:
            continue
        short = c.replace("cluster_", "")
        game_to_model[game] = f"{cluster_model_prefix}{short}"

    model_indexes = {}
    for model_name in sorted(set(game_to_model.values())):
        model_indexes[model_name] = _build_model_index(source_root, suite, model_name)

    out_model_name = out_run_dir.name
    out_model_root = out_run_dir / suite / out_model_name
    if out_model_root.exists():
        shutil.rmtree(out_model_root)
    out_model_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    copied_by_model = Counter()
    for game, model_name in sorted(game_to_model.items()):
        idx = model_indexes.get(model_name, {})
        # copy all tasks for this game from the selected model
        game_keys = [k for k in idx.keys() if k[0] == game]
        if not game_keys:
            missing.append({"game": game, "reason": "no_keys_for_game", "model": model_name})
        for (g, exp, gid) in sorted(game_keys):
            src = idx[(g, exp, gid)]
            dst = out_model_root / g / exp / f"instance_{gid:05d}"
            _copy_instance_dir(src, dst)
            copied += 1
            copied_by_model[model_name] += 1

    return {
        "copied_instances": copied,
        "missing": missing,
        "copied_by_model": dict(copied_by_model),
        "game_to_model": game_to_model,
    }


def _materialize_pergame_oracle(
    source_root: Path,
    out_run_dir: Path,
    suite: str,
    pergame_model_prefix: str,
):
    game_to_model = {g: f"{pergame_model_prefix}{g}" for g in _all_games_from_cluster_map()}
    model_indexes = {}
    for model_name in sorted(set(game_to_model.values())):
        model_indexes[model_name] = _build_model_index(source_root, suite, model_name)

    out_model_name = out_run_dir.name
    out_model_root = out_run_dir / suite / out_model_name
    if out_model_root.exists():
        shutil.rmtree(out_model_root)
    out_model_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    copied_by_model = Counter()
    for game, model_name in sorted(game_to_model.items()):
        idx = model_indexes.get(model_name, {})
        game_keys = [k for k in idx.keys() if k[0] == game]
        if not game_keys:
            missing.append({"game": game, "reason": "no_keys_for_game", "model": model_name})
        for (g, exp, gid) in sorted(game_keys):
            src = idx[(g, exp, gid)]
            dst = out_model_root / g / exp / f"instance_{gid:05d}"
            _copy_instance_dir(src, dst)
            copied += 1
            copied_by_model[model_name] += 1

    return {
        "copied_instances": copied,
        "missing": missing,
        "copied_by_model": dict(copied_by_model),
        "game_to_model": game_to_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Materialize oracle MoE results from existing adapter results.")
    parser.add_argument("--source-root", type=Path, required=True, help="Source root (cluster adapters or per-game adapters)")
    parser.add_argument("--out-run-dir", type=Path, required=True, help="Output run dir to create")
    parser.add_argument("--suite", type=str, default="clem", help="Suite to materialize (default: clem)")
    parser.add_argument("--oracle-type", choices=["cluster", "pergame"], required=True)
    parser.add_argument("--cluster-model-prefix", type=str, default="llama3-8b-sft-cluster_")
    parser.add_argument("--pergame-model-prefix", type=str, default="llama3-8b-sft-game_")
    parser.add_argument(
        "--negotiation-fallback-model",
        type=str,
        default="",
        help="Optional full model name used for clean_up/dond/hot_air_balloon in cluster oracle.",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_run_dir = args.out_run_dir.expanduser().resolve()
    out_run_dir.mkdir(parents=True, exist_ok=True)

    if args.oracle_type == "cluster":
        result = _materialize_cluster_oracle(
            source_root=source_root,
            out_run_dir=out_run_dir,
            suite=args.suite,
            cluster_model_prefix=args.cluster_model_prefix,
            negotiation_fallback_model=(args.negotiation_fallback_model.strip() or None),
        )
    else:
        result = _materialize_pergame_oracle(
            source_root=source_root,
            out_run_dir=out_run_dir,
            suite=args.suite,
            pergame_model_prefix=args.pergame_model_prefix,
        )

    summary = {
        "source_root": str(source_root),
        "out_run_dir": str(out_run_dir),
        "suite": args.suite,
        "oracle_type": args.oracle_type,
        **result,
    }
    (out_run_dir / f"{out_run_dir.name}.materialize.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
