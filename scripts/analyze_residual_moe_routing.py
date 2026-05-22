#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_label(name: str) -> str:
    label = str(name)
    for prefix in ("llama3-8b-sft-", "llama3-8b-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("explorationnavigation", "exploration navigation")
    return label.replace("_", " ")


def _load_expected_meta(playpen_eval_root: Path, model_name: str) -> Dict[str, object]:
    run_json = playpen_eval_root / "moe-residual-sweep" / model_name / "clem" / model_name / "run.json"
    if not run_json.exists():
        return {
            "model_name": model_name,
            "run_json": str(run_json),
            "expected_num_experts": np.nan,
            "expected_num_layers": np.nan,
            "expected_alpha": np.nan,
            "state_path": "",
            "has_expected_meta": False,
        }
    obj = json.loads(run_json.read_text(encoding="utf-8"))
    model_config = obj["player_models"]["0"]["model_spec"].get("model_config", {})
    return {
        "model_name": model_name,
        "run_json": str(run_json),
        "expected_num_experts": model_config.get("moe_num_experts", np.nan),
        "expected_num_layers": model_config.get("moe_num_moe_layers", np.nan),
        "expected_alpha": model_config.get("moe_alpha", np.nan),
        "state_path": model_config.get("moe_state_path", ""),
        "has_expected_meta": True,
    }


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _heatmap(
    values: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
) -> None:
    fig_h = max(4.5, 0.35 * len(row_labels))
    fig_w = max(6.0, 0.7 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7, color="#111827")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_one(routing_dir: Path, playpen_eval_root: Path, out_root: Path) -> Dict[str, object]:
    model_name = routing_dir.parent.name if routing_dir.name == "moe-routing" else routing_dir.name
    latent_csv = routing_dir / "latent_eval_usage.csv"
    if not latent_csv.exists():
        raise FileNotFoundError(f"missing {latent_csv}")

    df = pd.read_csv(latent_csv)
    if df.empty:
        raise ValueError(f"empty routing log: {latent_csv}")

    meta = _load_expected_meta(playpen_eval_root, model_name)
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df["expert_id"] = df["expert_id"].astype(int)
    df["layer"] = df["layer"].astype(int)
    df["top1_count"] = df["top1_count"].astype(float)
    df["top1_proportion"] = df["top1_proportion"].astype(float)
    df["tokens"] = df["tokens"].astype(float)
    df["avg_entropy"] = df["avg_entropy"].astype(float)

    observed_experts = sorted(df["expert_id"].unique().tolist())
    observed_layers = sorted(df["layer"].unique().tolist())
    validation = {
        **meta,
        "routing_dir": str(routing_dir),
        "observed_num_experts": len(observed_experts),
        "observed_num_layers": len(observed_layers),
        "observed_expert_ids": observed_experts,
        "observed_layers": observed_layers,
        "num_games": int(df["game"].nunique()),
        "games": sorted(df["game"].dropna().unique().tolist()),
    }
    expected_e = meta["expected_num_experts"]
    expected_l = meta["expected_num_layers"]
    validation["experts_match"] = bool(pd.notna(expected_e) and int(expected_e) == len(observed_experts))
    validation["layers_match"] = bool(pd.notna(expected_l) and int(expected_l) == len(observed_layers))
    validation["likely_default_fallback_used"] = (
        len(observed_experts) == 4 and len(observed_layers) == 6 and (
            (pd.notna(expected_e) and int(expected_e) != 4) or (pd.notna(expected_l) and int(expected_l) != 6)
        )
    )
    (out_dir / "routing_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    global_layer = (
        df.groupby(["layer", "expert_id"], as_index=False)
        .agg(top1_count=("top1_count", "sum"), tokens=("tokens", "sum"), avg_entropy=("avg_entropy", "mean"))
    )
    layer_totals = global_layer.groupby("layer")["top1_count"].transform("sum").replace(0, np.nan)
    global_layer["usage_share"] = global_layer["top1_count"] / layer_totals
    _save_csv(global_layer, out_dir / "global_layer_expert_usage.csv")

    game_expert = (
        df.groupby(["game", "expert_id"], as_index=False)
        .agg(top1_count=("top1_count", "sum"), tokens=("tokens", "sum"), avg_entropy=("avg_entropy", "mean"))
    )
    game_totals = game_expert.groupby("game")["top1_count"].transform("sum").replace(0, np.nan)
    game_expert["usage_share"] = game_expert["top1_count"] / game_totals
    _save_csv(game_expert, out_dir / "game_expert_usage.csv")

    game_layer = (
        df.groupby(["game", "layer", "expert_id"], as_index=False)
        .agg(top1_count=("top1_count", "sum"), tokens=("tokens", "sum"), avg_entropy=("avg_entropy", "mean"))
    )
    game_layer_totals = game_layer.groupby(["game", "layer"])["top1_count"].transform("sum").replace(0, np.nan)
    game_layer["usage_share"] = game_layer["top1_count"] / game_layer_totals
    _save_csv(game_layer, out_dir / "game_layer_expert_usage.csv")

    game_layer_entropy = (
        df.groupby(["game", "layer"], as_index=False)
        .agg(avg_entropy=("avg_entropy", "mean"), tokens=("tokens", "sum"))
        .sort_values(["game", "layer"])
    )
    _save_csv(game_layer_entropy, out_dir / "game_layer_entropy.csv")

    dominant = game_layer.sort_values(["game", "layer", "usage_share"], ascending=[True, True, False]).drop_duplicates(
        ["game", "layer"]
    )
    dominant = dominant[["game", "layer", "expert_id", "usage_share", "avg_entropy"]].rename(
        columns={"expert_id": "dominant_expert", "usage_share": "dominant_share"}
    )
    _save_csv(dominant, out_dir / "game_layer_dominant_expert.csv")

    # Plots
    gl_piv = global_layer.pivot(index="layer", columns="expert_id", values="usage_share").sort_index()
    _heatmap(
        gl_piv.values,
        [str(x) for x in gl_piv.index],
        [f"E{x}" for x in gl_piv.columns],
        f"{_plot_label(model_name)}: global expert usage by layer",
        out_dir / "global_layer_expert_share_heatmap.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annotate=True,
    )

    ge_piv = game_expert.pivot(index="game", columns="expert_id", values="usage_share")
    game_order = sorted(ge_piv.index.tolist())
    ge_piv = ge_piv.reindex(game_order)
    _heatmap(
        ge_piv.values,
        [_plot_label(x) for x in ge_piv.index],
        [f"E{x}" for x in ge_piv.columns],
        f"{_plot_label(model_name)}: game-level expert usage",
        out_dir / "game_expert_share_heatmap.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annotate=False,
    )

    ent_piv = game_layer_entropy.pivot(index="game", columns="layer", values="avg_entropy").reindex(game_order)
    _heatmap(
        ent_piv.values,
        [_plot_label(x) for x in ent_piv.index],
        [str(x) for x in ent_piv.columns],
        f"{_plot_label(model_name)}: routing entropy by game/layer",
        out_dir / "game_layer_entropy_heatmap.png",
        cmap="magma",
        annotate=False,
    )

    dom_piv = dominant.pivot(index="game", columns="layer", values="dominant_expert").reindex(game_order)
    _heatmap(
        dom_piv.values,
        [_plot_label(x) for x in dom_piv.index],
        [str(x) for x in dom_piv.columns],
        f"{_plot_label(model_name)}: dominant expert by game/layer",
        out_dir / "game_layer_dominant_expert_heatmap.png",
        cmap="tab10",
        annotate=True,
        fmt="{:.0f}",
    )

    top_games = (
        game_expert.sort_values(["expert_id", "usage_share"], ascending=[True, False])
        .groupby("expert_id")
        .head(5)
        .copy()
    )
    _save_csv(top_games, out_dir / "expert_top_games.csv")

    return validation


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze residual MoE routing logs and validate them against playpen-eval run metadata.")
    ap.add_argument(
        "--routing-dirs",
        nargs="+",
        required=True,
        help="One or more routing directories, typically analysis_outputs/<model>/moe-routing",
    )
    ap.add_argument("--playpen-eval-root", type=Path, default=Path("playpen-eval"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/residual_moe_routing_analysis"))
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.routing_dirs:
        routing_dir = Path(p).expanduser().resolve()
        rows.append(analyze_one(routing_dir, args.playpen_eval_root.expanduser().resolve(), out_dir))

    summary = pd.DataFrame(rows)
    _save_csv(summary, out_dir / "combined_validation.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
