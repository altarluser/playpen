#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLUSTER_MAP: Dict[str, List[str]] = {
    "wordguessing": [
        "codenames",
        "taboo",
        "guesswhat",
        "wordle",
        "wordle_withclue",
        "wordle_withcritic",
    ],
    "explorationnavigation": [
        "adventuregame",
        "textmapworld",
        "textmapworld_graphreasoning",
        "textmapworld_specificroom",
    ],
    "cooperation": [
        "imagegame",
        "matchit_ascii",
        "referencegame",
        "privateshared",
    ],
}

OOD_GAMES = ["dond", "clean_up", "hot_air_balloon"]
IN_DOMAIN_GAMES = [g for games in CLUSTER_MAP.values() for g in games]
ALL_ANALYSIS_GAMES = IN_DOMAIN_GAMES + OOD_GAMES
GAME_GROUP = {game: group for group, games in CLUSTER_MAP.items() for game in games}
GAME_GROUP.update({game: "outdomain_negotiation" for game in OOD_GAMES})

FAMILY_COLORS = {
    "base": "#5b6472",
    "generalist": "#2563eb",
    "single_adapter": "#8b5cf6",
    "oracle_moe": "#16a34a",
    "learned_router_moe": "#f59e0b",
    "loss_based_moe": "#ef4444",
    "merging": "#0f766e",
    "residual_moe": "#db2777",
    "unknown": "#94a3b8",
}

ROUTING_COLORS = {
    "Oracle": "#16a34a",
    "Learned Router": "#f59e0b",
    "Loss-based": "#ef4444",
}


class _FirstTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables_seen = 0
        self.in_first_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_text: List[str] = []
        self.current_row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self.tables_seen += 1
            self.in_first_table = self.tables_seen == 1
        if not self.in_first_table:
            return
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"th", "td"} and self.in_row:
            self.in_cell = True
            self.cell_text = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if not self.in_first_table:
            return
        if tag in {"th", "td"} and self.in_cell:
            self.current_row.append(" ".join("".join(self.cell_text).split()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif tag == "table":
            self.in_first_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)


def _warn(message: str):
    print(f"[warn] {message}")


def _to_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if math.isfinite(out) else np.nan


def _mean(values: Iterable[float]) -> float:
    finite = [_to_float(v) for v in values]
    finite = [v for v in finite if math.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def _score_from_played_quality(played: float, quality: float) -> float:
    p = _to_float(played)
    if not math.isfinite(p):
        return np.nan
    if p == 0.0:
        return 0.0
    q = _to_float(quality)
    if not math.isfinite(q):
        return np.nan
    return p * q / 100.0


def _plot_label(name: str) -> str:
    label = str(name)
    for prefix in ("llama3-8b-sft-", "llama3-8b-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = re.sub(r"(^|[-_\s])game([_\s-]+)", r"\1", label, flags=re.IGNORECASE)
    label = re.sub(r"(^|[-_\s])cluster([_\s-]+)", r"\1", label, flags=re.IGNORECASE)
    label = label.replace("explorationnavigation", "exploration navigation")
    label = label.replace("pergame", "per-game")
    label = re.sub(r"[_-]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def _scatter_label(row: "pd.Series") -> str:
    """Concise scatter-plot label: omits family type already shown by legend color."""
    family = str(row.get("family_group", "unknown"))
    method = str(row.get("method", ""))
    granularity = str(row.get("expert_granularity", ""))

    if family == "oracle_moe":
        return "per-game oracle" if granularity == "per_game" else "cluster oracle"

    lbl = _plot_label(method)

    if family == "residual_moe":
        lbl = re.sub(r"(?i)^moe\s+residual\s*", "", lbl).strip()
        lbl = re.sub(r"(?i)^residual\s+moe\s*", "", lbl).strip()
    elif family == "single_adapter":
        lbl = re.sub(r"(?i)^(per[-\s]?game|cluster)\s+adapter:\s*", "", lbl).strip()
    elif family in ("learned_router_moe", "loss_based_moe"):
        lbl = re.sub(r"(?i)^(per[-\s]?game|cluster)\s+", "", lbl).strip()
        lbl = re.sub(r"(?i)\s+moe$", "", lbl).strip()

    return lbl[:26]


def _read_csv_row(path: Path) -> Tuple[Dict[str, str], List[str]]:
    if not path.exists():
        return {}, []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        _warn(f"could not read CSV {path}: {exc}")
        return {}, []
    if not rows:
        return {}, []
    return dict(rows[0]), list(rows[0].keys())


def _read_val_json(run_dir: Path) -> Tuple[Dict, Optional[Path]]:
    exact = run_dir / f"{run_dir.name}.val.json"
    candidates = [exact] if exact.exists() else sorted(run_dir.glob("*.val.json"))
    if not candidates:
        return {}, None
    path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception as exc:
        _warn(f"could not read JSON {path}: {exc}")
        return {}, path


def _load_run_dirs(root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if "ignore" in d.parts:
            continue
        if d.name in {"clem", "static", "val", "plots", "correlation"}:
            continue
        has_results = (d / "clem" / "results.csv").exists() or (d / "static" / "results.csv").exists()
        has_val = (d / f"{d.name}.val.json").exists()
        if has_results or has_val:
            run_dirs.append(d)
    seen = set()
    deduped = []
    for d in sorted(run_dirs):
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    return deduped


def _game_metric(row: Dict[str, str], game: str, metric: str) -> float:
    suffix = "% Played" if metric == "played" else "Quality Score"
    return _to_float(row.get(f"{game}, {suffix}"))


def _group_metrics(row: Dict[str, str], games: List[str]) -> Tuple[float, float, float]:
    played = _mean(_game_metric(row, g, "played") for g in games)
    quality = _mean(_game_metric(row, g, "quality") for g in games)
    return _score_from_played_quality(played, quality), played, quality


def _single_game_metrics(row: Dict[str, str], game: str) -> Tuple[float, float, float]:
    played = _game_metric(row, game, "played")
    quality = _game_metric(row, game, "quality")
    return _score_from_played_quality(played, quality), played, quality


def _parse_residual_params(name: str) -> Tuple[float, float]:
    low = name.lower()
    experts = np.nan
    alpha = np.nan
    m_exp = re.search(r"residual-(\d+)e", low)
    if m_exp:
        experts = _to_float(m_exp.group(1))
    m_alpha = re.search(r"(?:^|[-_])a([0-9]+(?:\.[0-9]+)?)(?:[-_]|$)", low)
    if m_alpha:
        alpha = _to_float(m_alpha.group(1))
    return alpha, experts


def _granularity_from_name(text: str) -> str:
    low = text.lower()
    if "per-game" in low or "pergame" in low or "per_game" in low or "game_" in low:
        return "per_game"
    if "cluster" in low:
        return "cluster"
    if "residual" in low:
        return "residual_moe"
    return "none"


def _method_fields(run_dir: Path, root: Path) -> Dict[str, object]:
    rel = str(run_dir.resolve().relative_to(root.resolve())).replace("\\", "/")
    low = rel.lower()
    name = run_dir.name
    granularity = _granularity_from_name(rel)

    fields = {
        "method": _plot_label(name),
        "family_group": "unknown",
        "expert_granularity": granularity,
        "composition_level": "unknown",
        "selection_method": "unknown",
        "is_adapter_based": False,
        "is_residual_moe": False,
        "method_group": "unknown",
        "method_subtype": "unknown",
    }

    if name == "llama3-8b":
        fields.update(
            method="Base model",
            family_group="base",
            expert_granularity="none",
            composition_level="none",
            selection_method="none",
            method_group="base_model",
            method_subtype="base",
        )
    elif "cluster_all" in low:
        fields.update(
            method="Generalist LoRA",
            family_group="generalist",
            expert_granularity="generalist",
            composition_level="single_adapter",
            selection_method="none",
            is_adapter_based=True,
            method_group="generalist_adapter",
            method_subtype="generalist",
        )
    elif "residual" in low and "moe" in low:
        fields.update(
            method=_plot_label(name),
            family_group="residual_moe",
            expert_granularity="internal_experts",
            composition_level="token_level",
            selection_method="learned_token_router",
            is_adapter_based=False,
            is_residual_moe=True,
            method_group="residual_moe",
            method_subtype="residual_moe",
        )
    elif "oracle" in low and "moe" in low:
        prefix = "Per-game" if granularity == "per_game" else "Cluster"
        fields.update(
            method=f"{prefix} Oracle MoE",
            family_group="oracle_moe",
            composition_level="task_level",
            selection_method="oracle",
            is_adapter_based=True,
            method_group="oracle_routing",
            method_subtype=f"{granularity}_oracle",
        )
    elif "loss-router" in low or "lossrouter" in low:
        prefix = "Per-game" if granularity == "per_game" else "Cluster"
        fields.update(
            method=f"{prefix} Loss-based MoE",
            family_group="loss_based_moe",
            composition_level="task_level",
            selection_method="loss_based",
            is_adapter_based=True,
            method_group="loss_based_router",
            method_subtype=f"{granularity}_loss_based",
        )
    elif "router" in low or "classifier" in low or "tfidf" in low:
        prefix = "Per-game" if granularity == "per_game" else "Cluster"
        fields.update(
            method=f"{prefix} Learned Router MoE",
            family_group="learned_router_moe",
            composition_level="task_level",
            selection_method="learned_router",
            is_adapter_based=True,
            method_group="learned_router",
            method_subtype=f"{granularity}_learned_router",
        )
    elif "merge" in low:
        merge_method = "weight_averaging" if re.search(r"(^|[-_])wa($|[-_])", low) else "task_arithmetic"
        method_label = "WA" if merge_method == "weight_averaging" else "TA"
        prefix = "Per-game" if granularity == "per_game" else "Cluster"
        fields.update(
            method=f"{prefix} {method_label} merge",
            family_group="merging",
            composition_level="parameter_level",
            selection_method=merge_method,
            is_adapter_based=True,
            method_group="merging",
            method_subtype=merge_method,
        )
    elif "per-game adapters" in low:
        fields.update(
            method=f"Per-game adapter: {_plot_label(name)}",
            family_group="single_adapter",
            expert_granularity="per_game",
            composition_level="single_adapter",
            selection_method="none",
            is_adapter_based=True,
            method_group="per_game_adapter",
            method_subtype="single_per_game_adapter",
        )
    elif "cluster adapters" in low and "cluster_" in low:
        fields.update(
            method=f"Cluster adapter: {_plot_label(name)}",
            family_group="single_adapter",
            expert_granularity="cluster",
            composition_level="single_adapter",
            selection_method="none",
            is_adapter_based=True,
            method_group="cluster_adapter",
            method_subtype="single_cluster_adapter",
        )

    return fields


def _load_leaderboard(root: Path) -> pd.DataFrame:
    path = root / "leaderboard.html"
    if not path.exists():
        _warn(f"leaderboard not found: {path}")
        return pd.DataFrame()
    try:
        tables = pd.read_html(str(path))
    except Exception as exc:
        _warn(f"pandas could not parse leaderboard {path}: {exc}; using stdlib HTML parser")
        parser = _FirstTableParser()
        try:
            parser.feed(path.read_text(encoding="utf-8"))
        except Exception as fallback_exc:
            _warn(f"could not parse leaderboard {path}: {fallback_exc}")
            return pd.DataFrame()
        if len(parser.rows) < 2:
            _warn(f"leaderboard has no parseable table rows: {path}")
            return pd.DataFrame()
        header = parser.rows[0]
        data = [row for row in parser.rows[1:] if len(row) == len(header)]
        df = pd.DataFrame(data, columns=header)
    else:
        if not tables:
            _warn(f"leaderboard has no tables: {path}")
            return pd.DataFrame()
        df = tables[0].copy()
    print(f"[input][leaderboard] {path} columns={list(df.columns)} rows={len(df)}")
    if "Run" in df.columns:
        df["Run"] = df["Run"].astype(str).str.strip().str.replace("\\\\", "/", regex=True)
    return df


def _apply_leaderboard_scores(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    leaderboard = _load_leaderboard(root)
    if leaderboard.empty or "Run" not in leaderboard.columns:
        return df

    score_cols = {
        "ClemScore": "clemscore_overall",
        "StatScore": "statscore",
        "WG Score": "wordguessing_score",
        "WG Played": "wordguessing_played",
        "WG Q": "wordguessing_quality",
        "EN Score": "explorationnavigation_score",
        "EN Played": "explorationnavigation_played",
        "EN Q": "explorationnavigation_quality",
        "CO Score": "cooperation_score",
        "CO Played": "cooperation_played",
        "CO Q": "cooperation_quality",
        "NEG Score": "outdomain_score",
        "NEG Played": "outdomain_played",
        "NEG Q": "outdomain_quality",
    }
    available = {k: v for k, v in score_cols.items() if k in leaderboard.columns}
    lb = leaderboard[["Run"] + list(available.keys())].copy()
    for col in available:
        lb[col] = lb[col].apply(_to_float)
    lb_map = {
        str(row["Run"]): {target: row[source] for source, target in available.items()}
        for _, row in lb.iterrows()
    }

    out = df.copy()
    root_resolved = root.resolve()
    for idx, row in out.iterrows():
        keys = [str(row.get("run_path", "")), str(row.get("run_name", ""))]
        try:
            rel = str(Path(row["run_dir"]).resolve().relative_to(root_resolved)).replace("\\", "/")
            keys.insert(0, rel)
        except Exception:
            pass
        match = next((lb_map[k] for k in keys if k in lb_map), None)
        if not match:
            continue
        for col, val in match.items():
            if pd.notna(val):
                out.at[idx, col] = val
    return out


def _harmonize_oracle_proxy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Use source-adapter family means for oracle MoE StatScore and OOD metrics."""
    out = df.copy()

    def _mean_from(mask: pd.Series, col: str) -> float:
        if col not in out.columns:
            return np.nan
        vals = pd.to_numeric(out.loc[mask, col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else np.nan

    pools = {
        "per_game": out["method_group"].astype(str).eq("per_game_adapter"),
        "cluster": out["method_group"].astype(str).eq("cluster_adapter"),
    }
    targets = {
        "per_game": out["method_subtype"].astype(str).eq("per_game_oracle")
        | out["run_name"].astype(str).str.contains("pergame-oracle-moe|pergame_oracle_moe", case=False, regex=True, na=False),
        "cluster": out["method_subtype"].astype(str).eq("cluster_oracle")
        | out["run_name"].astype(str).str.contains("cluster-oracle-moe|cluster_oracle_moe", case=False, regex=True, na=False),
    }

    for family, pool_mask in pools.items():
        target_mask = targets[family]
        if not target_mask.any():
            continue
        statscore = _mean_from(pool_mask, "statscore")
        out_score = _mean_from(pool_mask, "outdomain_score")
        if pd.isna(out_score):
            out_score = _mean_from(pool_mask, "clemscore_outdomain")
        out_played = _mean_from(pool_mask, "outdomain_played")
        out_quality = _mean_from(pool_mask, "outdomain_quality")

        if pd.notna(statscore):
            out.loc[target_mask, "statscore"] = statscore
        if pd.notna(out_score):
            out.loc[target_mask, "clemscore_outdomain"] = out_score
            if "outdomain_score" in out.columns:
                out.loc[target_mask, "outdomain_score"] = out_score
        if pd.notna(out_played):
            out.loc[target_mask, "outdomain_played"] = out_played
        if pd.notna(out_quality):
            out.loc[target_mask, "outdomain_quality"] = out_quality

    return out


def build_method_summary(root: Path) -> pd.DataFrame:
    rows = []
    run_dirs = _load_run_dirs(root)
    print(f"[analysis] root={root}")
    print(f"[analysis] discovered_run_dirs={len(run_dirs)}")

    for run_dir in run_dirs:
        run_path = str(run_dir.resolve().relative_to(root.resolve())).replace("\\", "/")
        val, val_path = _read_val_json(run_dir)
        clem_csv = run_dir / "clem" / "results.csv"
        static_csv = run_dir / "static" / "results.csv"
        clem_row, clem_cols = _read_csv_row(clem_csv)
        static_row, static_cols = _read_csv_row(static_csv)

        if val_path:
            print(f"[input][val] {val_path} keys={list(val.keys())}")
        if clem_cols:
            print(f"[input][clem] {clem_csv} columns={clem_cols}")
        if static_cols:
            print(f"[input][static] {static_csv} columns={static_cols}")

        clem_overall = _to_float(val.get("clemscore"))
        if pd.isna(clem_overall):
            clem_overall = _to_float(clem_row.get("-, clemscore"))
        statscore = _to_float(val.get("statscore"))
        if pd.isna(statscore):
            statscore = _to_float(static_row.get("-, clemscore"))

        group_values = {}
        for group, games in CLUSTER_MAP.items():
            score, played, quality = _group_metrics(clem_row, games)
            group_values[f"{group}_score"] = score
            group_values[f"{group}_played"] = played
            group_values[f"{group}_quality"] = quality

        in_played = _mean(group_values[f"{g}_played"] for g in CLUSTER_MAP)
        in_quality = _mean(group_values[f"{g}_quality"] for g in CLUSTER_MAP)
        in_score = _score_from_played_quality(in_played, in_quality)
        out_score, out_played, out_quality = _group_metrics(clem_row, OOD_GAMES)

        fields = _method_fields(run_dir, root)
        alpha, experts = _parse_residual_params(run_dir.name)

        row = {
            "method": fields["method"],
            "family_group": fields["family_group"],
            "expert_granularity": fields["expert_granularity"],
            "composition_level": fields["composition_level"],
            "selection_method": fields["selection_method"],
            "is_adapter_based": bool(fields["is_adapter_based"]),
            "is_residual_moe": bool(fields["is_residual_moe"]),
            "clemscore_overall": clem_overall,
            "clemscore_indomain": in_score,
            "clemscore_outdomain": out_score,
            "statscore": statscore,
            "generalization_gap": in_score - out_score if pd.notna(in_score) and pd.notna(out_score) else np.nan,
            "indomain_played": in_played,
            "indomain_quality": in_quality,
            "outdomain_played": out_played,
            "outdomain_quality": out_quality,
            "residual_alpha": alpha,
            "num_experts": experts,
            "run_name": run_dir.name,
            "run_path": run_path,
            "run_dir": str(run_dir),
            "method_group": fields["method_group"],
            "method_subtype": fields["method_subtype"],
            **group_values,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No analyzable run directories found under {root}")
    df = _apply_leaderboard_scores(df, root)
    if "outdomain_score" in df.columns:
        df["clemscore_outdomain"] = df["outdomain_score"].where(df["outdomain_score"].notna(), df["clemscore_outdomain"])
    df = _harmonize_oracle_proxy_metrics(df)
    df["generalization_gap"] = df["clemscore_indomain"] - df["clemscore_outdomain"]

    # Compatibility aliases used by earlier helper scripts.
    df["method_name"] = df["method"]
    df["clemscore"] = df["clemscore_overall"]
    df["in_domain_score"] = df["clemscore_indomain"]
    df["out_domain_score"] = df["clemscore_outdomain"]
    df["in_domain_played"] = df["indomain_played"]
    df["out_domain_played"] = df["outdomain_played"]
    df["in_domain_quality"] = df["indomain_quality"]
    df["out_domain_quality"] = df["outdomain_quality"]
    df["residual_experts"] = df["num_experts"]
    return df.sort_values("clemscore_overall", ascending=False, na_position="last")


def _best(df: pd.DataFrame, mask: pd.Series, metric: str = "clemscore_overall") -> Optional[pd.Series]:
    part = df[mask].dropna(subset=[metric]).sort_values(metric, ascending=False)
    if part.empty:
        return None
    return part.iloc[0]


def build_main_comparison(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Base", df["family_group"].eq("base")),
        ("Generalist", df["family_group"].eq("generalist")),
        ("Per-game Oracle MoE", df["selection_method"].eq("oracle") & df["expert_granularity"].eq("per_game")),
        ("Cluster Oracle MoE", df["selection_method"].eq("oracle") & df["expert_granularity"].eq("cluster")),
        ("Best Learned Router MoE", df["selection_method"].eq("learned_router")),
        ("Best Loss-based MoE", df["selection_method"].eq("loss_based")),
        ("Best Merging", df["family_group"].eq("merging")),
        ("Best Residual MoE", df["family_group"].eq("residual_moe")),
    ]
    rows = []
    for category, mask in specs:
        row = _best(df, mask)
        if row is None:
            _warn(f"main comparison missing category: {category}")
            rows.append({"category": category, "method": np.nan})
            continue
        rows.append(
            {
                "category": category,
                "method": row["method"],
                "family_group": row["family_group"],
                "expert_granularity": row["expert_granularity"],
                "composition_level": row["composition_level"],
                "selection_method": row["selection_method"],
                "clemscore_overall": row["clemscore_overall"],
                "clemscore_indomain": row["clemscore_indomain"],
                "clemscore_outdomain": row["clemscore_outdomain"],
                "statscore": row["statscore"],
                "indomain_played": row["indomain_played"],
                "indomain_quality": row["indomain_quality"],
                "outdomain_played": row["outdomain_played"],
                "outdomain_quality": row["outdomain_quality"],
                "run_path": row["run_path"],
            }
        )
    return pd.DataFrame(rows)


def build_routing_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for granularity in ["per_game", "cluster"]:
        oracle = _best(df, df["expert_granularity"].eq(granularity) & df["selection_method"].eq("oracle"))
        learned = _best(df, df["expert_granularity"].eq(granularity) & df["selection_method"].eq("learned_router"))
        loss = _best(df, df["expert_granularity"].eq(granularity) & df["selection_method"].eq("loss_based"))
        oracle_score = oracle["clemscore_overall"] if oracle is not None else np.nan
        learned_score = learned["clemscore_overall"] if learned is not None else np.nan
        loss_score = loss["clemscore_overall"] if loss is not None else np.nan
        rows.append(
            {
                "granularity": granularity,
                "oracle_method": oracle["method"] if oracle is not None else np.nan,
                "learned_router_method": learned["method"] if learned is not None else np.nan,
                "loss_based_method": loss["method"] if loss is not None else np.nan,
                "oracle_score": oracle_score,
                "learned_router_score": learned_score,
                "loss_based_score": loss_score,
                "learned_router_gap": oracle_score - learned_score if pd.notna(oracle_score) and pd.notna(learned_score) else np.nan,
                "loss_based_gap": oracle_score - loss_score if pd.notna(oracle_score) and pd.notna(loss_score) else np.nan,
                "oracle_statscore": oracle["statscore"] if oracle is not None else np.nan,
                "learned_router_statscore": learned["statscore"] if learned is not None else np.nan,
                "loss_based_statscore": loss["statscore"] if loss is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_merging_comparison(df: pd.DataFrame) -> pd.DataFrame:
    base = _best(df, df["family_group"].eq("base"))
    generalist = _best(df, df["family_group"].eq("generalist"))
    base_score = base["clemscore_overall"] if base is not None else np.nan
    generalist_score = generalist["clemscore_overall"] if generalist is not None else np.nan

    rows = []
    for _, row in df[df["family_group"].eq("merging")].sort_values("clemscore_overall", ascending=False).iterrows():
        granularity = row["expert_granularity"]
        source_mask = df["method_group"].eq("per_game_adapter") if granularity == "per_game" else df["method_group"].eq("cluster_adapter")
        source = _best(df, source_mask)
        source_score = source["clemscore_overall"] if source is not None else np.nan
        rows.append(
            {
                "merge_method": row["selection_method"],
                "granularity": granularity,
                "method": row["method"],
                "score": row["clemscore_overall"],
                "statscore": row["statscore"],
                "comparison_to_base": row["clemscore_overall"] - base_score if pd.notna(base_score) else np.nan,
                "comparison_to_generalist": row["clemscore_overall"] - generalist_score if pd.notna(generalist_score) else np.nan,
                "best_source_expert": source["method"] if source is not None else np.nan,
                "best_source_expert_score": source_score,
                "merge_interference": source_score - row["clemscore_overall"] if pd.notna(source_score) else np.nan,
                "run_path": row["run_path"],
            }
        )
    return pd.DataFrame(rows)


def build_merging_detailed_comparison(df: pd.DataFrame) -> pd.DataFrame:
    merge_rows = df[df["family_group"].eq("merging")].copy()
    if merge_rows.empty:
        return pd.DataFrame()

    refs = {
        "base": _best(df, df["family_group"].eq("base")),
        "generalist": _best(df, df["family_group"].eq("generalist")),
        "per_game_oracle": _best(df, df["method"].eq("Per-game Oracle MoE")),
        "cluster_oracle": _best(df, df["method"].eq("Cluster Oracle MoE")),
        "best_learned_router": _best(df, df["family_group"].eq("learned_router_moe")),
        "best_loss_based": _best(df, df["family_group"].eq("loss_based_moe")),
        "best_residual": _best(df, df["family_group"].eq("residual_moe")),
    }
    rows = []
    for _, row in merge_rows.iterrows():
        out = {
            "method": row["method"],
            "merge_method": row["selection_method"],
            "granularity": row["expert_granularity"],
            "num_source_adapters": 3 if row["expert_granularity"] == "cluster" else 14,
            "clemscore_overall": row["clemscore_overall"],
            "clemscore_indomain": row["clemscore_indomain"],
            "clemscore_outdomain": row["clemscore_outdomain"],
            "statscore": row["statscore"],
            "indomain_played": row["indomain_played"],
            "indomain_quality": row["indomain_quality"],
            "outdomain_played": row["outdomain_played"],
            "outdomain_quality": row["outdomain_quality"],
            "wordguessing_score": row["wordguessing_score"],
            "explorationnavigation_score": row["explorationnavigation_score"],
            "cooperation_score": row["cooperation_score"],
        }
        for name, ref in refs.items():
            if ref is None:
                continue
            out[f"delta_clemscore_vs_{name}"] = row["clemscore_overall"] - ref["clemscore_overall"]
            out[f"delta_indomain_vs_{name}"] = row["clemscore_indomain"] - ref["clemscore_indomain"]
            out[f"delta_outdomain_vs_{name}"] = row["clemscore_outdomain"] - ref["clemscore_outdomain"]
            out[f"delta_statscore_vs_{name}"] = row["statscore"] - ref["statscore"]
        rows.append(out)
    return pd.DataFrame(rows).sort_values("clemscore_overall", ascending=False)


def build_residual_sweep(df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["family_group"].eq("residual_moe")].copy()
    if res.empty:
        return pd.DataFrame()
    out = res[
        [
            "method",
            "residual_alpha",
            "num_experts",
            "clemscore_overall",
            "clemscore_indomain",
            "clemscore_outdomain",
            "statscore",
            "run_path",
        ]
    ].rename(columns={"residual_alpha": "alpha"})
    out["rank"] = out["clemscore_overall"].rank(ascending=False, method="min")
    return out.sort_values("rank")


def build_residual_effect_tables(residual_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if residual_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    alpha_df = (
        residual_df.groupby("alpha", dropna=True)
        .agg(
            n=("method", "count"),
            clemscore_mean=("clemscore_overall", "mean"),
            clemscore_max=("clemscore_overall", "max"),
            indomain_mean=("clemscore_indomain", "mean"),
            outdomain_mean=("clemscore_outdomain", "mean"),
            outdomain_max=("clemscore_outdomain", "max"),
            statscore_mean=("statscore", "mean"),
            statscore_max=("statscore", "max"),
        )
        .reset_index()
        .sort_values("alpha")
    )
    expert_df = (
        residual_df.groupby("num_experts", dropna=True)
        .agg(
            n=("method", "count"),
            clemscore_mean=("clemscore_overall", "mean"),
            clemscore_max=("clemscore_overall", "max"),
            indomain_mean=("clemscore_indomain", "mean"),
            outdomain_mean=("clemscore_outdomain", "mean"),
            outdomain_max=("clemscore_outdomain", "max"),
            statscore_mean=("statscore", "mean"),
            statscore_max=("statscore", "max"),
        )
        .reset_index()
        .sort_values("num_experts")
    )
    return alpha_df, expert_df


def build_generalization_gap(df: pd.DataFrame) -> pd.DataFrame:
    return df[["method", "clemscore_indomain", "clemscore_outdomain", "generalization_gap", "run_path"]].sort_values(
        "generalization_gap", ascending=False, na_position="last"
    )


def build_metric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["clemscore_overall", "clemscore_indomain", "clemscore_outdomain"]:
        sub = df[[metric, "statscore"]].dropna()
        corr = sub[metric].corr(sub["statscore"]) if len(sub) >= 2 else np.nan
        rows.append({"metric_x": metric, "metric_y": "statscore", "pearson_r": corr, "n": len(sub)})
    return pd.DataFrame(rows)


def build_ranked_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["method", "run_path", "clemscore_indomain", "clemscore_outdomain", "clemscore_overall", "statscore"]].copy()
    rank_cols = {
        "clemscore_indomain": "rank_indomain_clemscore",
        "clemscore_outdomain": "rank_outdomain_clemscore",
        "clemscore_overall": "rank_overall_clemscore",
        "statscore": "rank_statscore",
    }
    for source, rank_col in rank_cols.items():
        out[rank_col] = out[source].rank(ascending=False, method="average")
    out["average_rank"] = out[list(rank_cols.values())].mean(axis=1, skipna=True)
    return out.sort_values("average_rank", na_position="last")


def _read_clem_results_for_method(row: pd.Series) -> Dict[str, str]:
    run_dir = Path(str(row.get("run_dir", "")))
    clem_row, _ = _read_csv_row(run_dir / "clem" / "results.csv")
    return clem_row


def _source_game_from_run_name(run_name: str) -> str:
    match = re.search(r"game_(.+)$", str(run_name))
    return match.group(1) if match else ""


def _source_cluster_from_run_name(run_name: str) -> str:
    match = re.search(r"cluster_(wordguessing|explorationnavigation|cooperation)", str(run_name))
    return match.group(1) if match else ""


def build_adapter_transfer_tables(method_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = _best(method_df, method_df["family_group"].eq("base"))
    generalist = _best(method_df, method_df["family_group"].eq("generalist"))
    if base is None or generalist is None:
        _warn("adapter transfer analysis needs both base and generalist rows")
        return pd.DataFrame(), pd.DataFrame()

    base_row = _read_clem_results_for_method(base)
    generalist_row = _read_clem_results_for_method(generalist)
    rows = []
    adapter_rows = method_df[method_df["method_group"].isin(["per_game_adapter", "cluster_adapter"])].copy()

    for _, method in adapter_rows.iterrows():
        result_row = _read_clem_results_for_method(method)
        adapter_type = "per_game" if method["method_group"] == "per_game_adapter" else "cluster"
        source_game = _source_game_from_run_name(method["run_name"]) if adapter_type == "per_game" else ""
        source_cluster = _source_cluster_from_run_name(method["run_name"]) if adapter_type == "cluster" else ""

        for game in ALL_ANALYSIS_GAMES:
            score, played, quality = _single_game_metrics(result_row, game)
            base_score, base_played, base_quality = _single_game_metrics(base_row, game)
            gen_score, gen_played, gen_quality = _single_game_metrics(generalist_row, game)
            group = GAME_GROUP.get(game, "unknown")
            rows.append(
                {
                    "adapter_type": adapter_type,
                    "method": method["method"],
                    "run_path": method["run_path"],
                    "source_game": source_game,
                    "source_cluster": source_cluster,
                    "eval_game": game,
                    "eval_group": group,
                    "is_own_game": bool(source_game and source_game == game),
                    "is_source_cluster": bool(source_cluster and source_cluster == group),
                    "is_outdomain": game in OOD_GAMES,
                    "score": score,
                    "played": played,
                    "quality": quality,
                    "base_score": base_score,
                    "base_played": base_played,
                    "base_quality": base_quality,
                    "generalist_score": gen_score,
                    "generalist_played": gen_played,
                    "generalist_quality": gen_quality,
                    "score_delta_vs_base": score - base_score if pd.notna(score) and pd.notna(base_score) else np.nan,
                    "played_delta_vs_base": played - base_played if pd.notna(played) and pd.notna(base_played) else np.nan,
                    "quality_delta_vs_base": quality - base_quality if pd.notna(quality) and pd.notna(base_quality) else np.nan,
                    "score_delta_vs_generalist": score - gen_score if pd.notna(score) and pd.notna(gen_score) else np.nan,
                    "played_delta_vs_generalist": played - gen_played if pd.notna(played) and pd.notna(gen_played) else np.nan,
                    "quality_delta_vs_generalist": quality - gen_quality if pd.notna(quality) and pd.notna(gen_quality) else np.nan,
                }
            )

    transfer_df = pd.DataFrame(rows)
    if transfer_df.empty:
        return transfer_df, pd.DataFrame()

    summary_rows = []
    for (adapter_type, method), part in transfer_df.groupby(["adapter_type", "method"], sort=False):
        own = part[part["is_own_game"]]
        source_cluster = part[part["is_source_cluster"]]
        other_indomain = part[(~part["is_outdomain"]) & (~part["is_own_game"])]
        outdomain = part[part["is_outdomain"]]
        summary_rows.append(
            {
                "adapter_type": adapter_type,
                "method": method,
                "source_game": part["source_game"].dropna().iloc[0] if (part["source_game"].astype(str) != "").any() else "",
                "source_cluster": part["source_cluster"].dropna().iloc[0] if (part["source_cluster"].astype(str) != "").any() else "",
                "own_game_score_delta_vs_generalist": _mean(own["score_delta_vs_generalist"]) if not own.empty else np.nan,
                "own_game_played_delta_vs_generalist": _mean(own["played_delta_vs_generalist"]) if not own.empty else np.nan,
                "own_game_quality_delta_vs_generalist": _mean(own["quality_delta_vs_generalist"]) if not own.empty else np.nan,
                "source_cluster_score_delta_vs_generalist": _mean(source_cluster["score_delta_vs_generalist"]) if not source_cluster.empty else np.nan,
                "other_indomain_score_delta_vs_generalist": _mean(other_indomain["score_delta_vs_generalist"]) if not other_indomain.empty else np.nan,
                "outdomain_score_delta_vs_generalist": _mean(outdomain["score_delta_vs_generalist"]) if not outdomain.empty else np.nan,
                "mean_score_delta_vs_base": _mean(part["score_delta_vs_base"]),
                "mean_score_delta_vs_generalist": _mean(part["score_delta_vs_generalist"]),
                "mean_played_delta_vs_generalist": _mean(part["played_delta_vs_generalist"]),
                "mean_quality_delta_vs_generalist": _mean(part["quality_delta_vs_generalist"]),
                "num_games_better_than_base": int((part["score_delta_vs_base"] > 0).sum()),
                "num_games_worse_than_base": int((part["score_delta_vs_base"] < 0).sum()),
                "num_games_better_than_generalist": int((part["score_delta_vs_generalist"] > 0).sum()),
                "num_games_worse_than_generalist": int((part["score_delta_vs_generalist"] < 0).sum()),
                "top_boost_game_vs_generalist": part.sort_values("score_delta_vs_generalist", ascending=False).iloc[0]["eval_game"],
                "top_boost_vs_generalist": part["score_delta_vs_generalist"].max(),
                "top_penalty_game_vs_generalist": part.sort_values("score_delta_vs_generalist", ascending=True).iloc[0]["eval_game"],
                "top_penalty_vs_generalist": part["score_delta_vs_generalist"].min(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return transfer_df, summary_df


def build_adapter_domain_transfer_table(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate single-adapter transfer by source domain and evaluation domain."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame()

    rows = []
    tmp = transfer_df.copy()
    tmp["source_domain"] = tmp["source_game"].map(GAME_GROUP)
    tmp.loc[tmp["adapter_type"].eq("cluster"), "source_domain"] = tmp.loc[
        tmp["adapter_type"].eq("cluster"), "source_cluster"
    ]

    metric_cols = [
        "score",
        "played",
        "quality",
        "score_delta_vs_base",
        "played_delta_vs_base",
        "quality_delta_vs_base",
        "score_delta_vs_generalist",
        "played_delta_vs_generalist",
        "quality_delta_vs_generalist",
    ]
    for (adapter_type, source_domain, eval_group), part in tmp.groupby(
        ["adapter_type", "source_domain", "eval_group"], dropna=True, sort=True
    ):
        row = {
            "adapter_type": adapter_type,
            "source_domain": source_domain,
            "eval_group": eval_group,
            "n": len(part),
        }
        for col in metric_cols:
            row[col] = _mean(part[col]) if col in part.columns else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_per_game_outdomain_tables(transfer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize which per-game adapters transfer to each out-of-domain game."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ood = transfer_df[
        transfer_df["adapter_type"].eq("per_game")
        & transfer_df["eval_group"].eq("outdomain_negotiation")
    ].copy()
    if ood.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ood["source_domain"] = ood["source_game"].map(GAME_GROUP)

    by_adapter = (
        ood.groupby(["source_game", "source_domain"], dropna=False)
        .agg(
            outdomain_score=("score", "mean"),
            outdomain_played=("played", "mean"),
            outdomain_quality=("quality", "mean"),
            score_delta_vs_base=("score_delta_vs_base", "mean"),
            played_delta_vs_base=("played_delta_vs_base", "mean"),
            quality_delta_vs_base=("quality_delta_vs_base", "mean"),
            score_delta_vs_generalist=("score_delta_vs_generalist", "mean"),
            played_delta_vs_generalist=("played_delta_vs_generalist", "mean"),
            quality_delta_vs_generalist=("quality_delta_vs_generalist", "mean"),
            n=("score", "size"),
        )
        .reset_index()
        .sort_values("outdomain_score", ascending=False)
    )

    by_game = ood[
        [
            "source_game",
            "source_domain",
            "eval_game",
            "score",
            "played",
            "quality",
            "base_score",
            "base_played",
            "base_quality",
            "generalist_score",
            "generalist_played",
            "generalist_quality",
            "score_delta_vs_base",
            "played_delta_vs_base",
            "quality_delta_vs_base",
            "score_delta_vs_generalist",
            "played_delta_vs_generalist",
            "quality_delta_vs_generalist",
        ]
    ].sort_values(["eval_game", "score"], ascending=[True, False])

    by_source_domain = (
        ood.groupby(["source_domain", "eval_game"], dropna=False)
        .agg(
            score=("score", "mean"),
            played=("played", "mean"),
            quality=("quality", "mean"),
            score_delta_vs_base=("score_delta_vs_base", "mean"),
            played_delta_vs_base=("played_delta_vs_base", "mean"),
            quality_delta_vs_base=("quality_delta_vs_base", "mean"),
            score_delta_vs_generalist=("score_delta_vs_generalist", "mean"),
            played_delta_vs_generalist=("played_delta_vs_generalist", "mean"),
            quality_delta_vs_generalist=("quality_delta_vs_generalist", "mean"),
            n=("score", "size"),
        )
        .reset_index()
        .sort_values(["eval_game", "score"], ascending=[True, False])
    )
    return by_adapter, by_game, by_source_domain


def build_own_cross_domain_summary(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether adapters help their own game/domain, cross-domain games, and OOD games."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame()

    tmp = transfer_df.copy()
    tmp["source_domain"] = tmp["source_game"].map(GAME_GROUP)
    tmp.loc[tmp["adapter_type"].eq("cluster"), "source_domain"] = tmp.loc[
        tmp["adapter_type"].eq("cluster"), "source_cluster"
    ]
    tmp["relation"] = "cross_domain_indomain"
    tmp.loc[tmp["is_outdomain"].astype(bool), "relation"] = "outdomain_negotiation"
    tmp.loc[
        tmp["adapter_type"].eq("per_game")
        & (~tmp["is_outdomain"].astype(bool))
        & tmp["source_domain"].eq(tmp["eval_group"])
        & (~tmp["is_own_game"].astype(bool)),
        "relation",
    ] = "same_domain_cross_game"
    tmp.loc[tmp["adapter_type"].eq("per_game") & tmp["is_own_game"].astype(bool), "relation"] = "own_game"
    tmp.loc[tmp["adapter_type"].eq("cluster") & tmp["is_source_cluster"].astype(bool), "relation"] = "source_domain"

    metric_cols = [
        "score",
        "played",
        "quality",
        "score_delta_vs_base",
        "played_delta_vs_base",
        "quality_delta_vs_base",
        "score_delta_vs_generalist",
        "played_delta_vs_generalist",
        "quality_delta_vs_generalist",
    ]
    rows = []
    for (adapter_type, source_domain, relation), part in tmp.groupby(
        ["adapter_type", "source_domain", "relation"], dropna=False, sort=True
    ):
        row = {
            "adapter_type": adapter_type,
            "source_domain": source_domain,
            "relation": relation,
            "n": len(part),
        }
        for col in metric_cols:
            row[col] = _mean(part[col]) if col in part.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_empirical_game_similarity(transfer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate game similarity from how similarly games respond to single-adapter transfer."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Per-game adapters are the cleanest probe for cross-game transfer: each row is a source game adapter.
    probe = transfer_df[transfer_df["adapter_type"].eq("per_game") & transfer_df["eval_game"].isin(IN_DOMAIN_GAMES)].copy()
    if probe.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    matrix = probe.pivot_table(index="source_game", columns="eval_game", values="score_delta_vs_base", aggfunc="mean")
    matrix = matrix.reindex(index=[g for g in IN_DOMAIN_GAMES if g in matrix.index], columns=[g for g in IN_DOMAIN_GAMES if g in matrix.columns])
    corr = matrix.corr(min_periods=3)

    long_rows = []
    games = list(corr.columns)
    for i, g1 in enumerate(games):
        for j, g2 in enumerate(games):
            if j <= i:
                continue
            r = corr.loc[g1, g2]
            long_rows.append(
                {
                    "game_a": g1,
                    "game_b": g2,
                    "domain_a": GAME_GROUP.get(g1, "unknown"),
                    "domain_b": GAME_GROUP.get(g2, "unknown"),
                    "same_semantic_domain": GAME_GROUP.get(g1) == GAME_GROUP.get(g2),
                    "pearson_r": r,
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values("pearson_r", ascending=False, na_position="last")

    summary_rows = []
    if not long_df.empty:
        for same, part in long_df.groupby("same_semantic_domain", dropna=False):
            summary_rows.append(
                {
                    "pair_type": "same_semantic_domain" if bool(same) else "cross_semantic_domain",
                    "n": len(part),
                    "mean_pearson_r": _mean(part["pearson_r"]),
                    "median_pearson_r": float(part["pearson_r"].median()) if part["pearson_r"].notna().any() else np.nan,
                    "max_pearson_r": float(part["pearson_r"].max()) if part["pearson_r"].notna().any() else np.nan,
                    "min_pearson_r": float(part["pearson_r"].min()) if part["pearson_r"].notna().any() else np.nan,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return corr, long_df, summary_df


def _save_csv(df: pd.DataFrame, out_dir: Path, name: str):
    path = out_dir / name
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def _annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _save_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, ylabel: str):
    sub = df.dropna(subset=[y]).copy()
    if sub.empty:
        _warn(f"cannot plot {out_path.name}: no data for {y}")
        return
    colors = [FAMILY_COLORS.get(str(v), FAMILY_COLORS["unknown"]) for v in sub.get("family_group", "unknown")]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(sub)), sub[y].values, color=colors)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels([_plot_label(v) for v in sub[x]], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _annotate_bars(ax, bars, sub[y].values)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_main(main_df: pd.DataFrame, out_dir: Path):
    _save_bar(main_df, "category", "clemscore_overall", out_dir / "main_comparison_clemscore.png", "Main comparison: ClemScore", "ClemScore")
    _save_bar(main_df, "category", "statscore", out_dir / "main_comparison_statscore.png", "Main comparison: StatScore", "StatScore")


def plot_routing(routing_df: pd.DataFrame, out_dir: Path):
    if routing_df.empty:
        _warn("cannot plot routing figures: routing_comparison is empty")
        return

    labels = ["Oracle", "Learned Router", "Loss-based"]
    value_cols = ["oracle_score", "learned_router_score", "loss_based_score"]
    x = np.arange(len(routing_df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, label, col in zip([-width, 0, width], labels, value_cols):
        vals = routing_df[col].astype(float).values
        plot_vals = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(x + offset, plot_vals, width=width, label=label, color=ROUTING_COLORS[label])
        _annotate_bars(ax, bars, vals)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", "-") for v in routing_df["granularity"]])
    ax.set_ylabel("ClemScore")
    ax.set_title("Routing reliability: oracle vs practical selectors")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "routing_reliability_clemscore.png", dpi=300)
    plt.close(fig)

    gap_rows = []
    for _, row in routing_df.iterrows():
        for label, col in [("Learned Router", "learned_router_gap"), ("Loss-based", "loss_based_gap")]:
            if pd.notna(row[col]):
                gap_rows.append({"label": f"{row['granularity']} {label}", "gap": row[col], "selector": label})
    gap_df = pd.DataFrame(gap_rows)
    if gap_df.empty:
        _warn("cannot plot routing_gap.png: no finite routing gaps")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ROUTING_COLORS.get(v, "#94a3b8") for v in gap_df["selector"]]
    bars = ax.bar(np.arange(len(gap_df)), gap_df["gap"], color=colors)
    ax.set_xticks(np.arange(len(gap_df)))
    ax.set_xticklabels([_plot_label(v) for v in gap_df["label"]], rotation=25, ha="right")
    ax.set_ylabel("Oracle - selector ClemScore")
    ax.set_title("Routing gap")
    _annotate_bars(ax, bars, gap_df["gap"].values)
    fig.tight_layout()
    fig.savefig(out_dir / "routing_gap.png", dpi=300)
    plt.close(fig)


def plot_merging(merging_df: pd.DataFrame, method_df: pd.DataFrame, out_dir: Path):
    if merging_df.empty:
        _warn("cannot plot merging_comparison.png: no merging rows")
        return
    base = _best(method_df, method_df["family_group"].eq("base"))
    generalist = _best(method_df, method_df["family_group"].eq("generalist"))
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = merging_df.sort_values("score", ascending=False)
    bars = ax.bar(np.arange(len(sub)), sub["score"], color=FAMILY_COLORS["merging"])
    if base is not None:
        ax.axhline(base["clemscore_overall"], color=FAMILY_COLORS["base"], linestyle="--", linewidth=1.5, label="Base")
    if generalist is not None:
        ax.axhline(generalist["clemscore_overall"], color=FAMILY_COLORS["generalist"], linestyle="--", linewidth=1.5, label="Generalist")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels([_plot_label(v) for v in sub["method"]], rotation=25, ha="right")
    ax.set_ylabel("ClemScore")
    ax.set_title("Parameter-level merging")
    _annotate_bars(ax, bars, sub["score"].values)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "merging_comparison.png", dpi=300)
    plt.close(fig)


def plot_residual_heatmap(residual_df: pd.DataFrame, metric: str, out_path: Path):
    if residual_df.empty:
        _warn(f"cannot plot {out_path.name}: residual sweep is empty")
        return
    tmp = residual_df.dropna(subset=["alpha", "num_experts", metric]).copy()
    if tmp.empty:
        _warn(f"cannot plot {out_path.name}: missing alpha/experts/{metric}")
        return
    piv = tmp.pivot_table(index="num_experts", columns="alpha", values=metric, aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{float(v):g}" for v in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{int(v)}" for v in piv.index])
    ax.set_xlabel("residual contribution alpha")
    ax.set_ylabel("number of experts")
    ax.set_title(metric.replace("_", " "))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            value = piv.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_in_vs_out(df: pd.DataFrame, out_dir: Path):
    sub = df.dropna(subset=["clemscore_indomain", "clemscore_outdomain"]).copy()
    if sub.empty:
        _warn("cannot plot in_domain_vs_out_domain.png: no data")
        return
    sub = sub.sort_values("clemscore_overall", ascending=False).head(22)
    x = np.arange(len(sub))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, sub["clemscore_indomain"], width=width, color="#2563eb", label="in-domain")
    ax.bar(x + width / 2, sub["clemscore_outdomain"], width=width, color="#f59e0b", label="out-of-domain")
    ax.set_xticks(x)
    ax.set_xticklabels([_plot_label(v) for v in sub["method"]], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ClemScore")
    ax.set_title("In-domain vs out-of-domain ClemScore")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "in_domain_vs_out_domain.png", dpi=300)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_dir: Path):
    sub = df.dropna(subset=["clemscore_overall"]).copy()
    if sub.empty:
        _warn("cannot plot clemscore_vs_statscore_scatter.png: no ClemScore data")
        return
    fig, ax = plt.subplots(figsize=(18, 12))
    has_stat = sub[sub["statscore"].notna()].copy()
    no_stat = sub[sub["statscore"].isna()].copy()
    fallback_y = float(has_stat["statscore"].min()) - 0.75 if not has_stat.empty else 0.0

    for family, part in has_stat.groupby("family_group"):
        ax.scatter(
            part["clemscore_overall"],
            part["statscore"],
            s=72,
            alpha=0.82,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["unknown"]),
            label=family.replace("_", " "),
        )
    if not no_stat.empty:
        ax.scatter(
            no_stat["clemscore_overall"],
            [fallback_y] * len(no_stat),
            s=78,
            marker="v",
            edgecolors="#111827",
            linewidths=0.5,
            color="#d1d5db",
            label="missing StatScore",
        )
        ax.axhline(fallback_y, color="#9ca3af", linestyle="--", linewidth=1)
        ax.text(sub["clemscore_overall"].min(), fallback_y + 0.08, "StatScore missing", fontsize=9, color="#4b5563")

    for i, (_, row) in enumerate(sub.iterrows()):
        y = row["statscore"] if pd.notna(row["statscore"]) else fallback_y
        dx = 0.08 * (1 if i % 2 == 0 else -1)
        dy = 0.06 * (1 if (i // 2) % 2 == 0 else -1)
        ax.text(row["clemscore_overall"] + dx, y + dy, _scatter_label(row), fontsize=8, alpha=0.85)
    ax.set_xlabel("ClemScore")
    ax.set_ylabel("StatScore")
    ax.set_title("ClemScore vs StatScore")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "clemscore_vs_statscore_scatter.png", dpi=300)
    plt.close(fig)


def plot_average_rank(ranked_df: pd.DataFrame, out_dir: Path):
    sub = ranked_df.dropna(subset=["average_rank"]).sort_values("average_rank").head(22)
    if sub.empty:
        _warn("cannot plot average_rank_methods.png: no rank data")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(sub)), sub["average_rank"], color="#475569")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels([_plot_label(v) for v in sub["method"]], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Average rank (lower is better)")
    ax.set_title("Average rank across overall, in-domain, out-of-domain, and StatScore")
    _annotate_bars(ax, bars, sub["average_rank"].values)
    fig.tight_layout()
    fig.savefig(out_dir / "average_rank_methods.png", dpi=300)
    plt.close(fig)


def plot_transfer_heatmaps(transfer_df: pd.DataFrame, out_dir: Path):
    if transfer_df.empty:
        _warn("cannot plot adapter transfer heatmaps: transfer table is empty")
        return

    specs = [
        ("score_delta_vs_base", "Score delta vs base"),
        ("played_delta_vs_base", "% Played delta vs base"),
        ("quality_delta_vs_base", "Quality delta vs base"),
        ("score_delta_vs_generalist", "Score delta vs generalist"),
        ("played_delta_vs_generalist", "% Played delta vs generalist"),
        ("quality_delta_vs_generalist", "Quality delta vs generalist"),
    ]
    for adapter_type in ["per_game", "cluster"]:
        part = transfer_df[transfer_df["adapter_type"].eq(adapter_type)].copy()
        if part.empty:
            continue
        for metric, title in specs:
            piv = part.pivot_table(index="method", columns="eval_game", values=metric, aggfunc="mean")
            cols = [g for g in ALL_ANALYSIS_GAMES if g in piv.columns]
            piv = piv.reindex(columns=cols)
            if piv.empty:
                continue
            values = piv.values.astype(float)
            finite = values[np.isfinite(values)]
            vmax = max(1.0, float(np.nanmax(np.abs(finite)))) if finite.size else 1.0
            fig_h = max(4.5, 0.42 * len(piv.index))
            fig, ax = plt.subplots(figsize=(14, fig_h))
            im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels([_plot_label(c) for c in piv.columns], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels([_plot_label(i) for i in piv.index], fontsize=8)
            ax.set_title(f"{adapter_type.replace('_', '-')} adapters: {title}")
            ax.set_xlabel("evaluation game")
            ax.set_ylabel("adapter")
            fig.colorbar(im, ax=ax, label=title)
            fig.tight_layout()
            out_name = f"{adapter_type}_adapter_{metric}_heatmap.png"
            fig.savefig(out_dir / out_name, dpi=300)
            plt.close(fig)


def plot_transfer_summary(transfer_summary: pd.DataFrame, out_dir: Path):
    if transfer_summary.empty:
        _warn("cannot plot adapter_transfer_summary.png: no transfer summary rows")
        return
    for adapter_type in ["per_game", "cluster"]:
        part = transfer_summary[transfer_summary["adapter_type"].eq(adapter_type)].copy()
        if part.empty:
            continue
        part = part.sort_values("mean_score_delta_vs_generalist", ascending=False)
        fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(part))))
        y = np.arange(len(part))
        bars = ax.barh(y, part["mean_score_delta_vs_generalist"], color="#64748b")
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([_plot_label(v) for v in part["method"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean per-game score delta vs generalist")
        ax.set_title(f"{adapter_type.replace('_', '-')} adapter average transfer")
        for bar, value in zip(bars, part["mean_score_delta_vs_generalist"]):
            if pd.notna(value):
                ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{adapter_type}_adapter_mean_transfer_vs_generalist.png", dpi=300)
        plt.close(fig)


def plot_domain_transfer(domain_transfer: pd.DataFrame, out_dir: Path):
    if domain_transfer.empty:
        _warn("cannot plot domain transfer matrices: no domain transfer rows")
        return

    source_order = ["wordguessing", "explorationnavigation", "cooperation"]
    eval_order = ["wordguessing", "explorationnavigation", "cooperation", "outdomain_negotiation"]
    specs = [
        ("score_delta_vs_base", "Score delta vs base"),
        ("played_delta_vs_base", "% Played delta vs base"),
        ("quality_delta_vs_base", "Quality delta vs base"),
        ("score_delta_vs_generalist", "Score delta vs generalist"),
    ]

    for adapter_type in ["cluster", "per_game"]:
        part = domain_transfer[domain_transfer["adapter_type"].eq(adapter_type)].copy()
        if part.empty:
            continue
        for metric, title in specs:
            piv = part.pivot_table(index="source_domain", columns="eval_group", values=metric, aggfunc="mean")
            piv = piv.reindex(index=[x for x in source_order if x in piv.index], columns=[x for x in eval_order if x in piv.columns])
            if piv.empty:
                continue
            values = piv.values.astype(float)
            finite = values[np.isfinite(values)]
            vmax = max(1.0, float(np.nanmax(np.abs(finite)))) if finite.size else 1.0
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels([_plot_label(c) for c in piv.columns], rotation=30, ha="right", fontsize=9)
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels([_plot_label(i) for i in piv.index], fontsize=9)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    val = values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="#111827")
            ax.set_title(f"{adapter_type.replace('_', '-')} adapters: {title} by source/eval domain")
            ax.set_xlabel("evaluation domain")
            ax.set_ylabel("training/source domain")
            fig.colorbar(im, ax=ax, label=title)
            fig.tight_layout()
            fig.savefig(out_dir / f"{adapter_type}_adapter_domain_{metric}_matrix.png", dpi=300)
            plt.close(fig)


def plot_own_cross_summary(own_cross_df: pd.DataFrame, out_dir: Path):
    if own_cross_df.empty:
        _warn("cannot plot own/cross-domain summary: no rows")
        return
    relation_order = ["own_game", "source_domain", "same_domain_cross_game", "cross_domain_indomain", "outdomain_negotiation"]
    from matplotlib.lines import Line2D
    for adapter_type in ["per_game", "cluster"]:
        part = own_cross_df[own_cross_df["adapter_type"].eq(adapter_type)].copy()
        if part.empty:
            continue
        part["relation"] = pd.Categorical(part["relation"], categories=relation_order, ordered=True)
        agg = (
            part.groupby("relation", observed=True)
            .agg(
                score_delta_vs_base=("score_delta_vs_base", "mean"),
                played_delta_vs_base=("played_delta_vs_base", "mean"),
                quality_delta_vs_base=("quality_delta_vs_base", "mean"),
                score_delta_vs_generalist=("score_delta_vs_generalist", "mean"),
                played_delta_vs_generalist=("played_delta_vs_generalist", "mean"),
                quality_delta_vs_generalist=("quality_delta_vs_generalist", "mean"),
            )
            .reset_index()
            .dropna(subset=["relation"])
        )
        if agg.empty:
            continue
        # generalist position in delta-vs-base space = adapter_delta_vs_base - adapter_delta_vs_generalist
        gen_score  = agg["score_delta_vs_base"]  - agg["score_delta_vs_generalist"]
        gen_played = agg["played_delta_vs_base"]  - agg["played_delta_vs_generalist"]
        gen_qual   = agg["quality_delta_vs_base"] - agg["quality_delta_vs_generalist"]

        x = np.arange(len(agg))
        width = 0.26
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x - width, agg["score_delta_vs_base"],  width=width, label="score",    color="#2563eb")
        ax.bar(x,         agg["played_delta_vs_base"],  width=width, label="% played", color="#16a34a")
        ax.bar(x + width, agg["quality_delta_vs_base"], width=width, label="quality",  color="#f59e0b")
        # generalist triangle markers (one per metric column)
        ax.scatter(x - width, gen_score,  marker="^", color="#2563eb", s=55, zorder=5, linewidths=0)
        ax.scatter(x,         gen_played, marker="^", color="#16a34a", s=55, zorder=5, linewidths=0)
        ax.scatter(x + width, gen_qual,   marker="^", color="#f59e0b", s=55, zorder=5, linewidths=0)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_plot_label(v) for v in agg["relation"]], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Delta vs base")
        ax.set_title(f"{adapter_type.replace('_', '-')} adapters: own/cross/OOD effect")
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#2563eb", label="score"),
            plt.Rectangle((0, 0), 1, 1, color="#16a34a", label="% played"),
            plt.Rectangle((0, 0), 1, 1, color="#f59e0b", label="quality"),
            Line2D([0], [0], marker="^", color="#555", linestyle="None", markersize=6, label="generalist"),
        ]
        ax.legend(handles=legend_handles, frameon=False, ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{adapter_type}_adapter_own_cross_domain_summary.png", dpi=300)
        plt.close(fig)


def plot_empirical_game_similarity(sim_matrix: pd.DataFrame, out_dir: Path):
    if sim_matrix.empty:
        _warn("cannot plot empirical game similarity: empty matrix")
        return
    order = [g for g in IN_DOMAIN_GAMES if g in sim_matrix.index and g in sim_matrix.columns]
    mat = sim_matrix.reindex(index=order, columns=order)
    values = mat.values.astype(float)
    n = len(order)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.5 else "#111827"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([_plot_label(g) for g in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([_plot_label(g) for g in order], fontsize=8)
    ax.set_xlabel("game", fontsize=10)
    ax.set_ylabel("game", fontsize=10)

    # Cluster boundary lines + group labels
    starts = []
    cursor = 0
    for group, games in CLUSTER_MAP.items():
        present = [g for g in games if g in order]
        if present:
            starts.append((cursor, cursor + len(present), group))
            cursor += len(present)
    for start, end, group in starts:
        for pos in (start - 0.5, end - 0.5):
            ax.axhline(pos, color="#111827", linewidth=1.2)
            ax.axvline(pos, color="#111827", linewidth=1.2)
        mid = (start + end - 1) / 2
        label = _plot_label(group)
        ax.text(-1.6, mid, label, ha="right", va="center", fontsize=7.5,
                color="#374151", fontstyle="italic")
        ax.text(mid, -1.6, label, ha="center", va="bottom", fontsize=7.5,
                color="#374151", fontstyle="italic")

    ax.set_title(
        "Game-to-game similarity (Pearson r of per-game adapter score profiles)\n"
        "Each cell: how similarly two games respond to all per-game adapters",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, label="Pearson r  (+1 = always helped/hurt together,  −1 = opposite effect)")
    fig.tight_layout()
    fig.savefig(out_dir / "empirical_game_similarity_heatmap.png", dpi=300)
    plt.close(fig)


def build_transfer_matrix(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """Directed source_game × eval_game matrix of mean score_delta_vs_base (IID rows, IID+OOD cols)."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame()
    probe = transfer_df[
        transfer_df["adapter_type"].eq("per_game")
        & transfer_df["eval_game"].isin(IN_DOMAIN_GAMES + OOD_GAMES)
    ].copy()
    if probe.empty:
        return pd.DataFrame()
    matrix = probe.pivot_table(index="source_game", columns="eval_game", values="score_delta_vs_base", aggfunc="mean")
    row_order = [g for g in IN_DOMAIN_GAMES if g in matrix.index]
    col_order = [g for g in IN_DOMAIN_GAMES if g in matrix.columns] + [g for g in OOD_GAMES if g in matrix.columns]
    return matrix.reindex(index=row_order, columns=col_order)


def plot_transfer_matrix_heatmap(matrix_df: pd.DataFrame, out_dir: Path):
    """Directed source_game × eval_game score_delta_vs_base heatmap (IID + OOD columns)."""
    if matrix_df.empty:
        _warn("cannot plot transfer_matrix_heatmap.png: empty matrix")
        return

    values = matrix_df.values.astype(float)
    row_games = list(matrix_df.index)
    col_games = list(matrix_df.columns)
    n_rows, n_cols = len(row_games), len(col_games)
    vmax = max(np.nanmax(np.abs(values[np.isfinite(values)])), 1.0)

    n_iid_cols = sum(1 for g in col_games if g in IN_DOMAIN_GAMES)

    # size so each cell is roughly square (~0.65 in per cell + fixed margins)
    cell = 0.65
    fig_w = n_cols * cell + 4.0   # extra for y-labels + colorbar
    fig_h = n_rows * cell + 3.5   # extra for x-labels + title
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            text_color = "white" if abs(v) / vmax > 0.55 else "#111827"
            weight = "bold" if row_games[i] == col_games[j] else "normal"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                    color=text_color, fontweight=weight)

    # box around own-game diagonal cells
    for i, rg in enumerate(row_games):
        if rg in col_games:
            j = col_games.index(rg)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#111827", linewidth=2.0))

    # cluster boundary lines (rows + IID columns only)
    for axis, games_list in [("row", row_games), ("col", col_games[:n_iid_cols])]:
        for group, g_list in CLUSTER_MAP.items():
            present = [g for g in g_list if g in games_list]
            if not present:
                continue
            start = games_list.index(present[0])
            end = games_list.index(present[-1]) + 1
            for pos in (start - 0.5, end - 0.5):
                if axis == "row":
                    ax.axhline(pos, color="#374151", linewidth=1.5)
                else:
                    ax.axvline(pos, color="#374151", linewidth=1.5)

    # OOD section — thick separator + header label
    if n_iid_cols < n_cols:
        ax.axvline(n_iid_cols - 0.5, color="#b91c1c", linewidth=2.5, linestyle="--")
        ood_mid = (n_iid_cols + n_cols - 1) / 2
        ax.text(ood_mid, -0.62, "OOD games", ha="center", va="bottom",
                fontsize=9, color="#b91c1c", fontweight="bold", clip_on=False)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([_plot_label(g) for g in col_games], rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_plot_label(g) for g in row_games], fontsize=8.5)
    ax.set_xlabel("Evaluated on  →", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("←  Adapter trained on", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cross-game transfer matrix  (score Δ vs base)\n"
        "Rows = adapter trained on · Cols = game evaluated · Bold box = own game",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Score Δ vs base  (positive = adapter helps)")
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_network(matrix_df: pd.DataFrame, out_dir: Path, threshold: float = 5.0):
    """Directed transfer network: draw an edge wherever cross-game score delta > threshold."""
    if matrix_df.empty:
        _warn("cannot plot transfer_network.png: empty matrix")
        return
    try:
        import networkx as nx
    except ImportError:
        _warn("networkx not installed – skipping transfer_network.png")
        return

    cluster_colors = {
        "wordguessing": "#3b82f6",
        "explorationnavigation": "#22c55e",
        "cooperation": "#f59e0b",
    }

    G: "nx.DiGraph" = nx.DiGraph()
    for g in matrix_df.index:
        G.add_node(g, cluster=GAME_GROUP.get(g, "unknown"))

    for src in matrix_df.index:
        for dst in matrix_df.columns:
            if src == dst:
                continue
            val = matrix_df.loc[src, dst]
            if pd.notna(val) and float(val) > threshold:
                G.add_edge(src, dst, weight=float(val))

    if G.number_of_edges() == 0:
        _warn(f"transfer network has no edges above threshold {threshold} — skipping")
        return

    cluster_centers: Dict[str, np.ndarray] = {
        "wordguessing":        np.array([0.0,  1.0]),
        "explorationnavigation": np.array([-0.95, -0.5]),
        "cooperation":         np.array([0.95, -0.5]),
    }
    rng = np.random.default_rng(42)
    init_pos = {
        g: cluster_centers.get(GAME_GROUP.get(g, "unknown"), np.zeros(2)) + rng.normal(0, 0.18, 2)
        for g in G.nodes()
    }
    pos = nx.spring_layout(G, pos=init_pos, k=2.0, iterations=100, seed=42)

    node_colors = [cluster_colors.get(GAME_GROUP.get(g, "unknown"), "#9ca3af") for g in G.nodes()]
    out_weights = {g: sum(d for _, _, d in G.out_edges(g, data="weight")) for g in G.nodes()}
    node_sizes = [280 + out_weights.get(g, 0) * 14 for g in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.8 + 2.8 * (w / max_w) for w in edge_weights]
    edge_alphas = [0.35 + 0.55 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(12, 9))

    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        color = cluster_colors.get(GAME_GROUP.get(u, "unknown"), "#9ca3af")
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            width=width, alpha=alpha, edge_color=[color],
            arrows=True, arrowsize=18,
            connectionstyle="arc3,rad=0.14",
            node_size=node_sizes,
        )

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={g: _plot_label(g) for g in G.nodes()},
                            font_size=7.5, font_weight="bold")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=cluster_colors["wordguessing"],          label="word guessing"),
        Patch(facecolor=cluster_colors["explorationnavigation"], label="exploration & nav."),
        Patch(facecolor=cluster_colors["cooperation"],           label="cooperation"),
        Line2D([0], [0], color="#6b7280", linewidth=2,
               label=f"cross-game transfer  >  {threshold:.0f} pts"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.9)
    ax.set_title(
        f"Cross-game learning transfer network  (threshold: Δ vs base > {threshold:.0f} pts)\n"
        "Node size = total outgoing transfer  ·  Edge color = source cluster  ·  Arrows = direction of benefit",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_pca(transfer_df: pd.DataFrame, out_dir: Path):
    """PCA of games in transfer-response space.

    Each game is represented as a vector of how every per-game adapter affects it
    (score_delta_vs_base). Proximity in PCA space = similar transfer sensitivity.
    """
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        _warn("cannot plot transfer_pca.png: empty data")
        return
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        _warn("sklearn not installed – skipping transfer_pca.png")
        return

    probe = transfer_df[
        transfer_df["adapter_type"].eq("per_game") & transfer_df["eval_game"].isin(IN_DOMAIN_GAMES)
    ].copy()
    if probe.empty or len(probe["eval_game"].unique()) < 3:
        _warn("cannot plot transfer_pca.png: too few games")
        return

    matrix = probe.pivot_table(index="eval_game", columns="source_game",
                                values="score_delta_vs_base", aggfunc="mean")
    matrix = matrix.reindex(index=[g for g in IN_DOMAIN_GAMES if g in matrix.index])
    matrix = matrix.fillna(0.0)

    pca = _PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix.values)
    ev = pca.explained_variance_ratio_
    games = list(matrix.index)

    cluster_colors = {
        "wordguessing":        "#3b82f6",
        "explorationnavigation": "#22c55e",
        "cooperation":         "#f59e0b",
    }
    node_colors = [cluster_colors.get(GAME_GROUP.get(g, "unknown"), "#9ca3af") for g in games]

    fig, ax = plt.subplots(figsize=(9, 7))

    # convex hulls per cluster
    for cluster, g_list in CLUSTER_MAP.items():
        idx = [i for i, g in enumerate(games) if g in g_list]
        if len(idx) < 3:
            continue
        pts = coords[idx]
        color = cluster_colors.get(cluster, "#9ca3af")
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[:1]]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.10)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color, linewidth=1.2,
                    linestyle="--", alpha=0.55)
        except Exception:
            pass
        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, _plot_label(cluster), ha="center", va="center",
                fontsize=8.5, color=color, fontstyle="italic", alpha=0.75, fontweight="bold")

    ax.scatter(coords[:, 0], coords[:, 1], c=node_colors, s=150,
               zorder=5, edgecolors="white", linewidth=0.9)

    from matplotlib import patheffects
    for g, (x, y) in zip(games, coords):
        ax.text(x, y + (coords[:, 1].ptp() * 0.025), _plot_label(g),
                ha="center", va="bottom", fontsize=7.5, zorder=6,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])

    ax.axhline(0, color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1  ({ev[0]:.1%} variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2  ({ev[1]:.1%} variance explained)", fontsize=10)
    ax.set_title(
        "Empirical game space: PCA of cross-game transfer responses\n"
        "Each point = a game, positioned by how all per-game adapters affect it  ·  proximity = similar transfer sensitivity",
        fontsize=10,
    )
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=cluster_colors[c], label=_plot_label(c)) for c in cluster_colors],
        loc="best", fontsize=8, frameon=True, framealpha=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adapter_transferability(transfer_df: pd.DataFrame, out_dir: Path):
    """Bars = avg cross-game delta per adapter, dots = individual eval-game deltas."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        _warn("cannot plot adapter_transferability.png: empty data")
        return

    probe = transfer_df[
        transfer_df["adapter_type"].eq("per_game")
        & transfer_df["eval_game"].isin(IN_DOMAIN_GAMES)
        & ~transfer_df["is_own_game"].astype(bool)
    ].copy()
    if probe.empty:
        _warn("cannot plot adapter_transferability.png: no cross-game data")
        return

    cluster_colors = {
        "wordguessing":          "#3b82f6",
        "explorationnavigation": "#22c55e",
        "cooperation":           "#f59e0b",
    }

    cross = (
        probe.groupby("source_game")["score_delta_vs_base"]
        .mean()
        .rename("avg_cross_game_delta")
        .reset_index()
        .sort_values("avg_cross_game_delta", ascending=True)
    )
    game_order = list(cross["source_game"])
    y_positions = {g: i for i, g in enumerate(game_order)}

    bar_colors = [cluster_colors.get(GAME_GROUP.get(g, "unknown"), "#9ca3af") for g in game_order]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # bars (average)
    bars = ax.barh(
        [_plot_label(g) for g in game_order],
        cross["avg_cross_game_delta"],
        color=bar_colors,
        alpha=0.35,
        height=0.62,
        zorder=2,
    )

    # dots — one per eval game, colored by eval game's cluster
    rng = np.random.default_rng(42)
    for _, row in probe.iterrows():
        src = row["source_game"]
        if src not in y_positions:
            continue
        yi = y_positions[src]
        eval_cluster = GAME_GROUP.get(row["eval_game"], "unknown")
        color = cluster_colors.get(eval_cluster, "#9ca3af")
        jitter = rng.uniform(-0.22, 0.22)
        ax.scatter(
            row["score_delta_vs_base"],
            yi + jitter,
            color=color,
            s=28,
            alpha=0.80,
            zorder=4,
            linewidths=0,
        )

    # average value labels
    x_range = cross["avg_cross_game_delta"].abs().max()
    offset = x_range * 0.02
    for bar, val in zip(bars, cross["avg_cross_game_delta"]):
        w = bar.get_width()
        ax.text(
            w + offset if w >= 0 else w - offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=8,
            color="#374151",
            fontweight="bold",
            zorder=5,
        )

    ax.axvline(0, color="#111827", linewidth=1.0, zorder=3)
    ax.xaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("ClemScore Δ vs base  (own game excluded)", fontsize=9)
    ax.set_title(
        "Which game's adapter transfers best to other games?\n"
        "Bar = average  ·  dots = individual eval games colored by their cluster",
        fontsize=10, fontweight="bold",
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=cluster_colors[c], label=f"{_plot_label(c)} games (eval)")
        for c in cluster_colors
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=6, label="each dot = one eval game")
    )
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "adapter_transferability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_game_transfer_breakdown(transfer_df: pd.DataFrame, out_dir: Path):
    """Small multiples: one panel per adapter showing delta on every individual eval game."""
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        _warn("cannot plot per_game_transfer_breakdown.png: empty data")
        return

    probe = transfer_df[
        transfer_df["adapter_type"].eq("per_game")
        & transfer_df["eval_game"].isin(IN_DOMAIN_GAMES)
        & ~transfer_df["is_own_game"].astype(bool)
    ].copy()
    if probe.empty:
        _warn("cannot plot per_game_transfer_breakdown.png: no cross-game data")
        return

    # order source games by avg transferability (best first)
    avg_order = (
        probe.groupby("source_game")["score_delta_vs_base"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # consistent eval game order (by cluster, then within cluster)
    eval_order = [g for g in IN_DOMAIN_GAMES if g in probe["eval_game"].values]

    ncols = 4
    nrows = math.ceil(len(avg_order) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.8, nrows * 2.8),
        sharey=False,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, src_game in enumerate(avg_order):
        ax = axes_flat[idx]
        rows = probe[probe["source_game"].eq(src_game)].copy()
        rows = rows.set_index("eval_game").reindex(eval_order).reset_index()

        deltas = rows["score_delta_vs_base"].tolist()
        labels = [_plot_label(g) for g in eval_order]
        colors = ["#16a34a" if (pd.notna(d) and d > 0) else "#dc2626" for d in deltas]
        x = np.arange(len(eval_order))

        ax.bar(x, [d if pd.notna(d) else 0 for d in deltas],
               color=colors, alpha=0.80, width=0.65, zorder=3)
        ax.axhline(0, color="#374151", linewidth=0.9, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(_plot_label(src_game), fontsize=8.5, fontweight="bold")
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        avg = probe.loc[probe["source_game"].eq(src_game), "score_delta_vs_base"].mean()
        ax.set_ylabel(f"avg {avg:+.1f}", fontsize=7, color="#6b7280")

        # cluster separator lines
        cursor = 0
        for group, g_list in CLUSTER_MAP.items():
            present = [g for g in g_list if g in eval_order]
            cursor += len(present)
            if cursor < len(eval_order):
                ax.axvline(cursor - 0.5, color="#9ca3af", linewidth=0.7,
                           linestyle="--", zorder=2)

    # hide unused panels
    for idx in range(len(avg_order), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#16a34a", alpha=0.80, label="helps (+ delta vs base)"),
            Patch(facecolor="#dc2626", alpha=0.80, label="hurts (− delta vs base)"),
        ],
        loc="lower right", fontsize=8, frameon=False,
        bbox_to_anchor=(0.98, 0.01),
    )
    fig.suptitle(
        "Per-game adapter: ClemScore Δ vs base on every other game\n"
        "Panels ordered best → worst average transfer  ·  dashed lines = cluster boundaries",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "per_game_transfer_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_cluster_adapter_in_out_tables(
    transfer_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """In-domain vs out-of-domain breakdown for each of the three cluster adapters.

    relation labels:
      in_domain       – games belonging to the adapter's own training cluster
      other_indomain  – in-domain games from the two other clusters
      ood_negotiation – OOD negotiation games (dond, clean_up, hot_air_balloon)
    """
    if transfer_df.empty or "adapter_type" not in transfer_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    cluster_df = transfer_df[transfer_df["adapter_type"].eq("cluster")].copy()
    if cluster_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _relation(row: pd.Series) -> str:
        if row["is_outdomain"]:
            return "ood_negotiation"
        if row["is_source_cluster"]:
            return "in_domain"
        return "other_indomain"

    cluster_df["relation"] = cluster_df.apply(_relation, axis=1)

    keep_cols = [
        "source_cluster",
        "eval_game",
        "eval_group",
        "relation",
        "score",
        "played",
        "quality",
        "base_score",
        "base_played",
        "base_quality",
        "generalist_score",
        "generalist_played",
        "generalist_quality",
        "score_delta_vs_base",
        "played_delta_vs_base",
        "quality_delta_vs_base",
        "score_delta_vs_generalist",
        "played_delta_vs_generalist",
        "quality_delta_vs_generalist",
    ]
    detailed_df = cluster_df[[c for c in keep_cols if c in cluster_df.columns]].copy()

    relation_order = ["in_domain", "other_indomain", "ood_negotiation"]
    metric_cols = [
        "score",
        "played",
        "quality",
        "score_delta_vs_base",
        "played_delta_vs_base",
        "quality_delta_vs_base",
        "score_delta_vs_generalist",
        "played_delta_vs_generalist",
        "quality_delta_vs_generalist",
    ]
    summary_rows = []
    for (src, rel), part in cluster_df.groupby(["source_cluster", "relation"], sort=False):
        row: Dict[str, Any] = {"source_cluster": src, "relation": rel, "n_games": len(part)}
        for col in metric_cols:
            row[f"avg_{col}"] = _mean(part[col]) if col in part.columns else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["relation"] = pd.Categorical(summary_df["relation"], categories=relation_order, ordered=True)
        summary_df = summary_df.sort_values(["source_cluster", "relation"])

    return summary_df, detailed_df


def plot_cluster_adapter_in_vs_out(detailed_df: pd.DataFrame, out_dir: Path):
    """One subplot per cluster adapter (stacked vertically), ClemScore only, colored by relation."""
    if detailed_df.empty:
        _warn("cannot plot cluster_adapter_in_vs_out.png: no data")
        return

    cluster_order = ["wordguessing", "explorationnavigation", "cooperation"]
    relation_colors = {
        "in_domain": "#2563eb",
        "other_indomain": "#f59e0b",
        "ood_negotiation": "#dc2626",
    }
    relation_labels = {
        "in_domain": "in-domain",
        "other_indomain": "other in-domain",
        "ood_negotiation": "OOD",
    }

    clusters_present = [c for c in cluster_order if c in detailed_df["source_cluster"].values]
    if not clusters_present:
        _warn("cannot plot cluster_adapter_in_vs_out.png: no source_cluster data")
        return

    relation_order = ["in_domain", "other_indomain", "ood_negotiation"]
    game_order_base = [g for g in ALL_ANALYSIS_GAMES if g in detailed_df["eval_game"].values]

    fig, axes = plt.subplots(len(clusters_present), 1, figsize=(13, 3.8 * len(clusters_present)), squeeze=False)

    for row_idx, cluster in enumerate(clusters_present):
        ax = axes[row_idx, 0]
        part = detailed_df[detailed_df["source_cluster"].eq(cluster)].copy()
        if part.empty:
            ax.set_visible(False)
            continue

        part["relation"] = pd.Categorical(part["relation"], categories=relation_order, ordered=True)
        game_order = []
        for rel in relation_order:
            games_in_rel = [g for g in game_order_base if g in part.loc[part["relation"] == rel, "eval_game"].values]
            game_order.extend(games_in_rel)
        game_order = [g for g in game_order if g in part["eval_game"].values]

        x = np.arange(len(game_order))
        width = 0.55

        scores = [part.loc[part["eval_game"].eq(g), "score"].mean() for g in game_order]
        base_scores = [part.loc[part["eval_game"].eq(g), "base_score"].mean() for g in game_order]
        gen_scores = [part.loc[part["eval_game"].eq(g), "generalist_score"].mean() for g in game_order]
        rels = [str(part.loc[part["eval_game"].eq(g), "relation"].iloc[0]) for g in game_order]
        colors = [relation_colors.get(r, "#9ca3af") for r in rels]

        ax.bar(x, scores, width=width, color=colors, alpha=0.78, zorder=3)

        for xi, (bv, gv) in enumerate(zip(base_scores, gen_scores)):
            if pd.notna(bv):
                ax.plot(xi, bv, marker="x", color="#111827", markersize=7, zorder=5, linewidth=0, markeredgewidth=2.0)
            if pd.notna(gv):
                ax.plot(xi, gv, marker="^", color="#7c3aed", markersize=6, zorder=5, linewidth=0)

        ax.set_ylim(0, 112)

        # section header labels and vertical separators
        prev_rel = None
        start_i = 0
        for i, rel in enumerate(rels + [None]):
            if rel != prev_rel:
                if prev_rel is not None:
                    mid = (start_i + i - 1) / 2
                    color = relation_colors.get(prev_rel, "#9ca3af")
                    ax.text(mid, 108, relation_labels.get(prev_rel, prev_rel),
                            ha="center", va="top", fontsize=9, fontweight="bold", color=color)
                    ax.axvspan(start_i - 0.5, i - 0.5, color=color, alpha=0.05, zorder=0)
                    if i < len(game_order):
                        ax.axvline(i - 0.5, color="#9ca3af", linewidth=0.9, linestyle="--", zorder=1)
                start_i = i
                prev_rel = rel

        ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([_plot_label(g) for g in game_order], rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("ClemScore", fontsize=9)
        ax.set_title(_plot_label(cluster), fontsize=11, fontweight="bold", loc="left", pad=4)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=relation_colors["in_domain"], alpha=0.78, label="in-domain"),
        Patch(facecolor=relation_colors["other_indomain"], alpha=0.78, label="other in-domain"),
        Patch(facecolor=relation_colors["ood_negotiation"], alpha=0.78, label="OOD negotiation"),
        Line2D([0], [0], marker="x", color="#111827", linestyle="None", markersize=7, markeredgewidth=2.0, label="base"),
        Line2D([0], [0], marker="^", color="#7c3aed", linestyle="None", markersize=6, label="generalist"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Cluster adapter ClemScore: in-domain vs out-of-domain", fontsize=12, y=1.005)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "cluster_adapter_in_vs_out.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_cluster_in_out_html(summary_df: pd.DataFrame, out_dir: Path):
    """Styled HTML table for the cluster adapter in/out domain summary."""
    if summary_df.empty:
        _warn("cannot write cluster_adapter_in_out_summary.html: no data")
        return

    cluster_names = {
        "wordguessing": "Word Guessing",
        "explorationnavigation": "Exploration & Navigation",
        "cooperation": "Cooperation",
    }
    relation_styles: Dict[str, tuple] = {
        "in_domain":       ("in-domain",        "#2563eb"),
        "other_indomain":  ("other in-domain",   "#d97706"),
        "ood_negotiation": ("OOD negotiation",   "#dc2626"),
    }
    cluster_bg = {
        "wordguessing":        "#eff6ff",
        "explorationnavigation": "#f0fdf4",
        "cooperation":         "#fdf4ff",
    }

    def _score_td(val: Any) -> str:
        if pd.isna(val):
            return "<td>—</td>"
        return f"<td>{float(val):.1f}</td>"

    def _delta_td(val: Any) -> str:
        if pd.isna(val):
            return "<td>—</td>"
        fval = float(val)
        color = "#16a34a" if fval > 0.05 else "#dc2626" if fval < -0.05 else "#6b7280"
        sign = "+" if fval > 0 else ""
        return f'<td style="color:{color};font-weight:600">{sign}{fval:.1f}</td>'

    rows_html: list = []
    cluster_order = ["wordguessing", "explorationnavigation", "cooperation"]
    for cluster in cluster_order:
        part = summary_df[summary_df["source_cluster"].eq(cluster)]
        if part.empty:
            continue
        bg = cluster_bg.get(cluster, "#f9fafb")
        label = cluster_names.get(cluster, cluster)
        rows_html.append(
            f'<tr><td colspan="8" style="background:{bg};font-weight:700;font-size:0.93em;'
            f'padding:10px 14px;border-top:2px solid #e5e7eb;letter-spacing:0.02em">{label}</td></tr>'
        )
        for _, row in part.iterrows():
            rel = str(row["relation"])
            rel_label, rel_color = relation_styles.get(rel, (rel, "#6b7280"))
            badge = (
                f'<span style="background:{rel_color}18;color:{rel_color};border:1px solid {rel_color}44;'
                f'border-radius:4px;padding:2px 9px;font-size:0.82em;font-weight:600;white-space:nowrap">'
                f"{rel_label}</span>"
            )
            n = int(row["n_games"])
            rows_html.append(
                f"<tr>"
                f'<td style="padding-left:20px">{badge}</td>'
                f'<td style="color:#6b7280;text-align:center">{n}</td>'
                f"{_score_td(row.get('avg_score'))}"
                f"{_score_td(row.get('avg_played'))}"
                f"{_score_td(row.get('avg_quality'))}"
                f"{_delta_td(row.get('avg_score_delta_vs_base'))}"
                f"{_delta_td(row.get('avg_score_delta_vs_generalist'))}"
                f"</tr>"
            )

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "<title>Cluster Adapter In/Out Domain Summary</title>\n"
        "<style>\n"
        "  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "padding:2rem;color:#111827;background:#fff;max-width:900px}\n"
        "  h2{font-size:1.15rem;margin-bottom:.3rem}\n"
        "  p.sub{color:#6b7280;font-size:.87rem;margin-bottom:1.4rem}\n"
        "  table{border-collapse:collapse;width:100%;font-size:.9rem}\n"
        "  th{background:#f3f4f6;font-weight:600;padding:9px 14px;text-align:left;"
        "border-bottom:2px solid #d1d5db;white-space:nowrap}\n"
        "  td{padding:7px 14px;border-bottom:1px solid #f0f0f0}\n"
        "  tr:hover td{background:#fafafa}\n"
        "  .note{font-size:.78rem;color:#9ca3af;margin-top:1rem;line-height:1.6}\n"
        "</style>\n</head>\n<body>\n"
        "<h2>Cluster Adapter &mdash; In-domain vs Out-of-domain Performance</h2>\n"
        "<p class=\"sub\">Averages over games in each relation category. "
        "&Delta;&nbsp;vs&nbsp;base and &Delta;&nbsp;vs&nbsp;generalist are ClemScore deltas.</p>\n"
        "<table>\n<thead>\n<tr>\n"
        "  <th>Relation</th>\n"
        "  <th style=\"text-align:center\">Games</th>\n"
        "  <th>Avg Score</th>\n"
        "  <th>Avg Played %</th>\n"
        "  <th>Avg Quality</th>\n"
        "  <th>&Delta; vs base</th>\n"
        "  <th>&Delta; vs generalist</th>\n"
        "</tr>\n</thead>\n<tbody>\n"
        + "\n".join(rows_html)
        + "\n</tbody>\n</table>\n"
        "<p class=\"note\">"
        "<b>in-domain</b> = games in the adapter&rsquo;s own training cluster &nbsp;&middot;&nbsp; "
        "<b>other in-domain</b> = clembench in-domain games from the other two clusters &nbsp;&middot;&nbsp; "
        "<b>OOD negotiation</b> = dond, clean_up, hot_air_balloon"
        "</p>\n</body>\n</html>"
    )

    path = out_dir / "cluster_adapter_in_out_summary.html"
    path.write_text(html, encoding="utf-8")
    print(f"[saved] {path}")


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.2f}"


def _row_by_category(main_df: pd.DataFrame, category: str) -> Optional[pd.Series]:
    part = main_df[main_df["category"].eq(category)]
    return None if part.empty or pd.isna(part.iloc[0].get("method")) else part.iloc[0]


def write_report(
    out_dir: Path,
    method_df: pd.DataFrame,
    main_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    merging_df: pd.DataFrame,
    merging_detail_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    domain_transfer: pd.DataFrame,
    per_game_ood_summary: pd.DataFrame,
    per_game_ood_by_game: pd.DataFrame,
    per_game_ood_by_source_domain: pd.DataFrame,
    residual_alpha_effects: pd.DataFrame,
    residual_expert_effects: pd.DataFrame,
    own_cross_summary: pd.DataFrame,
    game_similarity_long: pd.DataFrame,
    game_similarity_summary: pd.DataFrame,
):
    base = _row_by_category(main_df, "Base")
    generalist = _row_by_category(main_df, "Generalist")
    pergame_oracle = _row_by_category(main_df, "Per-game Oracle MoE")
    cluster_oracle = _row_by_category(main_df, "Cluster Oracle MoE")
    learned = _row_by_category(main_df, "Best Learned Router MoE")
    loss_based = _row_by_category(main_df, "Best Loss-based MoE")
    merging = _row_by_category(main_df, "Best Merging")
    residual = _row_by_category(main_df, "Best Residual MoE")

    def score(row, col="clemscore_overall"):
        return np.nan if row is None else row.get(col, np.nan)

    def diff(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else a - b

    corr_overall = corr_df[corr_df["metric_x"].eq("clemscore_overall")]
    corr_txt = "NA" if corr_overall.empty else _fmt(corr_overall.iloc[0]["pearson_r"])
    best_overall = method_df.sort_values("clemscore_overall", ascending=False).iloc[0]
    best_indomain = method_df.sort_values("clemscore_indomain", ascending=False).iloc[0]
    best_outdomain = method_df.sort_values("clemscore_outdomain", ascending=False).iloc[0]
    best_stat = method_df.sort_values("statscore", ascending=False).iloc[0]

    pg_vs_gen = diff(score(pergame_oracle), score(generalist))
    cl_vs_gen = diff(score(cluster_oracle), score(generalist))
    pg_vs_cl = diff(score(pergame_oracle), score(cluster_oracle))
    learned_gap = diff(score(cluster_oracle), score(learned))
    loss_gap = diff(score(cluster_oracle), score(loss_based))
    merge_vs_gen = diff(score(merging), score(generalist))
    residual_vs_gen = diff(score(residual), score(generalist))

    pg_transfer = transfer_summary[transfer_summary["adapter_type"].eq("per_game")].copy() if not transfer_summary.empty else pd.DataFrame()
    cl_transfer = transfer_summary[transfer_summary["adapter_type"].eq("cluster")].copy() if not transfer_summary.empty else pd.DataFrame()
    best_pg_transfer = pg_transfer.sort_values("mean_score_delta_vs_generalist", ascending=False).head(1)
    worst_pg_transfer = pg_transfer.sort_values("mean_score_delta_vs_generalist", ascending=True).head(1)
    best_cl_transfer = cl_transfer.sort_values("mean_score_delta_vs_generalist", ascending=False).head(1)
    worst_cl_transfer = cl_transfer.sort_values("mean_score_delta_vs_generalist", ascending=True).head(1)
    best_pg_own = pg_transfer.sort_values("own_game_score_delta_vs_generalist", ascending=False).head(1) if not pg_transfer.empty else pd.DataFrame()
    worst_pg_own = pg_transfer.sort_values("own_game_score_delta_vs_generalist", ascending=True).head(1) if not pg_transfer.empty else pd.DataFrame()
    best_cl_source = cl_transfer.sort_values("source_cluster_score_delta_vs_generalist", ascending=False).head(1) if not cl_transfer.empty else pd.DataFrame()
    worst_cl_source = cl_transfer.sort_values("source_cluster_score_delta_vs_generalist", ascending=True).head(1) if not cl_transfer.empty else pd.DataFrame()
    pg_domain = domain_transfer[domain_transfer["adapter_type"].eq("per_game")].copy() if not domain_transfer.empty else pd.DataFrame()
    cl_domain = domain_transfer[domain_transfer["adapter_type"].eq("cluster")].copy() if not domain_transfer.empty else pd.DataFrame()

    def domain_row(source_domain: str, eval_group: str, adapter_type: str = "per_game") -> Optional[pd.Series]:
        table = pg_domain if adapter_type == "per_game" else cl_domain
        if table.empty:
            return None
        part = table[
            table["source_domain"].eq(source_domain)
            & table["eval_group"].eq(eval_group)
        ]
        return None if part.empty else part.iloc[0]

    wg_own = domain_row("wordguessing", "wordguessing")
    en_own = domain_row("explorationnavigation", "explorationnavigation")
    co_own = domain_row("cooperation", "cooperation")
    wg_ood = domain_row("wordguessing", "outdomain_negotiation")
    en_ood = domain_row("explorationnavigation", "outdomain_negotiation")
    co_ood = domain_row("cooperation", "outdomain_negotiation")
    cl_wg_own = domain_row("wordguessing", "wordguessing", "cluster")
    cl_en_own = domain_row("explorationnavigation", "explorationnavigation", "cluster")
    cl_co_own = domain_row("cooperation", "cooperation", "cluster")

    def row_value(row: Optional[pd.Series], col: str):
        return np.nan if row is None else row.get(col, np.nan)

    same_cluster_corr = game_similarity_summary[
        game_similarity_summary["pair_type"].eq("same_semantic_domain")
    ].head(1) if not game_similarity_summary.empty else pd.DataFrame()
    cross_cluster_corr = game_similarity_summary[
        game_similarity_summary["pair_type"].eq("cross_semantic_domain")
    ].head(1) if not game_similarity_summary.empty else pd.DataFrame()
    strongest_cross_pair = game_similarity_long[
        ~game_similarity_long["same_semantic_domain"].astype(bool)
    ].sort_values("pearson_r", ascending=False).head(1) if not game_similarity_long.empty else pd.DataFrame()
    weakest_same_pair = game_similarity_long[
        game_similarity_long["same_semantic_domain"].astype(bool)
    ].sort_values("pearson_r", ascending=True).head(1) if not game_similarity_long.empty else pd.DataFrame()

    best_ood_adapter = per_game_ood_summary.sort_values("outdomain_score", ascending=False).head(1) if not per_game_ood_summary.empty else pd.DataFrame()
    worst_ood_adapter = per_game_ood_summary.sort_values("outdomain_score", ascending=True).head(1) if not per_game_ood_summary.empty else pd.DataFrame()

    def best_ood_game(game: str) -> pd.DataFrame:
        if per_game_ood_by_game.empty:
            return pd.DataFrame()
        part = per_game_ood_by_game[per_game_ood_by_game["eval_game"].eq(game)]
        return part.sort_values("score", ascending=False).head(1)

    best_clean = best_ood_game("clean_up")
    best_dond = best_ood_game("dond")
    best_balloon = best_ood_game("hot_air_balloon")
    low_alpha = residual_alpha_effects.sort_values("alpha").head(1) if not residual_alpha_effects.empty else pd.DataFrame()
    high_alpha = residual_alpha_effects.sort_values("alpha").tail(1) if not residual_alpha_effects.empty else pd.DataFrame()
    best_expert_mean = residual_expert_effects.sort_values("clemscore_mean", ascending=False).head(1) if not residual_expert_effects.empty else pd.DataFrame()
    best_expert_ood = residual_expert_effects.sort_values("outdomain_mean", ascending=False).head(1) if not residual_expert_effects.empty else pd.DataFrame()
    cluster_wa = merging_detail_df[
        merging_detail_df["granularity"].eq("cluster")
        & merging_detail_df["merge_method"].eq("weight_averaging")
    ].head(1) if not merging_detail_df.empty else pd.DataFrame()
    cluster_ta = merging_detail_df[
        merging_detail_df["granularity"].eq("cluster")
        & merging_detail_df["merge_method"].eq("task_arithmetic")
    ].head(1) if not merging_detail_df.empty else pd.DataFrame()
    pergame_wa = merging_detail_df[
        merging_detail_df["granularity"].eq("per_game")
        & merging_detail_df["merge_method"].eq("weight_averaging")
    ].head(1) if not merging_detail_df.empty else pd.DataFrame()
    pergame_ta = merging_detail_df[
        merging_detail_df["granularity"].eq("per_game")
        & merging_detail_df["merge_method"].eq("task_arithmetic")
    ].head(1) if not merging_detail_df.empty else pd.DataFrame()

    def first_value(df: pd.DataFrame, col: str):
        return np.nan if df.empty else df.iloc[0].get(col, np.nan)

    def first_text(df: pd.DataFrame, col: str):
        return "NA" if df.empty else str(df.iloc[0].get(col, "NA"))

    report = [
        "# Analysis of LoRA Expert Composition",
        "",
        "## 1. Project Framing",
        "This analysis compares expert granularity and composition strategies for LoRA-based LLM adaptation. Dialogue games are used as controlled task environments; the benchmark is the measurement setting, while the main contribution is the comparison of expert definitions and composition mechanisms.",
        "",
        "The pipeline separates baseline single-adapter models, parameter-level adapter merging, task-level adapter selection, and residual MoE token-level composition. Out-of-domain games are evaluation-only: no out-of-domain adapters or training data are assumed.",
        "",
        "## 2. Main Comparison",
        f"The strongest overall ClemScore in the current table is **{best_overall['method']}** at **{_fmt(best_overall['clemscore_overall'])}**. The strongest in-domain score is **{best_indomain['method']}** at **{_fmt(best_indomain['clemscore_indomain'])}**, while the strongest out-of-domain score is **{best_outdomain['method']}** at **{_fmt(best_outdomain['clemscore_outdomain'])}**. The highest StatScore is **{best_stat['method']}** at **{_fmt(best_stat['statscore'])}**.",
        "",
        f"The generalist LoRA reaches **{_fmt(score(generalist))}** overall ClemScore. Per-game oracle MoE reaches **{_fmt(score(pergame_oracle))}**, a difference of **{_fmt(pg_vs_gen)}** points relative to the generalist. Cluster oracle MoE reaches **{_fmt(score(cluster_oracle))}**, which is **{_fmt(cl_vs_gen)}** points relative to the generalist. This is the key granularity result: per-game specialization is competitive with the generalist under perfect expert selection, while the manual cluster oracle is lower.",
        "",
        f"Looking below aggregate ClemScore, the generalist combines **{_fmt(score(generalist, 'indomain_played'))}%** in-domain played with **{_fmt(score(generalist, 'indomain_quality'))}** in-domain quality. Per-game oracle increases in-domain completion to **{_fmt(score(pergame_oracle, 'indomain_played'))}%** but lowers average quality to **{_fmt(score(pergame_oracle, 'indomain_quality'))}**. Cluster oracle has **{_fmt(score(cluster_oracle, 'indomain_played'))}%** played and **{_fmt(score(cluster_oracle, 'indomain_quality'))}** quality. This means the per-game oracle gain is mainly a completion gain, while the cluster oracle loses quality relative to the generalist.",
        "",
        "## 3. Expert Granularity",
        f"Per-game oracle MoE scores **{_fmt(score(pergame_oracle))}**, while cluster oracle MoE scores **{_fmt(score(cluster_oracle))}**. The per-game oracle advantage is **{_fmt(pg_vs_cl)}** ClemScore points. This suggests that the manual clusters are too coarse for at least part of the benchmark: grouping games together loses useful specialization that the per-game adapters preserve.",
        "",
        "The single adapter tables still matter as diagnostics: the best individual per-game adapter and best individual cluster adapter show how much transfer each specialized adapter provides before any routing or merging is applied. Those rows are retained in `method_summary.csv` but are not used as the main modular upper bound.",
        "",
        f"Compared game by game against the generalist, the strongest average per-game adapter transfer is **{first_text(best_pg_transfer, 'method')}** with mean score delta **{_fmt(first_value(best_pg_transfer, 'mean_score_delta_vs_generalist'))}**. The weakest is **{first_text(worst_pg_transfer, 'method')}** with **{_fmt(first_value(worst_pg_transfer, 'mean_score_delta_vs_generalist'))}**. For cluster adapters, the strongest average transfer is **{first_text(best_cl_transfer, 'method')}** with **{_fmt(first_value(best_cl_transfer, 'mean_score_delta_vs_generalist'))}**, and the weakest is **{first_text(worst_cl_transfer, 'method')}** with **{_fmt(first_value(worst_cl_transfer, 'mean_score_delta_vs_generalist'))}**. The detailed boosts and penalties are in `adapter_game_transfer.csv`, with separate score, played, and quality deltas against both base and generalist.",
        "",
        f"The own-game view shows why oracle routing can help. The strongest own-game gain is **{first_text(best_pg_own, 'method')}** on its source game with score delta **{_fmt(first_value(best_pg_own, 'own_game_score_delta_vs_generalist'))}**, played delta **{_fmt(first_value(best_pg_own, 'own_game_played_delta_vs_generalist'))}**, and quality delta **{_fmt(first_value(best_pg_own, 'own_game_quality_delta_vs_generalist'))}**. The weakest own-game case is **{first_text(worst_pg_own, 'method')}** with score delta **{_fmt(first_value(worst_pg_own, 'own_game_score_delta_vs_generalist'))}**. For cluster adapters, the best within-cluster transfer is **{first_text(best_cl_source, 'method')}** at **{_fmt(first_value(best_cl_source, 'source_cluster_score_delta_vs_generalist'))}**, while the weakest is **{first_text(worst_cl_source, 'method')}** at **{_fmt(first_value(worst_cl_source, 'source_cluster_score_delta_vs_generalist'))}**. The pattern is local gains plus broad penalties: no single specialized adapter beats the generalist on average across all games.",
        "",
        f"For per-game adapters aggregated by source domain, the source-domain effects are uneven. Word-guessing adapters average **{_fmt(row_value(wg_own, 'score'))}** on word-guessing games, which is **{_fmt(row_value(wg_own, 'score_delta_vs_base'))}** vs base and **{_fmt(row_value(wg_own, 'score_delta_vs_generalist'))}** vs generalist; their quality is **{_fmt(row_value(wg_own, 'quality_delta_vs_generalist'))}** lower than the generalist on average. Exploration/navigation adapters average **{_fmt(row_value(en_own, 'score'))}** on exploration/navigation games, **{_fmt(row_value(en_own, 'score_delta_vs_base'))}** vs base but **{_fmt(row_value(en_own, 'score_delta_vs_generalist'))}** vs generalist, mainly because played drops by **{_fmt(row_value(en_own, 'played_delta_vs_generalist'))}** points. Cooperation adapters average **{_fmt(row_value(co_own, 'score'))}** on cooperation games, **{_fmt(row_value(co_own, 'score_delta_vs_base'))}** vs base and **{_fmt(row_value(co_own, 'score_delta_vs_generalist'))}** vs generalist. On out-of-domain negotiation games, per-game adapters beat the generalist average by **{_fmt(row_value(wg_ood, 'score_delta_vs_generalist'))}** for word-guessing-source adapters, **{_fmt(row_value(en_ood, 'score_delta_vs_generalist'))}** for exploration-source adapters, and **{_fmt(row_value(co_ood, 'score_delta_vs_generalist'))}** for cooperation-source adapters, but they remain below the base model on average.",
        "",
        f"For cluster adapters, the handmade domains are only partly supported. The word-guessing cluster scores **{_fmt(row_value(cl_wg_own, 'score'))}** on word-guessing games (**{_fmt(row_value(cl_wg_own, 'score_delta_vs_base'))}** vs base, **{_fmt(row_value(cl_wg_own, 'score_delta_vs_generalist'))}** vs generalist). The exploration/navigation cluster scores **{_fmt(row_value(cl_en_own, 'score'))}** on exploration/navigation games (**{_fmt(row_value(cl_en_own, 'score_delta_vs_base'))}** vs base, **{_fmt(row_value(cl_en_own, 'score_delta_vs_generalist'))}** vs generalist). The cooperation cluster scores **{_fmt(row_value(cl_co_own, 'score'))}** on cooperation games (**{_fmt(row_value(cl_co_own, 'score_delta_vs_base'))}** vs base, **{_fmt(row_value(cl_co_own, 'score_delta_vs_generalist'))}** vs generalist). This should be read with `% Played` and quality deltas in `cluster_adapter_domain_transfer.csv`, because some apparent score gains are completion-rate effects while others are quality effects.",
        "",
        f"The empirical game-similarity analysis tests whether semantic clusters match transfer behavior. Same-domain game pairs have mean transfer-profile correlation **{_fmt(first_value(same_cluster_corr, 'mean_pearson_r'))}**, while cross-domain pairs have mean correlation **{_fmt(first_value(cross_cluster_corr, 'mean_pearson_r'))}**. The strongest cross-domain pair is **{first_text(strongest_cross_pair, 'game_a')} ↔ {first_text(strongest_cross_pair, 'game_b')}** with r=**{_fmt(first_value(strongest_cross_pair, 'pearson_r'))}**; the weakest same-domain pair is **{first_text(weakest_same_pair, 'game_a')} ↔ {first_text(weakest_same_pair, 'game_b')}** with r=**{_fmt(first_value(weakest_same_pair, 'pearson_r'))}**. If cross-domain correlations are close to or above same-domain correlations, the poster claim should be that games share latent skills not captured by manual semantic labels. See `empirical_game_similarity_heatmap.png` and `empirical_game_similarity_long.csv`.",
        "",
        f"Out-of-domain transfer is not uniform. Averaged over `clean_up`, `dond`, and `hot_air_balloon`, the best individual per-game adapter is **{first_text(best_ood_adapter, 'source_game')}** with OOD score **{_fmt(first_value(best_ood_adapter, 'outdomain_score'))}**, played **{_fmt(first_value(best_ood_adapter, 'outdomain_played'))}%**, and quality **{_fmt(first_value(best_ood_adapter, 'outdomain_quality'))}**. The weakest is **{first_text(worst_ood_adapter, 'source_game')}** at **{_fmt(first_value(worst_ood_adapter, 'outdomain_score'))}**. The best adapter differs by OOD game: `clean_up` is best with **{first_text(best_clean, 'source_game')}** at **{_fmt(first_value(best_clean, 'score'))}**, `dond` is best with **{first_text(best_dond, 'source_game')}** at **{_fmt(first_value(best_dond, 'score'))}**, and `hot_air_balloon` is best with **{first_text(best_balloon, 'source_game')}** at **{_fmt(first_value(best_balloon, 'score'))}**. Because base/generalist have 0% played on `hot_air_balloon`, score deltas there are not meaningful; use played and raw score for that game.",
        "",
        "## 4. Routing Reliability",
    ]

    if not routing_df.empty:
        for _, row in routing_df.iterrows():
            report.append(
                f"For **{row['granularity']}** routing, oracle={_fmt(row['oracle_score'])}, learned router={_fmt(row['learned_router_score'])}, loss-based={_fmt(row['loss_based_score'])}. The learned-router gap is **{_fmt(row['learned_router_gap'])}** and the loss-based gap is **{_fmt(row['loss_based_gap'])}**."
            )
    else:
        report.append("No routing comparison rows were available.")

    report.extend(
        [
            "",
            f"In the current data, the best learned router is **{_fmt(score(learned))}** and the best loss-based router is **{_fmt(score(loss_based))}**. Against the cluster oracle, those gaps are **{_fmt(learned_gap)}** and **{_fmt(loss_gap)}** points respectively. This supports the interpretation that practical routing reliability is a bottleneck when the oracle score is meaningfully higher than the routed score.",
            "",
            "## 5. Merging",
        ]
    )

    if not merging_df.empty:
        best_merge = merging_df.sort_values("score", ascending=False).iloc[0]
        report.append(
            f"The best merge is **{best_merge['method']}** with ClemScore **{_fmt(best_merge['score'])}**. It is **{_fmt(best_merge['comparison_to_base'])}** points relative to base and **{_fmt(best_merge['comparison_to_generalist'])}** points relative to the generalist. Its merge interference estimate is **{_fmt(best_merge['merge_interference'])}** points relative to the best source expert of the same granularity."
        )
        report.append(
            f"Cluster merging is stronger than per-game merging despite using fewer source adapters. Cluster WA merges **3** adapters and reaches **{_fmt(first_value(cluster_wa, 'clemscore_overall'))}** overall, **{_fmt(first_value(cluster_wa, 'clemscore_indomain'))}** in-domain, **{_fmt(first_value(cluster_wa, 'clemscore_outdomain'))}** out-of-domain, and **{_fmt(first_value(cluster_wa, 'statscore'))}** StatScore. Per-game WA merges **14** adapters but reaches only **{_fmt(first_value(pergame_wa, 'clemscore_overall'))}** overall, **{_fmt(first_value(pergame_wa, 'clemscore_indomain'))}** in-domain, **{_fmt(first_value(pergame_wa, 'clemscore_outdomain'))}** out-of-domain, and **{_fmt(first_value(pergame_wa, 'statscore'))}** StatScore. This suggests that averaging many fine-grained adapters introduces more interference than averaging three broader cluster adapters."
        )
        report.append(
            f"WA is better than TA for both granularities in this run. Cluster WA beats cluster TA by **{_fmt(first_value(cluster_wa, 'clemscore_overall') - first_value(cluster_ta, 'clemscore_overall'))}** overall ClemScore and **{_fmt(first_value(cluster_wa, 'statscore') - first_value(cluster_ta, 'statscore'))}** StatScore. Per-game WA beats per-game TA by **{_fmt(first_value(pergame_wa, 'clemscore_overall') - first_value(pergame_ta, 'clemscore_overall'))}** overall and **{_fmt(first_value(pergame_wa, 'statscore') - first_value(pergame_ta, 'statscore'))}** StatScore, but per-game WA collapses on out-of-domain negotiation (**{_fmt(first_value(pergame_wa, 'clemscore_outdomain'))}** vs **{_fmt(first_value(pergame_ta, 'clemscore_outdomain'))}** for per-game TA)."
        )
        report.append(
            f"Against the strongest practical and oracle baselines, merging remains below the target. Cluster WA is **{_fmt(first_value(cluster_wa, 'delta_clemscore_vs_generalist'))}** vs generalist, **{_fmt(first_value(cluster_wa, 'delta_clemscore_vs_per_game_oracle'))}** vs per-game oracle, **{_fmt(first_value(cluster_wa, 'delta_clemscore_vs_cluster_oracle'))}** vs cluster oracle, and **{_fmt(first_value(cluster_wa, 'delta_clemscore_vs_best_residual'))}** vs the best residual MoE. Its main advantage is preservation of StatScore: **{_fmt(first_value(cluster_wa, 'statscore'))}**, which is **{_fmt(first_value(cluster_wa, 'delta_statscore_vs_generalist'))}** above the generalist and **{_fmt(first_value(cluster_wa, 'delta_statscore_vs_base'))}** above the base."
        )
    else:
        report.append("No WA/TA merging rows were available.")

    report.extend(
        [
            f"Overall, the best merge is **{_fmt(score(merging))}** versus the generalist at **{_fmt(score(generalist))}** (**{_fmt(merge_vs_gen)}** points). If this difference is negative, merging preserves some behavior but does not match the strongest single generalist baseline.",
            "",
            "## 6. Residual MoE",
        ]
    )

    if not residual_df.empty:
        best_res = residual_df.sort_values("clemscore_overall", ascending=False).iloc[0]
        best_res_stat = residual_df.sort_values("statscore", ascending=False).iloc[0]
        best_res_ood = residual_df.sort_values("clemscore_outdomain", ascending=False).iloc[0]
        report.append(
            f"The best residual MoE by overall ClemScore is **{best_res['method']}** with alpha={_fmt(best_res['alpha'])}, experts={_fmt(best_res['num_experts'])}, ClemScore **{_fmt(best_res['clemscore_overall'])}**, in-domain **{_fmt(best_res['clemscore_indomain'])}**, out-of-domain **{_fmt(best_res['clemscore_outdomain'])}**, and StatScore **{_fmt(best_res['statscore'])}**."
        )
        report.append(
            f"The best residual MoE by StatScore is **{best_res_stat['method']}** with StatScore **{_fmt(best_res_stat['statscore'])}**. The best residual overall score is **{_fmt(score(residual))}**, which is **{_fmt(residual_vs_gen)}** points relative to the generalist."
        )
        report.append(
            f"The alpha grid suggests that lower residual contribution is safer for task performance in this sweep. Alpha **{_fmt(first_value(low_alpha, 'alpha'))}** averages **{_fmt(first_value(low_alpha, 'clemscore_mean'))}** overall ClemScore and **{_fmt(first_value(low_alpha, 'outdomain_mean'))}** out-of-domain, while alpha **{_fmt(first_value(high_alpha, 'alpha'))}** averages **{_fmt(first_value(high_alpha, 'clemscore_mean'))}** overall and **{_fmt(first_value(high_alpha, 'outdomain_mean'))}** out-of-domain. StatScore moves only slightly in the opposite direction: **{_fmt(first_value(low_alpha, 'statscore_mean'))}** at low alpha vs **{_fmt(first_value(high_alpha, 'statscore_mean'))}** at high alpha."
        )
        report.append(
            f"The expert-count effect is non-monotonic. Averaged across alphas, **{_fmt(first_value(best_expert_mean, 'num_experts'))}** experts has the best mean overall ClemScore (**{_fmt(first_value(best_expert_mean, 'clemscore_mean'))}**), but the best OOD mean is with **{_fmt(first_value(best_expert_ood, 'num_experts'))}** experts (**{_fmt(first_value(best_expert_ood, 'outdomain_mean'))}**). The single best OOD cell is **{best_res_ood['method']}** at **{_fmt(best_res_ood['clemscore_outdomain'])}**. This means more experts help when the residual is weak enough (`8e, alpha=0.3`), but increasing alpha to 0.6 hurts the 8-expert model and does not produce a monotonic scaling trend."
        )
    else:
        report.append("No residual MoE sweep rows were available.")

    report.extend(
        [
            "Residual MoE is not adapter composition. It is token-level learned composition inside the model, so it should be discussed as learned skill discovery rather than external LoRA selection or merging.",
            "",
            "## 7. In-Domain vs Out-of-Domain Generalization",
            "The out-of-domain games are evaluation-only. A large positive gap means a method is much better on in-domain games than on the negotiation-style out-of-domain games.",
            f"The largest in-domain result is **{best_indomain['method']}** at **{_fmt(best_indomain['clemscore_indomain'])}**. The largest out-of-domain result is **{best_outdomain['method']}** at **{_fmt(best_outdomain['clemscore_outdomain'])}**. This distinction matters because high in-domain specialization does not necessarily imply out-of-domain robustness.",
            "",
            "## 8. ClemScore vs StatScore",
            f"The Pearson correlation between overall ClemScore and StatScore is **{corr_txt}** over methods with both metrics. This quantifies whether dialogue-game gains track general linguistic/model performance. Methods above the trend line preserve StatScore better for their ClemScore; methods below it trade general ability for task performance.",
            "",
            "## 9. Poster-Level Findings",
            f"- Generalist LoRA is a strong baseline: **{_fmt(score(generalist))}** overall ClemScore and **{_fmt(score(generalist, 'statscore'))}** StatScore.",
            f"- Per-game oracle MoE is generalist-level or better in this run: **{_fmt(score(pergame_oracle))}** vs **{_fmt(score(generalist))}**, a difference of **{_fmt(pg_vs_gen)}**.",
            f"- Manual cluster oracle underperforms per-game oracle by **{_fmt(pg_vs_cl)}** points, suggesting the clusters are too coarse for some games.",
            f"- Practical routing is the bottleneck when oracle selection is stronger: cluster learned-router gap **{_fmt(learned_gap)}**, cluster loss-based gap **{_fmt(loss_gap)}**.",
            f"- WA/TA merging does not match the generalist in the current best case: best merge **{_fmt(score(merging))}** vs generalist **{_fmt(score(generalist))}**.",
            f"- Residual MoE provides token-level learned composition; the best sweep point reaches **{_fmt(score(residual))}** overall ClemScore.",
        ]
    )

    path = out_dir / "analysis_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser(description="Build the LoRA expert-composition analysis tables, plots, and report.")
    parser.add_argument("--root", type=Path, default=Path("playpen-eval"), help="Root playpen-eval directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs"), help="Analysis output directory.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_df = build_method_summary(root)
    main_df = build_main_comparison(method_df)
    routing_df = build_routing_comparison(method_df)
    merging_df = build_merging_comparison(method_df)
    merging_detail_df = build_merging_detailed_comparison(method_df)
    residual_df = build_residual_sweep(method_df)
    residual_alpha_effects, residual_expert_effects = build_residual_effect_tables(residual_df)
    gap_df = build_generalization_gap(method_df)
    corr_df = build_metric_correlation(method_df)
    ranked_df = build_ranked_methods(method_df)
    transfer_df, transfer_summary = build_adapter_transfer_tables(method_df)
    domain_transfer = build_adapter_domain_transfer_table(transfer_df)
    per_game_ood_summary, per_game_ood_by_game, per_game_ood_by_source_domain = build_per_game_outdomain_tables(transfer_df)
    own_cross_summary = build_own_cross_domain_summary(transfer_df)
    game_similarity_matrix, game_similarity_long, game_similarity_summary = build_empirical_game_similarity(transfer_df)
    transfer_matrix = build_transfer_matrix(transfer_df)
    cluster_in_out_summary, cluster_in_out_detailed = build_cluster_adapter_in_out_tables(transfer_df)

    _save_csv(method_df, out_dir, "method_summary.csv")
    _save_csv(main_df, out_dir, "main_comparison.csv")
    _save_csv(routing_df, out_dir, "routing_comparison.csv")
    _save_csv(merging_df, out_dir, "merging_comparison.csv")
    _save_csv(merging_detail_df, out_dir, "merging_detailed_comparison.csv")
    _save_csv(residual_df, out_dir, "residual_moe_sweep.csv")
    _save_csv(residual_alpha_effects, out_dir, "residual_moe_alpha_effects.csv")
    _save_csv(residual_expert_effects, out_dir, "residual_moe_expert_effects.csv")
    _save_csv(gap_df, out_dir, "generalization_gap.csv")
    _save_csv(corr_df, out_dir, "metric_correlation.csv")
    _save_csv(ranked_df, out_dir, "ranked_methods.csv")
    _save_csv(transfer_df, out_dir, "adapter_game_transfer.csv")
    if "adapter_type" in transfer_df.columns:
        _save_csv(transfer_df[transfer_df["adapter_type"].eq("per_game")].copy(), out_dir, "per_game_adapter_transfer.csv")
        _save_csv(transfer_df[transfer_df["adapter_type"].eq("cluster")].copy(), out_dir, "cluster_adapter_transfer.csv")
    else:
        _save_csv(pd.DataFrame(), out_dir, "per_game_adapter_transfer.csv")
        _save_csv(pd.DataFrame(), out_dir, "cluster_adapter_transfer.csv")
    _save_csv(transfer_summary, out_dir, "adapter_transfer_summary.csv")
    _save_csv(domain_transfer, out_dir, "adapter_domain_transfer.csv")
    if "adapter_type" in domain_transfer.columns:
        _save_csv(
            domain_transfer[domain_transfer["adapter_type"].eq("per_game")].copy(),
            out_dir,
            "per_game_adapter_domain_transfer.csv",
        )
        _save_csv(
            domain_transfer[domain_transfer["adapter_type"].eq("cluster")].copy(),
            out_dir,
            "cluster_adapter_domain_transfer.csv",
        )
    else:
        _save_csv(pd.DataFrame(), out_dir, "per_game_adapter_domain_transfer.csv")
        _save_csv(pd.DataFrame(), out_dir, "cluster_adapter_domain_transfer.csv")
    _save_csv(per_game_ood_summary, out_dir, "per_game_adapter_outdomain_summary.csv")
    _save_csv(per_game_ood_by_game, out_dir, "per_game_adapter_outdomain_by_game.csv")
    _save_csv(per_game_ood_by_source_domain, out_dir, "per_game_adapter_outdomain_by_source_domain.csv")
    _save_csv(own_cross_summary, out_dir, "adapter_own_cross_domain_summary.csv")
    _save_csv(transfer_matrix.reset_index().rename(columns={"source_game": "adapter_trained_on"}),
              out_dir, "transfer_matrix.csv")
    _save_csv(cluster_in_out_summary, out_dir, "cluster_adapter_in_out_summary.csv")
    _save_csv(cluster_in_out_detailed, out_dir, "cluster_adapter_in_out_detailed.csv")
    _save_csv(game_similarity_matrix.reset_index().rename(columns={"index": "game"}), out_dir, "empirical_game_similarity_matrix.csv")
    _save_csv(game_similarity_long, out_dir, "empirical_game_similarity_long.csv")
    _save_csv(game_similarity_summary, out_dir, "empirical_game_similarity_summary.csv")

    plot_main(main_df, out_dir)
    plot_routing(routing_df, out_dir)
    plot_merging(merging_df, method_df, out_dir)
    plot_residual_heatmap(residual_df, "clemscore_overall", out_dir / "residual_moe_heatmap_clemscore.png")
    plot_residual_heatmap(residual_df, "statscore", out_dir / "residual_moe_heatmap_statscore.png")
    plot_in_vs_out(method_df, out_dir)
    plot_scatter(method_df, out_dir)
    plot_average_rank(ranked_df, out_dir)
    plot_transfer_heatmaps(transfer_df, out_dir)
    plot_transfer_summary(transfer_summary, out_dir)
    plot_domain_transfer(domain_transfer, out_dir)
    plot_own_cross_summary(own_cross_summary, out_dir)
    plot_empirical_game_similarity(game_similarity_matrix, out_dir)
    plot_transfer_matrix_heatmap(transfer_matrix, out_dir)
    plot_transfer_network(transfer_matrix, out_dir)
    plot_transfer_pca(transfer_df, out_dir)
    plot_adapter_transferability(transfer_df, out_dir)
    plot_per_game_transfer_breakdown(transfer_df, out_dir)
    plot_cluster_adapter_in_vs_out(cluster_in_out_detailed, out_dir)
    write_cluster_in_out_html(cluster_in_out_summary, out_dir)

    write_report(
        out_dir,
        method_df,
        main_df,
        routing_df,
        merging_df,
        merging_detail_df,
        residual_df,
        corr_df,
        ranked_df,
        transfer_df,
        transfer_summary,
        domain_transfer,
        per_game_ood_summary,
        per_game_ood_by_game,
        per_game_ood_by_source_domain,
        residual_alpha_effects,
        residual_expert_effects,
        own_cross_summary,
        game_similarity_long,
        game_similarity_summary,
    )


if __name__ == "__main__":
    main()
