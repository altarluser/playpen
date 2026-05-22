#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

CLUSTER_MAP = {
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

OOD_GAMES = ["clean_up", "dond", "hot_air_balloon"]


METRICS = [
    ("clemscore", "Clemscore"),
    ("statscore", "Statscore"),
    ("wg_score", "WG Score"),
    ("wg_played", "WG % Played"),
    ("wg_quality", "WG Quality"),
    ("en_score", "EN Score"),
    ("en_played", "EN % Played"),
    ("en_quality", "EN Quality"),
    ("co_score", "CO Score"),
    ("co_played", "CO % Played"),
    ("co_quality", "CO Quality"),
    ("out_domain_score", "Clemscore ↑"),
    ("out_domain_played", "% Played ↑"),
    ("out_domain_quality", "Quality ↑"),
]


def _bar_html(delta: float, max_abs_delta: float | None = None) -> str:
    if pd.isna(delta) or abs(float(delta)) < 1e-12:
        return ""
    if max_abs_delta is None or not np.isfinite(max_abs_delta) or max_abs_delta <= 0:
        width = 2
    else:
        frac = min(1.0, abs(float(delta)) / float(max_abs_delta))
        width = max(2, int(round(2 + frac * 24)))  # [2..26] proportional per metric
    color = "#0a8f1f" if float(delta) > 0 else "#ff1f1f"
    return f"<span style='display:inline-block;width:{width}px;height:8px;background:{color};margin-left:6px;vertical-align:middle;'></span>"


def _fmt(v: float) -> str:
    if pd.isna(v):
        return "NaN"
    return f"{float(v):.2f}"


def _display_name(raw: str) -> str:
    s = str(raw)
    for pref in ("llama3-8b-sft-", "llama3-8b-"):
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.replace("_", " ").strip()
    return s


def _infer_family(run_name: str) -> str:
    s = str(run_name).lower()
    if "qwen" in s:
        return "Qwen"
    if "llama-3.1-70b" in s or "llama3.1-70b" in s or "70b" in s:
        return "Llama-3.1-70B"
    if "llama3-8b" in s or "llama-3.1-8b" in s:
        return "Llama-3-8B"
    # Default to the main family in this project, avoid noisy "Other".
    return "Llama-3-8B"


def _compute_group_metrics(clem_row: pd.Series, games: List[str]):
    played_vals = []
    quality_vals = []
    for g in games:
        p = pd.to_numeric(clem_row.get(f"{g}, % Played"), errors="coerce")
        q = pd.to_numeric(clem_row.get(f"{g}, Quality Score"), errors="coerce")
        if pd.notna(p):
            played_vals.append(float(p))
        if pd.notna(q):
            quality_vals.append(float(q))
    # Pairwise for score
    score_vals = []
    for g in games:
        p = pd.to_numeric(clem_row.get(f"{g}, % Played"), errors="coerce")
        q = pd.to_numeric(clem_row.get(f"{g}, Quality Score"), errors="coerce")
        if pd.notna(p) and pd.notna(q):
            score_vals.append((float(p) / 100.0) * float(q))
    score = float(np.nanmean(score_vals)) if score_vals else np.nan
    played = float(np.nanmean(played_vals)) if played_vals else np.nan
    quality = float(np.nanmean(quality_vals)) if quality_vals else np.nan
    return score, played, quality


def _augment_from_results_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["wg_score", "wg_played", "wg_quality", "en_score", "en_played", "en_quality", "co_score", "co_played", "co_quality", "out_domain_score", "out_domain_played", "out_domain_quality"]:
        if col not in out.columns:
            out[col] = np.nan

    for idx, r in out.iterrows():
        # Oracle MoE rows use proxy StatScore/OOD metrics averaged from their
        # source adapter families. Do not overwrite those with materialized
        # oracle clem/results.csv values, which only represent in-domain replay.
        if str(r.get("method_group", "")).lower() == "oracle_routing":
            continue
        run_dir = Path(str(r.get("run_dir", "")))
        clem_csv = run_dir / "clem" / "results.csv"
        if not clem_csv.exists():
            continue
        try:
            cdf = pd.read_csv(clem_csv)
            if cdf.empty:
                continue
            crow = cdf.iloc[0]
        except Exception:
            continue

        wg_s, wg_p, wg_q = _compute_group_metrics(crow, CLUSTER_MAP["wordguessing"])
        en_s, en_p, en_q = _compute_group_metrics(crow, CLUSTER_MAP["explorationnavigation"])
        co_s, co_p, co_q = _compute_group_metrics(crow, CLUSTER_MAP["cooperation"])
        od_s, od_p, od_q = _compute_group_metrics(crow, OOD_GAMES)

        out.at[idx, "wg_score"] = wg_s
        out.at[idx, "wg_played"] = wg_p
        out.at[idx, "wg_quality"] = wg_q
        out.at[idx, "en_score"] = en_s
        out.at[idx, "en_played"] = en_p
        out.at[idx, "en_quality"] = en_q
        out.at[idx, "co_score"] = co_s
        out.at[idx, "co_played"] = co_p
        out.at[idx, "co_quality"] = co_q
        out.at[idx, "out_domain_score"] = od_s
        out.at[idx, "out_domain_played"] = od_p
        out.at[idx, "out_domain_quality"] = od_q
    return out


def _harmonize_oracle_proxy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _mean_from(mask: pd.Series, col: str):
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
    cols = ["statscore", "out_domain_score", "out_domain_played", "out_domain_quality"]
    for family, pool_mask in pools.items():
        target_mask = targets[family]
        if not target_mask.any():
            continue
        for col in cols:
            value = _mean_from(pool_mask, col)
            if pd.notna(value):
                out.loc[target_mask, col] = value
    return out


def build_table(df: pd.DataFrame, out_html: Path, title: str):
    needed = {"method_name", "run_name", "clemscore", "statscore", "in_domain_score", "out_domain_score", "in_domain_played", "out_domain_played"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = _augment_from_results_csv(df.copy())

    # Grouping by model family.
    df["family"] = df["run_name"].map(_infer_family)
    df["label"] = df["method_name"].map(_display_name)

    # Per-family baseline row heuristic
    def _family_rows(part: pd.DataFrame) -> pd.DataFrame:
        part = part.sort_values("clemscore", ascending=False).copy()
        baseline = part[part["run_name"].astype(str).str.lower().eq("llama3-8b")]
        if baseline.empty:
            baseline = part[part["label"].str.lower().eq("base model")]
        base_row = baseline.iloc[0] if len(baseline) else part.iloc[-1]
        out = []
        for _, r in part.iterrows():
            row = {"family": r["family"], "label": r["label"]}
            for k, _ in METRICS:
                row[k] = r.get(k, np.nan)
                row[f"{k}_delta"] = (r.get(k, np.nan) - base_row.get(k, np.nan)) if r["label"] != base_row["label"] else np.nan
            out.append(row)
        return pd.DataFrame(out)

    rows = []
    for fam, part in df.groupby("family", sort=False):
        rows.append(_family_rows(part))
    table_df = pd.concat(rows, ignore_index=True)

    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'><title>Gameplay Results</title>")
    parts.append(
        "<style>"
        "body{font-family:Times New Roman,serif;margin:24px;color:#111;}"
        "h1{font-size:28px;margin:0 0 14px;}"
        "table{border-collapse:collapse;width:100%;font-size:16px;}"
        "th,td{padding:4px 8px;border-bottom:1px solid #666;white-space:nowrap;}"
        "thead th{border-bottom:2px solid #333;}"
        ".fam{font-weight:700;background:#f7f7f7;border-top:2px solid #444;}"
        ".model{font-weight:600;}"
        ".caption{margin-top:16px;font-size:16px;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>{title}</h1>")
    parts.append("<table>")
    parts.append("<thead>")
    parts.append("<tr><th></th><th colspan='2'>Overall</th><th colspan='9'>In Domain (by group)</th><th colspan='3'>Out of Domain</th></tr>")
    parts.append("<tr><th style='text-align:left'>Model</th>")
    for _, name in METRICS:
        parts.append(f"<th>{name}</th>")
    parts.append("</tr></thead><tbody>")

    for fam, part in table_df.groupby("family", sort=False):
        parts.append(f"<tr class='fam'><td colspan='15'>{fam}</td></tr>")
        max_abs_by_metric = {}
        for k, _ in METRICS:
            dcol = f"{k}_delta"
            if dcol in part.columns:
                vals = pd.to_numeric(part[dcol], errors="coerce").abs()
                max_abs_by_metric[k] = float(vals.max()) if vals.notna().any() else np.nan
            else:
                max_abs_by_metric[k] = np.nan
        for _, r in part.iterrows():
            parts.append(f"<tr><td class='model'>{r['label']}</td>")
            for k, _ in METRICS:
                parts.append(f"<td>{_fmt(r[k])}{_bar_html(r.get(f'{k}_delta', np.nan), max_abs_by_metric.get(k))}</td>")
            parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append(
        "<div class='caption'><b>Table.</b> Gameplay results with in-domain and out-of-domain metrics. "
        "Green bars indicate improvement vs family baseline, red bars indicate regression.</div>"
    )
    parts.append("</body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")
    print(f"[ok] wrote {out_html}")


def _select_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = str(mode).strip().lower()
    base = df[df["run_name"].astype(str).str.lower().eq("llama3-8b")]
    if base.empty:
        base = df[df["method_name"].astype(str).str.lower().eq("base model")]

    if mode == "main":
        m = df.copy()
        # Exclude per-game, cluster, residual from main table.
        m = m[~m["method_group"].isin(["per_game_adapter", "cluster_adapter", "residual_moe"])].copy()
        # keep baseline, generalist, merging, and explicit router/oracle variants + best residual
        generalist = m[m["run_name"].astype(str).str.contains("cluster_all", case=False, na=False)]
        merging = m[m["method_group"].eq("merging")]
        routed = m[m["method_group"].isin(["oracle_routing", "learned_router", "loss_based_router"])]
        keep_named = m[
            m["run_name"].astype(str).str.contains(
                r"oracle|router|loss[-_ ]?based",
                case=False,
                na=False,
                regex=True,
            )
        ]
        core = m[m["method_group"].isin(["base_model", "generalist_adapter"])]
        residual_best = df[df["method_group"].eq("residual_moe")].sort_values("clemscore", ascending=False).head(1)
        out = pd.concat([base, generalist, merging, core, routed, keep_named, residual_best], ignore_index=True).drop_duplicates(subset=["run_name"])
        return out

    if mode == "pergame":
        part = df[df["method_group"].eq("per_game_adapter")].copy()
        return pd.concat([base, part], ignore_index=True).drop_duplicates(subset=["run_name"])

    if mode == "cluster":
        part = df[df["method_group"].isin(["cluster_adapter", "generalist_adapter"])].copy()
        return pd.concat([base, part], ignore_index=True).drop_duplicates(subset=["run_name"])

    if mode == "residual":
        part = df[df["method_group"].eq("residual_moe")].copy().sort_values("clemscore", ascending=False)
        return pd.concat([base, part], ignore_index=True).drop_duplicates(subset=["run_name"])

    return df


def main():
    ap = argparse.ArgumentParser(description="Build paper-style gameplay results table.")
    ap.add_argument("--summary-csv", type=Path, default=Path("analysis_outputs/method_summary.csv"))
    ap.add_argument("--out-html", type=Path, default=Path("analysis_outputs/gameplay_results_table.html"))
    ap.add_argument("--title", type=str, default="Gameplay Results")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    df = _harmonize_oracle_proxy_metrics(df)
    # Main table
    build_table(_select_rows(df, "main"), args.out_html, args.title)

    # Extra tables
    out_dir = args.out_html.parent
    build_table(_select_rows(df, "pergame"), out_dir / "gameplay_results_table.pergame.html", "Gameplay Results: Per-Game Adapters")
    build_table(_select_rows(df, "cluster"), out_dir / "gameplay_results_table.cluster.html", "Gameplay Results: Cluster Adapters")
    build_table(_select_rows(df, "residual"), out_dir / "gameplay_results_table.residual.html", "Gameplay Results: Residual MoE Sweep")


if __name__ == "__main__":
    main()
