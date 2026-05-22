#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Optional

import clemcore.cli as clem


CLEM_EXPECTED_GAMES = (
    "adventuregame",
    "clean_up",
    "codenames",
    "dond",
    "guesswhat",
    "hot_air_balloon",
    "imagegame",
    "matchit_ascii",
    "privateshared",
    "referencegame",
    "taboo",
    "textmapworld",
    "textmapworld_graphreasoning",
    "textmapworld_specificroom",
    "wordle",
    "wordle_withclue",
    "wordle_withcritic",
)

GAME_GROUPS = {
    "wordguessing": (
        "codenames",
        "taboo",
        "guesswhat",
        "wordle",
        "wordle_withclue",
        "wordle_withcritic",
    ),
    "expnavi": (
        "adventuregame",
        "textmapworld",
        "textmapworld_graphreasoning",
        "textmapworld_specificroom",
    ),
    "coop": (
        "imagegame",
        "matchit_ascii",
        "referencegame",
        "privateshared",
    ),
    "negotiation": (
        "clean_up",
        "dond",
        "hot_air_balloon",
    ),
}


def _extract_clemscore(df) -> float:
    # Robust extraction across dataframe layouts.
    try:
        series = df["-, clemscore"]
        val = float(series.iloc[0])
    except Exception:
        try:
            val = float(df.iloc[0]["-, clemscore"])
        except Exception:
            # Final fallback: 0.0 if columns are missing due to empty/aborted runs.
            val = 0.0
    return 0.0 if val != val else val


def _contains_val_component(path: Path) -> bool:
    return any(part == "val" for part in path.parts)


def _contains_ignore_component(path: Path) -> bool:
    return any(part == "ignore" for part in path.parts)


def _normalize_results_csv(suite_dir: Path, expected_games: tuple[str, ...]) -> None:
    results_csv = suite_dir / "results.csv"
    if not results_csv.exists():
        return
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list((rows and rows[0].keys()) or [])
    except Exception:
        return
    if not rows or not fieldnames:
        return

    # Clem/static outputs are typically single-row by model.
    row = rows[0]

    # Detect whether this file contains std columns and preserve existing naming.
    # Use a canonical name if missing and needed.
    for game in expected_games:
        played_col = f"{game}, % Played"
        q_col = f"{game}, Quality Score"
        q_std_col = f"{game}, Quality Score (std)"

        if played_col not in fieldnames:
            fieldnames.append(played_col)
            row[played_col] = "0"
        if q_col not in fieldnames:
            fieldnames.append(q_col)
            row[q_col] = "NaN"

        # If std columns are present in this file for any game, normalize this one too.
        has_any_std = any("quality score (std)" in str(c).lower() for c in fieldnames)
        if has_any_std and q_std_col not in fieldnames:
            fieldnames.append(q_std_col)
            row[q_std_col] = "NaN"

        played_val = str(row.get(played_col, "")).strip()
        try:
            played_num = float(played_val) if played_val != "" else 0.0
        except Exception:
            played_num = 0.0
        if played_val == "":
            row[played_col] = "0"

        # For unplayed/missing games, force NaN quality fields.
        if played_num <= 0.0:
            row[q_col] = "NaN"
            if q_std_col in fieldnames:
                row[q_std_col] = "NaN"
        else:
            # Keep measured values for played games.
            if str(row.get(q_col, "")).strip() == "":
                row[q_col] = "NaN"
            if q_std_col in fieldnames and str(row.get(q_std_col, "")).strip() == "":
                row[q_std_col] = "NaN"

    try:
        with results_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        return


def recompute_suite(suite_dir: Path) -> Optional[float]:
    # Use broad game selector; suite_dir already scopes what files are present.
    clem.score("all", str(suite_dir))
    clem.transcripts("all", str(suite_dir))
    try:
        df = clem.clemeval.perform_evaluation(str(suite_dir), return_dataframe=True)
        score = _extract_clemscore(df)
        if suite_dir.name == "clem":
            _normalize_results_csv(suite_dir, CLEM_EXPECTED_GAMES)
        return score
    except Exception:
        if suite_dir.name == "clem":
            _normalize_results_csv(suite_dir, CLEM_EXPECTED_GAMES)
        return None


def recompute_run_dir(run_dir: Path, dry_run: bool = False) -> Optional[Dict[str, Optional[float]]]:
    if _contains_val_component(run_dir):
        return None
    if _contains_ignore_component(run_dir):
        return None
    if not run_dir.is_dir():
        return None

    clem_dir = run_dir / "clem"
    static_dir = run_dir / "static"
    has_clem = clem_dir.is_dir()
    has_static = static_dir.is_dir()
    if not has_clem and not has_static:
        return None

    scores: Dict[str, Optional[float]] = {}
    if has_clem:
        scores["clemscore"] = 0.0 if dry_run else recompute_suite(clem_dir)
    if has_static:
        scores["statscore"] = 0.0 if dry_run else recompute_suite(static_dir)

    out_file = run_dir / f"{run_dir.name}.val.json"
    if not dry_run:
        out_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def discover_run_dirs(root: Path):
    # A run dir is any dir that directly contains clem/ and/or static/, excluding any .../val/...
    if root.is_dir() and (not _contains_val_component(root)) and (not _contains_ignore_component(root)):
        if (root / "clem").is_dir() or (root / "static").is_dir():
            yield root
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if d.name in {"clem", "static", "val", "plots", "correlation"}:
            continue
        if _contains_val_component(d):
            continue
        if _contains_ignore_component(d):
            continue
        if (d / "clem").is_dir() or (d / "static").is_dir():
            yield d


def _to_float(v):
    try:
        out = float(v)
        return None if out != out else out
    except Exception:
        return None


def _mean(vals):
    xs = [x for x in vals if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _read_group_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    clem_csv = run_dir / "clem" / "results.csv"
    if not clem_csv.exists():
        for group in GAME_GROUPS:
            out[f"{group}_played"] = None
            out[f"{group}_quality"] = None
            out[f"{group}_quality_std"] = None
        return out
    try:
        with clem_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        row = rows[0] if rows else {}
    except Exception:
        row = {}

    for group, games in GAME_GROUPS.items():
        played_vals = []
        quality_vals = []
        std_vals = []
        for g in games:
            p = _to_float(row.get(f"{g}, % Played"))
            q = _to_float(row.get(f"{g}, Quality Score"))
            s = _to_float(row.get(f"{g}, Quality Score (std)"))
            # Missing played means not played.
            played_vals.append(0.0 if p is None else p)
            # Keep quality/std only for games with some played portion.
            if (p is not None) and (p > 0):
                quality_vals.append(q)
                std_vals.append(s)
        out[f"{group}_played"] = _mean(played_vals)
        out[f"{group}_quality"] = _mean(quality_vals)
        out[f"{group}_quality_std"] = _mean(std_vals)
        gp = out[f"{group}_played"]
        gq = out[f"{group}_quality"]
        if gp is not None and gq is not None:
            out[f"{group}_score"] = float(gp) * float(gq) / 100.0
        else:
            out[f"{group}_score"] = None
    return out


def write_leaderboard(root: Path, entries: list[dict]) -> Path:
    rows = []
    for item in entries:
        run_dir = Path(item["run_dir"])
        if _contains_val_component(run_dir) or _contains_ignore_component(run_dir):
            continue
        clem = _to_float(item.get("clemscore"))
        stat = _to_float(item.get("statscore"))
        if clem is None and stat is None:
            continue
        rel = run_dir.relative_to(root)
        group_metrics = _read_group_metrics(run_dir)
        rows.append({"run": str(rel), "clemscore": clem, "statscore": stat, **group_metrics})

    # Oracle-MoE harmonization (user-requested):
    # Compute proxy values from the corresponding adapter family:
    # - cluster-oracle-moe <- mean over cluster adapters
    # - pergame-oracle-moe <- mean over per-game adapters
    # Applies to statscore and negotiation group metrics.
    def _mean_of(key: str, pred):
        vals = [r.get(key) for r in rows if pred(r) and (r.get(key) is not None)]
        return (sum(vals) / len(vals)) if vals else None

    def pergame_pool(r):
        parts = str(r["run"]).split("/")
        return len(parts) == 2 and parts[0] == "per-game adapters"

    def cluster_pool(r):
        parts = str(r["run"]).split("/")
        return (
            len(parts) == 2
            and parts[0] == "cluster adapters"
            and parts[1].startswith("llama3-8b-sft-cluster_")
            and parts[1] != "llama3-8b-sft-cluster_all"
        )

    cluster_statscore_mean = _mean_of("statscore", cluster_pool)
    cluster_neg_score_mean = _mean_of("negotiation_score", cluster_pool)
    cluster_neg_played_mean = _mean_of("negotiation_played", cluster_pool)
    cluster_neg_quality_mean = _mean_of("negotiation_quality", cluster_pool)
    cluster_neg_std_mean = _mean_of("negotiation_quality_std", cluster_pool)

    pergame_statscore_mean = _mean_of("statscore", pergame_pool)
    pergame_neg_score_mean = _mean_of("negotiation_score", pergame_pool)
    pergame_neg_played_mean = _mean_of("negotiation_played", pergame_pool)
    pergame_neg_quality_mean = _mean_of("negotiation_quality", pergame_pool)
    pergame_neg_std_mean = _mean_of("negotiation_quality_std", pergame_pool)

    for r in rows:
        run_lower = str(r["run"]).lower()
        if ("cluster-oracle-moe" in run_lower) or ("cluster_oracle_moe" in run_lower):
            if cluster_statscore_mean is not None:
                r["statscore"] = cluster_statscore_mean
            if cluster_neg_score_mean is not None:
                r["negotiation_score"] = cluster_neg_score_mean
            if cluster_neg_played_mean is not None:
                r["negotiation_played"] = cluster_neg_played_mean
            if cluster_neg_quality_mean is not None:
                r["negotiation_quality"] = cluster_neg_quality_mean
            if cluster_neg_std_mean is not None:
                r["negotiation_quality_std"] = cluster_neg_std_mean
            continue
        if ("pergame-oracle-moe" in run_lower) or ("pergame_oracle_moe" in run_lower):
            if pergame_statscore_mean is not None:
                r["statscore"] = pergame_statscore_mean
            if pergame_neg_score_mean is not None:
                r["negotiation_score"] = pergame_neg_score_mean
            if pergame_neg_played_mean is not None:
                r["negotiation_played"] = pergame_neg_played_mean
            if pergame_neg_quality_mean is not None:
                r["negotiation_quality"] = pergame_neg_quality_mean
            if pergame_neg_std_mean is not None:
                r["negotiation_quality_std"] = pergame_neg_std_mean

    by_clem = sorted(rows, key=lambda r: ((r["clemscore"] is None), -(r["clemscore"] or -10**9), -(r["statscore"] or -10**9), r["run"]))
    by_stat = sorted(rows, key=lambda r: ((r["statscore"] is None), -(r["statscore"] or -10**9), -(r["clemscore"] or -10**9), r["run"]))
    by_folder = {}
    for r in rows:
        folder = r["run"].split("/", 1)[0] if "/" in r["run"] else r["run"]
        by_folder.setdefault(folder, []).append(r)
    for folder, items in by_folder.items():
        by_folder[folder] = {
            "clem": sorted(items, key=lambda x: ((x["clemscore"] is None), -(x["clemscore"] or -10**9), -(x["statscore"] or -10**9), x["run"])),
            "stat": sorted(items, key=lambda x: ((x["statscore"] is None), -(x["statscore"] or -10**9), -(x["clemscore"] or -10**9), x["run"])),
        }

    def fmt(v):
        return "NaN" if v is None else f"{v:.2f}"

    def make_table(data, title):
        lines = [f"<h2>{title}</h2>", "<table>", "<thead><tr>"
                 "<th>#</th><th>Run</th><th>ClemScore</th><th>StatScore</th>"
                 "<th>WG Score</th><th>WG Played</th><th>WG Q</th>"
                 "<th>EN Score</th><th>EN Played</th><th>EN Q</th>"
                 "<th>CO Score</th><th>CO Played</th><th>CO Q</th>"
                 "<th>NEG Score</th><th>NEG Played</th><th>NEG Q</th>"
                 "</tr></thead>", "<tbody>"]
        for i, r in enumerate(data, start=1):
            lines.append(
                f"<tr><td>{i}</td><td>{r['run']}</td><td>{fmt(r['clemscore'])}</td><td>{fmt(r['statscore'])}</td>"
                f"<td>{fmt(r.get('wordguessing_score'))}</td>"
                f"<td>{fmt(r.get('wordguessing_played'))}</td><td>{fmt(r.get('wordguessing_quality'))}</td>"
                f"<td>{fmt(r.get('expnavi_score'))}</td>"
                f"<td>{fmt(r.get('expnavi_played'))}</td><td>{fmt(r.get('expnavi_quality'))}</td>"
                f"<td>{fmt(r.get('coop_score'))}</td>"
                f"<td>{fmt(r.get('coop_played'))}</td><td>{fmt(r.get('coop_quality'))}</td>"
                f"<td>{fmt(r.get('negotiation_score'))}</td>"
                f"<td>{fmt(r.get('negotiation_played'))}</td><td>{fmt(r.get('negotiation_quality'))}</td>"
                f"</tr>"
            )
        lines.append("</tbody></table>")
        return "\n".join(lines)

    folder_sections = []
    for folder in sorted(by_folder.keys()):
        folder_sections.append(f"<h2>Folder: {folder}</h2>")
        folder_sections.append(make_table(by_folder[folder]["clem"], f"{folder} — Ranked by ClemScore"))
        folder_sections.append(make_table(by_folder[folder]["stat"], f"{folder} — Ranked by StatScore"))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Playpen Leaderboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    thead th {{ background: #f3f4f6; position: sticky; top: 0; }}
    h1, h2 {{ margin: 8px 0 16px; }}
    .meta {{ color: #555; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>Playpen Leaderboard</h1>
  <div class="meta">Root: {root} | Entries: {len(rows)}</div>
  {make_table(by_clem, "Ranked by ClemScore (desc), StatScore tie-break")}
  {make_table(by_stat, "Ranked by StatScore (desc), ClemScore tie-break")}
  {' '.join(folder_sections)}
</body>
</html>
"""
    out = root / "leaderboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute playpen eval scores for run directories under a root. "
            "Ignores any path containing '/val/'. Writes <run_dir_name>.val.json per run dir."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("playpen-eval"),
        help="Root folder to scan (default: playpen-eval).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be recomputed.")
    parser.add_argument("--no-leaderboard", action="store_true", help="Skip leaderboard.html generation.")
    parser.add_argument("--leaderboard-only", action="store_true", help="Only rebuild leaderboard from existing val/results files.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root not found or not a directory: {root}")

    run_dirs = sorted(set(discover_run_dirs(root)))
    print(f"Found {len(run_dirs)} candidate run dirs under {root}")

    processed = 0
    leaderboard_entries = []
    if args.leaderboard_only:
        for run_dir in run_dirs:
            if _contains_val_component(run_dir) or _contains_ignore_component(run_dir):
                continue
            val_file = run_dir / f"{run_dir.name}.val.json"
            if not val_file.exists():
                continue
            try:
                payload = json.loads(val_file.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            leaderboard_entries.append(
                {
                    "run_dir": str(run_dir),
                    "clemscore": payload.get("clemscore"),
                    "statscore": payload.get("statscore"),
                }
            )
            processed += 1
    else:
        for run_dir in run_dirs:
            scores = recompute_run_dir(run_dir, dry_run=args.dry_run)
            if scores is None:
                continue
            processed += 1
            leaderboard_entries.append(
                {
                    "run_dir": str(run_dir),
                    "clemscore": scores.get("clemscore"),
                    "statscore": scores.get("statscore"),
                }
            )
            print(f"[ok] {run_dir}")
            print(json.dumps(scores, indent=2))

    if not args.dry_run and not args.no_leaderboard:
        lb = write_leaderboard(root, leaderboard_entries)
        print(f"Leaderboard written: {lb}")

    print(f"Done. Processed {processed} run dirs.")


if __name__ == "__main__":
    main()
