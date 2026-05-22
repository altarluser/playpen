"""
Analyze MoE residual routing statistics from latent_eval_usage.csv logs.

Usage:
    python scripts/analyze_moe_routing.py \
        --models llama3-8b-sft-moe-residual-8e-a0.3-aux0.003 \
        --routing-root analysis_outputs \
        --out-dir analysis_outputs/moe_routing_analysis
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── game taxonomy ────────────────────────────────────────────────────────────

CLUSTER_MAP: Dict[str, List[str]] = {
    "wordguessing": [
        "codenames", "taboo", "guesswhat",
        "wordle", "wordle_withclue", "wordle_withcritic",
    ],
    "explorationnavigation": [
        "adventuregame", "textmapworld",
        "textmapworld_graphreasoning", "textmapworld_specificroom",
    ],
    "cooperation": [
        "imagegame", "matchit_ascii", "referencegame", "privateshared",
    ],
}

OOD_GAMES = ["dond", "clean_up", "hot_air_balloon"]
IN_DOMAIN_GAMES = [g for games in CLUSTER_MAP.values() for g in games]
ALL_GAMES = IN_DOMAIN_GAMES + OOD_GAMES
CLEM_GAMES = ALL_GAMES

GAME_GROUP = {g: grp for grp, gs in CLUSTER_MAP.items() for g in gs}
GAME_GROUP.update({g: "ood_negotiation" for g in OOD_GAMES})

CLUSTER_COLORS = {
    "wordguessing": "#3b82f6",
    "explorationnavigation": "#22c55e",
    "cooperation": "#f59e0b",
    "ood_negotiation": "#dc2626",
}

# Default stronger colors for non-poster diagnostic plots.
EXPERT_COLORS = [
    "#2563eb", "#16a34a", "#f59e0b", "#dc2626",
    "#7c3aed", "#0891b2", "#be185d", "#374151",
]

# Poster palette: softer and less painful beside the transfer matrix.
EXPERT_COLORS_POSTER = [
    "#5B8DB8",  # E0 muted blue
    "#6FAA5F",  # E1 muted green
    "#E6B84E",  # E2 muted yellow
    "#D95F5F",  # E3 muted red
    "#A879A8",  # E4 muted purple
    "#69AFA9",  # E5 muted teal
    "#D77FA1",  # E6 muted pink
    "#A27A63",  # E7 muted brown
]


# ── shared style constants ───────────────────────────────────────────────────

CELL = 0.65
SEP_C = "#374151"
SEP_LW = 1.3
OOD_C = "#b91c1c"

FS_CELL = 7.2
FS_TICK = 8.5
FS_LABEL = 11
FS_TITLE = 10


# ── helpers ──────────────────────────────────────────────────────────────────

def _game_label(g: str) -> str:
    return g.replace("_", " ").replace("textmapworld", "tmw")


def _model_label(name: str) -> str:
    m = re.search(r"(\d+e)-a([\d.]+)", name)
    if m:
        return f"{m.group(1)}, α={m.group(2)}"
    return name


def _game_order(games: List[str]) -> List[str]:
    return [g for g in ALL_GAMES if g in games]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def _draw_cluster_separators(ax, games_present: List[str], axis: str) -> None:
    for _, g_list in CLUSTER_MAP.items():
        present = [g for g in g_list if g in games_present]
        if not present:
            continue

        end = games_present.index(present[-1]) + 1

        if end < len(games_present):
            if axis == "row":
                ax.axhline(end - 0.5, color=SEP_C, linewidth=SEP_LW)
            else:
                ax.axvline(end - 0.5, color=SEP_C, linewidth=SEP_LW)


def _draw_ood_separator(ax, games_present: List[str], axis: str) -> None:
    ood_present = [g for g in OOD_GAMES if g in games_present]
    if not ood_present:
        return

    ood_start = games_present.index(ood_present[0])

    if axis == "row":
        ax.axhline(
            ood_start - 0.5,
            color=OOD_C,
            linewidth=1.6,
            linestyle=(0, (4, 3)),
        )
        ax.text(
            -0.62,
            ood_start - 0.5,
            "OOD games",
            ha="right",
            va="center",
            fontsize=FS_TICK - 1,
            color=OOD_C,
            fontweight="bold",
            clip_on=False,
        )
    else:
        ax.axvline(
            ood_start - 0.5,
            color=OOD_C,
            linewidth=1.6,
            linestyle=(0, (4, 3)),
        )
        ax.text(
            ood_start + (len(ood_present) - 1) / 2,
            -0.62,
            "OOD games",
            ha="center",
            va="bottom",
            fontsize=FS_TICK - 1,
            color=OOD_C,
            fontweight="bold",
            clip_on=False,
        )


# ── data loading ─────────────────────────────────────────────────────────────

def load_model_data(routing_root: Path, model_name: str) -> Optional[pd.DataFrame]:
    path = routing_root / model_name / "moe-routing" / "latent_eval_usage.csv"

    if not path.exists():
        print(f"[warn] missing: {path}")
        return None

    df = pd.read_csv(path)
    df["model"] = model_name

    required = ["layer", "expert_id", "top1_proportion", "avg_entropy"]
    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["layer", "expert_id"])
    return df


# ── POSTER PLOT: stacked contribution per game-layer cell ────────────────────

def plot_poster_heatmap_only(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Poster-ready double-column stacked expert-contribution heatmap.

    Each game-layer cell is a horizontal stacked bar:
      - segment width = expert top-1 proportion
      - segment color = expert
      - number = dominant expert's top-1 share

    Layout:
      - left panel: wordguessing + exploration/navigation
      - right panel: cooperation + OOD negotiation

    No title. Designed to sit beside transfer_matrix_heatmap.
    """

    from matplotlib.patches import Patch

    pal = EXPERT_COLORS_POSTER

    def ordered_subset(groups: List[str], games_present: List[str]) -> List[str]:
        out = []
        for group in groups:
            if group == "ood_negotiation":
                candidates = OOD_GAMES
            else:
                candidates = CLUSTER_MAP[group]
            out.extend([g for g in candidates if g in games_present])
        return out

    def build_arrays(
        df_clem: pd.DataFrame,
        games_subset: List[str],
        layers: List[int],
        experts: List[int],
    ):
        n_games = len(games_subset)
        n_layers = len(layers)
        n_exp = len(experts)

        proportions = np.zeros((n_games, n_layers, n_exp))
        dom = np.full((n_games, n_layers), np.nan)
        top1 = np.full((n_games, n_layers), np.nan)

        for gi, game in enumerate(games_subset):
            for li, layer in enumerate(layers):
                sub = df_clem[
                    (df_clem["game"] == game) &
                    (df_clem["layer"] == layer)
                ]

                if sub.empty:
                    continue

                for ei, eid in enumerate(experts):
                    row = sub[sub["expert_id"] == eid]
                    if not row.empty:
                        proportions[gi, li, ei] = float(
                            row["top1_proportion"].mean()
                        )

                # Defensive normalization so each cell fills exactly one cell width.
                s = proportions[gi, li].sum()
                if s > 0:
                    proportions[gi, li] /= s

                best_idx = int(np.nanargmax(proportions[gi, li]))
                dom[gi, li] = experts[best_idx]
                top1[gi, li] = proportions[gi, li, best_idx]

        return proportions, dom, top1

    def draw_panel(
        ax,
        proportions: np.ndarray,
        dom: np.ndarray,
        top1: np.ndarray,
        games_subset: List[str],
        layers: List[int],
        experts: List[int],
        show_ylabel: bool,
        right_side_labels: bool = False,
    ) -> None:
        n_games = len(games_subset)
        n_layers = len(layers)

        # Draw stacked expert segments inside each game-layer cell.
        for gi in range(n_games):
            for li in range(n_layers):
                x_left = li - 0.5

                for ei, eid in enumerate(experts):
                    width = proportions[gi, li, ei]
                    if width <= 0:
                        continue

                    ax.add_patch(
                        plt.Rectangle(
                            (x_left, gi - 0.5),
                            width,
                            1.0,
                            facecolor=pal[eid % len(pal)],
                            edgecolor="none",
                            alpha=1.0,
                            zorder=1,
                        )
                    )
                    x_left += width

        # Thin white grid lines.
        for li in range(n_layers + 1):
            ax.axvline(li - 0.5, color="white", linewidth=0.75, zorder=2)

        for gi in range(n_games + 1):
            ax.axhline(gi - 0.5, color="white", linewidth=0.75, zorder=2)

        # Dominant share numbers: dark, simple, no outline.
        for gi in range(n_games):
            for li in range(n_layers):
                if not np.isfinite(top1[gi, li]):
                    continue

                dominant_eid = int(dom[gi, li])
                dominant_pos = experts.index(dominant_eid)

                x_start = li - 0.5 + proportions[gi, li, :dominant_pos].sum()
                x_text = x_start + proportions[gi, li, dominant_pos] / 2

                ax.text(
                    x_text,
                    gi,
                    f"{top1[gi, li]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FS_CELL,
                    color="#0F172A",
                    fontweight="normal",
                    zorder=3,
                )

        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(n_games - 0.5, -0.5)

        # Cluster separators inside this panel.
        for _, g_list in CLUSTER_MAP.items():
            present = [g for g in g_list if g in games_subset]
            if not present:
                continue

            end = games_subset.index(present[-1]) + 1
            if end < len(games_subset):
                ax.axhline(
                    end - 0.5,
                    color=SEP_C,
                    linewidth=1.0,
                    zorder=4,
                )

        # OOD separator inside right panel.
        ood_present = [g for g in OOD_GAMES if g in games_subset]
        if ood_present:
            ood_start = games_subset.index(ood_present[0])
            ax.axhline(
                ood_start - 0.5,
                color=OOD_C,
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                zorder=5,
            )

        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(
            [f"L{l}" for l in layers],
            fontsize=FS_TICK,
            fontweight="bold",
        )

        ax.set_yticks(np.arange(n_games))
        ax.set_yticklabels(
            [_game_label(g) for g in games_subset],
            fontsize=FS_TICK,
        )

        # This is the important fix:
        # right panel labels go to the right side, so they do not overlap the left panel.
        if right_side_labels:
            ax.yaxis.tick_right()
            ax.tick_params(
                axis="y",
                labelright=True,
                labelleft=False,
                right=False,
                left=False,
                pad=3,
                length=0,
            )
        else:
            ax.yaxis.tick_left()
            ax.tick_params(
                axis="y",
                labelleft=True,
                labelright=False,
                right=False,
                left=False,
                pad=3,
                length=0,
            )

        ax.tick_params(axis="x", length=0)

        ax.set_xlabel("Layer", fontsize=FS_LABEL, fontweight="bold", labelpad=5)

        if show_ylabel:
            ax.set_ylabel("Game", fontsize=FS_LABEL, fontweight="bold", labelpad=5)
        else:
            ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

    for model_name, df in dfs.items():
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        experts = sorted(df_clem["expert_id"].dropna().unique().astype(int))
        layers = sorted(df_clem["layer"].dropna().unique().astype(int))
        games_present = _game_order(df_clem["game"].unique().tolist())

        left_games = ordered_subset(
            ["wordguessing", "explorationnavigation"],
            games_present,
        )

        right_games = ordered_subset(
            ["cooperation", "ood_negotiation"],
            games_present,
        )

        left_props, left_dom, left_top1 = build_arrays(
            df_clem, left_games, layers, experts
        )

        right_props, right_dom, right_top1 = build_arrays(
            df_clem, right_games, layers, experts
        )

        # Two-column layout.
        # Height is driven by the taller left column.
        max_games = max(len(left_games), len(right_games))
        fig_w = 2 * (len(layers) * CELL + 1.6) + 1.0
        fig_h = max_games * CELL + 1.45

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            gridspec_kw={
                "width_ratios": [1.0, 1.0],
                "wspace": 0.12,
            },
        )

        draw_panel(
            axes[0],
            left_props,
            left_dom,
            left_top1,
            left_games,
            layers,
            experts,
            show_ylabel=True,
            right_side_labels=False
        )

        draw_panel(
            axes[1],
            right_props,
            right_dom,
            right_top1,
            right_games,
            layers,
            experts,
            show_ylabel=False,
            right_side_labels=True
        )

        # Small group labels above panels, not a main title.
        axes[0].set_title(
            "Word guessing + exploration/navigation",
            fontsize=FS_TICK,
            fontweight="bold",
            pad=4,
            color="#111827",
        )

        axes[1].set_title(
            "Cooperation + negotiation",
            fontsize=FS_TICK,
            fontweight="bold",
            pad=4,
            color="#111827",
        )

        # Compact legend.
        patches = [
            Patch(
                facecolor=pal[e % len(pal)],
                edgecolor="none",
                alpha=1.0,
                label=f"E{e}",
            )
            for e in experts
        ]

        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=len(experts),
            bbox_to_anchor=(0.5, -0.006),
            fontsize=FS_TICK - 1,
            frameon=False,
            title="Expert contribution",
            title_fontsize=FS_TICK - 1,
            columnspacing=0.75,
            handlelength=1.0,
        )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        _save(fig, out_dir / "moe_poster_heatmap.png")


# ── plot 1: game × expert heatmap ────────────────────────────────────────────

def plot_game_expert_heatmap(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Heatmap:
    rows = games
    cols = experts
    cell = avg top1_proportion across layers
    """

    for model_name, df in dfs.items():
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()
        games_present = _game_order(df_clem["game"].unique().tolist())
        experts = sorted(df_clem["expert_id"].dropna().unique().astype(int))

        n_games = len(games_present)
        n_experts = len(experts)
        uniform = 1.0 / n_experts

        mat = np.full((n_games, n_experts), np.nan)

        for gi, game in enumerate(games_present):
            for ei, eid in enumerate(experts):
                sub = df_clem[
                    (df_clem["game"] == game) &
                    (df_clem["expert_id"] == eid)
                ]

                if not sub.empty:
                    if "tokens" in sub.columns:
                        w = sub["tokens"].fillna(1.0)
                        mat[gi, ei] = float(
                            np.average(sub["top1_proportion"].fillna(0), weights=w)
                        )
                    else:
                        mat[gi, ei] = float(sub["top1_proportion"].mean())

        delta = mat - uniform
        vmax = max(np.nanmax(np.abs(delta)), 0.01)

        fig_w = n_experts * CELL + 4.0
        fig_h = n_games * CELL + 3.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        for gi in range(n_games):
            for ei in range(n_experts):
                v = mat[gi, ei]
                d = delta[gi, ei]

                if np.isfinite(v):
                    text_color = "white" if abs(d) / vmax > 0.55 else "#111827"
                    ax.text(
                        ei,
                        gi,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=FS_CELL,
                        color=text_color,
                    )

        # Box dominant expert per game.
        for gi in range(n_games):
            row = delta[gi]
            if np.any(np.isfinite(row)):
                dom_ei = int(np.nanargmax(row))
                ax.add_patch(
                    plt.Rectangle(
                        (dom_ei - 0.5, gi - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111827",
                        linewidth=1.8,
                    )
                )

        _draw_cluster_separators(ax, games_present, "row")
        _draw_ood_separator(ax, games_present, "row")

        ax.set_xticks(np.arange(n_experts))
        ax.set_xticklabels(
            [f"Expert {e}" for e in experts],
            fontsize=FS_TICK,
            fontweight="bold",
        )
        ax.set_yticks(np.arange(n_games))
        ax.set_yticklabels(
            [_game_label(g) for g in games_present],
            fontsize=FS_TICK,
        )

        ax.set_xlabel("Expert  →", fontsize=FS_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("←  Game", fontsize=FS_LABEL, fontweight="bold")

        ax.set_title(
            f"Game × Expert routing preference  —  {_model_label(model_name)}\n"
            "Cell = avg top-1 proportion across layers · Black box = dominant expert per game",
            fontsize=FS_TITLE,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(
            "Top-1 proportion − uniform\n(blue = over-selected, red = under-selected)",
            fontsize=FS_TICK,
        )

        fig.tight_layout()
        _save(fig, out_dir / "moe_game_expert_heatmap.png")


# ── plot 2: dominant expert heatmap ──────────────────────────────────────────

def plot_dominant_expert_heatmap(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Heatmap:
    rows = games
    cols = layers
    color = dominant expert
    value = dominant expert top1 share
    """

    from matplotlib.colors import ListedColormap, BoundaryNorm

    for model_name, df in dfs.items():
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        experts = sorted(df_clem["expert_id"].dropna().unique().astype(int))
        layers = sorted(df_clem["layer"].dropna().unique().astype(int))
        games_present = _game_order(df_clem["game"].unique().tolist())

        n_games = len(games_present)
        n_layers = len(layers)

        dom = np.full((n_games, n_layers), np.nan)
        prop = np.full((n_games, n_layers), np.nan)

        for gi, game in enumerate(games_present):
            for li, layer in enumerate(layers):
                sub = df_clem[
                    (df_clem["game"] == game) &
                    (df_clem["layer"] == layer)
                ]

                if sub.empty:
                    continue

                best = sub.loc[sub["top1_proportion"].idxmax()]
                dom[gi, li] = float(best["expert_id"])
                prop[gi, li] = float(best["top1_proportion"])

        cmap = ListedColormap(
            [EXPERT_COLORS_POSTER[int(e) % len(EXPERT_COLORS_POSTER)] for e in experts]
        )
        bounds = [e - 0.5 for e in experts] + [experts[-1] + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig_w = n_layers * CELL + 4.0
        fig_h = n_games * CELL + 3.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(dom, cmap=cmap, norm=norm, aspect="auto")

        for gi in range(n_games):
            for li in range(n_layers):
                if np.isfinite(prop[gi, li]):
                    ax.text(
                        li,
                        gi,
                        f"{prop[gi, li]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=FS_CELL,
                        color="#111827",
                    )

        _draw_cluster_separators(ax, games_present, "row")
        _draw_ood_separator(ax, games_present, "row")

        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(
            [f"L{l}" for l in layers],
            fontsize=FS_TICK,
            fontweight="bold",
        )
        ax.set_yticks(np.arange(n_games))
        ax.set_yticklabels(
            [_game_label(g) for g in games_present],
            fontsize=FS_TICK,
        )

        ax.set_xlabel("Layer  →", fontsize=FS_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("←  Game", fontsize=FS_LABEL, fontweight="bold")

        ax.set_title(
            f"Dominant expert per game × layer  —  {_model_label(model_name)}\n"
            "Cell value = top-1 proportion of the winning expert",
            fontsize=FS_TITLE,
        )

        cbar = fig.colorbar(im, ax=ax, ticks=experts, shrink=0.7)
        cbar.set_label("Dominant expert", fontsize=FS_TICK)
        cbar.set_ticklabels([f"E{e}" for e in experts])

        fig.tight_layout()
        _save(fig, out_dir / "moe_dominant_expert_heatmap.png")


# ── plot 3: routing entropy heatmap ──────────────────────────────────────────

def plot_entropy_heatmap(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    for model_name, df in dfs.items():
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        layers = sorted(df_clem["layer"].dropna().unique().astype(int))
        games_present = _game_order(df_clem["game"].unique().tolist())

        n_games = len(games_present)
        n_layers = len(layers)

        ent = np.full((n_games, n_layers), np.nan)

        for gi, game in enumerate(games_present):
            for li, layer in enumerate(layers):
                sub = df_clem[
                    (df_clem["game"] == game) &
                    (df_clem["layer"] == layer)
                ]

                if not sub.empty:
                    ent[gi, li] = sub["avg_entropy"].mean()

        vmin = np.nanmin(ent)
        vmax = np.nanmax(ent)

        fig_w = n_layers * CELL + 4.0
        fig_h = n_games * CELL + 3.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(ent, cmap="YlOrRd_r", vmin=vmin, vmax=vmax, aspect="auto")

        mid = (vmin + vmax) / 2
        for gi in range(n_games):
            for li in range(n_layers):
                if np.isfinite(ent[gi, li]):
                    c = "white" if ent[gi, li] < mid else "#111827"
                    ax.text(
                        li,
                        gi,
                        f"{ent[gi, li]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=FS_CELL,
                        color=c,
                    )

        _draw_cluster_separators(ax, games_present, "row")
        _draw_ood_separator(ax, games_present, "row")

        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(
            [f"L{l}" for l in layers],
            fontsize=FS_TICK,
            fontweight="bold",
        )
        ax.set_yticks(np.arange(n_games))
        ax.set_yticklabels(
            [_game_label(g) for g in games_present],
            fontsize=FS_TICK,
        )

        ax.set_xlabel("Layer  →", fontsize=FS_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("←  Game", fontsize=FS_LABEL, fontweight="bold")

        ax.set_title(
            f"Routing entropy per game × layer  —  {_model_label(model_name)}\n"
            "Low entropy → one expert wins confidently · High → spread across experts",
            fontsize=FS_TITLE,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(
            "Avg routing entropy\n(low = concentrated, high = spread)",
            fontsize=FS_TICK,
        )

        fig.tight_layout()
        _save(fig, out_dir / "moe_entropy_heatmap.png")


# ── plot 4: expert preference profiles ───────────────────────────────────────

def plot_expert_profiles(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    from matplotlib.patches import Patch

    for model_name, df in dfs.items():
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        experts = sorted(df_clem["expert_id"].dropna().unique().astype(int))
        games_present = _game_order(df_clem["game"].unique().tolist())

        n_exp = len(experts)
        uniform = 1.0 / n_exp

        ncols = min(n_exp, 4)
        nrows = math.ceil(n_exp / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.5, nrows * 3.2),
            sharey=False,
            squeeze=False,
        )

        axes_flat = axes.flatten()

        agg = (
            df_clem.groupby(["game", "expert_id"])["top1_proportion"]
            .mean()
            .reset_index()
        )

        bar_palette = [
            CLUSTER_COLORS.get(GAME_GROUP.get(g, "ood_negotiation"), "#9ca3af")
            for g in games_present
        ]

        for ei, eid in enumerate(experts):
            ax = axes_flat[ei]

            vals = []
            for g in games_present:
                v = agg.loc[
                    (agg["game"] == g) &
                    (agg["expert_id"] == eid),
                    "top1_proportion",
                ].mean()

                vals.append(v - uniform if pd.notna(v) else -uniform)

            x = np.arange(len(games_present))

            ax.bar(
                x,
                vals,
                color=bar_palette,
                alpha=0.82,
                width=0.7,
                edgecolor="none",
            )

            ax.axhline(0, color=SEP_C, linewidth=1.0)
            _draw_cluster_separators(ax, games_present, "col")

            ood_present = [g for g in OOD_GAMES if g in games_present]
            if ood_present:
                ood_x = games_present.index(ood_present[0])
                ax.axvline(
                    ood_x - 0.5,
                    color=OOD_C,
                    linewidth=1.5,
                    linestyle=(0, (4, 3)),
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [_game_label(g) for g in games_present],
                rotation=40,
                ha="right",
                fontsize=FS_TICK - 2,
            )

            ax.set_title(
                f"Expert {int(eid)}",
                fontsize=FS_TITLE,
                fontweight="bold",
                color="#111827",
            )

            ax.set_ylabel("Δ vs uniform", fontsize=FS_TICK)
            ax.yaxis.grid(True, linestyle=":", alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n_exp, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        legend_patches = [
            Patch(
                facecolor=c,
                alpha=0.82,
                label=k.replace("explorationnavigation", "explor./nav.")
                 .replace("ood_negotiation", "OOD"),
            )
            for k, c in CLUSTER_COLORS.items()
        ]

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(CLUSTER_COLORS),
            fontsize=FS_TICK,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

        fig.suptitle(
            f"Expert routing preference profiles  —  {_model_label(model_name)}\n"
            "Bars show usage above/below uniform baseline · Colour = game cluster",
            fontsize=FS_TITLE,
            fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        safe = model_name.replace("/", "_")
        _save(fig, out_dir / f"moe_expert_profiles_{safe}.png")


# ── plot 5: layer routing balance ────────────────────────────────────────────

def plot_layer_balance(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)

    if n == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, dfs.items()):
        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        agg = (
            df_clem.groupby(["layer", "expert_id"])["top1_proportion"]
            .mean()
            .reset_index()
        )

        layers = sorted(agg["layer"].unique().astype(int))
        experts = sorted(agg["expert_id"].unique().astype(int))

        x = np.arange(len(layers))
        bottom = np.zeros(len(layers))

        for ei, eid in enumerate(experts):
            vals = []
            for l in layers:
                v = agg.loc[
                    (agg["layer"] == l) &
                    (agg["expert_id"] == eid),
                    "top1_proportion",
                ].mean()

                vals.append(v if pd.notna(v) else 0.0)

            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=f"Expert {int(eid)}",
                color=EXPERT_COLORS_POSTER[eid % len(EXPERT_COLORS_POSTER)],
                alpha=1,
                edgecolor="none",
            )

            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=FS_TICK)

        ax.set_title(_model_label(model_name), fontsize=FS_TITLE, fontweight="bold")
        ax.set_ylabel(
            "Avg top-1 proportion" if ax == axes[0] else "",
            fontsize=FS_TICK,
        )

        ax.axhline(
            1 / len(experts),
            color=SEP_C,
            linewidth=1.2,
            linestyle="--",
            label="uniform",
        )

        ax.legend(fontsize=FS_TICK - 1, frameon=False, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Expert usage proportion per layer · Dashed = uniform baseline",
        fontsize=FS_TITLE,
        fontweight="bold",
    )

    fig.tight_layout()
    _save(fig, out_dir / "moe_layer_routing_balance.png")


# ── plot 6: per-game expert preference stacked bars ──────────────────────────

def plot_game_expert_preference(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), squeeze=False)

    for row_idx, (model_name, df) in enumerate(dfs.items()):
        ax = axes[row_idx, 0]

        df_clem = df[df["game"].isin(CLEM_GAMES)].copy()

        agg = (
            df_clem.groupby(["game", "expert_id"])["top1_proportion"]
            .mean()
            .reset_index()
        )

        games_present = _game_order(df_clem["game"].unique().tolist())
        experts = sorted(agg["expert_id"].unique().astype(int))

        x = np.arange(len(games_present))
        bottom = np.zeros(len(games_present))

        for ei, eid in enumerate(experts):
            vals = []
            for g in games_present:
                v = agg.loc[
                    (agg["game"] == g) &
                    (agg["expert_id"] == eid),
                    "top1_proportion",
                ].mean()

                vals.append(v if pd.notna(v) else 0.0)

            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=f"Expert {int(eid)}",
                color=EXPERT_COLORS_POSTER[eid % len(EXPERT_COLORS_POSTER)],
                alpha=1,
                edgecolor="none",
            )

            bottom += np.array(vals)

        _draw_cluster_separators(ax, games_present, "col")
        _draw_ood_separator(ax, games_present, "col")

        ax.axhline(
            1 / len(experts),
            color=SEP_C,
            linewidth=1.2,
            linestyle="--",
            label="uniform",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_game_label(g) for g in games_present],
            rotation=35,
            ha="right",
            fontsize=FS_TICK,
        )

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Avg top-1 proportion", fontsize=FS_TICK)
        ax.set_title(
            _model_label(model_name),
            fontsize=FS_TITLE,
            fontweight="bold",
            loc="left",
        )

        ax.legend(
            fontsize=FS_TICK - 1,
            frameon=False,
            ncol=len(experts) + 1,
            loc="upper right",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Per-game expert routing preference (avg across layers) · Dashed = uniform",
        fontsize=FS_TITLE,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir / "moe_game_expert_preference.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names, i.e. subdirs under --routing-root",
    )

    parser.add_argument(
        "--routing-root",
        type=Path,
        default=Path("analysis_outputs"),
        help="Root dir containing model subdirs with moe-routing/",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/moe_routing_analysis"),
    )

    args = parser.parse_args()

    dfs: Dict[str, pd.DataFrame] = {}

    for model_name in args.models:
        df = load_model_data(args.routing_root, model_name)
        if df is not None:
            dfs[model_name] = df

    if not dfs:
        print("No data loaded. Check --routing-root and --models.")
        return

    print(f"Loaded {len(dfs)} model(s): {list(dfs.keys())}")

    for name, df in dfs.items():
        games = df["game"].nunique()
        layers = sorted(df["layer"].dropna().unique().astype(int).tolist())
        experts = sorted(df["expert_id"].dropna().unique().astype(int).tolist())
        print(f"  {name}: {games} games, layers {layers}, experts {experts}")

    # Poster-ready plot.
    plot_poster_heatmap_only(dfs, args.out_dir)

    # Diagnostic plots.
    plot_game_expert_heatmap(dfs, args.out_dir)
    plot_dominant_expert_heatmap(dfs, args.out_dir)
    plot_entropy_heatmap(dfs, args.out_dir)
    plot_expert_profiles(dfs, args.out_dir)
    plot_layer_balance(dfs, args.out_dir)
    plot_game_expert_preference(dfs, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()