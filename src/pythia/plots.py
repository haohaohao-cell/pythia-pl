"""All figures shipped with the repository.

Charts use English labels on purpose: they are the part of the project most
likely to be read outside a Chinese-speaking audience, and font fallback for CJK
in matplotlib is a trap that only shows up on someone else's machine.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import REPORTS_DIR  # noqa: E402
from .eval.calibration import reliability_curve  # noqa: E402
from .eval.metrics import log_loss  # noqa: E402
from .models.base import outcome_index  # noqa: E402
from .teams import en_name  # noqa: E402

COLORS = {
    "market": "#111827",
    "blend": "#2563eb",
    "elo_logit": "#0891b2",
    "gbdt": "#7c3aed",
    "dixon_coles": "#ea580c",
    "base_rates": "#9ca3af",
}
LABELS = {
    "market": "Closing line",
    "blend": "Blend",
    "elo_logit": "Elo + softmax",
    "gbdt": "Gradient boosting",
    "dixon_coles": "Dixon-Coles",
    "base_rates": "Base rates",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def _save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or REPORTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def leaderboard_bars(lb: pd.DataFrame, outdir: Path | None = None) -> Path:
    _style()
    df = lb.sort_values("log_loss", ascending=False)  # best model ends up on top
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [COLORS.get(m, "#94a3b8") for m in df["model"]]
    bars = ax.barh([LABELS.get(m, m) for m in df["model"]], df["log_loss"], color=colors, height=0.62)

    market = df.loc[df["model"] == "market", "log_loss"]
    if not market.empty:
        x = float(market.iloc[0])
        ax.axvline(x, color=COLORS["market"], linestyle="--", linewidth=1.0)
        ax.annotate("closing line", xy=(x, len(df) - 0.55), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=COLORS["market"], ha="left", va="center")
    for bar, value in zip(bars, df["log_loss"]):
        ax.text(value + 0.0015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}",
                va="center", fontsize=8)
    ax.set_xlim(min(df["log_loss"]) - 0.02, max(df["log_loss"]) + 0.03)
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_title("2018-19 to 2026-27 walk-forward backtest")
    return _save(fig, "leaderboard.png", outdir)


def cumulative_logloss(preds: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Running mean log loss — reveals whether a model is stable or lucky."""
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))

    order = ["base_rates", "dixon_coles", "gbdt", "elo_logit", "blend", "market"]
    for model in order:
        block = preds.loc[preds["model"] == model].sort_values("date").reset_index(drop=True)
        if block.empty:
            continue
        y = outcome_index(block)
        p = block[["p_home", "p_draw", "p_away"]].to_numpy()
        losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))
        running = np.cumsum(losses) / np.arange(1, len(losses) + 1)
        ax.plot(np.arange(1, len(losses) + 1), running, label=LABELS.get(model, model),
                color=COLORS.get(model), linewidth=1.8 if model in ("market", "blend") else 1.1,
                alpha=1.0 if model in ("market", "blend", "elo_logit") else 0.75)

    ax.axhline(1.0, color="#d1d5db", linewidth=0.8, linestyle=":")
    ax.set_ylim(0.86, 1.12)
    ax.set_xlim(20, None)
    ax.set_xlabel("Matches (chronological)")
    ax.set_ylabel("Cumulative mean log loss")
    ax.set_title("Skill accumulates slowly and never overtakes the closing line")
    ax.legend(ncol=3, fontsize=7.5, loc="upper right")
    return _save(fig, "cumulative_logloss.png", outdir)


def calibration_plot(preds: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Reliability for home wins and draws: blend vs the closing line."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)

    specs = (
        ("p_home", axes[0], "P(home win)", "H", np.linspace(0.0, 1.0, 9), (0.0, 1.0)),
        ("p_draw", axes[1], "P(draw)", "D", np.linspace(0.05, 0.45, 6), (0.02, 0.5)),
    )
    for col, ax, title, code, edges, xlim in specs:
        ax.plot([0, 1], [0, 1], color="#9ca3af", linewidth=0.9, linestyle="--")
        for model in ("blend", "market"):
            block = preds.loc[preds["model"] == model]
            p = block[col].to_numpy()
            observed = (block["ftr"] == code).to_numpy(dtype=float)
            curve = reliability_curve(p, observed, edges=edges)
            ax.errorbar(curve["mean_pred"], curve["observed"], yerr=1.96 * curve["se"],
                        marker="o", markersize=4, capsize=2, linewidth=1.4,
                        color=COLORS[model], label=LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Forecast probability")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration on 3,044 out-of-sample matches", fontsize=10, y=1.02)
    return _save(fig, "calibration.png", outdir)


def _place_labels(fig, ax, points: np.ndarray, labels: list[str], fontsize: float = 6.4) -> None:
    """Greedy collision-aware label placement, most distinctive clubs first.

    Each label tries progressively farther offsets in all directions and is
    accepted once its bounding box clears every previously placed label and
    every data marker.  Far placements get a hairline connector.
    """
    from matplotlib.transforms import Bbox

    # Layout must be finalised before display coordinates mean anything;
    # without this the collision test silently passes for every candidate.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    display = ax.transData.transform(points)

    occupied: list[Bbox] = [
        Bbox.from_extents(x - 5, y - 5, x + 5, y + 5) for x, y in display
    ]
    offsets = [
        (r * np.cos(np.deg2rad(angle)), r * np.sin(np.deg2rad(angle)))
        for r in (7, 12, 18, 25, 34, 44)
        for angle in range(0, 360, 20)
    ]

    for i in np.argsort(-np.hypot(points[:, 0], points[:, 1])):
        x, y = points[i]
        placed = False
        for dx, dy in offsets:
            far = np.hypot(dx, dy) > 10
            text = ax.annotate(
                labels[i], (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color="#374151",
                arrowprops={"arrowstyle": "-", "color": "#9ca3af", "linewidth": 0.5} if far else None,
                annotation_clip=False,
            )
            box = text.get_window_extent(renderer)
            if all(not box.overlaps(other) for other in occupied):
                occupied.append(box)
                placed = True
                break
            text.remove()
        if not placed:
            occupied.append(
                ax.annotate(labels[i], (x, y), xytext=(6, 6), textcoords="offset points",
                            fontsize=fontsize, color="#374151",
                            annotation_clip=False).get_window_extent(renderer)
            )


def strength_map(dc, outdir: Path | None = None) -> Path:
    """Dixon-Coles attack vs defence — the league on one axis pair."""
    _style()
    table = dc.strength_table()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.scatter(table["attack"], table["defence"], s=34, color="#2563eb", alpha=0.85, zorder=3)

    ax.set_xlabel("Attack (higher = scores more)")
    ax.set_ylabel("Defence (higher = concedes more)")
    ax.set_title("Dixon-Coles parameters, current squad strength")
    ax.margins(x=0.16, y=0.12)

    points = table[["attack", "defence"]].to_numpy()
    _place_labels(fig, ax, points, [en_name(t) for t in table["team"]])
    return _save(fig, "strength_map.png", outdir)


def score_heatmap(dc, home: str, away: str, outdir: Path | None = None) -> Path:
    """Joint scoreline distribution for one fixture."""
    _style()
    matrix = dc.score_matrix(home, away)[:5, :5]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(matrix, cmap="Blues", origin="lower", aspect="equal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if value > matrix.max() * 0.55 else "#1f2937")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel(f"{en_name(away)} goals")
    ax.set_ylabel(f"{en_name(home)} goals")
    ax.set_title(f"{en_name(home)} vs {en_name(away)} — scoreline %", fontsize=9.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, "score_heatmap.png", outdir)


def simulation_plot(sim: pd.DataFrame, current_points: pd.Series, outdir: Path | None = None) -> Path:
    """Distribution of final points from the Monte Carlo, against the live table."""
    _style()
    teams = sim.groupby("team")["points"].median().sort_values().index.tolist()
    data = [sim.loc[sim["team"] == t, "points"].to_numpy() for t in teams]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    positions = np.arange(len(teams))
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={"color": "#111827", "linewidth": 1.2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#2563eb")
        patch.set_linewidth(0.8)

    actual = np.array([float(current_points.get(t, np.nan)) for t in teams])
    ax.scatter(actual, positions, color="#dc2626", s=18, zorder=3, label="Points so far")

    ax.set_yticks(positions)
    ax.set_yticklabels([en_name(t) for t in teams], fontsize=7.5)
    ax.set_xlabel("Projected final points")
    ax.set_title("Remaining fixtures simulated 20,000 times", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "simulated_table.png", outdir)


def season_bars(preds: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Log loss per season for the five headline models."""
    _style()
    models = ["market", "elo_logit", "blend", "dixon_coles", "gbdt"]
    seasons = sorted(preds["season"].unique())
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.8 / len(models)
    x = np.arange(len(seasons))
    collected: list[float] = []
    for k, model in enumerate(models):
        vals = []
        for season in seasons:
            block = preds.loc[(preds["model"] == model) & (preds["season"] == season)]
            y = outcome_index(block)
            vals.append(log_loss(block[["p_home", "p_draw", "p_away"]].to_numpy(), y) if len(block) else np.nan)
        collected.extend(v for v in vals if not np.isnan(v))
        ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width=width,
               color=COLORS.get(model), label=LABELS.get(model, model))
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Log loss (axis truncated)")
    low, high = min(collected), max(collected)
    ax.set_ylim(low - 0.02, high + 0.03)
    ax.set_title("Season by season: the gap to the market is consistent, not lucky")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    return _save(fig, "season_logloss.png", outdir)


OUTCOME_COLORS = {"H": "#2563eb", "D": "#9ca3af", "A": "#f59e0b"}


def calibration_study(lb: pd.DataFrame, decomp: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Left: does a temperature fix anything? Right: reliability vs resolution."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.3))

    order = ["market", "blend", "blend+T"]
    labels = {"market": "Closing line", "blend": "Blend", "blend+T": "Blend + temperature"}
    present = [m for m in order if m in set(lb["model"])]
    # barh draws the first row at the bottom: ascending values put the best on top.
    present.sort(key=lambda m: -float(lb.loc[lb["model"] == m, "log_loss"].iloc[0]))
    values = [float(lb.loc[lb["model"] == m, "log_loss"].iloc[0]) for m in present]
    colors = ["#93c5fd" if m.endswith("+T") else COLORS.get(m, "#94a3b8") for m in present]
    bars = ax1.barh([labels[m] for m in present], values, color=colors[: len(present)], height=0.55)
    for bar, value in zip(bars, values):
        ax1.text(value + 0.0006, bar.get_y() + bar.get_height() / 2, f"{value:.4f}",
                 va="center", fontsize=8)
    ax1.set_xlim(min(values) - 0.004, max(values) + 0.004)
    ax1.set_xlabel("Log loss (lower is better)")
    ax1.set_title("A temperature does not close the gap", fontsize=9.5)

    for row in decomp.itertuples(index=False):
        colour = COLORS.get(str(row.model).replace("+T", ""), "#94a3b8")
        marker = "D" if "+T" in str(row.model) else "o"
        ax2.scatter(row.reliability, row.resolution, s=70, color=colour, marker=marker, zorder=3)
        ax2.annotate(labels.get(row.model, row.model), (row.reliability, row.resolution),
                     xytext=(6, -3), textcoords="offset points", fontsize=7.5)
    ax2.set_xlabel("Reliability (lower = better calibrated)")
    ax2.set_ylabel("Resolution (higher = more information)")
    ax2.set_title("Same calibration, less information", fontsize=9.5)
    ax2.margins(x=0.45, y=0.35)

    return _save(fig, "calibration_study.png", outdir)


def fixtures_chart(forecast: pd.DataFrame, outdir: Path | None = None) -> Path:
    """Model vs market for the next round, as paired stacked bars."""
    from matplotlib.patches import Patch

    _style()
    df = forecast.reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * n + 1.7))
    y_model = np.arange(n) * 2.0
    y_market = y_model + 1.0

    for offset, cols, alpha in (
        (y_model, ("p_home", "p_draw", "p_away"), 1.0),
        (y_market, ("mkt_home", "mkt_draw", "mkt_away"), 0.42),
    ):
        left = np.zeros(n)
        for code, col in zip(("H", "D", "A"), cols):
            values = df[col].to_numpy(float)
            values = np.where(np.isfinite(values), values, 0.0)
            ax.barh(offset, values, left=left, height=0.70,
                    color=OUTCOME_COLORS[code], alpha=alpha,
                    edgecolor="white", linewidth=0.5)
            left += values

    for i in range(n):
        ax.text(0.006, y_model[i], "M", va="center", ha="left", fontsize=5.6, color="white")
        ax.text(0.006, y_market[i], "P", va="center", ha="left", fontsize=5.6, color="#111827")

    labels = []
    for row in df.itertuples(index=False):
        mark = ""
        if row.status == "played":
            mark = " ✅" if row.hit else " ❌"
        labels.append(f"{en_name(row.home)} v {en_name(row.away)}{mark}")

    ax.set_yticks(y_model + 0.5)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.grid(axis="y", visible=False)
    ax.set_title("M = model · P = de-vigged opening price", fontsize=9.5)
    ax.legend(
        handles=[Patch(color=OUTCOME_COLORS[c], label=lab) for c, lab in (("H", "Home"), ("D", "Draw"), ("A", "Away"))],
        ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.07),
    )
    return _save(fig, "next_round.png", outdir)
