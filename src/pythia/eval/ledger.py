"""Season ledger: the running account of model vs closing line, week by week.

The project's claim is that the model loses to the market, and the only way to
keep that claim honest over a season is to publish the running score: hits, log
loss, and how the gap moves as the sample grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_ORDER = np.array(["H", "D", "A"])


def _hit_and_loss(block: pd.DataFrame) -> pd.DataFrame:
    probs = block[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    y = block["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    rows = np.arange(len(block))
    picked = OUTCOME_ORDER[probs.argmax(axis=1)]
    return block.assign(
        pick=picked,
        hit=picked == block["ftr"].to_numpy(),
        log_loss=-np.log(np.clip(probs[rows, y], 1e-15, 1.0)),
    )


def build_ledger(preds: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Per-week statistics for the blend and the closing line, with running totals."""
    df = preds.loc[preds["model"].isin(["blend", "market"])].copy()
    if season is not None:
        df = df.loc[df["season"] == season]
    if df.empty:
        return pd.DataFrame()

    df = _hit_and_loss(df)
    df["week"] = pd.to_datetime(df["date"]).dt.strftime("%G-W%V")

    rows = []
    for (week, model), block in df.groupby(["week", "model"]):
        rows.append({
            "week": week,
            "model": model,
            "n": len(block),
            "hits": int(block["hit"].sum()),
            "log_loss": float(block["log_loss"].mean()),
        })
    weekly = pd.DataFrame(rows)

    wide = weekly.pivot(index="week", columns="model", values=["hits", "log_loss", "n"]).sort_index()
    out = pd.DataFrame({
        "week": wide.index,
        "n": wide[("n", "blend")].to_numpy(),
        "model_hits": wide[("hits", "blend")].to_numpy(),
        "market_hits": wide[("hits", "market")].to_numpy(),
        "model_ll": wide[("log_loss", "blend")].to_numpy(),
        "market_ll": wide[("log_loss", "market")].to_numpy(),
    })
    out["model_hits_cum"] = out["model_hits"].cumsum()
    out["market_hits_cum"] = out["market_hits"].cumsum()
    out["n_cum"] = out["n"].cumsum()
    # Running log loss, weighted by matches rather than averaged over weeks.
    out["model_ll_cum"] = (out["model_ll"] * out["n"]).cumsum() / out["n_cum"]
    out["market_ll_cum"] = (out["market_ll"] * out["n"]).cumsum() / out["n_cum"]
    return out.reset_index(drop=True)


def ledger_markdown(ledger: pd.DataFrame, season: str, *, lang: str = "zh") -> str:
    """README block: the current season's running account."""
    if ledger.empty:
        return "_赛季尚未开始。_" if lang == "zh" else "_Season has not started yet._"

    last = ledger.iloc[-1]
    model_rate = last["model_hits_cum"] / last["n_cum"]
    market_rate = last["market_hits_cum"] / last["n_cum"]
    zh = lang == "zh"

    head = (f"**{season} 赛季记账：已结算 {int(last['n_cum'])} 场。** "
            f"模型命中 {int(last['model_hits_cum'])} 场（{model_rate:.1%}），"
            f"收盘价命中 {int(last['market_hits_cum'])} 场（{market_rate:.1%}）；"
            f"累计 log loss 模型 `{last['model_ll_cum']:.4f}`、市场 `{last['market_ll_cum']:.4f}`。"
            if zh else
            f"**{season} ledger: {int(last['n_cum'])} matches settled.** "
            f"Model {int(last['model_hits_cum'])} correct ({model_rate:.1%}), "
            f"closing line {int(last['market_hits_cum'])} ({market_rate:.1%}); "
            f"running log loss {last['model_ll_cum']:.4f} vs {last['market_ll_cum']:.4f}.")

    header = ("| 周 | 场次 | 模型命中 | 市场命中 | 模型累计 LL | 市场累计 LL |" if zh
              else "| Week | Matches | Model hits | Market hits | Model LL | Market LL |")
    lines = [head, "", header, "| --- | --- | --- | --- | --- | --- |"]
    for row in ledger.tail(8).itertuples(index=False):
        lines.append(
            f"| {row.week} | {int(row.n)} | {int(row.model_hits)} | {int(row.market_hits)} | "
            f"{row.model_ll_cum:.4f} | {row.market_ll_cum:.4f} |"
        )
    return "\n".join(lines)
