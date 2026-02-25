import pandas as pd
import numpy as np

from src.signals.signal_engine import SignalEngine


def run_walk_forward(
    df: pd.DataFrame,
    horizon: int = 5,
    spread_col: str = None,
    yield_col: str = None,
    n_quantiles: int = 5,
    min_obs: int = 60,
    use_gated: bool = False,
) -> dict:
    """
    Annual walk-forward evaluation of the composite score.

    For each calendar year in the dataset:
      - Compute composite score and forward returns at `horizon` days
      - Bucket into quintiles within that year (within-year thresholds)
      - Report Q1 / Q5 performance and whether the full pattern is monotonic

    Note: the composite score formula was determined on the full dataset, so
    this is not a fully clean out-of-sample test of the formula itself.
    What it does validate is year-by-year consistency — i.e., in how many
    individual years did the signal actually work?

    Parameters
    ----------
    df          : enriched DataFrame (output of FeatureEngineer.add_curve_features)
    horizon     : forward return horizon in trading days (default 5)
    spread_col  : spread column for curve P&L (e.g. '2s10s')
    yield_col   : tenor for bond return calc (e.g. '10Y')
    n_quantiles : number of buckets (default 5)
    min_obs     : minimum observations per year to include (default 60)
    use_gated   : if True, use gated_score (macro-scaled) instead of composite_score

    Returns
    -------
    dict with two keys:
        'annual'  : pd.DataFrame — per-year metrics, indexed by year
        'summary' : pd.Series   — aggregate statistics across all years
    """
    df = df.copy()
    df["_score"] = SignalEngine.gated_score(df) if use_gated else SignalEngine.composite_score(df)
    df["_fwd"]   = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
    df = df.dropna(subset=["_score", "_fwd"])
    df["_year"]  = df.index.year

    annualize = np.sqrt(252 / horizon)
    q_labels  = [f"Q{i}" for i in range(1, n_quantiles + 1)]

    def _stats(series: pd.Series, sign: int = 1):
        """Return (mean, hit_rate, sharpe) for a position taken with `sign`."""
        s = series * sign
        mean_r = s.mean()
        hit    = (s > 0).mean()
        sharpe = mean_r / s.std() * annualize if s.std() > 0 else np.nan
        return round(mean_r, 4), round(hit, 4), round(sharpe, 4)

    rows = []

    for year, grp in df.groupby("_year"):
        if len(grp) < min_obs:
            continue

        grp = grp.copy()
        try:
            grp["_q"] = pd.qcut(
                grp["_score"],
                q=n_quantiles,
                labels=q_labels,
                duplicates="drop",
            )
        except ValueError:
            continue

        q1_ret = grp.loc[grp["_q"] == "Q1", "_fwd"]
        q5_ret = grp.loc[grp["_q"] == "Q5", "_fwd"]

        if len(q1_ret) < 5 or len(q5_ret) < 5:
            continue

        # Q1: trade is a FLATTENER (sign = -1), Q5: STEEPENER (sign = +1)
        q1_mean, q1_hit, q1_sharpe = _stats(q1_ret, sign=-1)
        q5_mean, q5_hit, q5_sharpe = _stats(q5_ret, sign=+1)

        # Full quintile means — check if pattern is monotonically increasing
        q_means = (
            grp.groupby("_q", observed=True)["_fwd"]
            .mean()
            .reindex(q_labels)
        )
        monotonic = bool(q_means.is_monotonic_increasing)

        rows.append({
            "year":             year,
            "q1_mean_ret":      round(q1_ret.mean(), 4),
            "q1_sharpe":        q1_sharpe,
            "q1_hit_rate":      q1_hit,
            "q5_mean_ret":      round(q5_ret.mean(), 4),
            "q5_sharpe":        q5_sharpe,
            "q5_hit_rate":      q5_hit,
            "q5_q1_spread":     round(q5_ret.mean() - q1_ret.mean(), 4),
            "monotonic":        monotonic,
            "n_obs":            len(grp),
        })

    if not rows:
        raise ValueError("No years met the minimum observation threshold.")

    annual = pd.DataFrame(rows).set_index("year")

    n        = len(annual)
    summary  = pd.Series({
        "years_evaluated":        n,
        "pct_q5_positive_return": round((annual["q5_mean_ret"] > 0).sum() / n, 3),
        "pct_q5_gt_q1":           round((annual["q5_q1_spread"] > 0).sum() / n, 3),
        "pct_monotonic":          round(annual["monotonic"].sum() / n, 3),
        "mean_q5_sharpe":         round(annual["q5_sharpe"].mean(), 3),
        "mean_q1_sharpe":         round(annual["q1_sharpe"].mean(), 3),
        "mean_q5_q1_spread":      round(annual["q5_q1_spread"].mean(), 4),
        "worst_q5_sharpe":        round(annual["q5_sharpe"].min(), 3),
        "worst_q5_year":          int(annual["q5_sharpe"].idxmin()),
        "best_q5_sharpe":         round(annual["q5_sharpe"].max(), 3),
        "best_q5_year":           int(annual["q5_sharpe"].idxmax()),
    })

    return {"annual": annual, "summary": summary}


def compare_walk_forward(
    df: pd.DataFrame,
    horizon: int = 5,
    spread_col: str = None,
    yield_col: str = None,
) -> pd.DataFrame:
    """
    Run walk-forward for three variants and return a side-by-side comparison:
      - Raw:     composite score, all days traded
      - Filtered: composite score, Hiking/Cutting Fast days excluded (no trade)
      - Gated:   gated_score (scale-adjusted), all days traded

    The filtered variant is the most actionable: it simply refuses to trade
    when the macro environment is strongly trending against mean-reversion.
    """
    # Build filtered DataFrame (exclude days with aggressive policy pricing)
    neutral_regimes = ["Neutral", "Hike Priced", "Cut Priced"]
    df_filtered = df[df["macro_regime"].isin(neutral_regimes)].copy()

    raw      = run_walk_forward(df,          horizon=horizon, spread_col=spread_col, yield_col=yield_col, use_gated=False)
    filtered = run_walk_forward(df_filtered, horizon=horizon, spread_col=spread_col, yield_col=yield_col, use_gated=False)
    gated    = run_walk_forward(df,          horizon=horizon, spread_col=spread_col, yield_col=yield_col, use_gated=True)

    # Show dominant non-neutral regime + what % of days were gated
    def _regime_label(group):
        vc = group.value_counts()
        pct_non_neutral = (group.isin(["Hike Priced+", "Cut Priced+"])).mean()
        if pct_non_neutral > 0.05:
            dominant = vc[vc.index.isin(["Hike Priced+", "Cut Priced+"])].idxmax() if any(
                v in ["Hike Priced+", "Cut Priced+"] for v in vc.index) else "Neutral"
            return f"{dominant} ({pct_non_neutral:.0%})"
        return "Neutral"

    cmp = pd.DataFrame({
        "raw_q5_sharpe":      raw["annual"]["q5_sharpe"],
        "filtered_q5_sharpe": filtered["annual"]["q5_sharpe"],
        "gated_q5_sharpe":    gated["annual"]["q5_sharpe"],
        "raw_q1_sharpe":      raw["annual"]["q1_sharpe"],
        "filtered_q1_sharpe": filtered["annual"]["q1_sharpe"],
        "macro_env":          df.groupby(df.index.year)["macro_regime"].apply(_regime_label),
        "n_obs_raw":          raw["annual"]["n_obs"],
        "n_obs_filtered":     filtered["annual"]["n_obs"],
    })

    print("=== Summary: Raw vs Filtered (no Hiking/Cutting Fast) vs Gated (scaled) ===")
    for label, res in [("Raw", raw), ("Filtered", filtered), ("Gated", gated)]:
        s = res["summary"]
        print(
            f"  {label:10s} | mean Q5 Sharpe: {s['mean_q5_sharpe']:+.2f} "
            f"| worst: {s['worst_q5_sharpe']:+.2f} ({int(s['worst_q5_year'])}) "
            f"| Q5>Q1: {s['pct_q5_gt_q1']*100:.0f}%"
        )

    return cmp
