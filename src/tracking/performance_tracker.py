import os
import pandas as pd
import numpy as np

from src.tracking.signal_logger import DEFAULT_LOG_PATH


def evaluate_signals(
    df: pd.DataFrame,
    log_path: str = DEFAULT_LOG_PATH,
    horizon: int = 5,
) -> dict:
    """
    Load the signal log and resolve any signals where `horizon` trading
    days have passed. Computes realized P&L and hit/miss for each entry.

    A signal is "resolved" when the current market data contains a row
    at least `horizon` trading days after the signal date.

    Parameters
    ----------
    df       : enriched DataFrame (output of FeatureEngineer.add_curve_features)
               Must contain 2s10s column.
    log_path : path to the signal log CSV
    horizon  : forward return horizon in trading days (default 5)

    Returns
    -------
    dict with keys:
        "resolved"   : pd.DataFrame — all resolved signals with P&L columns
        "pending"    : pd.DataFrame — signals not yet resolvable
        "summary"    : pd.Series   — overall performance stats
        "by_quintile": pd.DataFrame — performance broken down by Q1–Q5
    """
    if not os.path.exists(log_path):
        print("  No signal log found. Run 'python main.py log' first.")
        return {}

    log = pd.read_csv(log_path, parse_dates=["date"])
    log = log.sort_values("date").reset_index(drop=True)

    if log.empty:
        print("  Signal log is empty.")
        return {}

    # Index df by date for fast lookup
    prices = df[["2s10s"]].copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()

    resolved_rows = []
    pending_rows  = []

    for _, row in log.iterrows():
        signal_date  = pd.Timestamp(row["date"]).normalize()
        entry_2s10s  = row["2s10s"]
        direction    = row["direction"]  # "steepener", "flattener", "neutral"

        # Find the trading days available after signal_date
        future_dates = prices.index[prices.index > signal_date]

        if len(future_dates) < horizon:
            pending_rows.append(row.to_dict())
            continue

        exit_date      = future_dates[horizon - 1]
        exit_2s10s     = float(prices.loc[exit_date, "2s10s"])
        realized_bps   = (exit_2s10s - entry_2s10s) * 100  # bps

        # Aligned return: steepener profits from widening (+), flattener from narrowing (-)
        sign = {"steepener": +1, "flattener": -1, "neutral": 0}.get(direction, 0)
        aligned_bps = realized_bps * sign

        r = row.to_dict()
        r.update({
            "exit_date":     exit_date.date(),
            "exit_2s10s":    round(exit_2s10s, 6),
            "realized_bps":  round(realized_bps, 2),
            "aligned_bps":   round(aligned_bps, 2),
            "hit":           1 if aligned_bps > 0 else (0 if aligned_bps < 0 else np.nan),
        })
        resolved_rows.append(r)

    resolved = pd.DataFrame(resolved_rows) if resolved_rows else pd.DataFrame()
    pending  = pd.DataFrame(pending_rows)  if pending_rows  else pd.DataFrame()

    if resolved.empty:
        print(f"  No signals resolved yet (need {horizon}+ trading days after each signal).")
        return {"resolved": resolved, "pending": pending, "summary": None, "by_quintile": None}

    # ── Summary stats ─────────────────────────────────────────────────────────
    # Only include directional signals (steepener / flattener) in performance calcs
    directional = resolved[resolved["direction"] != "neutral"].copy()

    def _sharpe(s: pd.Series) -> float:
        return s.mean() / s.std() * np.sqrt(252 / horizon) if len(s) > 1 and s.std() > 0 else np.nan

    summary = pd.Series({
        "total_signals":   len(log),
        "resolved":        len(resolved),
        "pending":         len(pending),
        "directional":     len(directional),
        "hit_rate":        round(directional["hit"].mean(), 3) if not directional.empty else np.nan,
        "mean_aligned_bps":round(directional["aligned_bps"].mean(), 2) if not directional.empty else np.nan,
        "median_aligned_bps": round(directional["aligned_bps"].median(), 2) if not directional.empty else np.nan,
        "sharpe_aligned":  round(_sharpe(directional["aligned_bps"]), 3) if not directional.empty else np.nan,
        "total_pnl_bps":   round(directional["aligned_bps"].sum(), 2) if not directional.empty else np.nan,
        "best_trade_bps":  round(directional["aligned_bps"].max(), 2) if not directional.empty else np.nan,
        "worst_trade_bps": round(directional["aligned_bps"].min(), 2) if not directional.empty else np.nan,
    })

    # ── By-quintile breakdown ─────────────────────────────────────────────────
    if not directional.empty:
        by_q = directional.groupby("quintile")["aligned_bps"].agg(
            count="count",
            hit_rate=lambda x: (x > 0).mean(),
            mean_bps="mean",
            sharpe=_sharpe,
        ).round(3)
    else:
        by_q = pd.DataFrame()

    return {
        "resolved":    resolved,
        "pending":     pending,
        "summary":     summary,
        "by_quintile": by_q,
    }


def print_tracking_report(
    df: pd.DataFrame,
    log_path: str = DEFAULT_LOG_PATH,
    horizon: int = 5,
) -> None:
    """
    Print a formatted live performance tracking report to stdout.
    """
    result = evaluate_signals(df, log_path=log_path, horizon=horizon)
    if not result:
        return

    SEP  = "=" * 62
    THIN = "─" * 62
    resolved  = result["resolved"]
    pending   = result["pending"]
    summary   = result["summary"]
    by_q      = result["by_quintile"]

    print(SEP)
    print("  LIVE SIGNAL TRACKING REPORT")
    print(f"  Evaluation horizon: {horizon} trading days")
    print(SEP)

    if summary is None:
        print(f"\n  No resolved signals yet. {len(pending)} signal(s) pending resolution.")
        return

    print(f"\n  Signals logged:    {int(summary['total_signals'])}")
    print(f"  Resolved:          {int(summary['resolved'])}")
    print(f"  Pending:           {int(summary['pending'])}  (horizon not yet elapsed)")
    print(f"  Directional:       {int(summary['directional'])}  (excludes neutral signals)")

    print(f"\n{THIN}")
    print("  PERFORMANCE SUMMARY  (directional signals only)")
    print(THIN)
    print(f"  Hit rate:          {summary['hit_rate']:.1%}")
    print(f"  Mean return:       {summary['mean_aligned_bps']:+.2f} bps / trade")
    print(f"  Median return:     {summary['median_aligned_bps']:+.2f} bps / trade")
    print(f"  Sharpe (ann.):     {summary['sharpe_aligned']:.2f}")
    print(f"  Cumulative P&L:    {summary['total_pnl_bps']:+.1f} bps")
    print(f"  Best trade:        {summary['best_trade_bps']:+.1f} bps")
    print(f"  Worst trade:       {summary['worst_trade_bps']:+.1f} bps")

    if not by_q.empty:
        print(f"\n{THIN}")
        print("  BY QUINTILE")
        print(THIN)
        print(f"  {'Quintile':<10} {'Count':>6}  {'Hit Rate':>9}  {'Mean bps':>9}  {'Sharpe':>8}")
        for q, row in by_q.iterrows():
            print(
                f"  {q:<10} {int(row['count']):>6}  "
                f"{row['hit_rate']:>9.1%}  "
                f"{row['mean_bps']:>+9.2f}  "
                f"{row['sharpe']:>8.2f}"
            )

    if not resolved.empty:
        print(f"\n{THIN}")
        print("  RESOLVED SIGNALS  (most recent 10)")
        print(THIN)
        print(f"  {'Entry':>12}  {'Exit':>12}  {'Q':>3}  {'Dir':>10}  {'Realized':>10}  {'Aligned':>9}  {'Hit':>4}")
        for _, r in resolved.tail(10).iterrows():
            hit_str = "✓" if r["hit"] == 1 else ("✗" if r["hit"] == 0 else "─")
            print(
                f"  {str(r['date'])[:10]:>12}  {str(r['exit_date']):>12}  "
                f"{r['quintile']:>3}  {r['direction']:>10}  "
                f"{r['realized_bps']:>+10.2f}  {r['aligned_bps']:>+9.2f}  {hit_str:>4}"
            )

    if not pending.empty:
        print(f"\n{THIN}")
        print(f"  PENDING  ({len(pending)} signal(s) — horizon not yet elapsed)")
        print(THIN)
        for _, r in pending.iterrows():
            print(f"  {str(r['date'])[:10]}  Q={r['quintile']}  {r['direction']}  score={r['composite_score']:+.2f}")

    print(f"\n{SEP}")
