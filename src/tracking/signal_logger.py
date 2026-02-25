import os
import pandas as pd
import numpy as np

from src.signals.signal_engine import SignalEngine

DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "signal_log.csv"
)

# All columns written to the log
LOG_COLUMNS = [
    "date",
    # Score
    "composite_score", "gated_score", "quintile", "direction",
    # Macro context
    "macro_regime", "macro_scalar", "curve_regime",
    # 6 component values (what was driving the score)
    "c_regime", "c_zscore", "c_voladj", "c_funding", "c_vix", "c_frontend",
    # Market levels at signal time (needed to compute forward return later)
    "2s10s", "5s30s", "10Y", "2Y",
    # Key inputs
    "2s10s_z20", "2s10s_voladj", "VIX", "VIX_z20", "1y2y_z20", "2Y_vs_FF",
]


def _compute_quintile(score: float, all_scores: pd.Series) -> str:
    """Assign today's score to a quintile based on full history thresholds."""
    cuts = all_scores.quantile([0.2, 0.4, 0.6, 0.8])
    if score <= cuts[0.2]:
        return "Q1"
    elif score <= cuts[0.4]:
        return "Q2"
    elif score <= cuts[0.6]:
        return "Q3"
    elif score <= cuts[0.8]:
        return "Q4"
    else:
        return "Q5"


def _direction(score: float) -> str:
    if score >= 1.0:
        return "steepener"
    elif score <= -1.0:
        return "flattener"
    return "neutral"


def _component_values(last: pd.Series) -> dict:
    """Recompute the 6 component contributions for the snapshot row."""
    def _clip(val, scale=2.0):
        return float(np.clip(val / scale, -1, 1)) if pd.notna(val) else 0.0

    regime_map = {
        "Bear Steepener": -1, "Bull Steepener": -1,
        "Bear Flattener": +1, "Bull Flattener": +1,
        "Unknown": 0,
    }
    return {
        "c_regime":   float(regime_map.get(last.get("regime", "Unknown"), 0)),
        "c_zscore":   _clip(-last.get("2s10s_z20", 0)),
        "c_voladj":   _clip(-last.get("2s10s_voladj", 0)),
        "c_funding":  _clip(-last.get("SOFR_TBill_spread_z20", 0)),
        "c_vix":      _clip(last.get("VIX_z20", 0)),
        "c_frontend": _clip(last.get("1y2y_z20", 0)),
    }


def log_signal(df: pd.DataFrame, log_path: str = DEFAULT_LOG_PATH) -> dict:
    """
    Append today's signal to the log file (one row per trading day).

    Skips silently if today's date is already in the log — safe to call
    multiple times in the same day.

    Parameters
    ----------
    df       : enriched DataFrame (output of FeatureEngineer.add_curve_features)
    log_path : path to the CSV log file (created if it doesn't exist)

    Returns
    -------
    dict — the row that was logged (or the existing row if already logged today)
    """
    df = df.copy()
    df["composite_score"] = SignalEngine.composite_score(df)
    df["gated_score"]     = SignalEngine.gated_score(df)

    last     = df.iloc[-1]
    today    = df.index[-1].date()
    score    = float(last["composite_score"])
    gated    = float(last["gated_score"])
    quintile = _compute_quintile(score, df["composite_score"])
    comps    = _component_values(last)

    row = {
        "date":           str(today),
        "composite_score": round(score, 4),
        "gated_score":     round(gated, 4),
        "quintile":        quintile,
        "direction":       _direction(score),
        "macro_regime":    last.get("macro_regime", "N/A"),
        "macro_scalar":    round(float(last.get("macro_scalar", 1.0)), 2),
        "curve_regime":    last.get("regime", "Unknown"),
        **{k: round(v, 4) for k, v in comps.items()},
        "2s10s":           round(float(last.get("2s10s", np.nan)), 6),
        "5s30s":           round(float(last.get("5s30s", np.nan)), 6),
        "10Y":             round(float(last.get("10Y", np.nan)), 4),
        "2Y":              round(float(last.get("2Y", np.nan)), 4),
        "2s10s_z20":       round(float(last.get("2s10s_z20", np.nan)), 4),
        "2s10s_voladj":    round(float(last.get("2s10s_voladj", np.nan)), 4),
        "VIX":             round(float(last.get("VIX", np.nan)), 2),
        "VIX_z20":         round(float(last.get("VIX_z20", np.nan)), 4),
        "1y2y_z20":        round(float(last.get("1y2y_z20", np.nan)), 4),
        "2Y_vs_FF":        round(float(last.get("2Y_vs_FF", np.nan)), 4),
    }

    # Load existing log or create empty DataFrame
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, dtype={"date": str})
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        existing = pd.DataFrame(columns=LOG_COLUMNS)

    # Skip if already logged today
    if str(today) in existing["date"].values:
        print(f"  Already logged for {today} — skipping.")
        return existing[existing["date"] == str(today)].iloc[0].to_dict()

    new_row = pd.DataFrame([row], columns=LOG_COLUMNS)
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(log_path, index=False)

    print(f"  Logged signal for {today}:")
    print(f"    Score: {score:+.2f}  |  Quintile: {quintile}  |  Direction: {_direction(score)}")
    print(f"    Macro: {row['macro_regime']}  |  Curve: {row['curve_regime']}")
    print(f"    2s10s: {row['2s10s']*100:+.1f} bps  |  VIX: {row['VIX']:.1f}  |  2Y_vs_FF: {row['2Y_vs_FF']:+.3f}%")

    return row
