import pandas as pd
import numpy as np
from src.data.return_calculator import ReturnCalculator


HORIZONS = [5, 10, 21]


def run_backtest(df: pd.DataFrame, yield_col: str, price_col: str = None) -> dict:
    """
    Runs a backtest correlating 5-day yield changes against forward returns.

    If price_col is provided (e.g. Bloomberg prices), uses actual price returns.
    Otherwise falls back to synthetic duration-approximated returns.
    """
    df = df.copy()
    df["yield_change_5d"] = df[yield_col].diff(5)

    results = {}

    for h in HORIZONS:
        if price_col and price_col in df.columns:
            fwd_returns = ReturnCalculator.from_prices(df, price_col, h)
        else:
            fwd_returns = ReturnCalculator.synthetic(df, yield_col, h)

        corr = df["yield_change_5d"].corr(fwd_returns)
        results[f"{h}d_corr"] = round(corr, 4)

    return results
