import pandas as pd

# Approximate modified durations (in years) for each tenor
DURATIONS = {
    "2Y": 1.9,
    "5Y": 4.5,
    "10Y": 8.5,
    "30Y": 18.0,
}


class ReturnCalculator:

    @staticmethod
    def synthetic(df: pd.DataFrame, yield_col: str, horizon: int) -> pd.Series:
        """
        Approximate bond return using duration approximation:
            return ≈ -duration × Δyield

        yield_col must be one of: 2Y, 5Y, 10Y, 30Y
        horizon is in trading days.
        Yields are assumed to be in percent (e.g. 4.08 not 0.0408).
        """
        duration = DURATIONS[yield_col]
        delta_yield = df[yield_col].diff(horizon) / 100
        return -duration * delta_yield

    @staticmethod
    def spread_return(df: pd.DataFrame, spread_col: str, horizon: int) -> pd.Series:
        """
        Forward change in a yield spread over `horizon` trading days.

        This is the correct P&L measure for a duration-neutral curve trade
        (e.g. 2s10s steepener: long 10Y / short 2Y, DV01-neutral).

        Return units match the spread column units:
            - If spread stored in % (e.g. 2s10s = 0.62): result in percentage points
            - Positive return = curve steepened = steepener trade profited
            - Negative return = curve flattened = steepener trade lost

        Caller must NOT apply .shift(-horizon) — this method returns the raw
        diff(horizon) consistent with ReturnCalculator.synthetic() convention.
        Apply .shift(-horizon) when constructing forward-looking returns.
        """
        return df[spread_col].diff(horizon)

    @staticmethod
    def from_prices(df: pd.DataFrame, price_col: str, horizon: int) -> pd.Series:
        """
        Actual forward return from a price series (e.g. Bloomberg bond prices, ETF).
        Used when real price data is available.
        """
        return df[price_col].shift(-horizon) / df[price_col] - 1
