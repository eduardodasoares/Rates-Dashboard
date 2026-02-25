from fredapi import Fred
import pandas as pd
import os
from src.data.base_loader import BaseLoader


class DataLoader(BaseLoader):
    def __init__(self, fred_api_key: str):
        self.fred = Fred(api_key=fred_api_key)

    def load_treasury_data(self) -> pd.DataFrame:
        series_ids = {
            # ── Treasury yields (nominal) ──────────────────────────────────
            "3M":       "DGS3MO",
            "6M":       "DGS6MO",
            "1Y":       "DGS1",
            "2Y":       "DGS2",
            "5Y":       "DGS5",
            "10Y":      "DGS10",
            "30Y":      "DGS30",
            # ── Overnight / money-market rates ─────────────────────────────
            "SOFR":     "SOFR",
            "3M_TBill": "DTB3",
            # OBFR (Overnight Bank Funding Rate, NY Fed) is the best available
            # GC repo proxy on FRED. BGCR/TGCR are not published on FRED.
            # When Bloomberg is integrated, swap this for actual BGCR data.
            "GC_proxy": "OBFR",
            "Fed_Funds": "DFF",
            # ── TIPS / real yields — starts 2003 ───────────────────────────
            "5Y_TIPS":  "DFII5",
            "10Y_TIPS": "DFII10",
            # ── Breakeven inflation — starts 2003 ──────────────────────────
            "5Y_BE":    "T5YIE",
            "10Y_BE":   "T10YIE",
            # ── Risk sentiment ─────────────────────────────────────────────
            # VIX starts 1990; credit OAS starts 1997
            "VIX":      "VIXCLS",
            "IG_OAS":   "BAMLC0A0CM",
            "HY_OAS":   "BAMLH0A0HYM2",
        }

        data = pd.DataFrame()
        for name, sid in series_ids.items():
            data[name] = self.fred.get_series(sid)

        # Forward-fill short gaps (weekends, holidays, publication lags).
        # Newer series are allowed to have NaN before their start dates so
        # full yield history is preserved for backtesting.
        _ffill_cols = [
            "SOFR", "GC_proxy", "Fed_Funds",
            "5Y_TIPS", "10Y_TIPS",
            "5Y_BE", "10Y_BE",
            "VIX", "IG_OAS", "HY_OAS",
        ]
        for col in _ffill_cols:
            data[col] = data[col].ffill(limit=3)

        # Only core yield columns must be non-null.
        # SOFR is intentionally excluded: it only starts Apr 2018.
        # Pre-2018 rows have SOFR=NaN; FeatureEngineer falls back to Fed_Funds
        # for carry calculations, giving ~36yr backtest vs 8yr.
        data = data.dropna(subset=["2Y", "5Y", "10Y", "30Y", "3M_TBill"])

        # ── Curve spreads ──────────────────────────────────────────────────
        data["2s10s"] = data["10Y"] - data["2Y"]    # classic belly slope
        data["5s30s"] = data["30Y"] - data["5Y"]    # long-end slope
        data["3m10y"] = data["10Y"] - data["3M"]    # recession monitor
        data["1y2y"]  = data["2Y"]  - data["1Y"]    # front-end hike/cut pricing

        # ── Daily changes (bps) ───────────────────────────────────────────
        for col in ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]:
            data[f"{col}_chg"] = data[col].diff() * 100

        for spread in ["2s10s", "5s30s", "3m10y"]:
            data[f"{spread}_chg"] = data[spread].diff() * 100

        os.makedirs("data/raw", exist_ok=True)
        data.to_csv("data/raw/treasury_data.csv")

        return data
