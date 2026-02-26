import pandas as pd
import numpy as np


class FeatureEngineer:

    @staticmethod
    def add_curve_features(df: pd.DataFrame) -> pd.DataFrame:

        # ── Helper: rolling z-score ───────────────────────────────────────
        def _zscore(series: pd.Series, window: int) -> pd.Series:
            return (series - series.rolling(window).mean()) / series.rolling(window).std()

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 1 — CURVE SLOPE FEATURES (2s10s, 5s30s)
        # ═══════════════════════════════════════════════════════════════════

        # Z-scores: 20d and 60d
        df["2s10s_z20"] = _zscore(df["2s10s"], 20)
        df["5s30s_z20"] = _zscore(df["5s30s"], 20)
        df["2s10s_z60"] = _zscore(df["2s10s"], 60)
        df["5s30s_z60"] = _zscore(df["5s30s"], 60)

        # Momentum: 5d and 20d
        df["2s10s_mom_5d"]  = df["2s10s"].diff(5)
        df["5s30s_mom_5d"]  = df["5s30s"].diff(5)
        df["2s10s_mom_20d"] = df["2s10s"].diff(20)
        df["5s30s_mom_20d"] = df["5s30s"].diff(20)

        # Rolling vol: 20d std of daily bps changes
        df["2Y_vol20"]    = df["2Y_chg"].rolling(20).std()
        df["10Y_vol20"]   = df["10Y_chg"].rolling(20).std()
        df["2s10s_vol20"] = df["2s10s_chg"].rolling(20).std()
        df["5s30s_vol20"] = df["5s30s_chg"].rolling(20).std()

        # Vol-adjusted moves (today's move / typical-day vol)
        df["2Y_voladj"]    = df["2Y_chg"]    / df["2Y_vol20"].replace(0, np.nan)
        df["10Y_voladj"]   = df["10Y_chg"]   / df["10Y_vol20"].replace(0, np.nan)
        df["2s10s_voladj"] = df["2s10s_chg"] / df["2s10s_vol20"].replace(0, np.nan)
        df["5s30s_voladj"] = df["5s30s_chg"] / df["5s30s_vol20"].replace(0, np.nan)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 2 — REGIME CLASSIFICATION
        # Based on 5-day direction of 10Y level and 2s10s slope
        # ═══════════════════════════════════════════════════════════════════
        chg_10y   = df["10Y"].diff(5)
        chg_2s10s = df["2s10s"].diff(5)

        df["regime"] = np.select(
            condlist=[
                (chg_10y > 0) & (chg_2s10s > 0),
                (chg_10y > 0) & (chg_2s10s <= 0),
                (chg_10y <= 0) & (chg_2s10s > 0),
                (chg_10y <= 0) & (chg_2s10s <= 0),
            ],
            choicelist=[
                "Bear Steepener",
                "Bear Flattener",
                "Bull Steepener",
                "Bull Flattener",
            ],
            default="Unknown",
        )

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 3 — FUNDING STRESS SPREADS
        # ═══════════════════════════════════════════════════════════════════
        # SOFR − 3M TBill: classic market stress proxy (TED-like), in bps
        df["SOFR_TBill_spread"] = (df["SOFR"] - df["3M_TBill"]) * 100

        # SOFR − GC_proxy (OBFR): repo market internal basis, in bps
        # GC_proxy = OBFR (Overnight Bank Funding Rate) — best available
        # FRED proxy for GC repo. Replace with BGCR when Bloomberg is live.
        # Will be NaN for pre-2016 rows where OBFR is unavailable.
        df["SOFR_GC_spread"] = (df["SOFR"] - df["GC_proxy"]) * 100

        for col in ["SOFR_TBill_spread", "SOFR_GC_spread"]:
            df[f"{col}_z20"] = _zscore(df[col], 20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 4 — FED EXPECTATIONS PROXY
        # 2Y minus Fed Funds: how much easing (negative) or tightening
        # (positive) is priced into the front end vs current policy rate.
        # ═══════════════════════════════════════════════════════════════════
        df["2Y_vs_FF"]     = df["2Y"] - df["Fed_Funds"]
        df["2Y_vs_FF_z20"] = _zscore(df["2Y_vs_FF"], 20)
        df["2Y_vs_FF_z60"] = _zscore(df["2Y_vs_FF"], 60)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 5 — FRONT-END CURVE (3m10y, 1y2y)
        # ═══════════════════════════════════════════════════════════════════
        df["3m10y_z20"] = _zscore(df["3m10y"], 20)
        df["3m10y_z60"] = _zscore(df["3m10y"], 60)
        df["1y2y_z20"]  = _zscore(df["1y2y"],  20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 6 — REAL YIELDS AND BREAKEVENS
        # Only populated where TIPS/BE data exists (post-2003).
        # ═══════════════════════════════════════════════════════════════════
        # Real yield slope (5s10s in real space)
        df["real_5s10s"]   = df["10Y_TIPS"] - df["5Y_TIPS"]

        # Breakeven inflation slope (forward inflation premium)
        df["BE_slope"]     = df["10Y_BE"] - df["5Y_BE"]

        # Z-scores
        df["10Y_TIPS_z20"] = _zscore(df["10Y_TIPS"], 20)
        df["10Y_BE_z20"]   = _zscore(df["10Y_BE"],   20)
        df["5Y_BE_z20"]    = _zscore(df["5Y_BE"],    20)
        df["BE_slope_z20"] = _zscore(df["BE_slope"],  20)

        # Momentum
        df["10Y_BE_mom_5d"]  = df["10Y_BE"].diff(5)
        df["10Y_BE_mom_20d"] = df["10Y_BE"].diff(20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 7 — RISK SENTIMENT (VIX + CREDIT SPREADS)
        # Only populated where data exists (VIX post-1990, credit post-1997).
        # ═══════════════════════════════════════════════════════════════════

        # VIX z-scores and momentum
        df["VIX_z20"]     = _zscore(df["VIX"], 20)
        df["VIX_z60"]     = _zscore(df["VIX"], 60)
        df["VIX_mom_5d"]  = df["VIX"].diff(5)
        df["VIX_mom_20d"] = df["VIX"].diff(20)

        # IG and HY OAS z-scores and momentum
        df["IG_OAS_z20"]     = _zscore(df["IG_OAS"], 20)
        df["IG_OAS_z60"]     = _zscore(df["IG_OAS"], 60)
        df["HY_OAS_z20"]     = _zscore(df["HY_OAS"], 20)
        df["HY_OAS_z60"]     = _zscore(df["HY_OAS"], 60)
        df["IG_OAS_mom_5d"]  = df["IG_OAS"].diff(5)
        df["IG_OAS_mom_20d"] = df["IG_OAS"].diff(20)
        df["HY_OAS_mom_5d"]  = df["HY_OAS"].diff(5)
        df["HY_OAS_mom_20d"] = df["HY_OAS"].diff(20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 8 — MACRO OVERLAY GATE
        # Gate trigger: 2Y_vs_FF (2Y yield minus Fed Funds rate).
        #
        # This is MARKET-IMPLIED and forward-looking: it captures what the
        # market expects the Fed to do, not what the Fed has already done.
        # FF_chg_63d is backward-looking and misses anticipatory moves —
        # e.g., in Q1 2022 the curve was already inverting before the first
        # hike, and in March 2020 the 2Y collapsed before the emergency cut.
        #
        # Thresholds (absolute level of 2Y_vs_FF, in percentage points):
        #   > +1.50% → market pricing aggressive hikes  → macro_scalar = 0.25
        #   > +1.00% → market pricing moderate hikes    → macro_scalar = 0.50
        #   < -1.50% → market pricing aggressive cuts   → macro_scalar = 0.25
        #   < -1.00% → market pricing moderate cuts     → macro_scalar = 0.50
        #   otherwise → neutral policy environment      → macro_scalar = 1.00
        # ═══════════════════════════════════════════════════════════════════
        # Keep FF_chg_63d as a reference feature (not used as gate trigger)
        df["FF_chg_63d"] = df["Fed_Funds"].diff(63)

        df["macro_scalar"] = np.select(
            [df["2Y_vs_FF"].abs() > 1.50,
             df["2Y_vs_FF"].abs() > 1.00],
            [0.25, 0.50],
            default=1.0,
        )

        df["macro_regime"] = np.select(
            [df["2Y_vs_FF"] >  1.50,
             df["2Y_vs_FF"] >  1.00,
             df["2Y_vs_FF"] < -1.50,
             df["2Y_vs_FF"] < -1.00],
            ["Hike Priced+", "Hike Priced", "Cut Priced+", "Cut Priced"],
            default="Neutral",
        )

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 9 — CARRY & ROLL (DV01-neutral 2s10s curve trade, 5d hold)
        #
        # Carry: the net yield income minus repo cost for holding a DV01-neutral
        # 2s10s steepener (long 10Y / short 2Y) for 5 trading days.
        #
        #   carry_annual = (y10 − repo) / dur10  −  (y2 − repo) / dur2
        #   carry_5d     = carry_annual × 100 × 5 / 252          [bps]
        #
        # Roll: the gain/loss from each leg rolling down the curve over 5 days.
        #   slope_10Y = (y10 − y5) / 5        [%/yr at the 10Y point]
        #   slope_2Y  = 1y2y / 1              [%/yr at the 2Y point]
        #
        #   Net roll = (slope_10Y − slope_2Y): positive if the 10Y point is
        #   steeper than the 2Y point (steepener benefits), negative otherwise.
        #   roll_5d  = (slope_10Y − slope_2Y) × 100 × 5 / 252    [bps]
        #
        # Convention: positive = STEEPENER collects carry/roll.
        #             For a flattener, multiply by −1.
        #
        # Duration constants (approximate modified duration for ~par bonds):
        #   DUR_10Y ≈ 8.5 yr   DUR_2Y ≈ 1.9 yr
        # Repo rate = SOFR (pre-filled to handle weekends/holidays in loader).
        # Repo rate: SOFR when available (post-Apr 2018), Fed Funds otherwise.
        # Correlation over overlap period = 0.998 — no material seam.
        # Extends carry history from 8yr to ~36yr (back to 1990 with VIX).
        # ═══════════════════════════════════════════════════════════════════
        _DUR_10Y = 8.5
        _DUR_2Y  = 1.9
        _H       = 5      # standard signal hold period in trading days

        repo = df["SOFR"].fillna(df["Fed_Funds"])

        carry_annual = (df["10Y"] - repo) / _DUR_10Y - (df["2Y"] - repo) / _DUR_2Y
        df["carry_5d"] = carry_annual * 100 * _H / 252

        slope_10Y = (df["10Y"] - df["5Y"]) / 5   # %/yr at the 10Y point
        slope_2Y  = df["1y2y"]                    # %/yr at the 2Y point
        df["roll_5d"]       = (slope_10Y - slope_2Y) * 100 * _H / 252
        df["carry_roll_5d"] = df["carry_5d"] + df["roll_5d"]

        # 60-day z-score of carry+roll — used by SignalEngine.effective_threshold()
        # to tighten/loosen entry bars based on the current carry environment.
        df["carry_roll_5d_z60"] = _zscore(df["carry_roll_5d"], 60)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 10 — BUTTERFLY SPREADS (2s5s10s and 5s10s30s)
        #
        # fly_2s5s10s = 2*y5 − y2 − y10
        #   > 0 → 5Y yield is low relative to the 2Y/10Y wing average (belly rich)
        #   < 0 → 5Y yield is high relative to the 2Y/10Y wing average (belly cheap)
        #
        # fly_5s10s30s = 2*y10 − y5 − y30
        #   > 0 → 10Y yield is low relative to the 5Y/30Y wing average (belly rich)
        #   < 0 → 10Y yield is high relative to the 5Y/30Y wing average (belly cheap)
        #
        # Units: percentage points (same as underlying yields).
        # Convention: positive = belly rich. A steepener trade is equivalent to
        # being long the wings and short the belly → gains when fly moves negative.
        # ═══════════════════════════════════════════════════════════════════
        df["fly_2s5s10s"]  = 2 * df["5Y"]  - df["2Y"]  - df["10Y"]
        df["fly_5s10s30s"] = 2 * df["10Y"] - df["5Y"]  - df["30Y"]

        # Daily changes (bps)
        df["fly_2s5s10s_chg"]  = df["fly_2s5s10s"].diff()  * 100
        df["fly_5s10s30s_chg"] = df["fly_5s10s30s"].diff() * 100

        # Rolling vol: 20d std of daily changes (bps)
        df["fly_2s5s10s_vol20"]  = df["fly_2s5s10s_chg"].rolling(20).std()
        df["fly_5s10s30s_vol20"] = df["fly_5s10s30s_chg"].rolling(20).std()

        # Vol-adjusted moves (today's move / recent typical day)
        df["fly_2s5s10s_voladj"]  = df["fly_2s5s10s_chg"]  / df["fly_2s5s10s_vol20"].replace(0, np.nan)
        df["fly_5s10s30s_voladj"] = df["fly_5s10s30s_chg"] / df["fly_5s10s30s_vol20"].replace(0, np.nan)

        # Z-scores: 20d (short-term mean reversion) and 60d (medium-term)
        df["fly_2s5s10s_z20"]  = _zscore(df["fly_2s5s10s"],  20)
        df["fly_2s5s10s_z60"]  = _zscore(df["fly_2s5s10s"],  60)
        df["fly_5s10s30s_z20"] = _zscore(df["fly_5s10s30s"], 20)
        df["fly_5s10s30s_z60"] = _zscore(df["fly_5s10s30s"], 60)

        # Momentum: 5d and 20d level changes (in %)
        df["fly_2s5s10s_mom_5d"]   = df["fly_2s5s10s"].diff(5)
        df["fly_2s5s10s_mom_20d"]  = df["fly_2s5s10s"].diff(20)
        df["fly_5s10s30s_mom_5d"]  = df["fly_5s10s30s"].diff(5)
        df["fly_5s10s30s_mom_20d"] = df["fly_5s10s30s"].diff(20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 11 — MOVE INDEX (Bloomberg only; skipped if column absent)
        #
        # MOVE = ICE BofA Merrill Lynch Option Volatility Estimate.
        # Measures 1-month implied vol of Treasury options — the rates
        # equivalent of VIX. Available from 1990.
        #
        # MOVE_VIX_ratio: when MOVE is high relative to VIX, rates markets
        # are more stressed than equities. A ratio z-score > 0 historically
        # coincides with forced rates selling / liquidity stress → flattening.
        # ═══════════════════════════════════════════════════════════════════
        if "MOVE" in df.columns:
            df["MOVE_z20"]     = _zscore(df["MOVE"], 20)
            df["MOVE_z60"]     = _zscore(df["MOVE"], 60)
            df["MOVE_mom_5d"]  = df["MOVE"].diff(5)
            df["MOVE_mom_20d"] = df["MOVE"].diff(20)

            if "VIX" in df.columns:
                df["MOVE_VIX_ratio"]     = df["MOVE"] / df["VIX"].replace(0, np.nan)
                df["MOVE_VIX_ratio_z20"] = _zscore(df["MOVE_VIX_ratio"], 20)

        # ═══════════════════════════════════════════════════════════════════
        # BLOCK 12 — SWAP SPREADS (Bloomberg only; skipped if columns absent)
        #
        # Swap spread = Treasury yield − USD IRS swap rate (same maturity), bps.
        # Computed in enhance_fred_data() and carried into df before this point.
        #
        # Negative swap spread = Treasuries yield MORE than swaps — unusual.
        # Historically associated with supply pressure, balance-sheet stress,
        # or forced Treasury selling. Tends to precede flattening as real-money
        # buyers step in to buy cheap Treasuries.
        #
        # swap_2s10s: the slope of the swap curve (SW_10Y − SW_2Y).
        # Comparing it to Treasury 2s10s reveals who is driving the curve:
        # Treasuries cheap vs swaps at the long end → supply story.
        # ═══════════════════════════════════════════════════════════════════
        _swap_spreads = [c for c in
                         ["swap_spread_2Y", "swap_spread_5Y",
                          "swap_spread_10Y", "swap_spread_30Y"]
                         if c in df.columns]

        for col in _swap_spreads:
            df[f"{col}_z20"] = _zscore(df[col], 20)
            df[f"{col}_z60"] = _zscore(df[col], 60)
            df[f"{col}_mom_5d"] = df[col].diff(5)

        if "SW_10Y" in df.columns and "SW_2Y" in df.columns:
            df["swap_2s10s"]     = df["SW_10Y"] - df["SW_2Y"]
            df["swap_2s10s_z20"] = _zscore(df["swap_2s10s"], 20)

        return df
