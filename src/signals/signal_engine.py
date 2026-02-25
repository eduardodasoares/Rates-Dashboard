import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from src.data.return_calculator import ReturnCalculator


class SignalEngine:
    """
    Computes signal analysis outputs from the enriched feature DataFrame.

    All methods accept the DataFrame produced by:
        DataLoader.load_treasury_data() -> FeatureEngineer.add_curve_features()

    All methods return DataFrames so results can be piped to the dashboard,
    exported to CSV, or used in further analysis.

    Two evaluation tracks:
        Curve / spread trades  → pass spread_col (e.g. '2s10s', '5s30s')
                                  P&L = Δspread in percentage points
                                  Positive = steepened = steepener trade won
        Outright bond trades   → pass yield_col (e.g. '10Y', '2Y')
                                  P&L = synthetic duration-approximated return
    """

    @staticmethod
    def _fwd_return(
        df: pd.DataFrame,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
    ) -> pd.Series:
        """
        Unified forward return helper. Pass either yield_col OR spread_col.

        spread_col → ReturnCalculator.spread_return (curve trade P&L)
        yield_col  → ReturnCalculator.synthetic     (bond level return)

        Returns a Series aligned to df.index, shifted so that on day t
        the value reflects the return over the NEXT `horizon` trading days.
        """
        if spread_col is not None:
            return ReturnCalculator.spread_return(df, spread_col, horizon).shift(-horizon)
        if yield_col is not None:
            return ReturnCalculator.synthetic(df, yield_col, horizon).shift(-horizon)
        raise ValueError("Must provide either yield_col or spread_col.")

    @staticmethod
    def conditional_returns(
        df: pd.DataFrame,
        signal_col: str,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Bucket signal_col into n quantiles and compute forward return stats per bucket.

        Answers: "When the signal was in its top quintile, what did returns look like?"

        Pass yield_col for bond-level evaluation (outright duration trade).
        Pass spread_col for curve trade evaluation (e.g. '2s10s' steepener P&L).

        Parameters
        ----------
        df          : enriched DataFrame from FeatureEngineer
        signal_col  : column to bucket (e.g. '2s10s_z20', '10Y_voladj')
        horizon     : forward return horizon in trading days
        yield_col   : tenor for synthetic return calc (e.g. '10Y', '2Y')
        spread_col  : spread column for curve trade P&L (e.g. '2s10s', '5s30s')
        n_quantiles : number of buckets (default 5 = quintiles)

        Returns
        -------
        pd.DataFrame indexed by quantile label (Q1..Q5)
        """
        df = df.copy()
        df["_fwd"] = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
        df["_sig"] = df[signal_col]
        df = df.dropna(subset=["_sig", "_fwd"])

        df["_q"] = pd.qcut(
            df["_sig"],
            q=n_quantiles,
            labels=[f"Q{i}" for i in range(1, n_quantiles + 1)],
            duplicates="drop",
        )

        result = df.groupby("_q", observed=True).agg(
            median_return=("_fwd", "median"),
            mean_return=("_fwd", "mean"),
            count=("_fwd", "count"),
            signal_mean=("_sig", "mean"),
            signal_min=("_sig", "min"),
            signal_max=("_sig", "max"),
        )

        return result

    @staticmethod
    def regime_performance(
        df: pd.DataFrame,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
    ) -> pd.DataFrame:
        """
        Mean, median, hit rate, and approx Sharpe by regime.

        Pass yield_col to evaluate how outright bond returns vary by regime.
        Pass spread_col to evaluate how curve steepening/flattening varies by regime.

        Parameters
        ----------
        df         : enriched DataFrame (must have 'regime' column)
        horizon    : forward return horizon in trading days
        yield_col  : tenor for bond return calc (e.g. '10Y')
        spread_col : spread column for curve P&L (e.g. '2s10s')

        Returns
        -------
        pd.DataFrame indexed by regime label
        """
        df = df.copy()
        df["_fwd"] = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
        df = df.dropna(subset=["regime", "_fwd"])
        df = df[df["regime"] != "Unknown"]

        def _sharpe(s: pd.Series) -> float:
            return s.mean() / s.std() * np.sqrt(252 / horizon) if s.std() != 0 else np.nan

        result = df.groupby("regime")["_fwd"].agg(
            mean_return="mean",
            median_return="median",
            count="count",
            hit_rate=lambda x: (x > 0).mean(),
            sharpe_approx=_sharpe,
        ).round(4)

        return result

    @staticmethod
    def hit_rate(
        df: pd.DataFrame,
        signal_col: str,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
    ) -> pd.DataFrame:
        """
        Fraction of observations where signal direction matches forward return direction.

        For yield_col: positive signal → expect yields to fall (bond bullish).
        For spread_col: positive signal → expect curve to steepen.

        Parameters
        ----------
        df         : enriched DataFrame
        signal_col : signal column to evaluate (e.g. '2s10s_z20', '10Y_voladj')
        horizon    : forward return horizon in trading days
        yield_col  : tenor for bond return calc
        spread_col : spread column for curve P&L

        Returns
        -------
        Single-row pd.DataFrame with hit rate statistics
        """
        df = df.copy()
        df["_fwd"] = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
        df["_sig"] = df[signal_col]
        df = df.dropna(subset=["_sig", "_fwd"])

        sig_dir = np.sign(df["_sig"])
        ret_dir = np.sign(df["_fwd"])
        hit = sig_dir == ret_dir
        bull = sig_dir > 0
        bear = sig_dir < 0

        return pd.DataFrame([{
            "signal_col":         signal_col,
            "yield_col":          yield_col,
            "horizon_d":          horizon,
            "hit_rate":           round(hit.mean(), 4),
            "n_obs":              len(df),
            "bull_hit_rate":      round(hit[bull].mean(), 4) if bull.any() else np.nan,
            "bear_hit_rate":      round(hit[bear].mean(), 4) if bear.any() else np.nan,
            "avg_return_hit":     round(df.loc[hit,  "_fwd"].mean(), 6),
            "avg_return_miss":    round(df.loc[~hit, "_fwd"].mean(), 6),
        }])

    @staticmethod
    def signal_decay(
        df: pd.DataFrame,
        signal_col: str,
        max_horizon: int = 63,
        yield_col: str = None,
        spread_col: str = None,
    ) -> pd.DataFrame:
        """
        Pearson correlation between signal_col and forward return at each
        horizon from 1 to max_horizon days.

        Used to visualize how quickly a signal's predictive power decays.
        A signal with slow decay is more actionable at longer horizons.

        Parameters
        ----------
        df          : enriched DataFrame
        signal_col  : signal column (e.g. '2s10s_z20')
        max_horizon : last horizon to evaluate (default 63 ≈ 3 months)
        yield_col   : tenor for bond return calc
        spread_col  : spread column for curve P&L

        Returns
        -------
        pd.DataFrame indexed by horizon (int days),
        columns: correlation, p_value, n_obs
        """
        signal = df[signal_col].dropna()
        rows = []

        for h in range(1, max_horizon + 1):
            fwd = SignalEngine._fwd_return(df, h, yield_col=yield_col, spread_col=spread_col)
            combined = pd.concat([signal, fwd], axis=1).dropna()
            combined.columns = ["signal", "ret"]

            if len(combined) < 30:
                rows.append({"horizon": h, "correlation": np.nan, "p_value": np.nan, "n_obs": len(combined)})
                continue

            corr, pval = pearsonr(combined["signal"], combined["ret"])
            rows.append({"horizon": h, "correlation": round(corr, 4), "p_value": round(pval, 4), "n_obs": len(combined)})

        return pd.DataFrame(rows).set_index("horizon")

    @staticmethod
    def composite_score(df: pd.DataFrame) -> pd.Series:
        """
        Build a time-series composite signal score for curve steepening/flattening.

        Positive score (+) → expect curve to STEEPEN (2s10s widens).
        Negative score (−) → expect curve to FLATTEN (2s10s narrows).

        Score range: −4 to +4, four equally-weighted components (each ±1).

        Components removed after validation (see notebooks/component_validation.ipynb):
        - Funding (SOFR−TBill z20): 0/5 tests passed, wrong sign, p=0.13, hurt Q5 Sharpe.
        - Z-score (2s10s_z20): r=0.64 overlap with Regime. 3-way comparison showed
          dropping Z-score raised Q5 Sharpe from 1.471 → 1.642 and Q5 mean from
          +1.75 → +2.16 bps. Regime carries the non-redundant information.

        CONTRARIAN / MEAN-REVERSION components:
        1. Regime component:
               Steepening regime (Bear/Bull Steepener) → −1 (momentum fades, expect reversion)
               Flattening regime (Bear/Bull Flattener) → +1 (expect snap-back steepen)
               Validation: r=0.064, p=0.004, ablation ΔQ5=−0.354.

        2. Vol-adjusted move component:
               Curve flattened sharply today (voladj << 0) → +1 (expect snap-back steepen)
               Curve steepened sharply today (voladj >> 0) → −1 (expect snap-back flatten)
               Capped at ±1: -clip(2s10s_voladj / 2, −1, +1)
               Validation: r=−0.086, p=0.0002, ablation ΔQ5=−0.469 (largest contributor).

        MOMENTUM components (empirically validated at 5–10d horizon):
        3. VIX component:
               Elevated VIX vs recent history (z20 >> 0) → risk-off → flight-to-quality → +1
               (10Y rallies more than 2Y → curve steepens; significant through ~16d)
               Capped at ±1: +clip(VIX_z20 / 2, −1, +1)
               Validation: r=0.097, p<0.0001, ablation ΔQ5=−0.156.

        4. Front-end momentum component:
               Steep 1y2y spread vs recent history (z20 >> 0) → market pushing out cuts
               → front end rising → whole curve steepening → +1
               (Peak correlation at 9–10d horizon; significant through ~18d)
               Capped at ±1: +clip(1y2y_z20 / 2, −1, +1)
               Validation: r=0.115 (highest), p<0.0001, ablation ΔQ5=−0.468.

        Returns
        -------
        pd.Series named 'composite_score', same index as df
        """
        def _clip(series: pd.Series, scale: float = 2.0) -> pd.Series:
            return series.div(scale).clip(-1, 1).fillna(0)

        # Contrarian components
        regime_map = {
            "Bear Steepener": -1,
            "Bull Steepener": -1,
            "Bear Flattener": +1,
            "Bull Flattener": +1,
            "Unknown":         0,
        }
        regime_component = df["regime"].map(regime_map).fillna(0)
        voladj_component = _clip(-df["2s10s_voladj"])
        # Funding (SOFR−TBill z20) removed: failed all 5 validation tests,
        # wrong direction, p=0.13, ablation showed it hurt Q5 Sharpe.
        # Z-score (2s10s_z20) removed: r=0.64 overlap with Regime; dropping
        # raised Q5 Sharpe 1.471→1.642 and Q5 mean +1.75→+2.16 bps.

        # Momentum components (positive clip — signal and direction agree)
        vix_component       = _clip(df["VIX_z20"])
        front_end_component = _clip(df["1y2y_z20"])

        score = (
            regime_component
            + voladj_component
            + vix_component
            + front_end_component
        ).rename("composite_score")

        return score

    @staticmethod
    def gated_score(df: pd.DataFrame) -> pd.Series:
        """
        Composite score scaled by macro_scalar from feature_engineering.

        During active Fed hiking/cutting cycles, mean-reversion signals are
        less reliable. The macro_scalar (1.0 / 0.5 / 0.25) dampens the score
        proportionally, reducing position sizing signals in trending environments.

        Requires feature_engineering.add_curve_features() to have been run
        (produces macro_scalar and macro_regime columns).

        Returns
        -------
        pd.Series named 'gated_score', same index as df
        """
        score  = SignalEngine.composite_score(df)
        scalar = df["macro_scalar"].reindex(score.index).fillna(1.0)
        return (score * scalar).rename("gated_score")

    @staticmethod
    def conditional_regime_zscore(
        df: pd.DataFrame,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
    ) -> pd.DataFrame:
        """
        Build a regime × z-score bucket grid of mean forward returns.

        Answers: "What happens in a Bear Flattener when the curve z-score
                  is in its most extreme quintile?"

        Pass spread_col='2s10s' to grade on curve trade P&L (correct for
        a curve signal). Pass yield_col to grade on bond level returns.

        Parameters
        ----------
        df         : enriched DataFrame (must have 'regime', '2s10s_z20')
        horizon    : forward return horizon in trading days
        yield_col  : tenor for bond return calc (e.g. '10Y')
        spread_col : spread column for curve P&L (e.g. '2s10s')

        Returns
        -------
        pd.DataFrame — rows: regime labels, columns: z-score quintile buckets
        Values: mean forward return annotated with observation count
        """
        df = df.copy()
        df["_fwd"] = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
        df = df.dropna(subset=["regime", "2s10s_z20", "_fwd"])
        df = df[df["regime"] != "Unknown"]

        df["_z_bucket"] = pd.qcut(
            df["2s10s_z20"],
            q=5,
            labels=["Q1\n(very flat)", "Q2", "Q3", "Q4", "Q5\n(very steep)"],
            duplicates="drop",
        )

        mean_grid = df.pivot_table(
            values="_fwd",
            index="regime",
            columns="_z_bucket",
            aggfunc="mean",
            observed=True,
        ).round(4)

        count_grid = df.pivot_table(
            values="_fwd",
            index="regime",
            columns="_z_bucket",
            aggfunc="count",
            observed=True,
        )

        # Annotate cells: "return (n=count)"
        annotated = mean_grid.copy().astype(str)
        for r in mean_grid.index:
            for c in mean_grid.columns:
                ret = mean_grid.loc[r, c]
                cnt = count_grid.loc[r, c] if r in count_grid.index and c in count_grid.columns else 0
                annotated.loc[r, c] = f"{ret:.4f} (n={int(cnt)})"

        return annotated

    @staticmethod
    def score_performance(
        df: pd.DataFrame,
        horizon: int,
        yield_col: str = None,
        spread_col: str = None,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Compute forward return statistics bucketed by composite score quintile.

        Answers: "Does the combined signal have edge that regime or z-score
                  alone didn't?"

        Pass spread_col='2s10s' for the correct curve-signal evaluation.
        Pass yield_col for bond-level return evaluation.

        If the composite score is working, Q1 (most flattening signal) should
        show negative spread returns and Q5 (most steepening) positive —
        a monotonic pattern across quintiles is the sign of a real signal.

        Parameters
        ----------
        df          : enriched DataFrame
        horizon     : forward return horizon in trading days
        yield_col   : tenor for bond return calc (e.g. '10Y')
        spread_col  : spread column for curve P&L (e.g. '2s10s')
        n_quantiles : number of buckets (default 5 = quintiles)

        Returns
        -------
        pd.DataFrame indexed by score quintile (Q1..Q5)
        """
        df = df.copy()
        df["_score"] = SignalEngine.composite_score(df)
        df["_fwd"] = SignalEngine._fwd_return(df, horizon, yield_col=yield_col, spread_col=spread_col)
        df = df.dropna(subset=["_score", "_fwd"])

        df["_q"] = pd.qcut(
            df["_score"],
            q=n_quantiles,
            labels=[f"Q{i}" for i in range(1, n_quantiles + 1)],
            duplicates="drop",
        )

        def _sharpe(s: pd.Series) -> float:
            return s.mean() / s.std() * np.sqrt(252 / horizon) if s.std() != 0 else np.nan

        # Signal direction: Q1..Q(mid) = short (flattener), Q(mid+1)..Q5 = long (steepener)
        # Aligning the trade with the signal so all quintiles show the Sharpe
        # of the *correct* trade, not a fixed long position.
        mid = n_quantiles / 2
        q_labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
        q_rank = {q: i + 1 for i, q in enumerate(q_labels)}

        def _aligned_sharpe(group):
            q = group.name
            sign = -1 if q_rank.get(q, mid + 1) <= mid else 1
            s = group * sign
            return s.mean() / s.std() * np.sqrt(252 / horizon) if s.std() != 0 else np.nan

        base = df.groupby("_q", observed=True).agg(
            mean_return=("_fwd", "mean"),
            median_return=("_fwd", "median"),
            hit_rate=("_fwd", lambda x: (x > 0).mean()),
            sharpe_long=("_fwd", _sharpe),
            score_mean=("_score", "mean"),
            score_min=("_score", "min"),
            score_max=("_score", "max"),
            count=("_fwd", "count"),
        ).round(4)

        base["sharpe_aligned"] = (
            df.groupby("_q", observed=True)["_fwd"].apply(_aligned_sharpe).round(4)
        )

        return base

    @staticmethod
    def effective_threshold(
        df: pd.DataFrame,
        base: float = 1.3,
        beta: float = 0.10,
        cap:  float = 0.30,
    ) -> tuple:
        """
        Carry-adjusted entry thresholds for steepener and flattener signals.

        Shifts the static base threshold by beta × carry_roll_5d_z60, capped
        at ±cap so the adjustment never moves the threshold by more than 0.30.

        When carry is above its recent norm (z60 > 0):
            steepener threshold FALLS  → easier to enter (carry tailwind)
            flattener threshold RISES  → harder to enter (carry headwind for flat)

        When carry is below its recent norm (z60 < 0):
            steepener threshold RISES  → harder to enter (carry headwind for steep)
            flattener threshold FALLS  → easier to enter (carry tailwind for flat)

        Parameters
        ----------
        df   : enriched DataFrame (must have carry_roll_5d_z60 from FeatureEngineer)
        base : static threshold baseline (default 1.3)
        beta : sensitivity per σ of carry z-score (default 0.10)
        cap  : maximum absolute shift from base (default 0.30)

        Returns
        -------
        (thresh_steep, thresh_flat) — two pd.Series aligned to df.index,
        each floored at max(1.0, base - cap).
        """
        z = df["carry_roll_5d_z60"].clip(-2, 2).fillna(0)
        floor = max(1.0, base - cap)
        thresh_steep = (base - beta * z).clip(lower=floor, upper=base + cap)
        thresh_flat  = (base + beta * z).clip(lower=floor, upper=base + cap)
        return thresh_steep, thresh_flat

    @staticmethod
    def non_overlapping_performance(
        df: pd.DataFrame,
        horizon: int = 5,
        spread_col: str = "2s10s",
        score_threshold: float = 1.3,
    ) -> pd.DataFrame:
        """
        Evaluate signal on fully independent, non-overlapping windows.

        Samples the signal every `horizon` days so no two observations share
        any part of the same forward return window. This gives the true
        independent-trade Sharpe, free of autocorrelation from overlapping holds.

        The overlapping-daily Sharpe inflates the t-stat because consecutive
        daily signals share 4/5 of the same forward return. With 1,969 daily
        obs and horizon=5, the effective independent sample is ~394.

        Parameters
        ----------
        df              : enriched DataFrame
        horizon         : hold period and sampling interval in trading days
        spread_col      : curve spread to measure returns on (default '2s10s')
        score_threshold : min |score| to take a directional trade

        Returns
        -------
        pd.DataFrame with one row per independent window:
            date, score, direction, fwd_bps, carry_roll_bps,
            aligned_bps, aligned_adj_bps, hit, hit_adj

        carry_roll_bps  : carry + roll contribution to this trade (bps, signed)
        aligned_adj_bps : spread P&L + carry_roll (total economics of the trade)
        hit_adj         : 1 if total carry-adjusted P&L > 0
        """
        score = SignalEngine.composite_score(df)
        spread = df[spread_col]
        has_cr = "carry_roll_5d" in df.columns

        # Use carry-adjusted thresholds if the feature is available;
        # fall back to static score_threshold if carry_roll_5d_z60 is missing.
        if "carry_roll_5d_z60" in df.columns:
            thresh_steep_s, thresh_flat_s = SignalEngine.effective_threshold(
                df, base=score_threshold
            )
        else:
            thresh_steep_s = pd.Series(score_threshold, index=df.index)
            thresh_flat_s  = pd.Series(score_threshold, index=df.index)

        rows = []
        indices = range(0, len(df) - horizon, horizon)
        for i in indices:
            date = df.index[i]
            s = float(score.iloc[i])
            fwd_bps = (float(spread.iloc[i + horizon]) - float(spread.iloc[i])) * 100

            t_steep = float(thresh_steep_s.iloc[i])
            t_flat  = float(thresh_flat_s.iloc[i])

            if s >= t_steep:
                direction, sign = "steepener", +1
            elif s <= -t_flat:
                direction, sign = "flattener", -1
            else:
                direction, sign = "flat", 0

            aligned = fwd_bps * sign

            # Carry+roll contribution: positive = helps the trade
            cr_raw = float(df["carry_roll_5d"].iloc[i]) if has_cr else np.nan
            cr_trade = sign * cr_raw if sign != 0 and pd.notna(cr_raw) else (0.0 if sign == 0 else np.nan)
            aligned_adj = aligned + cr_trade if pd.notna(cr_trade) else np.nan

            rows.append({
                "date":             date,
                "score":            round(s, 4),
                "direction":        direction,
                "thresh_steep":     round(t_steep, 3),
                "thresh_flat":      round(t_flat, 3),
                "fwd_bps":          round(fwd_bps, 2),
                "carry_roll_bps":   round(cr_trade, 3) if pd.notna(cr_trade) else np.nan,
                "aligned_bps":      round(aligned, 2),
                "aligned_adj_bps":  round(aligned_adj, 2) if pd.notna(aligned_adj) else np.nan,
                "hit":              (1 if aligned > 0 else 0) if sign != 0 else np.nan,
                "hit_adj":          (1 if aligned_adj > 0 else 0) if sign != 0 and pd.notna(aligned_adj) else np.nan,
            })

        return pd.DataFrame(rows).set_index("date")

    @staticmethod
    def portfolio_pnl(
        df: pd.DataFrame,
        horizon: int = 5,
        spread_col: str = "2s10s",
        max_dv01: float = 100_000,
        target_risk_bps: float = 10.0,
        score_threshold: float = 1.3,
    ) -> pd.Series:
        """
        Simulate running the strategy with explicitly overlapping positions.

        At each day t, up to `horizon` positions may be concurrently open
        (entered on days t-1, t-2, ..., t-horizon). Each position contributes
        its own daily mark-to-market to the total daily P&L:

            daily_pnl(t) = Σ_i [ sign_i × dv01_i × spread_change(t) ]

        where the sum is over all positions still open on day t.

        This is the correct way to simulate a strategy where you signal daily
        but hold for multiple days. The resulting P&L series properly reflects
        the concentration risk from running several concurrent same-direction
        trades (a streak of 5 steepener signals = 5x the DV01 exposure).

        The Sharpe on this daily series is the honest risk-adjusted return —
        it accounts for the serial correlation automatically because it operates
        on the actual daily P&L, not per-trade returns.

        Returns
        -------
        pd.Series of daily P&L, normalised to fraction of max_dv01
        (multiply by max_dv01 for dollar P&L)
        """
        sizing = SignalEngine.position_size(
            df, max_dv01=max_dv01, target_risk_bps=target_risk_bps,
            horizon=horizon, score_threshold=score_threshold,
        )

        chg_col = spread_col + "_chg"   # already in bps from feature engineering
        daily_chg = df[chg_col].reindex(sizing.index).fillna(0)
        has_cr = "carry_roll_5d" in df.columns

        daily_pnl = pd.Series(0.0, index=sizing.index)

        for i, (date, row) in enumerate(sizing.iterrows()):
            if row["direction"] == "flat":
                continue
            sign = +1 if row["direction"] == "steepener" else -1
            dv01_frac = float(row["recommended_dv01"]) / max_dv01

            # Daily carry+roll accrual for this position.
            # carry_roll_5d is the total C+R for `horizon` days → accrue evenly.
            cr_daily = 0.0
            if has_cr and date in df.index:
                cr_val = df.loc[date, "carry_roll_5d"]
                if pd.notna(cr_val):
                    cr_daily = sign * dv01_frac * float(cr_val) / horizon

            # Mark this position to market for each of the next `horizon` days
            future_slice = sizing.index[i + 1: i + 1 + horizon]
            for future_date in future_slice:
                if future_date in daily_pnl.index:
                    daily_pnl[future_date] += (
                        sign * dv01_frac * daily_chg.get(future_date, 0)
                        + cr_daily
                    )

        return daily_pnl.rename("portfolio_pnl_bps_frac")

    @staticmethod
    def position_size(
        df: pd.DataFrame,
        max_dv01: float = 100_000,
        target_risk_bps: float = 10.0,
        horizon: int = 5,
        score_threshold: float = 1.3,
    ) -> pd.DataFrame:
        """
        Translate composite score + current vol into a recommended DV01 for a 2s10s trade.

        Three factors multiplied together:

        1. Score fraction — signal strength
               |score| / 4 → 0.0 (flat) to 1.0 (max score ±4)
               At threshold ±1.3 (Q5/Q1 boundary): fraction ≈ 0.33
               At ±2.5: fraction ≈ 0.63

        2. Vol scalar — curve vol adjustment
               target_risk_bps / (2s10s_vol20 × √horizon)
               Low vol → size up; high vol → size down
               Clipped to [0.25, 2.0] to prevent extreme sizing

        3. Gate — silence weak signals
               |score| < score_threshold → DV01 = 0

        recommended_dv01 = max_dv01 × score_fraction × vol_scalar × gate
        risk_bps         = expected bps at risk = fwd_vol × score_fraction × vol_scalar × gate

        Trade direction:
            score > 0  → STEEPENER (long 2s10s spread: long 10Y, short 2Y DV01-neutral)
            score < 0  → FLATTENER (short 2s10s spread: short 10Y, long 2Y DV01-neutral)

        Parameters
        ----------
        df               : enriched DataFrame (must contain 2s10s_vol20)
        max_dv01         : max DV01 at full score (e.g. 100_000 = $100k/bp at score ±4)
        target_risk_bps  : bps at risk per trade at vol-adjusted full size (default 10)
        horizon          : intended hold in trading days (default 5)
        score_threshold  : min |score| to trade (default 1.3 ≈ Q5/Q1 boundary)

        Returns
        -------
        pd.DataFrame with columns:
            score, direction, score_fraction, vol_20d_bps,
            fwd_vol_bps, vol_scalar, recommended_dv01, risk_bps
        """
        score = SignalEngine.composite_score(df)
        vol20 = df["2s10s_vol20"].reindex(score.index)
        fwd_vol = vol20 * np.sqrt(horizon)

        # Use carry-adjusted thresholds (per-date) when available.
        if "carry_roll_5d_z60" in df.columns:
            thresh_steep_s, thresh_flat_s = SignalEngine.effective_threshold(
                df, base=score_threshold
            )
        else:
            thresh_steep_s = pd.Series(score_threshold, index=score.index)
            thresh_flat_s  = pd.Series(score_threshold, index=score.index)

        score_fraction = (score.abs() / 4.0).clip(upper=1.0)
        vol_scalar = (target_risk_bps / fwd_vol).clip(lower=0.25, upper=2.0)

        # Gate: open if score clears the side-specific dynamic threshold.
        gate = pd.Series(0.0, index=score.index)
        gate[(score >= thresh_steep_s)] = 1.0
        gate[(score <= -thresh_flat_s)] = 1.0

        direction = pd.Series("flat", index=score.index)
        direction[score >= thresh_steep_s]  = "steepener"
        direction[score <= -thresh_flat_s]  = "flattener"

        recommended_dv01 = (max_dv01 * score_fraction * vol_scalar * gate).round(0)
        risk_bps = (fwd_vol * score_fraction * vol_scalar * gate).round(2)

        return pd.DataFrame({
            "score":            score,
            "direction":        direction,
            "thresh_steep":     thresh_steep_s.round(3),
            "thresh_flat":      thresh_flat_s.round(3),
            "score_fraction":   score_fraction.round(3),
            "vol_20d_bps":      vol20.round(3),
            "fwd_vol_bps":      fwd_vol.round(3),
            "vol_scalar":       vol_scalar.round(3),
            "recommended_dv01": recommended_dv01,
            "risk_bps":         risk_bps,
        })

    @staticmethod
    def butterfly_score(df: pd.DataFrame, fly_col: str = "fly_2s5s10s") -> pd.Series:
        """
        Candidate composite score for a butterfly belly trade.

        Positive (+) → belly cheap (fly z-score negative), expect richening.
                        Trade: receive belly (long 5Y, pay 2Y + 10Y DV01-neutral).
        Negative (−) → belly rich (fly z-score positive), expect cheapening.
                        Trade: pay belly (short 5Y, receive 2Y + 10Y DV01-neutral).

        Score range: −3 to +3, three equally-weighted candidate components (each ±1).

        ⚠  UNVALIDATED — run through component_validation.ipynb before use.
           Validate each component with conditional_returns(), signal_decay(),
           and ablation tests (same protocol used for composite_score()).

        Candidate components:
        1. Z-score (contrarian): cheap belly (z20 << 0) → +1, rich belly → −1.
               −clip(fly_z20 / 2, −1, +1)
        2. Vol-adjusted move (contrarian): sharp cheapening today → +1.
               −clip(fly_voladj / 2, −1, +1)
        3. Regime context:
               Bull Flattener  → 10Y rallies hard, 5Y follows → belly tends to richen → −1
               Bear Steepener  → 10Y sells off, 5Y lags then follows → belly cheapens → +1
               Other regimes → 0 (ambiguous belly behaviour, skip)

        Parameters
        ----------
        df      : enriched DataFrame from FeatureEngineer.add_curve_features()
        fly_col : butterfly column to score (default 'fly_2s5s10s')

        Returns
        -------
        pd.Series named 'butterfly_score', aligned to df.index
        """
        z_col    = fly_col + "_z20"
        vadj_col = fly_col + "_voladj"

        def _clip(s: pd.Series, scale: float = 2.0) -> pd.Series:
            return s.div(scale).clip(-1, 1).fillna(0)

        # Component 1: z-score mean reversion
        z_component = _clip(-df[z_col])

        # Component 2: vol-adjusted move mean reversion
        vadj_component = _clip(-df[vadj_col])

        # Component 3: regime context
        # Bear Steepener: 10Y selling off fastest, 5Y lags → belly cheapens → expect
        #   belly to continue cheapening (momentum) or snap back (reversion)?
        #   Empirical question for the notebook. Initial hypothesis: momentum → −1.
        # Bull Flattener: 10Y rallying hard → 5Y follows but belly enriches → −1.
        # Using mean-reversion hypothesis: regime that cheapened belly → expect richen.
        regime_map = {
            "Bear Steepener": +1,   # belly cheapened during regime → expect richen
            "Bull Flattener": -1,   # belly enriched during regime → expect cheapen
            "Bear Flattener":  0,
            "Bull Steepener":  0,
            "Unknown":         0,
        }
        regime_component = df["regime"].map(regime_map).fillna(0)

        score = (z_component + vadj_component + regime_component).rename("butterfly_score")
        return score

    @staticmethod
    def butterfly_non_overlapping_performance(
        df: pd.DataFrame,
        horizon: int = 5,
        fly_col: str = "fly_2s5s10s",
        score_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Evaluate butterfly_score on fully independent, non-overlapping windows.

        Mirrors non_overlapping_performance() but for butterfly (belly) trades.

        Direction convention:
            positive score → receive belly (long fly_2s5s10s: long 5Y, pay 2Y+10Y)
            negative score → pay belly    (short fly_2s5s10s: short 5Y, receive 2Y+10Y)

        fwd_bps is the change in the fly over the next `horizon` days × 100.
        aligned_bps = fwd_bps × sign, so positive aligned_bps = trade won.

        ⚠  UNVALIDATED — use for notebook validation, not live signals.

        Parameters
        ----------
        df              : enriched DataFrame from FeatureEngineer.add_curve_features()
        horizon         : hold period and sampling interval in trading days
        fly_col         : butterfly column to measure returns on
        score_threshold : min |score| required to take a trade

        Returns
        -------
        pd.DataFrame indexed by date, columns:
            score, direction, fwd_bps, aligned_bps, hit
        """
        score  = SignalEngine.butterfly_score(df, fly_col=fly_col)
        spread = df[fly_col]

        rows = []
        for i in range(0, len(df) - horizon, horizon):
            date    = df.index[i]
            s       = float(score.iloc[i])
            fwd_bps = (float(spread.iloc[i + horizon]) - float(spread.iloc[i])) * 100

            if s >= score_threshold:
                direction, sign = "receive_belly", +1
            elif s <= -score_threshold:
                direction, sign = "pay_belly", -1
            else:
                direction, sign = "flat", 0

            aligned = fwd_bps * sign

            rows.append({
                "date":        date,
                "score":       round(s, 4),
                "direction":   direction,
                "fwd_bps":     round(fwd_bps, 2),
                "aligned_bps": round(aligned, 2),
                "hit":         (1 if aligned > 0 else 0) if sign != 0 else np.nan,
            })

        return pd.DataFrame(rows).set_index("date")

    @staticmethod
    def snapshot(df: pd.DataFrame) -> pd.DataFrame:
        """
        Single-row DataFrame of the most recent date's signal state.

        Used by the dashboard bottom table and for daily reporting.

        Returns
        -------
        pd.DataFrame with one row (indexed by date)
        """
        df = df.copy()
        df["composite_score"] = SignalEngine.composite_score(df)

        # Inject sizing columns into df so they appear in the snapshot row
        sizing = SignalEngine.position_size(df)
        for col in ["direction", "score_fraction", "vol_20d_bps", "fwd_vol_bps",
                    "vol_scalar", "recommended_dv01", "risk_bps"]:
            df[col] = sizing[col]

        cols = [
            # Regime & composite
            "regime",
            "composite_score",
            # Position sizing
            "direction", "score_fraction", "vol_20d_bps", "fwd_vol_bps",
            "vol_scalar", "recommended_dv01", "risk_bps",
            # Carry & roll
            "carry_5d", "roll_5d", "carry_roll_5d",
            # Curve shape
            "2s10s_z20", "5s30s_z20",
            "2s10s_z60", "5s30s_z60",
            "2s10s_mom_5d", "2s10s_mom_20d",
            "5s30s_mom_5d", "5s30s_mom_20d",
            "2Y_voladj", "10Y_voladj",
            "2s10s_voladj", "5s30s_voladj",
            # Funding stress
            "SOFR_TBill_spread", "SOFR_GC_spread",
            "SOFR_TBill_spread_z20", "SOFR_GC_spread_z20",
            # Fed expectations
            "2Y_vs_FF", "2Y_vs_FF_z20",
            # Front-end curve
            "3m10y", "3m10y_z20",
            "1y2y",  "1y2y_z20",
            # Real yields & breakevens
            "10Y_TIPS", "10Y_BE", "5Y_BE",
            "real_5s10s", "BE_slope",
            "10Y_BE_z20", "10Y_TIPS_z20",
            "10Y_BE_mom_5d",
            # Risk sentiment
            "VIX", "VIX_z20", "VIX_mom_5d",
            "IG_OAS", "IG_OAS_z20", "IG_OAS_mom_5d",
            "HY_OAS", "HY_OAS_z20", "HY_OAS_mom_5d",
        ]
        available = [c for c in cols if c in df.columns]
        snap = df[available].iloc[[-1]].copy()
        snap.index.name = "date"
        return snap
