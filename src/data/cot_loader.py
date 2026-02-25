"""
cot_loader.py
─────────────
CFTC Commitments of Traders (COT) data loader for Treasury futures.

Data source : CFTC Disaggregated COT — financial futures.
              https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
Frequency   : Weekly (positions as of Tuesday close; published every Friday).
Coverage    : 2006–present (disaggregated report).

Treasury futures tracked:
    TU  — 2-Year Treasury Note futures  (CBOT)
    FV  — 5-Year Treasury Note futures  (CBOT)
    TY  — 10-Year Treasury Note futures (CBOT)
    US  — Classic 30-Year Treasury Bond (CBOT)
    WN  — Ultra Treasury Bond (30Y+ dur) (CBOT)

Positioning categories (Disaggregated report):
    Dealer/Intermediary — primary dealers, market makers
    Asset Manager       — real money (pension, insurance, mutual funds)
    Leveraged Funds     — hedge funds, CTAs, other leveraged vehicles

Key signals built here:
    lev_net_pct  — leveraged fund net long as % of OI → contrarian at extremes
    am_net_pct   — asset manager net long as % of OI  → momentum (real money)
    positioning_signal — aggregated 2s10s curve implication from lev-fund extremes

Workflow example:
    loader   = COTLoader()
    cot_raw  = loader.load(start_year=2010)
    features = COTLoader.positioning_features(cot_raw)

    # Merge into daily treasury DataFrame (forward-fills weekly COT to daily)
    enriched = COTLoader.merge_with_treasury_data(treasury_df, features)
"""

import io
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests


# ── CFTC market name patterns for Treasury futures ─────────────────────────
# These are substrings of the "Market_and_Exchange_Names" column in the
# CFTC disaggregated report. Matching is case-insensitive.

_TREASURY_MARKETS: dict[str, str] = {
    "TU": "2-YEAR U.S. TREASURY NOTES",
    "FV": "5-YEAR TREASURY NOTES",
    "TY": "10-YEAR U.S. TREASURY NOTES",
    "US": "U.S. TREASURY BONDS",          # Classic 30Y bond
    "WN": "ULTRA U.S. TREASURY BONDS",    # Ultra Bond (30Y+ duration)
}

# Relevant columns in the CFTC disaggregated report CSV
_DATE_COL   = "Report_Date_as_MM_DD_YYYY"
_MARKET_COL = "Market_and_Exchange_Names"

_RAW_COLS: dict[str, str] = {
    "dealer_long": "Dealer_Positions_Long_All",
    "dealer_short": "Dealer_Positions_Short_All",
    "am_long":     "Asset_Mgr_Positions_Long_All",
    "am_short":    "Asset_Mgr_Positions_Short_All",
    "lev_long":    "Lev_Money_Positions_Long_All",
    "lev_short":   "Lev_Money_Positions_Short_All",
    "oi":          "Open_Interest_All",
}

# CFTC file URL patterns
_HIST_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
_CURR_URL = "https://www.cftc.gov/dea/newcot/FinFutCombined.txt"


class COTLoader:
    """
    Loads and processes CFTC Disaggregated COT data for Treasury futures.

    Annual historical files are cached locally in cache_dir so they are
    only downloaded once. The current-year file is always re-fetched
    (it is updated every Friday by the CFTC).

    Usage:
        loader   = COTLoader()
        cot      = loader.load(start_year=2010)
        features = COTLoader.positioning_features(cot)
        enriched = COTLoader.merge_with_treasury_data(treasury_df, features)
    """

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ── Raw data fetching ──────────────────────────────────────────────────

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """
        Download and parse one year's disaggregated financial futures COT file.
        Cached to cache_dir/cot_fin_{year}.csv after first download.
        """
        cache_path = os.path.join(self.cache_dir, f"cot_fin_{year}.csv")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, low_memory=False)

        url  = _HIST_URL.format(year=year)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return pd.DataFrame()   # year not yet available
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
            with zf.open(txt_name) as f:
                df = pd.read_csv(f, low_memory=False)

        df.to_csv(cache_path, index=False)
        return df

    def _fetch_current(self) -> pd.DataFrame:
        """
        Fetch the current in-progress year's COT data from CFTC (updated weekly).
        Not cached — always pulled fresh to get the latest weekly report.
        """
        resp = requests.get(_CURR_URL, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), low_memory=False)

    def load_raw(
        self,
        start_year: int = 2006,
        include_current: bool = True,
    ) -> pd.DataFrame:
        """
        Load raw CFTC disaggregated COT data for all years >= start_year.

        Parameters
        ----------
        start_year      : first year to load (disaggregated data starts 2006)
        include_current : whether to append the in-progress current year

        Returns
        -------
        pd.DataFrame — raw CFTC data, all markets (not yet filtered to Treasury)
        """
        current_year = date.today().year
        frames = []

        for yr in range(start_year, current_year):
            df_yr = self._fetch_year(yr)
            if not df_yr.empty:
                frames.append(df_yr)

        if include_current:
            try:
                frames.append(self._fetch_current())
            except Exception as exc:
                print(f"[COTLoader] Warning: could not fetch current COT data: {exc}")

        if not frames:
            raise RuntimeError(
                "No COT data loaded. Check network access and CFTC URL."
            )
        return pd.concat(frames, ignore_index=True)

    # ── Treasury filtering & processing ───────────────────────────────────

    @staticmethod
    def _filter_treasury(raw: pd.DataFrame) -> pd.DataFrame:
        """Keep only Treasury futures rows and parse the report date."""
        mask = pd.Series(False, index=raw.index)
        for pattern in _TREASURY_MARKETS.values():
            mask |= raw[_MARKET_COL].str.contains(pattern, na=False, case=False)

        df = raw[mask].copy()
        df[_DATE_COL] = pd.to_datetime(df[_DATE_COL], format="%m/%d/%Y")
        df = df.rename(columns={_DATE_COL: "date"}).set_index("date").sort_index()
        return df

    @staticmethod
    def _identify_contract(market_name: str) -> str:
        """Map a CFTC market name string to a short contract label (TU/FV/…)."""
        for label, pattern in _TREASURY_MARKETS.items():
            if pattern.lower() in market_name.lower():
                return label
        return "UNKNOWN"

    def load(self, start_year: int = 2006) -> pd.DataFrame:
        """
        Load, filter, and clean Treasury COT data into a wide daily DataFrame.

        Returns a DataFrame indexed by report date (Tuesday of each week)
        with columns named {contract}_{metric}, e.g.:
            TY_oi, TY_lev_net, TY_lev_long, TY_lev_short,
            TY_am_net,  TY_am_long,  TY_am_short,
            TY_dealer_net, TY_dealer_long, TY_dealer_short

        Note: COT is WEEKLY. Use merge_with_treasury_data() to forward-fill
              to daily for use alongside treasury_data.
        """
        raw = self.load_raw(start_year=start_year)
        tsy = self._filter_treasury(raw)
        tsy["contract"] = tsy[_MARKET_COL].apply(self._identify_contract)

        contract_frames = []
        for contract in ["TU", "FV", "TY", "US", "WN"]:
            sub = tsy[tsy["contract"] == contract].copy()
            if sub.empty:
                continue
            # Keep last entry per date (duplicates can occur at year boundaries)
            sub = sub[~sub.index.duplicated(keep="last")]

            # Extract raw columns, coercing to numeric
            cols = {}
            for short, long_name in _RAW_COLS.items():
                cols[short] = (
                    pd.to_numeric(sub[long_name], errors="coerce")
                    if long_name in sub.columns
                    else pd.Series(np.nan, index=sub.index)
                )

            contract_df = pd.DataFrame({
                f"{contract}_oi":          cols["oi"],
                f"{contract}_lev_net":     cols["lev_long"]    - cols["lev_short"],
                f"{contract}_lev_long":    cols["lev_long"],
                f"{contract}_lev_short":   cols["lev_short"],
                f"{contract}_am_net":      cols["am_long"]     - cols["am_short"],
                f"{contract}_am_long":     cols["am_long"],
                f"{contract}_am_short":    cols["am_short"],
                f"{contract}_dealer_net":  cols["dealer_long"] - cols["dealer_short"],
                f"{contract}_dealer_long": cols["dealer_long"],
                f"{contract}_dealer_short":cols["dealer_short"],
            })
            contract_frames.append(contract_df)

        result = pd.concat(contract_frames, axis=1).sort_index()
        result.to_csv(os.path.join(self.cache_dir, "cot_treasury.csv"))
        return result

    # ── Signal features ───────────────────────────────────────────────────

    @staticmethod
    def positioning_features(
        cot: pd.DataFrame,
        windows: tuple = (13, 52),
    ) -> pd.DataFrame:
        """
        Transform raw COT net positioning into normalised signal features.

        For each contract × category (lev, am, dealer):
            {contract}_{cat}_net_pct     : net long as % of open interest
            {contract}_{cat}_z{w}w       : rolling z-score over w weeks
            {contract}_{cat}_extreme_{w}w: 1 if |z| > 2 (historically stretched)

        Normalising by open interest (net_pct) removes scale drift as contract
        notional changes over time, making z-scores comparable across eras.

        Windows default to 13w (≈ 1 quarter) and 52w (≈ 1 year).

        Parameters
        ----------
        cot     : output of COTLoader.load()
        windows : rolling windows in *weeks* (COT is weekly data)

        Returns
        -------
        pd.DataFrame — weekly frequency, same index as cot
        """
        feature_frames = []

        for contract in ["TU", "FV", "TY", "US", "WN"]:
            oi_col = f"{contract}_oi"
            if oi_col not in cot.columns:
                continue
            oi = cot[oi_col].replace(0, np.nan)

            for category in ["lev", "am", "dealer"]:
                net_col = f"{contract}_{category}_net"
                if net_col not in cot.columns:
                    continue
                net = cot[net_col]

                # Net as % of OI
                pct = (net / oi * 100).rename(f"{contract}_{category}_net_pct")
                feature_frames.append(pct)

                # Rolling z-scores and extreme flags
                for w in windows:
                    roll_mean = pct.rolling(w).mean()
                    roll_std  = pct.rolling(w).std()
                    z = ((pct - roll_mean) / roll_std).rename(
                        f"{contract}_{category}_z{w}w"
                    )
                    extreme = (z.abs() > 2).astype(int).rename(
                        f"{contract}_{category}_extreme_{w}w"
                    )
                    feature_frames.append(z)
                    feature_frames.append(extreme)

        return pd.concat(feature_frames, axis=1).sort_index()

    @staticmethod
    def positioning_signal(
        features: pd.DataFrame,
        window: int = 13,
    ) -> pd.Series:
        """
        Aggregate leveraged-fund positioning into a contrarian curve signal.

        Logic (contrarian):
            Lev funds extremely net SHORT 10Y (TY_lev_z << 0)
                → TY squeeze likely → 10Y rallies → 2s10s STEEPENS → +
            Lev funds extremely net SHORT 2Y  (TU_lev_z << 0)
                → TU squeeze likely → 2Y rallies  → 2s10s FLATTENS → −

        Signal = clip(TU_lev_z − TY_lev_z, −2, +2) / 2
            → range [−1, +1] consistent with other SignalEngine components.

        Positive → curve steepening expected.
        Negative → curve flattening expected.

        ⚠  UNVALIDATED — validate with SignalEngine.conditional_returns()
           before adding to composite_score().

        Parameters
        ----------
        features : output of COTLoader.positioning_features()
        window   : which z-score window to use (13 or 52)

        Returns
        -------
        pd.Series — weekly frequency, range [−1, +1]
        """
        tu_z = features.get(f"TU_lev_z{window}w")
        ty_z = features.get(f"TY_lev_z{window}w")

        if tu_z is None or ty_z is None:
            raise KeyError(
                f"Columns TU_lev_z{window}w / TY_lev_z{window}w not found. "
                f"Run positioning_features() with window={window} included."
            )

        # tu_z − ty_z:
        #   TY very short (ty_z << 0), TU neutral → 0 − (−2) = +2 → steepen ✓
        #   TU very short (tu_z << 0), TY neutral → −2 − 0 = −2 → flatten ✓
        raw = (tu_z - ty_z).clip(-2, 2) / 2
        return raw.rename("cot_positioning_signal")

    @staticmethod
    def merge_with_treasury_data(
        treasury_df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill weekly COT positioning features into the daily treasury DataFrame.

        COT positions are as of Tuesday close; the report is released Friday.
        We forward-fill each Tuesday reading to fill the following Mon–Mon period
        (limit=7 days so a missing week does not bleed further).

        Parameters
        ----------
        treasury_df : daily DataFrame from DataLoader.load_treasury_data()
        features    : weekly DataFrame from COTLoader.positioning_features()

        Returns
        -------
        Daily DataFrame with COT positioning columns appended.
        """
        daily = features.reindex(treasury_df.index, method="ffill", limit=7)
        return pd.concat([treasury_df, daily], axis=1)
