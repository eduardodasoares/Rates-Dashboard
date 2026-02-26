"""
bloomberg_loader.py
───────────────────
Full Bloomberg Terminal integration using blpapi.

Usage:
    from src.data.bloomberg_loader import BloombergLoader, enhance_fred_data

    with BloombergLoader() as bbg:
        rates  = bbg.load_rates_data(start_date="2005-01-01")
        oi     = bbg.load_futures_oi(start_date="2005-01-01")
        ois    = bbg.load_sofr_ois_path()

    # Merge Bloomberg data into existing FRED DataFrame
    enhanced_df = enhance_fred_data(fred_df, rates)

Requires:
    pip install blpapi   (Bloomberg Python SDK — from Bloomberg terminal)
    Bloomberg Terminal running and accessible at localhost:8194 (default)

Key Bloomberg series loaded:
    BGCR         Broad General Collateral Rate      (replaces OBFR GC_proxy)
    TGCR         Tri-Party General Collateral Rate
    MOVE         ICE BofA MOVE Index                (rates vol, like VIX for bonds)
    USSW{n}      USD IRS swap rates 2Y/5Y/10Y/30Y
    TU/FV/TY/US/WN Comdty  Treasury futures prices + open interest
    USOSFR{n}Z   SOFR OIS curve for Fed path pricing
"""

import os
from datetime import date
from typing import Dict, List, Union

import numpy as np
import pandas as pd

try:
    import blpapi
    _BLPAPI_AVAILABLE = True
except ImportError:
    _BLPAPI_AVAILABLE = False


# ── Bloomberg series catalog ───────────────────────────────────────────────

_RATES_SERIES: Dict[str, str] = {
    # Repo / funding — replaces OBFR (GC_proxy) in data_loader.py
    "BGCR":   "BGCR Index",          # Broad General Collateral Rate
    "TGCR":   "TGCR Index",          # Tri-Party General Collateral Rate
    # Rates volatility
    "MOVE":   "MOVE Index",          # ICE BofA MOVE Index (1M Treasury implied vol)
    # USD IRS swap rates (mid, BGN composite)
    "SW_2Y":  "USSW2 Curncy",
    "SW_5Y":  "USSW5 Curncy",
    "SW_10Y": "USSW10 Curncy",
    "SW_30Y": "USSW30 Curncy",
}

_FUTURES_SERIES: Dict[str, str] = {
    # Generic front-month contract tickers
    "TU": "TU1 Comdty",    # 2Y Treasury note futures
    "FV": "FV1 Comdty",    # 5Y Treasury note futures
    "TY": "TY1 Comdty",    # 10Y Treasury note futures
    "US": "US1 Comdty",    # Classic 30Y Treasury bond futures
    "WN": "WN1 Comdty",    # Ultra Bond (30Y+ duration) futures
}

_SOFR_OIS_TENORS: Dict[str, str] = {
    # SOFR OIS curve — each tenor gives the implied O/N SOFR average
    # compounded over that period; used to back out meeting-by-meeting pricing
    "SOFR_OIS_1M":  "USOSFR1Z BGN Curncy",
    "SOFR_OIS_2M":  "USOSFR2Z BGN Curncy",
    "SOFR_OIS_3M":  "USOSFR3Z BGN Curncy",
    "SOFR_OIS_6M":  "USOSFR6Z BGN Curncy",
    "SOFR_OIS_9M":  "USOSFR9Z BGN Curncy",
    "SOFR_OIS_12M": "USOSFR12Z BGN Curncy",
    "SOFR_OIS_18M": "USOSFR18Z BGN Curncy",
    "SOFR_OIS_24M": "USOSFR24Z BGN Curncy",
}


# ── Helper ─────────────────────────────────────────────────────────────────

def _to_bbg_date(d: Union[str, date]) -> str:
    """Convert a date to Bloomberg's YYYYMMDD string format."""
    if isinstance(d, str):
        return pd.Timestamp(d).strftime("%Y%m%d")
    return d.strftime("%Y%m%d")


# ── Main loader class ──────────────────────────────────────────────────────

class BloombergLoader:
    """
    Bloomberg BDH / BDP client for the rates dashboard.

    Use as a context manager so the session is always cleanly closed:

        with BloombergLoader() as bbg:
            df = bbg.bdh("TY1 Comdty", "PX_LAST", "20200101", "20251231")

    All high-level methods (load_rates_data, load_futures_oi, etc.) save CSVs
    to data/raw/ following the same convention as DataLoader.
    """

    def __init__(self, host: str = "localhost", port: int = 8194):
        if not _BLPAPI_AVAILABLE:
            raise ImportError(
                "blpapi is not installed. Run: pip install blpapi\n"
                "The Bloomberg Python SDK is available from your terminal at:\n"
                "  C:\\blp\\DAPI\\APIv3\\Python\\os\\win32  (Windows)\n"
                "or download from the Bloomberg Developer Portal."
            )
        self.host = host
        self.port = port
        self._session = None
        self._svc = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "BloombergLoader":
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._disconnect()

    # ── Session management ─────────────────────────────────────────────────

    def _connect(self) -> None:
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        self._session = blpapi.Session(opts)
        if not self._session.start():
            raise ConnectionError(
                f"Bloomberg session failed to start. "
                f"Ensure Bloomberg Terminal is running at {self.host}:{self.port}."
            )
        if not self._session.openService("//blp/refdata"):
            raise ConnectionError(
                "Failed to open Bloomberg reference data service (//blp/refdata)."
            )
        self._svc = self._session.getService("//blp/refdata")

    def _disconnect(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
            self._svc = None

    # ── Core request methods ───────────────────────────────────────────────

    def bdh(
        self,
        securities: Union[str, List[str]],
        fields: Union[str, List[str]],
        start_date: Union[str, date],
        end_date: Union[str, date, None] = None,
        periodicity: str = "DAILY",
        fill_option: str = "ACTIVE_DAYS_ONLY",
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Historical data request (BDH).

        Parameters
        ----------
        securities  : Bloomberg ticker(s). Single str returns a DataFrame;
                      list of str returns a dict keyed by ticker.
        fields      : field name(s), e.g. "PX_LAST" or ["PX_LAST", "OPEN_INT"]
        start_date  : "YYYY-MM-DD" str or date object
        end_date    : end date (defaults to today)
        periodicity : "DAILY" | "WEEKLY" | "MONTHLY"
        fill_option : "ACTIVE_DAYS_ONLY" (skip non-trading days) |
                      "ALL_CALENDAR_DAYS"

        Returns
        -------
        pd.DataFrame       — DatetimeIndex × fields, if single security
        dict[str, DataFrame] — keyed by ticker, if multiple securities
        """
        single = isinstance(securities, str)
        if single:
            securities = [securities]
        if isinstance(fields, str):
            fields = [fields]
        if end_date is None:
            end_date = date.today()

        request = self._svc.createRequest("HistoricalDataRequest")
        for sec in securities:
            request.getElement("securities").appendValue(sec)
        for fld in fields:
            request.getElement("fields").appendValue(fld)
        request.set("startDate", _to_bbg_date(start_date))
        request.set("endDate",   _to_bbg_date(end_date))
        request.set("periodicitySelection", periodicity)
        request.set("nonTradingDayFillOption", fill_option)

        self._session.sendRequest(request)

        results: Dict[str, pd.DataFrame] = {}
        while True:
            ev = self._session.nextEvent(timeout=30_000)
            for msg in ev:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    if msg.hasElement("responseError"):
                        err_msg = (
                            msg.getElement("responseError")
                            .getElement("message")
                            .getValue()
                        )
                        raise RuntimeError(f"Bloomberg error: {err_msg}")

                    sec_data   = msg.getElement("securityData")
                    ticker     = sec_data.getElement("security").getValue()
                    field_data = sec_data.getElement("fieldData")

                    rows = []
                    for i in range(field_data.numValues()):
                        pt  = field_data.getValue(i)
                        row = {"date": pd.Timestamp(pt.getElement("date").getValue())}
                        for fld in fields:
                            try:
                                row[fld] = pt.getElement(fld).getValue()
                            except blpapi.NotFoundException:
                                row[fld] = np.nan
                        rows.append(row)

                    results[ticker] = (
                        pd.DataFrame(rows).set_index("date")
                        if rows
                        else pd.DataFrame(columns=fields)
                    )

            if ev.eventType() == blpapi.Event.RESPONSE:
                break
            if ev.eventType() == blpapi.Event.TIMEOUT:
                raise TimeoutError("Bloomberg BDH request timed out after 30s.")

        return results[securities[0]] if single else results

    def bdp(
        self,
        securities: Union[str, List[str]],
        fields: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Reference / point-in-time data request (BDP).

        Returns a DataFrame indexed by security ticker, columns by field.
        Useful for static attributes: FUT_DLV_DT_LAST, DUR_MID, COUPON, etc.
        """
        if isinstance(securities, str):
            securities = [securities]
        if isinstance(fields, str):
            fields = [fields]

        request = self._svc.createRequest("ReferenceDataRequest")
        for sec in securities:
            request.getElement("securities").appendValue(sec)
        for fld in fields:
            request.getElement("fields").appendValue(fld)

        self._session.sendRequest(request)

        rows: Dict[str, dict] = {}
        while True:
            ev = self._session.nextEvent(timeout=30_000)
            for msg in ev:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_arr = msg.getElement("securityData")
                    for i in range(sec_arr.numValues()):
                        sec    = sec_arr.getValue(i)
                        ticker = sec.getElement("security").getValue()
                        fdata  = sec.getElement("fieldData")
                        row    = {}
                        for fld in fields:
                            try:
                                row[fld] = fdata.getElement(fld).getValue()
                            except Exception:
                                row[fld] = np.nan
                        rows[ticker] = row

            if ev.eventType() == blpapi.Event.RESPONSE:
                break
            if ev.eventType() == blpapi.Event.TIMEOUT:
                raise TimeoutError("Bloomberg BDP request timed out after 30s.")

        return pd.DataFrame(rows).T

    # ── High-level data loaders ────────────────────────────────────────────

    def load_rates_data(
        self,
        start_date: Union[str, date] = "2000-01-01",
        end_date: Union[str, date, None] = None,
    ) -> pd.DataFrame:
        """
        Load Bloomberg-sourced rates series into a single DataFrame.

        Columns returned:
            BGCR, TGCR        — repo / GC funding rates (replace OBFR GC_proxy)
            MOVE              — ICE BofA MOVE Index (rates implied vol)
            SW_2Y/5Y/10Y/30Y  — USD IRS par swap rates

        Saves to data/raw/bloomberg_rates.csv. Returns the DataFrame.
        """
        frames = {}
        for name, ticker in _RATES_SERIES.items():
            try:
                tmp = self.bdh(ticker, "PX_LAST", start_date, end_date)
                frames[name] = tmp["PX_LAST"].rename(name)
            except Exception as exc:
                print(f"[BloombergLoader] Warning: could not load {name} ({ticker}): {exc}")

        if not frames:
            raise RuntimeError("No Bloomberg rates data loaded. Check terminal connection.")

        result = pd.concat(frames.values(), axis=1).sort_index()
        result.index = pd.to_datetime(result.index)
        os.makedirs("data/raw", exist_ok=True)
        result.to_csv("data/raw/bloomberg_rates.csv")
        return result

    def load_futures_oi(
        self,
        start_date: Union[str, date] = "2000-01-01",
        end_date: Union[str, date, None] = None,
    ) -> pd.DataFrame:
        """
        Load daily price and open interest for front-month Treasury futures.

        Columns: TU_px, TU_oi, FV_px, FV_oi, TY_px, TY_oi, US_px, US_oi,
                 WN_px, WN_oi.

        Saves to data/raw/futures_oi.csv. Returns the DataFrame.
        """
        frames = {}
        for name, ticker in _FUTURES_SERIES.items():
            try:
                tmp = self.bdh(ticker, ["PX_LAST", "OPEN_INT"], start_date, end_date)
                # Rename Series before concat so column names survive pd.concat
                frames[f"{name}_px"] = tmp["PX_LAST"].rename(f"{name}_px")
                frames[f"{name}_oi"] = tmp["OPEN_INT"].rename(f"{name}_oi")
            except Exception as exc:
                print(f"[BloombergLoader] Warning: could not load {name} ({ticker}): {exc}")

        if not frames:
            raise RuntimeError("No futures data loaded. Check terminal connection.")

        result = pd.concat(frames.values(), axis=1).sort_index()
        result.index = pd.to_datetime(result.index)
        os.makedirs("data/raw", exist_ok=True)
        result.to_csv("data/raw/futures_oi.csv")
        return result

    def load_sofr_ois_path(
        self,
        start_date: Union[str, date] = "2018-04-01",
        end_date: Union[str, date, None] = None,
    ) -> pd.DataFrame:
        """
        Load SOFR OIS curve tenors for market-implied Fed path pricing.

        Returns the compounded SOFR rate implied at each tenor (1M through 24M).
        Used to build a precise "cuts/hikes priced over next N meetings" feature
        that replaces the blunt 2Y_vs_FF threshold gate in FeatureEngineer.

        Note: SOFR OIS data starts ~April 2018 (SOFR inception).

        Saves to data/raw/sofr_ois.csv. Returns the DataFrame.
        """
        frames = {}
        for name, ticker in _SOFR_OIS_TENORS.items():
            try:
                tmp = self.bdh(ticker, "PX_LAST", start_date, end_date)
                frames[name] = tmp["PX_LAST"].rename(name)
            except Exception as exc:
                print(f"[BloombergLoader] Warning: could not load {name} ({ticker}): {exc}")

        if not frames:
            raise RuntimeError("No SOFR OIS data loaded. Check terminal connection.")

        result = pd.concat(frames.values(), axis=1).sort_index()
        result.index = pd.to_datetime(result.index)
        os.makedirs("data/raw", exist_ok=True)
        result.to_csv("data/raw/sofr_ois.csv")
        return result


# ── Merge helper ───────────────────────────────────────────────────────────

def enhance_fred_data(
    fred_df: pd.DataFrame,
    bbg_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Bloomberg rates data into the FRED-sourced treasury DataFrame.

    Actions:
    - Replaces GC_proxy (OBFR) with BGCR where available (BGCR starts ~2014).
      Pre-BGCR rows keep the OBFR value so carry history remains intact.
    - Adds MOVE, TGCR, and USD swap rates as new columns.
    - Computes swap spreads (Treasury yield minus swap rate, in bps).
      A negative swap spread means Treasuries yield MORE than swaps — unusual
      and historically associated with forced selling / supply stress.

    Parameters
    ----------
    fred_df   : daily DataFrame from DataLoader.load_treasury_data()
    bbg_rates : daily DataFrame from BloombergLoader.load_rates_data()

    Returns
    -------
    Enhanced DataFrame — same index as fred_df, Bloomberg columns added.
    """
    df  = fred_df.copy()
    bbg = bbg_rates.reindex(df.index)

    # Replace GC_proxy (OBFR) with BGCR where BGCR is available.
    # SOFR_GC_spread in feature_engineering.py is now computed against
    # the actual broad GC rate rather than the OBFR proxy.
    if "BGCR" in bbg.columns:
        mask = bbg["BGCR"].notna()
        df.loc[mask, "GC_proxy"] = bbg.loc[mask, "BGCR"]

    # Append Bloomberg-only columns directly
    for col in ["TGCR", "MOVE", "SW_2Y", "SW_5Y", "SW_10Y", "SW_30Y"]:
        if col in bbg.columns:
            df[col] = bbg[col]

    # Swap spreads: Treasury yield minus par swap rate (bps).
    # Negative = Treasuries yield more than equivalent-maturity swaps.
    for tsy_col, sw_col in [("2Y", "SW_2Y"), ("5Y", "SW_5Y"),
                             ("10Y", "SW_10Y"), ("30Y", "SW_30Y")]:
        if sw_col in df.columns:
            df[f"swap_spread_{tsy_col}"] = (df[tsy_col] - df[sw_col]) * 100  # bps

    return df
