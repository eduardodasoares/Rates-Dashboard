from abc import ABC, abstractmethod
from datetime import date
from typing import List, Union

import numpy as np
import pandas as pd

try:
    import blpapi
    _BLPAPI_AVAILABLE = True
except ImportError:
    _BLPAPI_AVAILABLE = False


class BaseFuturesLoader(ABC):
    """
    Abstract interface for loading interest rate futures data.

    Implementors must return a DataFrame indexed by DatetimeIndex with columns:
        contract  : str   — futures contract ticker (e.g. 'TYH5')
        price     : float — settlement price
        expiry    : date  — contract expiry date
        yield_imp : float — implied yield (optional, derive if unavailable)
        oi        : int   — open interest (optional)

    To integrate a real data source, subclass this and implement load_futures_data().
    """

    @abstractmethod
    def load_futures_data(self, tickers: list) -> pd.DataFrame:
        """
        Parameters
        ----------
        tickers : list of str
            Futures contract tickers to load (e.g. ['TYH5', 'TYM5', 'USH5']).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns: contract, price, expiry, yield_imp, oi.
        """
        pass


class BloombergFuturesLoader(BaseFuturesLoader):
    """
    Bloomberg BDH integration for Treasury futures data.

    Loads settlement price, open interest, and expiry for a list of
    contract tickers using blpapi. Results are returned in the shape
    expected by BaseFuturesLoader (one row per date × contract).

    Usage:
        loader = BloombergFuturesLoader()
        df = loader.load_futures_data(["TYH6", "TYM6", "USH6"])

        # Or use generic front-month tickers for a continuous series:
        df = loader.load_futures_data(["TY1 Comdty", "US1 Comdty"])

    Bloomberg fields used:
        PX_LAST         — settlement / last price
        OPEN_INT        — open interest (contracts)
        FUT_DLV_DT_LAST — last delivery date (used as expiry proxy)

    Requires:
        pip install blpapi
        Bloomberg Terminal running at localhost:8194
    """

    _FIELDS = ["PX_LAST", "OPEN_INT", "FUT_DLV_DT_LAST"]

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        start_date: Union[str, date] = "2000-01-01",
        end_date: Union[str, date, None] = None,
    ):
        if not _BLPAPI_AVAILABLE:
            raise ImportError(
                "blpapi is not installed. Run: pip install blpapi\n"
                "The Bloomberg Python SDK ships with your terminal."
            )
        self.host = host
        self.port = port
        self.start_date = start_date
        self.end_date   = end_date or date.today()

    # ── Session helpers ────────────────────────────────────────────────────

    def _open_session(self) -> tuple:
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        session = blpapi.Session(opts)
        if not session.start():
            raise ConnectionError(
                f"Bloomberg session failed to start at {self.host}:{self.port}."
            )
        if not session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdata service.")
        svc = session.getService("//blp/refdata")
        return session, svc

    @staticmethod
    def _bbg_date(d: Union[str, date]) -> str:
        if isinstance(d, str):
            return pd.Timestamp(d).strftime("%Y%m%d")
        return d.strftime("%Y%m%d")

    # ── Core BDH request ───────────────────────────────────────────────────

    def _bdh_single(
        self,
        session,
        svc,
        ticker: str,
        fields: List[str],
    ) -> pd.DataFrame:
        """Send a single BDH request and parse the response into a DataFrame."""
        request = svc.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(ticker)
        for fld in fields:
            request.getElement("fields").appendValue(fld)
        request.set("startDate", self._bbg_date(self.start_date))
        request.set("endDate",   self._bbg_date(self.end_date))
        request.set("periodicitySelection", "DAILY")
        request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")

        session.sendRequest(request)

        rows = []
        while True:
            ev = session.nextEvent(timeout=30_000)
            for msg in ev:
                if msg.messageType() != blpapi.Name("HistoricalDataResponse"):
                    continue
                if msg.hasElement("responseError"):
                    err = (
                        msg.getElement("responseError")
                        .getElement("message")
                        .getValue()
                    )
                    raise RuntimeError(f"Bloomberg error for {ticker}: {err}")

                field_data = (
                    msg.getElement("securityData").getElement("fieldData")
                )
                for i in range(field_data.numValues()):
                    pt  = field_data.getValue(i)
                    row = {"date": pd.Timestamp(pt.getElement("date").getValue())}
                    for fld in fields:
                        try:
                            row[fld] = pt.getElement(fld).getValue()
                        except blpapi.NotFoundException:
                            row[fld] = np.nan
                    rows.append(row)

            if ev.eventType() == blpapi.Event.RESPONSE:
                break
            if ev.eventType() == blpapi.Event.TIMEOUT:
                raise TimeoutError(f"Bloomberg request timed out for {ticker}.")

        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    # ── Public interface ───────────────────────────────────────────────────

    def load_futures_data(self, tickers: list) -> pd.DataFrame:
        """
        Load settlement price, OI, and expiry for each ticker.

        Parameters
        ----------
        tickers : list of Bloomberg futures tickers.
                  Specific contracts: ['TYH6', 'TYM6', 'TYU6']
                  Generic front-month: ['TY1 Comdty', 'US1 Comdty']
                  (append ' Comdty' if the ticker has no suffix)

        Returns
        -------
        pd.DataFrame — DatetimeIndex, columns:
            contract, price, oi, expiry, yield_imp

        yield_imp is left as NaN here; derive it from CTD analytics
        or use BloombergLoader.bdh() with FUT_YLD_CNV_MID if needed.
        """
        # Normalise tickers: Bloomberg needs ' Comdty' suffix for futures
        normalised = [
            t if " " in t else f"{t} Comdty"
            for t in tickers
        ]

        session, svc = self._open_session()
        try:
            frames = []
            for ticker in normalised:
                raw = self._bdh_single(session, svc, ticker, self._FIELDS)
                if raw.empty:
                    continue

                df = pd.DataFrame({
                    "contract":  ticker.replace(" Comdty", ""),
                    "price":     pd.to_numeric(raw.get("PX_LAST"),         errors="coerce"),
                    "oi":        pd.to_numeric(raw.get("OPEN_INT"),        errors="coerce"),
                    "expiry":    raw.get("FUT_DLV_DT_LAST"),
                    "yield_imp": np.nan,
                }, index=raw.index)
                frames.append(df)
        finally:
            session.stop()

        if not frames:
            raise RuntimeError(
                f"No futures data returned for tickers: {tickers}. "
                "Check ticker format and Bloomberg terminal connection."
            )

        return pd.concat(frames).sort_index()
