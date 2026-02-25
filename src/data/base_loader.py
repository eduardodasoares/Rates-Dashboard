from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):

    @abstractmethod
    def load_treasury_data(self) -> pd.DataFrame:
        """
        Must return a DataFrame with at minimum these columns:
            2Y, 5Y, 10Y, 30Y, SOFR, 3M_TBill
        Index must be a DatetimeIndex.
        """
        pass
