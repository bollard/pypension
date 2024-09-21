import abc

import pandas as pd


class AbstractPortfolio(abc.ABC):
    def __init__(
        self, asset_returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex = None
    ):
        self.asset_returns = asset_returns
        self.rebalance_dates = rebalance_dates

    @property
    def rebalance_schedule(self) -> pd.DatetimeIndex:
        if self.rebalance_dates is None:
            return self.asset_returns.index.take([-1])

        return self.rebalance_dates

    def allocate_weights(self, *args, **kwargs) -> pd.DataFrame:
        weights_t = {}

        for rebalance_date in self.rebalance_schedule:
            asset_returns_t = self.asset_returns.loc[
                self.asset_returns.index < rebalance_date
            ]
            weights_t[rebalance_date] = self.allocate_weights_t(
                asset_returns_t, *args, **kwargs
            )

        return pd.DataFrame.from_dict(weights_t, orient="index")

    @abc.abstractmethod
    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, *args, **kwargs
    ) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def calculate_returns(df_close: pd.DataFrame) -> pd.DataFrame:
        return df_close.pct_change().dropna()
