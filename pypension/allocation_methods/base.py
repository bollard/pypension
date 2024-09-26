import abc

import pandas as pd


class AbstractPortfolio(abc.ABC):
    def __init__(self, asset_returns: pd.DataFrame):
        self.asset_returns = asset_returns

    def allocate_weights(
        self, rebalance_dates: pd.DatetimeIndex | None = None, *args, **kwargs
    ) -> pd.DataFrame:
        if rebalance_dates is None:
            rebalance_dates = self.asset_returns.index.take([-1])

        weights_t = {}

        for rebalance_date in rebalance_dates:
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
        raise NotImplementedError()

    @staticmethod
    def calculate_returns(df_close: pd.DataFrame) -> pd.DataFrame:
        return df_close.pct_change().dropna()

    @staticmethod
    def calculate_covariance(df_returns: pd.DataFrame) -> pd.DataFrame:
        # index = df_returns.index
        # df_cov = df_returns.ewm(halflife="30 days", times=index).cov()
        # return df_cov.loc[index[-1]]
        return df_returns.cov()
