import numpy as np
import pandas as pd

from pypension.allocation_methods.base import AbstractPortfolio


class FixedWeight(AbstractPortfolio):
    def allocate_weights_t(self, weights: dict[str, np.float32]):
        return pd.Series(weights)


class EqualWight(AbstractPortfolio):
    def allocate_weights_t(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        tickers = asset_returns.columns
        return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
