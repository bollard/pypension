import pandas as pd

from pypension.allocation_methods.base import AbstractPortfolio


class FixedWeight(AbstractPortfolio):
    def __init__(self, asset_returns: pd.DataFrame, weights: dict[str, float]):
        super().__init__(asset_returns=asset_returns)
        self.weights = weights

    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, **kwargs
    ) -> pd.Series:
        return pd.Series(self.weights)


class EqualWight(FixedWeight):
    def __init__(self, asset_returns: pd.DataFrame):
        assets = asset_returns.columns
        weights = {asset: 1 / len(assets) for asset in assets}
        super().__init__(asset_returns=asset_returns, weights=weights)
