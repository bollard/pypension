import pandas as pd

from pypension.allocation_methods.base import AbstractPortfolio


class FixedWeight(AbstractPortfolio):
    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, weights: dict[str, float], **kwargs
    ) -> pd.Series:
        return pd.Series(weights)


class EqualWight(FixedWeight):
    def allocate_weights_t(self, asset_returns: pd.DataFrame, **kwargs) -> pd.Series:
        assets = asset_returns.columns
        weights = {asset: 1 / len(assets) for asset in assets}
        return super().allocate_weights_t(asset_returns, weights=weights)
