import numpy as np
import pandas as pd
import scipy.optimize as spo

from pypension.allocation_methods.base import AbstractPortfolio


class MinimumVariance(AbstractPortfolio):
    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, target_return: float = None, **kwargs
    ) -> pd.Series:
        """
        Compute the Minimum Variance Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.

        Returns:
        - pd.Series: Optimal portfolio weights for minimum variance.
        """

        idx = asset_returns.isna().all()
        asset_returns_active = asset_returns.loc[:, ~idx]

        # Compute covariance matrix
        cov_matrix = self.calculate_covariance(asset_returns_active)

        # Define the objective function (Minimize portfolio variance)
        def objective(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        n_assets = len(asset_returns_active.columns)
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets

        # Optimization
        result = spo.minimize(
            objective,
            x0=initial_weights,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = {asset: 0 for asset in asset_returns.columns}
            for i, asset in enumerate(asset_returns_active.columns):
                weights[asset] = result.x[i]

            return pd.Series(weights)
        else:
            raise ValueError(result.message)


class MaximumReturn(AbstractPortfolio):
    def allocate_weights_t(self, asset_returns: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute the Maximum Return Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.

        Returns:
        - pd.Series: Optimal portfolio weights for maximum return.
        """

        idx = asset_returns.isna().all()
        asset_returns_active = asset_returns.loc[:, ~idx]

        # Compute mean returns
        mean_returns = asset_returns_active.mean()

        # Define the objective function (Maximize portfolio return)
        def objective(weights, mean_returns):
            # Negative because we minimize by default
            return -np.dot(weights, mean_returns)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        n_assets = len(asset_returns_active.columns)
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets

        # Optimization
        result = spo.minimize(
            objective,
            x0=initial_weights,
            args=(mean_returns,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = {asset: 0 for asset in asset_returns.columns}
            for i, asset in enumerate(asset_returns_active.columns):
                weights[asset] = result.x[i]

            return pd.Series(weights)
        else:
            raise ValueError(result.message)


class TangencyPortfolio(AbstractPortfolio):
    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, risk_free_rate: float = 0.0, **kwargs
    ) -> pd.Series:
        """
        Compute the Tangency (Maximum Sharpe Ratio) Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

        Returns:
        - pd.Series: Optimal portfolio weights for maximum Sharpe ratio.
        """

        idx = asset_returns.isna().all()
        asset_returns_active = asset_returns.loc[:, ~idx]

        # Compute mean returns and covariance matrix
        mean_returns = asset_returns_active.mean()
        cov_matrix = self.calculate_covariance(asset_returns_active)

        # Define the objective function (Maximize Sharpe Ratio)
        def objective(weights, cov_matrix, mean_returns, risk_free_rate):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative because we minimize by default

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        n_assets = len(asset_returns_active.columns)
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets

        # Optimization
        result = spo.minimize(
            objective,
            x0=initial_weights,
            args=(cov_matrix, mean_returns, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = {asset: 0 for asset in asset_returns.columns}
            for i, asset in enumerate(asset_returns_active.columns):
                weights[asset] = result.x[i]

            return pd.Series(weights)
        else:
            raise ValueError(result.message)
