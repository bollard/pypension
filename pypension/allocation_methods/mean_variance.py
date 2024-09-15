import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pypension.allocation_methods.base import AbstractPortfolio


class MinimumVariance(AbstractPortfolio):
    def allocate_weights(self, target_return=None):
        """
        Compute the Minimum Variance Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.

        Returns:
        - pd.Series: Optimal portfolio weights for minimum variance.
        """

        # Compute covariance matrix
        returns = self.df_returns
        cov_matrix = returns.cov()

        # Define the objective function (Minimize portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(returns.columns))]

        # Initial guess (equal weights)
        initial_weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Optimization
        result = minimize(
            objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            raise ValueError("Optimization failed.")


class MaximumReturn(AbstractPortfolio):
    def allocate_weights(self):
        """
        Compute the Maximum Return Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.

        Returns:
        - pd.Series: Optimal portfolio weights for maximum return.
        """

        # Compute mean returns
        returns = self.df_returns
        mean_returns = returns.mean()

        # Define the objective function (Maximize portfolio return)
        def objective(weights):
            return -np.dot(
                weights, mean_returns
            )  # Negative because we minimize by default

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(returns.columns))]

        # Initial guess (equal weights)
        initial_weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Optimization
        result = minimize(
            objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            raise ValueError("Optimization failed.")


class TangencyPortfolio(AbstractPortfolio):
    def allocate_weights(self, risk_free_rate=0.0):
        """
        Compute the Tangency (Maximum Sharpe Ratio) Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

        Returns:
        - pd.Series: Optimal portfolio weights for maximum Sharpe ratio.
        """

        # Compute mean returns and covariance matrix
        returns = self.df_returns
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Define the objective function (Maximize Sharpe Ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative because we minimize by default

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(returns.columns))]

        # Initial guess (equal weights)
        initial_weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Optimization
        result = minimize(
            objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            raise ValueError("Optimization failed.")
