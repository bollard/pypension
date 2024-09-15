import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pypension.allocation_methods.base import AbstractPortfolio


class RiskBudgeting(AbstractPortfolio):
    def allocate_weights(self, target_risk_contributions: dict[str, np.float32]):
        """
        Compute the Risk Budgeting Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - target_risk_contributions (dict): Desired risk contributions as a proportion (sum to 1).

        Returns:
        - pd.Series: Optimal portfolio weights for risk budgeting.
        """

        # Compute covariance matrix
        returns = self.df_returns
        cov_matrix = returns.cov()

        # Define the objective function (Minimize the deviation of risk contributions from target)
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contribs = np.dot(cov_matrix, weights)
            total_contribs = weights * marginal_contribs / portfolio_variance
            risk_contribs = total_contribs / total_contribs.sum()
            deviation = np.array(
                [
                    risk_contribs[i] - target_risk_contributions[asset]
                    for i, asset in enumerate(returns.columns)
                ]
            )
            return np.sum(deviation**2)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(returns.columns))]

        # Initial guess (equal weights)
        initial_weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Optimization
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            raise ValueError("Optimization failed.")


class RiskParity(RiskBudgeting):
    def allocate_weights(self):
        tickers = self.df_returns.columns
        equal = {ticker: 1 / len(tickers) for ticker in tickers}
        return super().allocate_weights(target_risk_contributions=equal)
