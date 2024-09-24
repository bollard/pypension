import numpy as np
import pandas as pd
import scipy.optimize as spo

from pypension.allocation_methods.base import AbstractPortfolio


class RiskBudgeting(AbstractPortfolio):
    def allocate_weights_t(
        self,
        asset_returns: pd.DataFrame,
        target_risk_contributions: dict[str, float],
        **kwargs,
    ) -> pd.Series:
        """
        Compute the Risk Budgeting Portfolio.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - target_risk_contributions (dict): Desired risk contributions as a proportion (sum to 1).

        Returns:
        - pd.Series: Optimal portfolio weights for risk budgeting.
        """

        idx = asset_returns.isna().all()
        asset_returns_active = asset_returns.loc[:, ~idx]

        # Compute covariance matrix
        cov_matrix = self.calculate_covariance(asset_returns_active)

        # Define the objective function (Minimize the deviation of risk contributions from target)
        def objective(weights, cov_matrix, target_risk_contributions):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contribs = np.dot(cov_matrix, weights)
            total_contribs = weights * marginal_contribs / portfolio_variance
            risk_contribs = total_contribs / total_contribs.sum()
            deviation = np.array(
                [
                    risk_contribs[i] - target_risk_contributions[asset]
                    for i, asset in enumerate(cov_matrix.columns)
                ]
            )
            return np.sum(deviation**2)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no shorting (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(asset_returns_active.columns))]

        # Initial guess (equal weights)
        initial_weights = np.ones(len(asset_returns_active.columns)) / len(
            asset_returns_active.columns
        )

        # Optimization
        result = spo.minimize(
            objective,
            x0=initial_weights,
            args=(cov_matrix, target_risk_contributions),
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


class RiskParity(RiskBudgeting):
    def allocate_weights_t(self, asset_returns: pd.DataFrame, **kwargs) -> pd.Series:
        assets = asset_returns.columns
        equal_rc = {asset: 1 / len(assets) for asset in assets}
        return super().allocate_weights_t(
            asset_returns, target_risk_contributions=equal_rc
        )
