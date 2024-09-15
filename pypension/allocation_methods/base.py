import abc

import numpy as np
import pandas as pd


class AbstractPortfolio(abc.ABC):
    def __init__(self, df_returns: pd.DataFrame, *args, **kwargs):
        self.df_returns = df_returns

    @staticmethod
    def calculate_returns(df_close: pd.DataFrame) -> pd.DataFrame:
        return df_close.pct_change().dropna()

    @abc.abstractmethod
    def allocate_weights(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_performance(returns, weights, periods_per_year=252):
        """
        Evaluate portfolio performance on both annualized and discrete annual basis.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - weights (pd.Series): Portfolio weights.
        - periods_per_year (int): Number of periods per year (default is 252 trading days).

        Returns:
        - dict: Portfolio performance metrics including annualized return, annualized volatility, and Sharpe ratio.
        """
        # Portfolio returns
        portfolio_daily_returns = returns @ weights
        annualized_return = portfolio_daily_returns.mean() * periods_per_year

        # Portfolio volatility (annualized)
        annualized_volatility = portfolio_daily_returns.std() * np.sqrt(
            periods_per_year
        )

        # Portfolio Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
        sharpe_ratio = annualized_return / annualized_volatility

        # Discrete annual returns and volatility
        discrete_annual_returns = portfolio_daily_returns.groupby(
            portfolio_daily_returns.index.year
        ).apply(lambda x: (1 + x).prod() - 1)
        discrete_annual_volatility = portfolio_daily_returns.groupby(
            portfolio_daily_returns.index.year
        ).std()

        return {
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Discrete Annual Returns": discrete_annual_returns.mean(),
            "Discrete Annual Volatility": discrete_annual_volatility.mean(),
        }
