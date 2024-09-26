import pandas as pd
import yfinance as yf

from pypension.allocation_methods import (
    EqualWight,
    HierarchicalRiskParity,
    RiskParity,
    TangencyPortfolio,
)
from pypension.backtest import BacktestResult
from pypension.config import END_DATE, START_DATE, TICKERS


def main() -> None:
    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE).convert_dtypes()
    df_returns = df_data["Adj Close"].apply(
        lambda x: x[~x.isna()].pct_change()
    )  # in decimal

    ser_rebalance_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="BME")

    portfolios = {
        "VTI": RiskParity(df_returns.loc[:, ["VTI", "VT"]]),
        "Test": EqualWight(df_returns.loc[:, ["VTI"]]),
        "HRP": HierarchicalRiskParity(df_returns),
        "Risk Parity": RiskParity(df_returns),
        "Tangency Portfolio": TangencyPortfolio(df_returns),
        "EqualWight": EqualWight(df_returns),
    }

    for label, portfolio in portfolios.items():
        df_weights = portfolio.allocate_weights(rebalance_dates=ser_rebalance_dates)

        result = BacktestResult(portfolio.asset_returns, df_weights)
        result.plot_portfolio_returns(label)


if __name__ == "__main__":
    main()
