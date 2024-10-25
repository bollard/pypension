import pandas as pd
import yfinance as yf

from pypension.allocation_methods import (
    EqualWight,
    FixedWeight,
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

    weights = {
        "BUT.L": 0.15,
        "JGGI.L": 0.15,
        "PCT.L": 0.075,
        "SMT.L": 0.075,
        "MYI.L": 0.075,
        "BRK-B": 0.075,
        "VT": 0.20,
        "VTI": 0.20,
    }
    ser_rebalance_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="BME")

    portfolios = {
        "Fixed Weight": FixedWeight(df_returns.loc[:, weights.keys()], weights=weights),
        "VTI": RiskParity(df_returns.loc[:, ["VTI", "VT"]]),
        "HRP": HierarchicalRiskParity(df_returns.loc[:, weights.keys()]),
        "Risk Parity": RiskParity(df_returns.loc[:, weights.keys()]),
        "Tangency Portfolio": TangencyPortfolio(df_returns.loc[:, weights.keys()]),
        "Equal Wight": EqualWight(df_returns.loc[:, weights.keys()]),
    }

    for label, portfolio in portfolios.items():
        df_weights = portfolio.allocate_weights(rebalance_dates=ser_rebalance_dates)

        result = BacktestResult(portfolio.asset_returns, df_weights)
        result.plot_portfolio_returns(label)


if __name__ == "__main__":
    main()
