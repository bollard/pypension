import matplotlib.pyplot as plt
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
from pypension.config import END_DATE, PLOT_DIR, START_DATE, TICKERS, TIME_ZONE


def main() -> None:
    df_data = (
        # ignore dividends (& splits?) to more closely match online sources
        yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
        .convert_dtypes()
        .tz_localize(TIME_ZONE)
    )
    df_returns = df_data["Close"].apply(
        lambda x: x[~x.isna()].pct_change()
    )  # in decimal

    weights: dict[str, float] = {
        "BUT.L": 0.17,
        "JGGI.L": 0.17,
        "PCT.L": 0.075,
        "SMT.L": 0.075,
        "VT": 0.17,
        "FCIT.L": 0.17,
        "ALW.L": 0.17,
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

    for i, (label, portfolio) in enumerate(portfolios.items(), start=1):
        print(f"Running [{label}] ({i}/{len(portfolios)})")
        df_weights = portfolio.allocate_weights(rebalance_dates=ser_rebalance_dates)

        result = BacktestResult(portfolio.asset_returns, df_weights)
        fig = result.plot_portfolio_returns(label)

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_DIR / f"{label}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
