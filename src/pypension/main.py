import warnings

import matplotlib.pyplot as plt
import pandas as pd

from pypension.allocation_methods import (
    EqualWight,
    FixedWeight,
    HierarchicalRiskParity,
    RiskParity,
    TangencyPortfolio,
)
from pypension.allocation_methods.base import AbstractPortfolio
from pypension.backtest import BacktestResult
from pypension.config import END_DTTM, PLOT_DIR, START_DTTM
from pypension.data import download


def main() -> None:
    # define fixed weight portfolio
    weights: dict[str, float] = {
        # index
        "VT": 0.30,
        "FCIT.L": 0.15,
        "ALW.L": 0.15,
        # active
        "BRK-B": 0.125,
        "JGGI.L": 0.125,
        # tech
        "PCT.L": 0.075,
        "SMT.L": 0.075,
    }

    # download close prices
    tickers: list[str] = list(weights.keys())
    tickers_vt = ["VT", "VTI"]
    tickers_download = list(set(tickers) | set(tickers_vt))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
        df_data = download(tickers_download, start_dttm=START_DTTM, end_dttm=END_DTTM)

    # compute daily percentage changes (in decimal)
    df_close: pd.DataFrame = df_data.loc[:, pd.IndexSlice["Close"]]
    df_returns = df_close.apply(lambda x: x[~x.isna()].pct_change())

    # define a (month end) rebalance schedule
    ser_rebalance_dates = pd.date_range(start=START_DTTM, end=END_DTTM, freq="BME")

    # define portfolio allocation methods
    portfolios: dict[str, AbstractPortfolio] = {
        "Fixed Weight": FixedWeight(df_returns.loc[:, tickers], weights=weights),
        "VTI": RiskParity(df_returns.loc[:, tickers_vt]),
        "HRP": HierarchicalRiskParity(df_returns.loc[:, tickers]),
        "Risk Parity": RiskParity(df_returns.loc[:, tickers]),
        "Tangency Portfolio": TangencyPortfolio(df_returns.loc[:, tickers]),
        "Equal Wight": EqualWight(df_returns.loc[:, tickers]),
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
