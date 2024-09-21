if __name__ == "__main__":
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

    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE).convert_dtypes()
    df_returns = df_data["Adj Close"].apply(
        lambda x: x[~x.isna()].pct_change()
    )  # in decimal

    ser_rebalance_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="BME")

    methods = {
        "VTI": RiskParity(
            df_returns.loc[:, ["VTI", "VT"]], rebalance_dates=ser_rebalance_dates
        ),
        "Test": EqualWight(
            df_returns.loc[:, ["VTI"]], rebalance_dates=ser_rebalance_dates
        ),
        "HRP": HierarchicalRiskParity(df_returns, rebalance_dates=ser_rebalance_dates),
        "Risk Parity": RiskParity(df_returns, rebalance_dates=ser_rebalance_dates),
        "Tangency Portfolio": TangencyPortfolio(
            df_returns, rebalance_dates=ser_rebalance_dates
        ),
        "EqualWight": EqualWight(df_returns, rebalance_dates=ser_rebalance_dates),
    }

    for label, method in methods.items():
        df_weights = method.allocate_weights()

        result = BacktestResult(method.asset_returns, df_weights)
        result.plot_portfolio_returns(label)
