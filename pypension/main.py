if __name__ == "__main__":
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt

    from pypension.allocation_methods import (
        EqualWight,
        HierarchicalRiskParity,
        RiskParity,
        TangencyPortfolio,
    )
    from pypension.backtest import BacktestResult
    from pypension.config import END_DATE, START_DATE, TICKERS

    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE).convert_dtypes()
    df_returns = df_data["Adj Close"].apply(lambda x: x[~x.isna()].pct_change()) # in decimal

    methods = {
        # "HRP": HierarchicalRiskParity(df_returns),
        # "Risk Parity": RiskParity(df_returns),
        # "Tangency Portfolio": TangencyPortfolio(df_returns),
        # "EqualWight": EqualWight(df_returns),
        # "VTI": RiskParity(df_returns.loc[:, ["VTI", "VT"]]),
        "Test": EqualWight(df_returns.loc[:, ["VTI"]])
    }

    for label, method in methods.items():
        weights = method.allocate_weights()

        result = BacktestResult(df_returns, weights)
        result.plot_portfolio_returns(label)
