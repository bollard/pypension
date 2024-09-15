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

    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    df_returns = df_data["Adj Close"].pct_change().dropna()
    df_returns_all = df_returns.copy()

    # keep vt as a benchmark
    df_returns_bm = pd.DataFrame([df_returns.pop(x) for x in ["VT", "VTI"]]).T

    methods = {
        "HRP": HierarchicalRiskParity(df_returns),
        "Risk Parity": RiskParity(df_returns),
        "Tangency Portfolio": TangencyPortfolio(df_returns),
        "EqualWight": EqualWight(df_returns),
        "VTI": RiskParity(df_returns_bm),
    }

    for label, method in methods.items():
        weights = method.allocate_weights()

        result = BacktestResult(df_returns_all, weights)
        result.plot_portfolio_returns(label)
        result.summarise_portfolio_performance()
