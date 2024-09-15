if __name__ == "__main__":
    import yfinance as yf

    from pypension.allocation_methods import (
        EqualWight,
        HierarchicalRiskParity,
        RiskParity,
        TangencyPortfolio,
    )
    from pypension.backtest import BacktestResult
    from pypension.config import END_DATE, START_DATE, TICKERS

    # Download historical price data
    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    df_returns = df_data["Adj Close"].pct_change().dropna()

    methods = {
        "HRP": HierarchicalRiskParity,
        "Risk Parity": RiskParity,
        "Tangency Portfolio": TangencyPortfolio,
        "EqualWight": EqualWight,
    }

    for label, klass in methods.items():
        instance = klass(df_returns)
        weights = instance.allocate_weights()

        BacktestResult(df_returns, weights).plot_portfolio_returns(label)
