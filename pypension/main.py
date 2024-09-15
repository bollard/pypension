if __name__ == "__main__":
    import yfinance as yf

    from pypension.allocation_methods import (
        EqualWight,
        HierarchicalRiskParity,
        RiskParity,
        TangencyPortfolio,
    )
    from pypension.config import END_DATE, START_DATE, TICKERS

    # Download historical price data
    df_data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    df_returns = df_data["Adj Close"].pct_change().dropna()

    hrp = HierarchicalRiskParity(df_returns)
    hrp_weights = hrp.allocate_weights()
    print(hrp.evaluate_performance(df_returns, hrp_weights))

    rp = RiskParity(df_returns)
    rp_weights = rp.allocate_weights()
    print(rp.evaluate_performance(df_returns, rp_weights))

    mv = TangencyPortfolio(df_returns)
    mv_weights = mv.allocate_weights()
    print(mv.evaluate_performance(df_returns, mv_weights))

    eq = EqualWight(df_returns)
    eq_weights = eq.allocate_weights()
    print(eq.evaluate_performance(df_returns, eq_weights))
