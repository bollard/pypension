import calendar
from typing import TypeVar

import numpy as np
import pandas as pd

P = TypeVar("P", pd.DataFrame, pd.Series)


def _resample_returns(ser_returns: P, freq: str) -> P:
    return ser_returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)


def _subset_returns(ser_returns: P, offset: str) -> P:
    tn = ser_returns.index[-1]

    t0 = {
        "YTD": tn - pd.offsets.YearBegin(),
        "MTD": tn - pd.offsets.MonthBegin(),
        "1Y": tn - pd.DateOffset(years=1),
        "3Y": tn - pd.DateOffset(years=3),
        "5Y": tn - pd.DateOffset(years=5),
    }[offset.upper()]

    idx = ser_returns.index >= t0
    return ser_returns.loc[idx,]


def _compute_equity_curve(ser_returns: pd.Series) -> pd.Series:
    return (1 + ser_returns.fillna(0.0)).cumprod()


def _compute_total_return(ser_returns: pd.Series) -> np.float64:
    # calculate cumulative returns (equity curve)
    ser_equity_curve = _compute_equity_curve(ser_returns)

    # calculate total return
    equity_final, equity_initial = (
        ser_equity_curve.iloc[-1],
        ser_equity_curve.iloc[0],
    )

    return (equity_final - equity_initial) / equity_initial


def _compute_annualised_return(ser_returns: pd.Series) -> np.float64:
    # calculate cumulative returns (equity curve)
    ser_equity_curve = _compute_equity_curve(ser_returns)

    # calculate compound annual growth rate (cagr) / annualised return
    n_days = ser_equity_curve.index[-1] - ser_equity_curve.index[0]
    growth = ser_equity_curve.iloc[-1] / ser_equity_curve.iloc[0]
    cagr = growth ** (pd.Timedelta(days=365) / n_days) - 1

    return cagr


def _compute_annualised_volatility(ser_returns: pd.Series) -> np.float64:
    return ser_returns.std() * np.sqrt(252)


def _compute_drawdowns(ser_returns: pd.Series) -> pd.Series:
    # calculate cumulative returns (equity curve)
    ser_equity_curve = _compute_equity_curve(ser_returns)

    # calculate drawdowns
    ser_hwm = ser_equity_curve.cummax()
    ser_drawdowns = (ser_equity_curve - ser_hwm) / ser_hwm
    ser_drawdowns = ser_drawdowns.ffill().fillna(0)

    return ser_drawdowns


def _pivot_monthly_returns(ser_returns: pd.Series) -> pd.DataFrame:
    # calculate monthly returns
    ser_returns_monthly = _resample_returns(ser_returns, "ME")

    # convert to dataframe
    label = ser_returns_monthly.name
    df_returns_monthly = ser_returns_monthly.to_frame(label)

    # convert to pivot table of monthly returns (year as rows, month as columns)
    df_returns_monthly = df_returns_monthly.pivot_table(
        index=df_returns_monthly.index.year,
        columns=df_returns_monthly.index.month,
        values=label,
    )

    # add annual returns
    ser_returns_annual = _resample_returns(ser_returns, "YE")
    df_returns_monthly["Annual"] = ser_returns_annual.values

    # pretty column names
    columns = df_returns_monthly.columns
    columns = [calendar.month_abbr[c] if isinstance(c, int) else c for c in columns]
    df_returns_monthly.columns = columns

    return df_returns_monthly


def _compute_summary_statistics(ser_returns: pd.Series) -> pd.Series:
    stats = {
        "YTD": _compute_total_return(_subset_returns(ser_returns, "YTD")),
        "1Y": _compute_total_return(_subset_returns(ser_returns, "1Y")),
        "3Y": _compute_total_return(_subset_returns(ser_returns, "3Y")),
        "5Y": _compute_total_return(_subset_returns(ser_returns, "5Y")),
        "ITD": _compute_total_return(ser_returns),
        "CAGR": _compute_annualised_return(ser_returns),
        "Vol": _compute_annualised_volatility(ser_returns),
        "Max DD": _compute_drawdowns(_resample_returns(ser_returns, "ME")).min(),
    }

    stats["SR"] = stats["CAGR"] / stats["Vol"]

    return pd.Series(stats, name=ser_returns.name, dtype=ser_returns.dtype)
