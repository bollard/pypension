import pandas as pd
import yfinance as yf

from pypension.config import END_DTTM, START_DTTM, TIME_ZONE


def download(
    tickers: list[str], start_dttm: pd.Timestamp = None, end_dttm: pd.Timestamp = None
) -> pd.DataFrame:
    if start_dttm is None:
        start_dttm = START_DTTM

    if end_dttm is None:
        end_dttm = END_DTTM

    df_data = (
        # ignore dividends (& splits?) to more closely match online sources
        yf.download(tickers, start=start_dttm, end=end_dttm, auto_adjust=False)
        .convert_dtypes()
        .tz_localize(TIME_ZONE)
    )

    return clean(df_data)


def clean(df_data: pd.DataFrame) -> pd.DataFrame:
    overrides = [
        ("IGET.L", pd.Timestamp(2025, 1, 10, tzinfo=TIME_ZONE), 100),
        ("FCIT.L", pd.Timestamp(2025, 4, 25, tzinfo=TIME_ZONE), 100),
    ]

    for ticker, date, factor in overrides:
        if ticker in df_data.columns.get_level_values("Ticker"):
            df_data.loc[date, pd.IndexSlice[:, ticker]] *= factor

    return df_data
