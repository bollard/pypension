import importlib.resources
from zoneinfo import ZoneInfo

import pandas as pd

TICKERS = [
    "0P0000XYWM.L",  # Royal London Sustainable Leaders C Acc
    "0P0001FE43.L",  # Rathbone Global Opportunities Fund S Acc
    "0P00017461.L",  # Premier Miton European Opports B Acc
    "SMT.L",  # Scottish Mortgage Ord
    "EWI.L",  # Edinburgh Worldwide Ord
    "PHI.L",  # Pacific Horizon Ord
    "VTI",  # Vanguard Total Stock Market Index Fund ETF Shares
    "VT",  # Vanguard Total World Stock Index Fund ETF Shares
    "BUT.L",  # The Brunner Investment Trust PLC
    "CGT.L",  # Capital Gearing Trust p.l.c
    "FCIT.L",  # F&C Investment Trust PLC
    "JGGI.L",  # JPMorgan Global Growth & Income plc
    "PCT.L",  # Polar Capital Technology Trust plc
    "PNL.L",  # Personal Assets Trust plc
    "RICA.L",  # Ruffer Investment Company Limited
    "QQQM",  # Invesco NASDAQ 100 ETF
    "VXUS",  # Vanguard Total International Stock Index Fund ETF Shares
    "BRK-B",  # Berkshire Hathaway Inc.
    "MYI.L",  # Murray International Trust PLC
    "MWY.L",  # Mid Wynd International Invest Trust PLC
    "ALW.L",  # Alliance Witan Ord (ALW.L)
]

TIME_ZONE = ZoneInfo("UTC")
END_DATE = pd.Timestamp.now(tz=TIME_ZONE).normalize()
START_DATE = (END_DATE - pd.DateOffset(years=10)) - pd.offsets.MonthBegin()

PLOT_DIR = importlib.resources.files(__package__).joinpath("plots")
