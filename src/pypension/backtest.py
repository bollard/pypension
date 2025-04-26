from collections import defaultdict
from functools import partial
from typing import Callable

import matplotlib.colors as mcolours
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import pypension.analytics as pa


class BacktestResult:
    def __init__(self, asset_returns: pd.DataFrame, asset_weights: pd.DataFrame):
        self.asset_returns = asset_returns
        self.asset_weights = asset_weights

    @staticmethod
    def _plot_cumulative_growth_and_drawdown(
        ax: plt.Axes,
        ser_equity_curve: pd.Series,
        ser_portfolio_drawdowns: pd.Series,
        assets_cumulative_returns: pd.DataFrame = None,
    ):
        # Plot the cumulative growth of the portfolio (thicker)
        ax.plot(
            ser_equity_curve.index,
            ser_equity_curve,
            color="g",
            lw=2,
            label=ser_equity_curve.name,
        )

        # Plot each individual asset's cumulative growth (thinner, fainter)
        if assets_cumulative_returns is not None:
            for i, col in enumerate(assets_cumulative_returns.columns):
                ax.plot(
                    assets_cumulative_returns.index,
                    assets_cumulative_returns[col],
                    lw=1,
                    alpha=0.6,
                    label=col,
                )

        ax.set_title("Cumulative Growth and Drawdowns")
        ax.set_ylabel("Cumulative Growth")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)  # Add faint grid lines

        # Create a twiny axis for the drawdown plot
        ax_dd = ax.twinx()
        ax_dd.fill_between(
            ser_portfolio_drawdowns.index,
            ser_portfolio_drawdowns,
            color="red",
            alpha=0.3,
        )
        ax_dd.set_xlim(ax.get_xlim())  # Ensure both x-axes share the same range
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    @staticmethod
    def _plot_asset_weights_over_time(ax: plt.Axes, weights_df: pd.DataFrame):
        weights_df.plot.area(ax=ax, stacked=True, alpha=0.6)

        ax.set_title("Portfolio Asset Weights Over Time")
        ax.set_ylabel("Weight")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, alpha=0.3)  # Add faint grid lines
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)

    @staticmethod
    def _column_format_func(
        df: pd.DataFrame,
        column_format: dict[str, str] = None,
        default_format: str = "{:,.2%}",
    ) -> dict[str, Callable]:
        column_default = defaultdict(lambda: default_format)
        column_format_with_default = column_default | (column_format or {})

        def safe_format(value, fmt: str):
            if pd.isna(value):
                return "-"
            try:
                return fmt.format(value)
            except (ValueError, TypeError):
                return value

        # use `functools.partial` to bind the format string for each column
        format_func = {
            column: partial(safe_format, fmt=column_format_with_default[column])
            for column in df.columns
        }

        return format_func

    @classmethod
    def _plot_summary_statistics_table(
        cls, ax: plt.Axes, df_stats: pd.DataFrame, column_format: dict[str, str] = None
    ):
        format_func = cls._column_format_func(df_stats, column_format)

        ax.table(
            cellText=df_stats.transform(format_func).values,
            rowLabels=df_stats.index,
            colLabels=df_stats.columns,
            cellLoc="center",
            loc="center",
            alpha=0.3,
        )

        ax.axis("tight")
        ax.axis("off")

    @classmethod
    def _plot_monthly_returns_table(
        cls,
        ax: plt.Axes,
        df_returns_table: pd.DataFrame,
        n_years: int = 10,
        column_format: dict[str, str] = None,
    ):
        # put most recent years at the top of the table
        format_func = cls._column_format_func(df_returns_table, column_format)
        df_returns_table = df_returns_table.sort_index(ascending=False).head(n_years)

        # normalise returns (excluding annual, which would dominate) for colour map
        values = df_returns_table.fillna(0.0).to_numpy(np.float64)
        values[::, -1] = 0
        normalise = plt.Normalize(values.min(), values.max())

        # initialise colour map
        cmap = mcolours.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        colours = cmap(normalise(values))

        ax.table(
            cellText=df_returns_table.transform(format_func).values,
            rowLabels=df_returns_table.index,
            colLabels=df_returns_table.columns,
            cellLoc="center",
            cellColours=colours,
            loc="center",
            alpha=0.3,
        )

        ax.axis("tight")
        ax.axis("off")

    def plot_portfolio_returns(self, label: str = None) -> plt.Figure:
        if label is None:
            label = "Portfolio"

        if label in self.asset_weights.columns:
            label = f"{label} Portfolio"

        assert label not in self.asset_weights.columns

        # put asset weights (updated monthly) onto same grid as returns (updated daily)
        df_weights = self.asset_weights.reindex(
            self.asset_returns.index, method="bfill"
        ).ffill()

        # append (daily) portfolio returns
        df_returns = self.asset_returns.loc[:, df_weights.columns]
        df_returns[label] = (df_weights * df_returns).sum(axis="columns")

        # calculate cumulative returns (equity curve)
        df_equity_curve = df_returns.apply(pa.compute_equity_curve)

        # calculate (daily) drawdowns
        df_drawdowns = df_returns.apply(pa.compute_drawdowns)

        # calculate monthly portfolio returns
        df_portfolio_returns_monthly = pa.pivot_monthly_returns(df_returns[label])

        # summary statistics
        df_summary_statistics = df_returns.apply(
            pa.compute_summary_statistics, n_annual_returns=5
        ).T

        # prepare figure (A4 size)
        plt.rcParams.update({"font.size": 8})
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(11.69, 8.27),
            gridspec_kw={"height_ratios": [2, 2, 1, 1]},
            layout="constrained",
        )

        # 1) cumulative growth and drawdown (line plot)
        self._plot_cumulative_growth_and_drawdown(
            axs[0], df_equity_curve[label], df_drawdowns[label]
        )

        # 2) stats table
        self._plot_summary_statistics_table(
            axs[1],
            df_summary_statistics,
            column_format={"SR": "{:,.2f}", "CAGR / DD": "{:,.2f}"},
        )

        # 3) asset weights over time (area plot)
        self._plot_asset_weights_over_time(axs[2], df_weights)

        # 4) monthly returns table (table)
        self._plot_monthly_returns_table(axs[3], df_portfolio_returns_monthly)

        plt.suptitle(label)

        return fig
