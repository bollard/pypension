import calendar
import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


class BacktestResult:
    def __init__(self, asset_returns: pd.DataFrame, asset_weights: pd.DataFrame):
        self.asset_returns = asset_returns
        self.asset_weights = asset_weights

    # Define the private plotting functions for each type of plot
    @staticmethod
    def _plot_cumulative_growth_and_drawdown(
        ax: plt.Axes,
        ser_equity_curve: pd.Series,
        ser_portfolio_drawdowns: pd.Series,
        assets_cumulative_returns=None,
    ):
        """
        Plots cumulative growth for the portfolio and individual assets, with drawdowns using twiny.
        """
        # Plot the cumulative growth of the portfolio (thicker)
        ax.plot(
            ser_equity_curve.index,
            ser_equity_curve,
            color="g",
            lw=2,
            label="Portfolio",
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
    def _plot_discrete_annual_performance(
        ax, annual_returns: pd.Series, ser_asset_annual_returns: pd.Series
    ):
        """
        Plots discrete annual performance for both the portfolio and individual assets as a bar plot.
        """
        combined_annual_returns = pd.concat(
            [annual_returns, ser_asset_annual_returns], axis="columns"
        )
        combined_annual_returns.columns = ["Portfolio"] + list(
            ser_asset_annual_returns.columns
        )

        width = dt.timedelta(days=30)

        # Plot as a bar chart
        for i, asset in enumerate(combined_annual_returns.columns):
            offset = (i * width) - (
                (len(combined_annual_returns.columns) - 1) / 2 * width
            )
            ax.bar(
                combined_annual_returns.index + offset,
                combined_annual_returns.loc[:, asset],
                width=width,
                label=asset,
            )

        ax.set_title("Discrete Annual Performance")
        ax.set_ylabel("Return")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xticks(combined_annual_returns.index)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)  # Add faint grid lines

    @staticmethod
    def _plot_rolling_metrics(
        ax: plt.Axes,
        ser_rolling_volatility: pd.Series,
        ser_rolling_return: pd.Series,
        ser_rolling_sharpe: pd.Series,
    ):
        """
        Plots rolling 30-day volatility, return, and Sharpe ratio for the portfolio.
        """
        ax.plot(
            ser_rolling_volatility.index,
            ser_rolling_volatility,
            color="purple",
            label="30-Day Rolling Volatility (Annualized)",
        )

        ax_return = ax.twinx()
        ax_return.plot(
            ser_rolling_return.index,
            ser_rolling_return,
            color="orange",
            label="30-Day Rolling Return (Annualized)",
        )
        ax_return.set_ylabel("Sharpe Ratio")
        ax_return.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

        ax.set_title("Rolling 30-Day Volatility, Return, and Sharpe Ratio (Annualized)")
        ax.set_ylabel("Metrics")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)  # Add faint grid lines

    @staticmethod
    def _plot_asset_weights_over_time(ax: plt.Axes, weights_df: pd.DataFrame):
        """
        Plots a shaded area plot showing the evolution of asset weights over time.
        """

        weights_df.plot.area(ax=ax, stacked=True, alpha=0.6)

        ax.set_title("Portfolio Asset Weights Over Time")
        ax.set_ylabel("Weight")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, alpha=0.3)  # Add faint grid lines
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)

    @staticmethod
    def _plot_monthly_returns_table(
        ax: plt.Axes, df_portfolio_returns_monthly: pd.DataFrame, n_years: int = 10
    ):
        """
        Plots a table inside the figure showing monthly returns and annual totals.
        """
        df_portfolio_returns_monthly = df_portfolio_returns_monthly.sort_index(
            ascending=False
        ).head(n_years)
        df_portfolio_returns_monthly = df_portfolio_returns_monthly.mul(100).round(2)

        values = df_portfolio_returns_monthly.fillna(0.0).to_numpy(np.float64)
        values[::, -1] = 0  # otherwise annual values will dominate
        normalise = plt.Normalize(values.min(), values.max())

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)

        colours = cmap(normalise(values))

        ax.axis("tight")
        ax.axis("off")
        ax.table(
            cellText=df_portfolio_returns_monthly.map('{:,.2f}'.format).values,
            rowLabels=df_portfolio_returns_monthly.index,
            colLabels=df_portfolio_returns_monthly.columns,
            cellLoc="center",
            cellColours=colours,
            loc="center",
            alpha=0.3,
        )

    # Main function to plot the portfolio returns and metrics
    def plot_portfolio_returns(self, label: str = None) -> plt.Figure:
        """
        Plots various portfolio performance metrics including:
        - Cumulative growth and drawdown
        - Discrete annual performance
        - Rolling 30-day volatility, return, and Sharpe ratio
        - Asset weights over time
        - A table of monthly returns with annual totals inside the plot

        Parameters:
        - returns: A DataFrame where each column is the daily returns of an individual asset.
        - weights: A numpy array representing the portfolio weights for each asset.
        """

        # put asset weights (updated monthly) onto same grid as returns (updated daily)
        df_asset_weights = self.asset_weights.reindex(
            self.asset_returns.index, method="bfill"
        )

        # calculate (daily) portfolio returns
        df_asset_returns = self.asset_returns.loc[:, df_asset_weights.columns]
        ser_portfolio_returns = (df_asset_weights * df_asset_returns).sum(
            axis="columns"
        )

        # calculate cumulative portfolio returns (equity curve)
        ser_equity_curve = (1 + ser_portfolio_returns).cumprod()

        # calculate portfolio compound annual growth rate (cagr)
        n_years = (
            ser_equity_curve.index[-1] - ser_equity_curve.index[0]
        ) / pd.Timedelta(days=365)
        growth = ser_equity_curve.iloc[-1] / ser_equity_curve.iloc[0]
        cagr = growth ** (1 / n_years) - 1

        # calculate portfolio drawdowns
        ser_hwm = ser_equity_curve.cummax()
        ser_portfolio_drawdowns = (ser_equity_curve - ser_hwm) / ser_hwm
        ser_portfolio_drawdowns = ser_portfolio_drawdowns.ffill().fillna(0)

        # calculate discrete annual performance
        ser_portfolio_annual_returns = ser_portfolio_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        ser_asset_annual_returns = self.asset_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )

        # calculate rolling 30-day metrics (not annualised)
        idx = ~ser_portfolio_returns.isna()
        window = pd.Timedelta(days=30)

        ser_rolling_volatility = (
            ser_portfolio_returns.loc[idx].rolling(window=window).std() * np.sqrt(252)
        ).reindex(idx.index)

        ser_rolling_return = (
            ser_portfolio_returns.loc[idx]
            .rolling(window=window)
            .apply(lambda x: (1 + x).prod() - 1)
        ).reindex(idx.index)

        ser_rolling_sharpe = (ser_rolling_return / ser_rolling_volatility).ffill()

        # calculate monthly portfolio returns
        ser_portfolio_returns_monthly = ser_portfolio_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )

        # convert to pivot table of monthly returns (year as rows, month as columns)
        df_portfolio_returns_monthly = ser_portfolio_returns_monthly.to_frame(
            "Portfolio"
        ).pivot_table(
            index=ser_portfolio_returns_monthly.index.year,
            columns=ser_portfolio_returns_monthly.index.month,
            values="Portfolio",
        )

        # add portfolio annual returns
        ser_portfolio_returns_annual = ser_portfolio_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        df_portfolio_returns_monthly["Annual"] = ser_portfolio_returns_annual.values

        # pretty column names
        columns = df_portfolio_returns_monthly.columns
        columns = [calendar.month_abbr[c] if isinstance(c, int) else c for c in columns]
        df_portfolio_returns_monthly.columns = columns

        # prepare figure (A4 size)
        plt.rcParams.update({"font.size": 8})
        fig, axs = plt.subplots(
            nrows=5,
            ncols=1,
            figsize=(11.69, 8.27),
            gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]},
            layout="constrained",
        )

        # 1) cumulative growth and drawdown (line plot)
        self._plot_cumulative_growth_and_drawdown(
            axs[0], ser_equity_curve, ser_portfolio_drawdowns
        )

        # 2) discrete annual performance (bar plot)
        self._plot_discrete_annual_performance(
            axs[1], ser_portfolio_annual_returns, ser_asset_annual_returns
        )

        # 3) rolling 30-day volatility, return, sharpe ratio
        self._plot_rolling_metrics(
            axs[2], ser_rolling_volatility, ser_rolling_return, ser_rolling_sharpe
        )

        # 4) asset weights over time (area plot)
        self._plot_asset_weights_over_time(axs[3], df_asset_weights)

        # 5) monthly returns table (table)
        self._plot_monthly_returns_table(axs[4], df_portfolio_returns_monthly)

        plot_title = f"[{cagr:.2%}]"
        if label is not None:
            plot_title = f"{label} {plot_title}"
        plt.suptitle(plot_title)

        return fig
