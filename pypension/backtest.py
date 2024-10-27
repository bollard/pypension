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
        ax1,
        portfolio_cumulative_returns,
        portfolio_drawdowns,
        assets_cumulative_returns=None,
    ):
        """
        Plots cumulative growth for the portfolio and individual assets, with drawdowns using twiny.
        """
        # Plot the cumulative growth of the portfolio (thicker)
        ax1.plot(
            portfolio_cumulative_returns.index,
            portfolio_cumulative_returns,
            color="g",
            lw=2,
            label="Portfolio",
        )

        # Plot each individual asset's cumulative growth (thinner, fainter)
        if assets_cumulative_returns is not None:
            for i, col in enumerate(assets_cumulative_returns.columns):
                ax1.plot(
                    assets_cumulative_returns.index,
                    assets_cumulative_returns[col],
                    lw=1,
                    alpha=0.6,
                    label=col,
                )

        ax1.set_title("Cumulative Growth and Drawdowns")
        ax1.set_ylabel("Cumulative Growth")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)  # Add faint grid lines

        # Create a twiny axis for the drawdown plot
        ax2 = ax1.twinx()
        ax2.fill_between(
            portfolio_drawdowns.index, portfolio_drawdowns, color="red", alpha=0.3
        )
        ax2.set_xlim(ax1.get_xlim())  # Ensure both x-axes share the same range
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    @staticmethod
    def _plot_discrete_annual_performance(ax, annual_returns, asset_annual_returns):
        """
        Plots discrete annual performance for both the portfolio and individual assets as a bar plot.
        """
        combined_annual_returns = pd.concat(
            [annual_returns, asset_annual_returns], axis="columns"
        )
        combined_annual_returns.columns = ["Portfolio"] + list(
            asset_annual_returns.columns
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
        ax, rolling_30_volatility, rolling_30_return, rolling_30_sharpe
    ):
        """
        Plots rolling 30-day volatility, return, and Sharpe ratio for the portfolio.
        """
        ax.plot(
            rolling_30_volatility.index,
            rolling_30_volatility,
            color="purple",
            label="30-Day Rolling Volatility (Annualized)",
        )
        ax.plot(
            rolling_30_return.index,
            rolling_30_return,
            color="blue",
            label="30-Day Rolling Return (Annualized)",
        )

        ax_sr = ax.twinx()
        ax_sr.plot(
            rolling_30_sharpe.index,
            rolling_30_sharpe,
            color="orange",
            label="30-Day Rolling Sharpe Ratio",
        )
        ax_sr.set_ylabel("Sharpe Ratio")
        ax_sr.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

        ax.set_title("Rolling 30-Day Volatility, Return, and Sharpe Ratio (Annualized)")
        ax.set_ylabel("Metrics")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)  # Add faint grid lines

    @staticmethod
    def _plot_asset_weights_over_time(ax, weights_df):
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
    def _plot_monthly_returns_table(ax, monthly_returns_table, n=5):
        """
        Plots a table inside the figure showing monthly returns and annual totals.
        """
        monthly_returns_table = monthly_returns_table.sort_index(ascending=False).head(
            n
        )
        monthly_returns_table = monthly_returns_table.mul(100).round(2)

        values = monthly_returns_table.fillna(0.0).to_numpy(np.float64)
        values[::, -1] = 0  # otherwise annual values will dominate
        normalise = plt.Normalize(values.min(), values.max())

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)

        colours = cmap(normalise(values))

        ax.axis("tight")
        ax.axis("off")
        ax.table(
            cellText=monthly_returns_table.values,
            rowLabels=monthly_returns_table.index,
            colLabels=monthly_returns_table.columns,
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

        # Ensure weights are a numpy array
        df_weights = self.asset_weights.reindex(
            self.asset_returns.index, method="bfill"
        )

        # Calculate portfolio returns by multiplying individual asset returns by weights and summing them
        df_returns = self.asset_returns.loc[:, df_weights.columns]
        portfolio_returns = (df_weights * df_returns).sum(axis="columns")

        # Calculate cumulative returns (Cumulative growth) for the portfolio and each individual asset
        portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
        assets_cumulative_returns = (1 + self.asset_returns).cumprod()

        # Calculate drawdowns
        cumulative_max = portfolio_cumulative_returns.cummax()
        portfolio_drawdowns = (
            portfolio_cumulative_returns - cumulative_max
        ) / cumulative_max
        portfolio_drawdowns = portfolio_drawdowns.ffill().fillna(0)

        # Calculate annual discrete performance
        annual_returns = portfolio_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        asset_annual_returns = self.asset_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )

        # Calculate rolling 30-day metrics (not annualised)
        idx = ~portfolio_returns.isna()
        rolling_30_volatility = (
            portfolio_returns.loc[idx].rolling(window=30).std() * np.sqrt(252)
        ).reindex(idx.index)
        rolling_30_return = (
            portfolio_returns.loc[idx]
            .rolling(window=30)
            .apply(lambda x: (1 + x).prod() - 1)
        ).reindex(idx.index)
        rolling_30_sharpe = (rolling_30_return / rolling_30_volatility).ffill()

        # Calculate monthly returns
        portfolio_returns_df = portfolio_returns.to_frame(
            name="Portfolio"
        )  # Convert Series to DataFrame
        monthly_returns = portfolio_returns_df.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )

        # Pivot table for monthly returns with year as rows and month as columns
        monthly_returns_table = monthly_returns.pivot_table(
            index=monthly_returns.index.year,
            columns=monthly_returns.index.month,
            values="Portfolio",
        )

        # Add a final column for annual returns
        annual_returns_by_year = portfolio_returns_df.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns_table["Annual"] = annual_returns_by_year.values

        # Set column names for the months
        months = [month for month in calendar.month_abbr if len(month)]
        monthly_returns_table.columns = months + ["Annual"]

        # Create the figure and axes for the subplots (A4 size)
        plt.rcParams.update({"font.size": 8})

        fig, axs = plt.subplots(
            5,
            1,
            figsize=(11.69, 8.27),
            gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]},
            layout="constrained",
        )

        # ---- First Plot: Cumulative Growth and Drawdown (Subplot) ----
        self._plot_cumulative_growth_and_drawdown(
            axs[0], portfolio_cumulative_returns, portfolio_drawdowns
        )

        # ---- Second Plot: Discrete Annual Performance (Bar Plot) ----
        self._plot_discrete_annual_performance(
            axs[1], annual_returns, asset_annual_returns
        )

        # ---- Third Plot: Rolling 30-Day Volatility, Return, and Sharpe Ratio ----
        self._plot_rolling_metrics(
            axs[2], rolling_30_volatility, rolling_30_return, rolling_30_sharpe
        )

        # ---- Fourth Plot: Asset Weights Over Time (Shaded Area Plot) ----
        self._plot_asset_weights_over_time(axs[3], df_weights)

        # ---- Fifth: Monthly Returns Table (Inside Plot) ----
        self._plot_monthly_returns_table(axs[4], monthly_returns_table)

        if label is not None:
            plt.suptitle(label)

        return fig
