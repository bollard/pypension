import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestResult:
    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights

    # Define the private plotting functions for each type of plot
    @staticmethod
    def _plot_cumulative_growth_and_drawdown(
        ax1, portfolio_cumulative_returns, assets_cumulative_returns, drawdowns
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
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)  # Add faint grid lines

        # Create a twiny axis for the drawdown plot
        ax2 = ax1.twiny()
        ax2.fill_between(drawdowns.index, drawdowns, color="red", alpha=0.3)
        ax2.set_xlim(ax1.get_xlim())  # Ensure both x-axes share the same range
        ax2.set_ylabel("Drawdown")

    @staticmethod
    def _plot_discrete_annual_performance(ax, annual_returns, asset_annual_returns):
        """
        Plots discrete annual performance for both the portfolio and individual assets as a bar plot.
        """
        combined_annual_returns = pd.concat(
            [annual_returns, asset_annual_returns], axis=1
        )
        combined_annual_returns.columns = ["Portfolio"] + list(
            asset_annual_returns.columns
        )

        # Plot as a bar chart
        combined_annual_returns.plot(kind="bar", ax=ax)
        ax.set_title("Discrete Annual Performance")
        ax.set_ylabel("Return")
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

        ax.set_title("Rolling 30-Day Volatility, Return, and Sharpe Ratio (Annualized)")
        ax.set_ylabel("Metrics")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)  # Add faint grid lines

    @staticmethod
    def _plot_asset_weights_over_time(ax, returns, weights):
        """
        Plots a shaded area plot showing the evolution of asset weights over time.
        """
        weights_df = pd.DataFrame(
            np.repeat(weights[np.newaxis, :], len(returns), axis=0),
            index=returns.index,
            columns=returns.columns,
        )
        weights_df.plot.area(ax=ax, stacked=True, alpha=0.6)

        ax.set_title("Portfolio Asset Weights Over Time")
        ax.set_ylabel("Weight")

    @staticmethod
    def _plot_monthly_returns_table(ax, monthly_returns_table):
        """
        Plots a table inside the figure showing monthly returns and annual totals.
        """
        monthly_returns_table = monthly_returns_table.mul(100).round(2)

        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=monthly_returns_table.values,
            rowLabels=monthly_returns_table.index,
            colLabels=monthly_returns_table.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

    # Main function to plot the portfolio returns and metrics
    def plot_portfolio_returns(self, label: str = None):
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
        weights = np.array(self.weights)

        # Calculate portfolio returns by multiplying individual asset returns by weights and summing them
        returns = self.returns.loc[:, self.weights.index]
        portfolio_returns = returns.dot(weights)

        # Calculate cumulative returns (Cumulative growth) for the portfolio and each individual asset
        portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
        assets_cumulative_returns = (1 + returns).cumprod()

        # Calculate drawdowns
        cumulative_max = portfolio_cumulative_returns.cummax()
        drawdowns = (portfolio_cumulative_returns - cumulative_max) / cumulative_max

        # Calculate annual discrete performance
        annual_returns = portfolio_returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )
        asset_annual_returns = returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        )

        # Calculate rolling 30-day metrics
        rolling_30_volatility = portfolio_returns.rolling(window=30).std() * np.sqrt(
            252
        )
        rolling_30_return = portfolio_returns.rolling(window=30).apply(
            lambda x: (1 + x).prod() ** (252 / 30) - 1
        )
        rolling_30_sharpe = rolling_30_return / rolling_30_volatility

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
        monthly_returns_table.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "Annual",
        ]

        # Create the figure and axes for the subplots (A4 size)
        fig, axs = plt.subplots(
            5,
            1,
            figsize=(11.69, 8.27),
            gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]},
            layout="constrained",
        )

        # ---- First Plot: Cumulative Growth and Drawdown (Subplot) ----
        self._plot_cumulative_growth_and_drawdown(
            axs[0], portfolio_cumulative_returns, assets_cumulative_returns, drawdowns
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
        self._plot_asset_weights_over_time(axs[3], returns, weights)

        # ---- Fifth: Monthly Returns Table (Inside Plot) ----
        self._plot_monthly_returns_table(axs[4], monthly_returns_table)

        if label is not None:
            plt.suptitle(label)

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
