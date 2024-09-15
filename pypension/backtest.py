import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BacktestResult:
    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights

    def plot_portfolio_returns(self, title=None):
        """
        Plots the daily returns, cumulative growth, and the underwater plot (drawdowns) of the portfolio.

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

        # Calculate trailing 252-day realized volatility (annualized)
        rolling_volatility = portfolio_returns.rolling(window=252).std() * np.sqrt(252)

        # Set up the plot style
        sns.set(style="whitegrid")

        # Create a figure with 4 subplots (daily returns, cumulative growth, drawdowns, and trailing volatility)
        fig, ax = plt.subplots(
            4, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [2, 2, 1, 1]}
        )

        # Plot portfolio daily returns
        ax[0].plot(
            portfolio_returns.index,
            portfolio_returns,
            color="b",
            lw=1.5,
            label="Daily Returns",
        )
        ax[0].set_title("Portfolio Daily Returns", fontsize=14)
        ax[0].set_ylabel("Daily Return")
        ax[0].legend()
        ax[0].grid(True)

        # Plot cumulative growth (portfolio + individual assets)
        ax[1].plot(
            portfolio_cumulative_returns.index,
            portfolio_cumulative_returns,
            color="g",
            lw=2,
            label="Portfolio",
            zorder=2,
        )

        # Plot each individual asset's cumulative growth with thinner and fainter lines
        for i, col in enumerate(assets_cumulative_returns.columns):
            ax[1].plot(
                assets_cumulative_returns.index,
                assets_cumulative_returns[col],
                lw=1,
                alpha=0.6,
                label=col,
                zorder=1,
            )

        ax[1].set_title(
            "Portfolio Cumulative Growth vs. Individual Assets", fontsize=14
        )
        ax[1].set_ylabel("Cumulative Growth")
        ax[1].legend()
        ax[1].grid(True)

        # Plot underwater (drawdowns) plot
        ax[2].plot(drawdowns.index, drawdowns, color="r", lw=1.5, label="Drawdown")
        ax[2].set_title("Portfolio Drawdowns (Underwater Plot)", fontsize=14)
        ax[2].set_ylabel("Drawdown")
        ax[2].set_xlabel("Date")
        ax[2].legend()
        ax[2].grid(True)

        # Plot trailing 252-day realized volatility (annualized)
        ax[3].plot(
            rolling_volatility.index,
            rolling_volatility,
            color="purple",
            lw=1.5,
            label="Annualized Volatility (252-day)",
        )
        ax[3].set_title(
            "Trailing 252-Day Realized Volatility (Annualized)", fontsize=14
        )
        ax[3].set_ylabel("Volatility")
        ax[3].set_xlabel("Date")
        ax[3].legend()
        ax[3].grid(True)

        if title is not None:
            plt.suptitle(title)

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()

    def summarise_portfolio_performance(self, risk_free_rate=0):
        """
        Generates a table showing the annualized return, annualized volatility, Sharpe ratio, and maximum drawdown
        for each discrete year, including Year-To-Date (YTD) if the current year is incomplete.

        Parameters:
        - returns: A DataFrame where each column is the daily returns of an individual asset.
        - weights: A numpy array representing the portfolio weights for each asset.
        - risk_free_rate: The risk-free rate used in Sharpe ratio calculation (default is 0).

        Returns:
        A DataFrame summarizing the portfolio statistics for each year.
        """
        # Ensure weights are a numpy array
        weights = np.array(self.weights)

        # Calculate portfolio returns by multiplying individual asset returns by weights and summing them
        returns = self.returns.loc[:, self.weights.index]
        portfolio_returns = returns.dot(weights)

        # Resample the data to daily frequency to ensure it's daily returns
        portfolio_returns = portfolio_returns.resample("D").ffill().dropna()

        # Group the returns by year
        yearly_groups = portfolio_returns.groupby(portfolio_returns.index.year)

        # Create an empty list to store the results
        results = []

        # Loop through each year and calculate the metrics
        for year, yearly_returns in yearly_groups:
            # Annualized return: (1 + avg daily return)^252 - 1
            total_return = (1 + yearly_returns).prod() - 1
            trading_days = yearly_returns.count()
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1

            # Annualized volatility: std dev of daily returns * sqrt(252)
            annualized_volatility = yearly_returns.std() * np.sqrt(252)

            # Sharpe ratio: (annualized return - risk-free rate) / annualized volatility
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

            # Maximum drawdown: maximum peak-to-trough decline
            cumulative_returns = (1 + yearly_returns).cumprod()
            cumulative_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
            max_drawdown = drawdowns.min()

            # Store the result in a list (for now)
            results.append(
                [
                    year,
                    annualized_return,
                    annualized_volatility,
                    sharpe_ratio,
                    max_drawdown,
                ]
            )

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(
            results,
            columns=[
                "Year",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
            ],
        )

        # Set 'Year' as the index
        results_df.set_index("Year", inplace=True)

        return results_df
