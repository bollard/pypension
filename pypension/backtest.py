import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BacktestResult:
    def __init__(self, returns: pd.DataFrame, weights: np.ndarray):
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
        portfolio_returns = self.returns.dot(weights)

        # Calculate cumulative returns (Cumulative growth)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate drawdowns
        cumulative_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - cumulative_max) / cumulative_max

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

        # Plot cumulative growth
        ax[1].plot(
            cumulative_returns.index,
            cumulative_returns,
            color="g",
            lw=1.5,
            label="Cumulative Growth",
        )
        ax[1].set_title("Portfolio Cumulative Growth", fontsize=14)
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
