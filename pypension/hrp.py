import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, linkage


class HierarchicalRiskParity:
    def __init__(self, returns):
        self.returns = returns
        self.correlation_matrix = None
        self.linkage_matrix = None
        self.weights = None

    def calculate_correlation_matrix(self):
        self.correlation_matrix = self.returns.corr()

    def perform_clustering(self):
        self.linkage_matrix = linkage(self.correlation_matrix, method="single")

    def allocate_weights(self):
        inverse_variance = np.diag(np.linalg.inv(self.correlation_matrix))
        risk_contributions = inverse_variance / np.sum(inverse_variance)
        self.weights = risk_contributions / np.sum(risk_contributions)

    def plot_dendrogram(self):
        dendrogram(self.linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Asset")
        plt.ylabel("Distance")

        plt.show()


if __name__ == "__main__":
    # Define the tickers
    tickers = ["JPM", "BAC", "C", "WFC"]

    # Download historical price data
    data = yf.download(tickers, start="2010-01-01", end="2023-08-31")["Adj Close"]

    # Calculate the returns
    returns = data.pct_change().dropna()

    # Create an instance of HierarchicalRiskParity
    hrp = HierarchicalRiskParity(returns)

    # Calculate the correlation matrix
    hrp.calculate_correlation_matrix()

    # Perform clustering
    hrp.perform_clustering()

    # Allocate weights
    hrp.allocate_weights()

    # Plot the dendrogram
    hrp.plot_dendrogram()

    # Calculate the mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the weights for the mean-variance portfolio
    weights_mean_variance = np.array([0.25, 0.25, 0.25, 0.25])

    # Calculate the mean-variance portfolio returns
    mean_variance_returns = np.dot(mean_returns, weights_mean_variance)

    # Calculate the mean-variance portfolio risk
    mean_variance_risk = np.sqrt(
        np.dot(weights_mean_variance, np.dot(cov_matrix, weights_mean_variance))
    )

    # Calculate the HRP portfolio returns
    hrp_returns = np.dot(mean_returns, hrp.weights)

    # Calculate the HRP portfolio risk
    hrp_risk = np.sqrt(np.dot(hrp.weights, np.dot(cov_matrix, hrp.weights)))

    print("Mean-Variance Portfolio:")
    print(f"Returns: {mean_variance_returns:.2%}")
    print(f"Risk: {mean_variance_risk:.2%}")
    print()
    print("HRP Portfolio:")
    print(f"Returns: {hrp_returns:.2%}")
    print(f"Risk: {hrp_risk:.2%}")
