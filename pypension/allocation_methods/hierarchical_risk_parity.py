# https://thepythonlab.medium.com/hierarchical-risk-parity-portfolio-optimization-f40584d7481d

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform

from pypension.allocation_methods.base import AbstractPortfolio


class HierarchicalRiskParity(AbstractPortfolio):
    def allocate_weights(self, n_clusters=3):
        """
        Perform Hierarchical Risk Parity (HRP) for portfolio optimization.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - n_clusters (int): Number of clusters for hierarchical clustering.

        Returns:
        - pd.Series: Optimal portfolio weights.
        """

        # Step 1: Compute the covariance matrix
        df_returns = self.df_returns
        cov_matrix = df_returns.cov()

        # Step 2: Compute the distance matrix
        dist_matrix = distance_matrix(cov_matrix.values, cov_matrix.values)

        # Step 3: Perform hierarchical clustering
        condensed_dist_matrix = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist_matrix, method="ward")

        # Step 4: Form clusters
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

        # Step 5: Compute cluster variances
        cluster_variances = pd.Series(index=df_returns.columns)
        for cluster_id in np.unique(clusters):
            cluster_assets = df_returns.columns[clusters == cluster_id]
            cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]
            cluster_variances[cluster_assets] = np.diag(cluster_cov)

        # Step 6: Assign weights based on inverse variance within clusters
        inv_var = 1 / cluster_variances
        inv_var /= inv_var.sum()

        # Step 7: Create the portfolio weights based on hierarchical risk parity
        portfolio_weights = pd.Series(index=df_returns.columns)
        for cluster_id in np.unique(clusters):
            cluster_assets = df_returns.columns[clusters == cluster_id]
            cluster_weight = inv_var[cluster_assets].sum()
            portfolio_weights[cluster_assets] = inv_var[cluster_assets] / cluster_weight

        return portfolio_weights / portfolio_weights.sum()

    @staticmethod
    def perform_clustering(correlation_matrix):
        return linkage(correlation_matrix, method="single")

    def plot_dendrogram(self, correlation_matrix, **kwargs):
        linkage_matrix = self.perform_clustering(correlation_matrix)
        dendrogram(linkage_matrix, **kwargs)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Asset")
        plt.ylabel("Distance")

        plt.show()
