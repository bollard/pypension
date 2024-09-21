import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sph
import scipy.spatial as sps
import scipy.spatial.distance as spd

from pypension.allocation_methods.base import AbstractPortfolio


class HierarchicalRiskParity(AbstractPortfolio):
    def allocate_weights_t(
        self, asset_returns: pd.DataFrame, n_clusters: int = 3, **kwargs
    ) -> pd.Series:
        """
        Perform Hierarchical Risk Parity (HRP) for portfolio optimization.

        Parameters:
        - returns (pd.DataFrame): Asset returns data.
        - n_clusters (int): Number of clusters for hierarchical clustering.

        Returns:
        - pd.Series: Optimal portfolio weights.
        """

        idx = asset_returns.isna().all()
        asset_returns_active = asset_returns.loc[:, ~idx]

        # Step 1: Compute the covariance matrix
        cov_matrix = asset_returns_active.cov()

        # Step 3: Perform hierarchical clustering
        linkage_matrix = self.perform_clustering(cov_matrix)

        # Step 4: Form clusters
        clusters = sph.fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

        # Step 5: Compute cluster variances
        cluster_variances = pd.Series(index=asset_returns_active.columns)
        for cluster_id in np.unique(clusters):
            cluster_assets = asset_returns_active.columns[clusters == cluster_id]
            cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]
            cluster_variances[cluster_assets] = np.diag(cluster_cov)

        # Step 6: Assign weights based on inverse variance within clusters
        inv_var = 1 / cluster_variances
        inv_var /= inv_var.sum()

        # Step 7: Create the portfolio weights based on hierarchical risk parity
        portfolio_weights = pd.Series(index=asset_returns.columns)
        for cluster_id in np.unique(clusters):
            cluster_assets = asset_returns_active.columns[clusters == cluster_id]
            cluster_weight = inv_var[cluster_assets].sum()
            portfolio_weights[cluster_assets] = inv_var[cluster_assets] / cluster_weight

        return portfolio_weights / portfolio_weights.sum()

    @staticmethod
    def perform_clustering(covariance_matrix):
        dist_matrix = sps.distance_matrix(covariance_matrix.values, covariance_matrix.values)
        condensed_dist_matrix = spd.squareform(dist_matrix)
        return sph.linkage(condensed_dist_matrix, method="ward")

    def plot_dendrogram(self, correlation_matrix, **kwargs):
        linkage_matrix = self.perform_clustering(correlation_matrix)
        sph.dendrogram(linkage_matrix, **kwargs)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Asset")
        plt.ylabel("Distance")

        plt.show()
