import numpy as np
from collections import defaultdict
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import importlib

def _npmle_em_hd(f, Z, mu, covInv, em_iter, eps=1e-5):
    """
    Performs Nonparametric Maximum Likelihood Estimation (NPMLE) using EM.

    - Ensures no division by zero in posterior probabilities.
    - Ensures `pi` remains well-defined.

    Parameters
    ----------
    f : ndarray (n_samples, d)
        Observed data points.
    Z : ndarray (n_support, d)
        Support points.
    mu : ndarray (d, d)
        Mean transformation matrix.
    covInv : ndarray (d, d)
        Inverse covariance matrix.
    em_iter : int
        Number of EM iterations.
    eps : float, optional (default=1e-8)
        Small positive value to prevent division by zero.

    Returns
    -------
    pi : ndarray (n_support,)
        Estimated prior probabilities.
    """
    nsupp = Z.shape[0]
    pi = np.full(nsupp, 1 / nsupp, dtype=np.float64)  # Initialize uniform prior

    W = _get_W(f, Z, mu, covInv, eps=eps)

    Wt = np.array(W.T, order='C')

    for _ in range(em_iter):
        denom = np.clip(W @ pi, eps, np.inf)  # Prevent division by zero
        update_factor = np.mean(Wt / denom, axis=1)
        update_factor = np.nan_to_num(update_factor, nan=1/nsupp, posinf=1/nsupp, neginf=1/nsupp)

        pi *= update_factor
        pi /= np.sum(pi)  # Normalize to sum to 1

    return pi

# consider another get W using broadcast
# W[i,j] = f(x_i | z_j)
def _get_W(f, z, mu, covInv, clip_max=50, eps=1e-5):
    """
    Compute likelihood matrix W[i, j] = P(X_i | Z_j), ensuring numerical stability.

    - Ensures all exponentiation is within safe limits.
    - Prevents division by zero.

    Parameters
    ----------
    f : ndarray (n_samples, d)
        Observed data points.
    z : ndarray (n_support, d)
        Support points.
    mu : ndarray (d, d)
        Mean matrix.
    covInv : ndarray (d, d)
        Inverse covariance matrix.
    clip_max : float, optional (default=50)
        Maximum absolute value for exponentiation to prevent numerical issues.
    eps : float, optional (default=1e-8)
        Small value to prevent division by zero.

    Returns
    -------
    W : ndarray (n_samples, n_support)
        Likelihood matrix.
    """
    f, z, mu, covInv = [arr.astype(np.float64) for arr in [f, z, mu, covInv]]

    fsq = (np.einsum("ij,ij->i", f @ covInv, f) / 2)[:, np.newaxis]  # Shape: (n_samples, 1)
    mz = z @ mu.T  # Shape: (n_support, d)
    zsq = (np.einsum("ij,ij->i", mz @ covInv, mz) / 2)[:, np.newaxis]  # Shape: (n_support, 1)
    
    exponent = -fsq - zsq.T + (f @ covInv @ mz.T)  # Shape: (n_samples, n_support)

    # Clip exponent to prevent overflow in exp
    exponent = np.clip(exponent, -clip_max, clip_max)

    # Compute likelihood matrix W and ensure no zero values
    W = np.exp(exponent)
    W = np.maximum(W, eps)  # Prevent exact zero values

    return W


# P[i,j] = P(Z_j | X_i)
def _get_P(f, z, mu, covInv, pi, eps=1e-5):
    """
    Compute posterior probability matrix P[i, j] = P(Z_j | X_i).

    - Ensures no division by zero.
    - Prevents `NaN` values.

    Parameters
    ----------
    f : ndarray (n_samples, d)
        Observed data points.
    z : ndarray (n_support, d)
        Support points.
    mu : ndarray (d, d)
        Mean transformation matrix.
    covInv : ndarray (d, d)
        Inverse covariance matrix.
    pi : ndarray (n_support,)
        Prior probabilities.
    eps : float, optional (default=1e-8)
        Small positive value to prevent division by zero.

    Returns
    -------
    P : ndarray (n_samples, n_support)
        Posterior probability matrix.
    """
    W = _get_W(f, z, mu, covInv, eps=eps)
    
    denom = np.clip(W @ pi, eps, np.inf)  # Prevent division by zero
    num = W * pi  # Element-wise multiplication

    P = num / denom[:, np.newaxis]  # Normalize probabilities
    P = np.nan_to_num(P, nan=1/len(pi))  # Replace NaNs with uniform distribution

    return P

def _get_P_from_W(W, pi, eps=1e-8):
    """
    Compute posterior probability matrix P[i, j] = P(Z_j | X_i) from likelihood matrix W.

    - Ensures no division by zero.
    - Prevents `NaN` values.

    Parameters
    ----------
    W : ndarray (n_samples, n_support)
        Likelihood matrix.
    pi : ndarray (n_support,)
        Prior probabilities.
    eps : float, optional (default=1e-8)
        Small positive value to prevent division by zero.

    Returns
    -------
    P : ndarray (n_samples, n_support)
        Posterior probability matrix.
    """
    denom = np.clip(W @ pi, eps, np.inf)  # Prevent division by zero
    num = W * pi  # Element-wise multiplication

    P = num / denom[:, np.newaxis]  
    P = np.nan_to_num(P, nan=1/len(pi))  # Replace NaNs with uniform distribution

    return P

def make_cluster_denoisers(eb):
    return {
        "denoise": lambda f, mu, cov: eb.denoise(f, mu, cov),
        "ddenoise": lambda f, mu, cov: eb.ddenoise(f, mu, cov),
    }


matrix_outer = lambda A, B: np.einsum("bi,bo->bio", A, B)

class _BaseEmpiricalBayes(ABC):
    """
    Abstract base class for Empirical Bayes estimation.

    Methods
    -------
    estimate_prior(f, mu, cov):
        Abstract method to estimate the prior distribution.
    denoise(f, mu, cov):
        Abstract method to denoise posterior observations.
    ddenoise(f, mu, cov):
        Abstract method to compute the derivative of the denoising function.
    """

    def __init__(self):
        self.rank = 0

    @abstractmethod
    def estimate_prior(self, f, mu, cov):
        pass

    @abstractmethod
    def denoise(self, f, mu, cov):
        pass

    @abstractmethod
    def ddenoise(self, f, mu, cov):
        pass


class NonparEB(_BaseEmpiricalBayes):
    """
    NPMLE-based empirical Bayes (only supports EM optimizer).

    Methods
    -------
    estimate_prior(f, mu, cov):
        Estimates the prior distribution using the EM algorithm.
    denoise(f, mu, cov):
        Computes the posterior mean estimates with NaN handling.
    ddenoise(f, mu, cov):
        Computes the derivative of the denoising function.
    """

    def __init__(self, max_nsupp=2000, nsupp_ratio=1, em_iter=500):
        super().__init__()
        self.em_iter = em_iter
        self.nsupp_ratio = nsupp_ratio
        self.max_nsupp = max_nsupp
        self.pi = None
        self.Z = None

    def _check_init(self, f, mu):
        self.rank = mu.shape[1]
        self.nsample = f.shape[0]
        self.nsupp = min(int(self.nsupp_ratio * self.nsample), self.max_nsupp or float('inf'))
        self.pi = np.full((self.nsupp,), 1 / self.nsupp)

        # Compute support points (Z) with `pinv` for stability
        if self.nsupp_ratio >= 1:
            self.Z = f @ np.linalg.pinv(mu).T
        else:
            idx = np.random.choice(f.shape[0], self.nsupp, replace=False)
            self.Z = f[idx, :] @ np.linalg.pinv(mu).T

    def estimate_prior(self, f, mu, cov):
        self._check_init(f, mu)
        covInv = np.linalg.pinv(cov)  # Use pseudo-inverse for stability
        self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter)
        return self.Z, self.pi  # Return support points and probability weights

    def denoise(self, f, mu, cov, eps=1e-8):
        """
        Compute the denoised posterior estimates with NaN handling.

        - Ensures rows with NaN values are replaced by prior mean.

        Parameters
        ----------
        f : ndarray (n, d)
            Observed data points.
        mu : ndarray (d, d)
            Mean transformation matrix.
        cov : ndarray (d, d)
            Covariance matrix of the prior.

        Returns
        -------
        denoised_values : ndarray (n, d)
            Posterior mean estimates with NaN handling.
        """
        covInv = np.linalg.pinv(cov)  # Use pseudo-inverse for numerical stability
        P = _get_P(f, self.Z, mu, covInv, self.pi)

        denoised_values = P @ self.Z  # Compute posterior mean

        # Identify rows with NaN values and replace them with prior mean
        nan_rows = np.isnan(denoised_values).any(axis=1)
        if np.any(nan_rows):
            warnings.warn("NaN detected in denoised output. Replacing affected rows with prior mean.", RuntimeWarning)
            denoised_values[nan_rows] = np.average(self.Z, axis=0, weights=self.pi)

        return np.nan_to_num(denoised_values, nan=0.0)  # Ensure no NaNs remain

    def ddenoise(self, f, mu, cov):
        """
        Compute the derivative of the denoising function.

        Parameters
        ----------
        f : ndarray (n, d)
            Observed data points.
        mu : ndarray (d, d)
            Mean transformation matrix.
        cov : ndarray (d, d)
            Covariance matrix of the prior.

        Returns
        -------
        derivative : ndarray
            The derivative of the denoising function at the posterior observations.
        """
        covInv = np.linalg.pinv(cov)
        P = _get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = np.einsum("ijk, kl -> ijl", matrix_outer(self.Z, self.Z @ mu.T), covInv)
        E1 = np.einsum("ij, jkl -> ikl", P, ZouterMZ)
        E2a = P @ self.Z
        E2 = np.einsum("ijk, kl -> ijl", matrix_outer(E2a, E2a @ mu.T), covInv)

        return E1 - E2

class NonparBayes(NonparEB):
    """
    Nonparametric Bayes with a Known Prior.

    This class extends `NonparEB` but does not estimate a prior from data. 
    Instead, it takes a known prior (locations and weights) as input.

    Attributes
    ----------
    Z : ndarray
        The known prior locations (support points).
    pi : ndarray
        The weights associated with the prior locations.
    rank : int
        Dimensionality of the prior distribution.
    """

    def __init__(self, truePriorLoc, truePriorWeight=None):
        """
        Initialize Nonparametric Bayes model with a known prior.

        Parameters
        ----------
        truePriorLoc : ndarray of shape (n, k)
            The known prior locations, where `n` is the number of support points 
            and `k` is the dimensionality.
        truePriorWeight : ndarray of shape (n,), optional
            The probability weights associated with `truePriorLoc`. If not provided, 
            a uniform distribution over `n` points is assumed.

        Raises
        ------
        ValueError
            If the provided prior locations and weights do not match dimensions.
        """
        super().__init__()

        # Ensure `truePriorLoc` is a 2D array
        if truePriorLoc.ndim != 2:
            raise ValueError("truePriorLoc must be a 2D array of shape (n, k)")

        n, k = truePriorLoc.shape

        self.Z = truePriorLoc.astype(np.float64)  # Ensure numerical stability
        self.rank = k  # Dimensionality of the prior distribution

        # Store prior weights (uniform if not provided)
        if truePriorWeight is None:
            self.pi = np.full((n,), 1 / n, dtype=np.float64)
        else:
            if truePriorWeight.ndim != 1:
                raise ValueError("truePriorWeight must be a 1D array of shape (n,)")
            if truePriorWeight.shape[0] != n:
                raise ValueError(f"truePriorWeight must match truePriorLoc in size ({n},)")

            self.pi = np.array(truePriorWeight, dtype=np.float64)

    def estimate_prior(self, f, mu, cov):
        """
        No prior estimation is needed since the prior is already given.
        """
        pass


class ClusterEmpiricalBayes:
    """
    Handles clustering of modalities, aggregation of data, and estimation of empirical Bayes priors.

    Attributes
    ----------
    cluster_data : dict
        Maps each cluster to its concatenated data matrix.
    cluster_M : dict
        Maps each cluster to its block-diagonal M matrix.
    cluster_S : dict
        Maps each cluster to its block-diagonal S matrix.
    cluster_priors : dict
        Maps each cluster to (support_points, prior_weights).
    cluster_denoisers : dict
        Maps each cluster to a function that extracts its denoised values for all modalities in that cluster.
    modality_denoisers : dict
        Maps each modality index to a function that extracts its denoised values from the cluster denoiser.
    """

    def __init__(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        """
        Initialize the ClusterEmpiricalBayes class.

        Parameters
        ----------
        data_matrices : list of ndarrays
            List of m data matrices X_k of shape (n, p_k).
        M_matrices : list of ndarrays
            List of m transformation matrices M_k of shape (r_k, r_k).
        S_matrices : list of ndarrays
            List of m noise matrices S_k of shape (r_k, r_k).
        cluster_labels : list or ndarray
            Cluster labels of length m, indicating the cluster index for each modality.
        """
        if not (len(data_matrices) == len(M_matrices) == len(S_matrices) == len(cluster_labels)):
            raise ValueError("Mismatch in number of modalities among data_matrices, M_matrices, S_matrices, and cluster_labels.")

        self.data_matrices = data_matrices  # Store raw data per modality
        self.cluster_labels = cluster_labels  # Store cluster assignments for modalities

        # Aggregate cluster data
        self.cluster_data, self.cluster_M, self.cluster_S = self.aggregate_cluster_data(
            data_matrices, M_matrices, S_matrices, cluster_labels
        )

        # Dictionary to store cluster priors
        self.cluster_priors = {}

        # Dictionary to store cluster-wise denoisers
        self.cluster_denoisers = {}

        # Dictionary to store denoising functions for each modality
        self.modality_denoisers = {}

    def aggregate_cluster_data(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        """
        Aggregates data, M, and S matrices based on cluster labels and constructs block-diagonal M and S.

        Returns
        -------
        cluster_data, cluster_M, cluster_S : dict
            Dictionaries mapping each cluster index to its aggregated data, block-diagonal M, and block-diagonal S.
        """
        cluster_data = defaultdict(list)
        cluster_M = defaultdict(list)
        cluster_S = defaultdict(list)

        for k, cluster in enumerate(cluster_labels):
            cluster_data[cluster].append(data_matrices[k].astype(np.float64))  # Ensure numerical stability
            cluster_M[cluster].append(M_matrices[k].astype(np.float64))
            cluster_S[cluster].append(S_matrices[k].astype(np.float64))

        for cluster in cluster_data:
            sample_sizes = [X.shape[0] for X in cluster_data[cluster]]
            if len(set(sample_sizes)) > 1:
                raise ValueError(f"Mismatch in sample sizes for cluster {cluster}: {sample_sizes}")

            cluster_data[cluster] = np.concatenate(cluster_data[cluster], axis=1)
            cluster_M[cluster] = block_diag(*cluster_M[cluster])
            cluster_S[cluster] = block_diag(*cluster_S[cluster])

        return cluster_data, cluster_M, cluster_S

    def estimate_cluster_priors(self, em_iter=500, nsupp_ratio=0.5, max_nsupp=100):
        """
        Estimates priors (per-cluster) and denoisers (per-modality) using Nonparametric Empirical Bayes.

        Returns
        -------
        cluster_priors : dict
            Dictionary mapping each cluster to (support_points, prior_weights).
        cluster_denoisers : dict
            Dictionary mapping each cluster to denoising functions that work on **full cluster data**.
        modality_denoisers : dict
            Dictionary mapping each modality index to a function that extracts its denoised values from the cluster denoiser.
        """
        # Estimate priors at the cluster level
        for cluster in self.cluster_data:
            X_cluster = self.cluster_data[cluster]
            M_cluster = self.cluster_M[cluster]
            S_cluster = self.cluster_S[cluster]

            if X_cluster.shape[1] != M_cluster.shape[0]:
                raise ValueError(f"Mismatch in dimensions for cluster {cluster}: X ({X_cluster.shape}) and M ({M_cluster.shape})")
            if S_cluster.shape[0] != S_cluster.shape[1]:
                raise ValueError(f"Noise matrix S for cluster {cluster} is not square: {S_cluster.shape}")
            if S_cluster.shape[0] != M_cluster.shape[0]:
                raise ValueError(f"Mismatch in S ({S_cluster.shape}) and M ({M_cluster.shape}) for cluster {cluster}")

            # Estimate prior using empirical Bayes
            nonpar_eb = NonparEB(em_iter=em_iter, nsupp_ratio=nsupp_ratio, max_nsupp=max_nsupp)
            support_points, prior_weights = nonpar_eb.estimate_prior(X_cluster, M_cluster, S_cluster)

            # Store prior per cluster
            self.cluster_priors[cluster] = (support_points, prior_weights)

            self.cluster_denoisers[cluster] = make_cluster_denoisers(nonpar_eb)

        # Define modality-specific denoisers from cluster denoisers
        for modality_idx, cluster in enumerate(self.cluster_labels):
            start_col = sum(
                self.data_matrices[i].shape[1] for i in range(modality_idx) if self.cluster_labels[i] == cluster
            )
            end_col = start_col + self.data_matrices[modality_idx].shape[1]

            cluster_denoiser = self.cluster_denoisers[cluster]  # Use shared cluster denoiser

            def create_modality_denoise(cluster_denoiser, start_col, end_col):
                def denoise_func(f, mu, cov):
                    denoised_cluster = cluster_denoiser["denoise"](f, mu, cov)
                    return denoised_cluster[:, start_col:end_col]
                return denoise_func

            def create_modality_ddenoise(cluster_denoiser, start_col, end_col):
                def ddenoise_func(f, mu, cov):
                    ddenoised_cluster = cluster_denoiser["ddenoise"](f, mu, cov)
                    return ddenoised_cluster[:, start_col:end_col, start_col:end_col]
                return ddenoise_func

            # Store the modality-specific denoisers
            self.modality_denoisers[modality_idx] = {
                "denoise": create_modality_denoise(cluster_denoiser, start_col, end_col),
                "ddenoise": create_modality_ddenoise(cluster_denoiser, start_col, end_col),
            }

        return self.cluster_priors, self.cluster_denoisers, self.modality_denoisers

def generate_synthetic_data(num_modalities=6, num_clusters=3, n=100, r_range=(3, 7), noise_scale=0.1, seed=42):
    """
    Generate synthetic data matrices X_k = M_k U_k + S_k^{1/2} Z_k with cluster-correlated latent factors.

    - Ensures numerical stability (`np.float64`).
    - Prevents zero or negative values in noise covariance matrices.
    - Avoids `NaN` values in generated data.
    - Ensures each cluster has at least one modality.

    Parameters
    ----------
    num_modalities : int
        Number of different modalities.
    num_clusters : int
        Number of clusters.
    n : int
        Number of samples (same across all modalities).
    r_range : tuple
        Range of values for r_k (dimensionality of each modality).
    noise_scale : float
        Standard deviation of noise components.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_matrices : list of ndarrays
        Generated X_k data matrices of different dimensions.
    M_matrices : list of ndarrays
        Transformation matrices M_k of different sizes.
    S_matrices : list of ndarrays
        Diagonal noise matrices S_k of different sizes.
    cluster_labels : ndarray
        Cluster assignments for each modality.
    """
    np.random.seed(seed)  # Set seed for reproducibility

    # Assign each modality to a cluster
    cluster_labels = np.random.randint(0, num_clusters, size=num_modalities).astype(np.int64)

    # Determine r_k for each modality, ensuring positive dimensions
    modality_dims = np.random.randint(r_range[0], r_range[1] + 1, size=num_modalities)

    # Ensure each cluster has at least one modality
    unique_clusters = np.unique(cluster_labels)

    # Compute the minimum r_c for each cluster
    cluster_min_dims = {c: min(modality_dims[cluster_labels == c]) for c in unique_clusters}

    # Generate shared cluster-wise latent variables U_c
    cluster_latents = {c: np.random.randn(n, cluster_min_dims[c]).astype(np.float64) for c in unique_clusters}

    # Initialize lists for data matrices, transformation matrices, and noise matrices
    data_matrices = []
    M_matrices = []
    S_matrices = []

    for k in range(num_modalities):
        r_k = modality_dims[k]  # Dimension of modality k
        cluster_idx = cluster_labels[k]  # Assigned cluster
        r_c = cluster_min_dims[cluster_idx]  # Minimum r_c in the cluster

        # Generate `U_k`, ensuring numerical stability
        U_k = np.hstack([
            cluster_latents[cluster_idx],  # First r_c columns from U_c
            np.random.randn(n, r_k - r_c).astype(np.float64) if r_k > r_c else np.empty((n, 0))
        ])

        # Generate transformation matrix M_k (r_k × r_k), ensuring no zero diagonal elements
        M_k = np.diag(np.random.uniform(0.5, 1.5, size=r_k).astype(np.float64))

        # Generate noise matrix S_k (diagonal, r_k × r_k) with minimum values
        S_k_diag = np.clip(np.random.uniform(0.05, noise_scale, size=r_k), 1e-4, np.inf)
        S_k = np.diag(S_k_diag).astype(np.float64)

        # Generate noise Z_k, ensuring numerical stability
        Z_k = np.random.randn(n, r_k).astype(np.float64)

        # Compute X_k = M_k U_k + S_k^{1/2} Z_k
        X_k = (U_k @ M_k.T) + (Z_k @ np.sqrt(S_k))

        # Store results
        data_matrices.append(X_k)
        M_matrices.append(M_k)
        S_matrices.append(S_k)

    return data_matrices, M_matrices, S_matrices, cluster_labels