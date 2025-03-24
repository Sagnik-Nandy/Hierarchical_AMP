import numpy as np
import importlib
from scipy.linalg import block_diag

# Dynamically import required modules
pca_pack = importlib.import_module("pca_pack")
emp_bayes = importlib.import_module("emp_bayes")
amp = importlib.import_module("amp")
preprocessing = importlib.import_module("preprocessing")

def extract_normalized_U(pca_model, X_list):
    """
    Extracts and normalizes the left singular vectors (U) for each modality
    from a fitted MultiModalityPCA object.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        Fitted PCA model containing PCA results.
    X_list : list of ndarray
        List of original modality matrices.

    Returns
    -------
    normalized_U_list : list of ndarray
        List of normalized U_k matrices for each modality.
    """
    normalized_U_list = []
    for k in range(len(X_list)):
        U_k = pca_model.pca_results[k].U
        n = U_k.shape[0]
        U_k_normalized = U_k / np.sqrt((U_k**2).sum(axis=0)) * np.sqrt(n)
        normalized_U_list.append(U_k_normalized)
    return normalized_U_list

def extract_normalized_V(pca_model, X_list):
    """
    Extracts and normalizes the right singular vectors (V) for each modality
    from a fitted MultiModalityPCA object.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        Fitted PCA model containing PCA results.
    X_list : list of ndarray
        List of original modality matrices.

    Returns
    -------
    normalized_U_list : list of ndarray
        List of normalized U_k matrices for each modality.
    """
    normalized_V_list = []
    for k in range(len(X_list)):
        V_k = pca_model.pca_results[k].V
        p = V_k.shape[0]
        V_k_normalized = V_k / np.sqrt((V_k**2).sum(axis=0)) * np.sqrt(p)
        normalized_V_list.append(V_k_normalized)
    return normalized_V_list

class MultimodalPCAPipeline:
    """
    Implements a full pipeline for multimodal PCA denoising using Gaussian Bayes AMP.

    Steps:
    1. Preprocesses raw modality matrices (normalize observations and PCs).
    2. Runs PCA to extract principal components and estimates noise structure.
    3. Constructs cluster-based empirical Bayes models for U and per-modality denoisers for V.
    4. Runs AMP to obtain denoised U and V matrices.

    Attributes
    ----------
    pca_model : pca_pack.MultiModalityPCA
        PCA results after fitting.
    cluster_model_u : emp_bayes.ClusterEmpiricalBayes
        Cluster-based empirical Bayes model for U (shared across modalities in the same cluster).
    cluster_model_v : emp_bayes.ClusterEmpiricalBayes
        Empirical Bayes model where each modality is assigned a unique cluster for V.
    amp_results : dict
        Stores U, V, denoised and raw versions.

    Methods
    -------
    denoise_amp(X_list, K_list, cluster_labels_U, amp_iters=5, muteu=False, mutev=False):
        Executes the full denoising pipeline on input data.
    """

    def __init__(self):
        self.pca_model = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results = None

    def denoise_amp(self, X_list, K_list, cluster_labels_U, amp_iters=10, muteu=False, mutev=False):
        """
        Runs the full denoising pipeline: Preprocessing, PCA, empirical Bayes modeling, and AMP.

        Parameters
        ----------
        X_list : list of ndarray
            List of m data matrices X_k of shape (n, r_k).
        K_list : list of int
            List of m values specifying the number of principal components per modality.
        cluster_labels_U : ndarray
            Cluster labels for U, indicating which modalities share the same cluster.
        amp_iters : int, optional
            Number of AMP iterations (default is 5).
        muteu, mutev : bool, optional
            If True, disables denoising in U or V direction (default is False).
        
        Returns
        -------
        amp_results : dict
            Contains the following structured results:
            - "U_non_denoised": dict of non-denoised U matrices
            - "U_denoised": dict of denoised U matrices
            - "V_non_denoised": dict of non-denoised V matrices
            - "V_denoised": dict of denoised V matrices
        """

        print("\n=== Step 1: Preprocessing ===")
        #diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
        #X_preprocessed = diagnostic_tool.normalize_obs(X_list, K_list)

        print("\n=== Step 2: PCA ===")
        self.pca_model = pca_pack.MultiModalityPCA()
        #self.pca_model.fit(X_preprocessed, K_list, plot_residual=False)
        self.pca_model.fit(X_list, K_list, plot_residual=False)

        print("\n=== Step 3: Constructing Empirical Bayes Models ===")
        n = X_list[0].shape[0]
        r_list = [X.shape[1] for X in X_list]
        U_normalized_list = extract_normalized_U(self.pca_model, X_list)
        V_normalized_list = extract_normalized_V(self.pca_model, X_list)

        # Construct M and S matrices using PCA feature aligns and sample aligns
        M_matrices_v = [np.diag(self.pca_model.pca_results[k].feature_aligns) for k in range(len(X_list))]
        S_matrices_v = [np.diag(1 - self.pca_model.pca_results[k].feature_aligns**2) for k in range(len(X_list))]
        M_matrices_u = [np.diag(self.pca_model.pca_results[k].sample_aligns) for k in range(len(X_list))]
        S_matrices_u = [np.diag(1 - self.pca_model.pca_results[k].sample_aligns**2) for k in range(len(X_list))]

        # Generate V clusters: Assign each modality to a distinct cluster
        cluster_labels_V = np.arange(len(X_list))

        # Construct empirical Bayes models for U and V
        self.cluster_model_u = emp_bayes.ClusterEmpiricalBayes(U_normalized_list, M_matrices_u, S_matrices_u, cluster_labels_U)
        self.cluster_model_v = emp_bayes.ClusterEmpiricalBayes(V_normalized_list, M_matrices_v, S_matrices_v, cluster_labels_V)

        # Estimate priors and denoisers
        self.cluster_model_u.estimate_cluster_priors()
        self.cluster_model_v.estimate_cluster_priors()

        print("\n=== Step 4: Running AMP ===")
        self.amp_results = amp.ebamp_multimodal(self.pca_model, self.cluster_model_v, self.cluster_model_u,
                                                amp_iters=amp_iters, muteu=muteu, mutev=mutev)

        print("\n=== Denoising Complete! ===")
        return self.amp_results