import numpy as np
import importlib
from scipy.linalg import block_diag
import torch
import torch.nn as nn

# Dynamically import required modules
pca_pack = importlib.import_module("pca_pack")
emp_bayes = importlib.import_module("emp_bayes")
amp = importlib.import_module("amp")
preprocessing = importlib.import_module("preprocessing")
clusterer = importlib.import_module("preprocessing")

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


# === Nonlinear Regression Modules ===
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_nonlinear_regressor(pipeline, X_list, y, epochs=2000, lr=1e-3, hidden_dim=32):
    """
    Train cluster-specific two-layer networks f_theta_c to minimize
    sum_i (y_i - E[f_theta(U_i)|hat U_i])^2, where the expectation
    is taken w.r.t. EB posterior weights.
    Saves trained models in pipeline.cluster_models_nonlinear and
    the final predicted y in pipeline.y_pred_nonlinear.
    """
    # extract needed objects
    V_dict = pipeline.amp_results["V_denoised"]
    D_dict = pipeline.amp_results["signal_diag_dict"]
    cluster_u = pipeline.cluster_model_u
    labels = cluster_u.cluster_labels
    priors = cluster_u.cluster_priors    # {c: (Z_c, pi_c)}
    M_maps = cluster_u.cluster_M        # {c: M_c}
    S_maps = cluster_u.cluster_S        # {c: S_c}

    n = X_list[0].shape[0]
    # reconstruct U_hat per modality
    U_hat = {}
    for k, Xk in enumerate(X_list):
        V_k = V_dict[k][:,:, -1]
        D_k = D_dict[k]
        A_k = (1.0 / n) * V_k @ D_k
        AtA_inv = np.linalg.inv(A_k.T @ A_k)
        U_hat[k] = Xk @ A_k @ AtA_inv

    # assemble cluster-specific inputs
    cluster_inputs = {}
    for c in np.unique(labels):
        mods = [k for k in range(len(X_list)) if labels[k] == c]
        Uc = np.hstack([U_hat[k] for k in mods])         # shape: (n, d_c)
        Zc, pic = priors[c]                              # Zc: (m_c, d_c), pic: (m_c,)
        # build cluster M and S same as in AMP
        M_cluster = np.eye(Uc.shape[1])
        S_blocks = []
        for k in mods:
            V_k = V_dict[k][:, :, -1]
            D_k = D_dict[k]
            D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
            Sigma_k = (1 / n) * (V_k.T @ V_k)
            Sigma_k_inv = np.linalg.inv(Sigma_k)
            S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
            S_k = S_k @ S_k
            S_blocks.append(S_k)
        S_cluster = block_diag(*S_blocks)
        covInv = np.linalg.pinv(S_cluster)
        P = emp_bayes._get_P(Uc, Zc, M_cluster, covInv, pic)
        # store tensors
        cluster_inputs[c] = {
            "P": torch.tensor(P, dtype=torch.float32),
            "Z": torch.tensor(Zc, dtype=torch.float32)
        }

    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)
    # create cluster models
    cluster_models = {c: TwoLayerNN(cluster_inputs[c]["Z"].shape[1], hidden_dim) for c in cluster_inputs}
    optimizers = {c: torch.optim.Adam(cluster_models[c].parameters(), lr=lr) for c in cluster_models}
    loss_fn = nn.MSELoss()

    # training loop
    for ep in range(epochs):
        total_loss = 0.0
        # zero grads
        for opt in optimizers.values():
            opt.zero_grad()
        # compute total expectation
        E_sum = torch.zeros_like(y_tensor)
        for c, inp in cluster_inputs.items():
            fZ = cluster_models[c](inp["Z"])        # (m_c,1)
            E_c = inp["P"] @ fZ                     # (n,1)
            E_sum = E_sum + E_c
        loss = loss_fn(E_sum, y_tensor)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        total_loss = loss.item()
        if ep % 100 == 0:
            print(f"[Nonlinear] Ep {ep}, loss {total_loss:.4f}", flush=True)

    # store results
    pipeline.cluster_models_nonlinear = cluster_models
    pipeline.y_pred_nonlinear = E_sum.detach().cpu().numpy().flatten()
    return pipeline.cluster_models_nonlinear, pipeline.y_pred_nonlinear

class MultimodalPCAPipeline:
    """
    Implements a full pipeline for multimodal PCA denoising using Gaussian Bayes AMP.

    Steps:
    1. Preprocesses raw modality matrices (normalize observations and PCs).
    2. Runs PCA to extract principal components and estimates noise structure.
    3. Constructs cluster-based empirical Bayes models for U and per-modality denoisers for V.
    4. Runs AMP to obtain denoised U and V matrices.
    5. Optionally, estimate regression coefficients if response vector y is provided via `pipeline.y_train`.

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
        self.y_train = None  # For supervised regression after AMP
<<<<<<< HEAD
=======
        self.relation = "linear"
>>>>>>> f6fef4f (Added the complete pipeline)

    def denoise_amp(self, X_list, K_list, cluster_labels_U, amp_iters=10, muteu=False, mutev=False, preprocess = False):
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
            - "beta_hat": estimated regression coefficients (if y_train is provided)
        """
        
        if preprocess:
            print("\n=== Step 1: Preprocessing ===")
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_preprocessed = diagnostic_tool.normalize_obs(X_list, K_list)

        print("\n=== Step 2: PCA ===")
        self.pca_model = pca_pack.MultiModalityPCA()

        if preprocess:
           self.pca_model.fit(X_preprocessed, K_list, plot_residual=False)
        else:
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

        print("hello", flush=True)

        print("\n=== Step 4: Running AMP ===")
        self.amp_results = amp.ebamp_multimodal(self.pca_model, self.cluster_model_v, self.cluster_model_u,
                                                amp_iters=amp_iters, muteu=muteu, mutev=mutev)

<<<<<<< HEAD
        print("\n=== Denoising Complete! ===")
        if hasattr(self, "y_train") and self.y_train is not None:
            # --- Estimate beta from U_denoised and y ---
            print("\n=== Step 6: Estimating Beta via Least Squares ===")
            U_concat = np.hstack([self.amp_results["U_denoised"][k][:, :, -1] for k in sorted(self.amp_results["U_denoised"].keys())])
            y = self.y_train
            beta_hat = np.linalg.pinv(U_concat) @ y  # least squares solution
            self.amp_results["beta_hat"] = beta_hat
=======
        print("\n=== Denoising Complete! ===", flush=True)

        if hasattr(self, "y_train") and self.y_train is not None:
            V_dict = self.amp_results["V_denoised"]
            D_dict = self.amp_results["signal_diag_dict"]
            cluster_model_u = self.cluster_model_u
            cluster_labels_u = cluster_model_u.cluster_labels
            cluster_denoisers = cluster_model_u.cluster_denoisers

            m = len(X_list)
            n = X_list[0].shape[0]
            U_hat_dict = {}

            for k in range(m):
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                A_k = (1.0 / n) * V_k @ D_k
                X_k = X_list[k]
                AtA_inv = np.linalg.inv(A_k.T @ A_k)
                U_hat_k = X_k @ A_k @ AtA_inv
                U_hat_dict[k] = U_hat_k

            U_denoised_dict = {}
            for cluster_id in np.unique(cluster_labels_u):
                modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_id]
                u_denoiser = cluster_denoisers[cluster_id]["denoise"]
                U_concat = np.hstack([U_hat_dict[k] for k in modalities])
                M_cluster = np.eye(U_concat.shape[1])
                S_blocks = []
                for k in modalities:
                    V_k = V_dict[k][:, :, -1]
                    D_k = D_dict[k]
                    D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
                    Sigma_k = (1 / n) * (V_k.T @ V_k)
                    Sigma_k_inv = np.linalg.inv(Sigma_k)
                    S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
                    S_k = S_k @ S_k
                    S_blocks.append(S_k)
                S_cluster = block_diag(*S_blocks)
                U_denoised_concat = u_denoiser(U_concat, M_cluster, S_cluster)
                split_sizes = [U_hat_dict[k].shape[1] for k in modalities]
                starts = np.cumsum([0] + split_sizes[:-1])
                ends = np.cumsum(split_sizes)
                for k, start, end in zip(modalities, starts, ends):
                    U_denoised_dict[k] = U_denoised_concat[:, start:end]

            relation_type = getattr(self, "relation", "linear")
            if relation_type == "linear":
                U_concat = np.hstack([U_denoised_dict[k] for k in sorted(U_denoised_dict.keys())])
                y = self.y_train
                beta_hat = np.linalg.pinv(U_concat) @ y
                self.amp_results["beta_hat"] = beta_hat
            if relation_type == "non-linear":
                # train and apply nonlinear regressor
                models, y_pred = train_nonlinear_regressor(self, X_list, self.y_train)
                self.amp_results["cluster_models_nonlinear"] = models
                self.amp_results["y_pred_nonlinear"] = y_pred
>>>>>>> f6fef4f (Added the complete pipeline)
        return self.amp_results

    # estimate_beta_from_consistent_denoising method removed as per instructions.
    
class MultimodalPCAPipelineClustering:
    """
    Full pipeline for multimodal PCA denoising using AMP with clustering-based Empirical Bayes.

    Steps:
    1. Perform PCA on modality matrices.
    2. Cluster modalities based on normalized sample PCs (U).
    3. Construct empirical Bayes denoisers for U (cluster-based) and V (per-modality).
    4. Run AMP to denoise U and V.
    5. Optionally, estimate regression coefficients if response vector y is provided via `pipeline.y_train`.

    Attributes
    ----------
    pca_model : MultiModalityPCA
        PCA results after fitting.
    cluster_model_u : ClusterEmpiricalBayes
        Empirical Bayes model for U (shared across clusters).
    cluster_model_v : ClusterEmpiricalBayes
        Empirical Bayes model for V (per-modality).
    amp_results : dict
        Dictionary of AMP outputs (raw/denoised U and V).
    y_train : ndarray
        Response vector for supervised regression (optional).
    """

    def __init__(self):
        self.pca_model = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results = None
        self.y_train = None  # For supervised regression after AMP
<<<<<<< HEAD
=======
        self.relation = "linear"
>>>>>>> f6fef4f (Added the complete pipeline)

    def denoise_amp(
        self, X_list, K_list,
        cluster_labels_U=None, compute_clusters=True, num_clusters=None,
        threshold=None, amp_iters=10, muteu=False, mutev=False, preprocess=False, similarity_method="hss"
    ):
        """
        Run the complete AMP denoising pipeline.

        Parameters
        ----------
        X_list : list of np.ndarray
            List of m data matrices of shape (n, r_k) per modality.
        K_list : list of int
            List of number of PCs to retain per modality.
        cluster_labels_U : array-like, optional
            Cluster labels for U. Required if compute_clusters is False.
        compute_clusters : bool
            Whether to compute clusters using similarity of U.
        num_clusters : int
            Number of clusters to compute if compute_clusters is True.
        threshold : float, optional
            Threshold for hierarchical clustering.
        amp_iters : int
            Number of AMP iterations.
        muteu : bool
            If True, disables denoising in U direction.
        mutev : bool
            If True, disables denoising in V direction.
        preprocess : bool
            Whether to normalize input data before PCA.
        similarity_method : string
            Method to compute similarity

        Returns
        -------
        amp_results : dict
            Contains the following structured results:
            - "U_non_denoised": dict of non-denoised U matrices
            - "U_denoised": dict of denoised U matrices
            - "V_non_denoised": dict of non-denoised V matrices
            - "V_denoised": dict of denoised V matrices
            - "beta_hat": estimated regression coefficients (if y_train is provided)
        """
        # Step 1: Preprocessing (optional)
        if preprocess:
            print("\n=== Step 1: Preprocessing ===")
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_list = diagnostic_tool.normalize_obs(X_list, K_list)

        # Step 2: PCA
        print("\n=== Step 2: PCA ===")
        self.pca_model = pca_pack.MultiModalityPCA()
        self.pca_model.fit(X_list, K_list, plot_residual=False)

        # Step 3: Clustering
        print("\n=== Step 3: Clustering Modalities via U ===")
        U_normalized_list = extract_normalized_U(self.pca_model, X_list)
        V_normalized_list = extract_normalized_V(self.pca_model, X_list)

        if compute_clusters:
            from hierarchical_clustering_modalities import ModalityClusterer
            clusterer_obj = ModalityClusterer(U_normalized_list)
            similarity_matrix = clusterer_obj.compute_similarity_matrix(similarity_method, epsilon=0.1, sigma=1.0)
            print(f"Similarity Matrix ({similarity_method.upper()}):\n", similarity_matrix)
            cluster_labels_U = clusterer_obj.cluster_modalities(similarity_method, num_clusters=num_clusters, threshold=threshold)
        elif cluster_labels_U is None:
            raise ValueError("Either enable compute_clusters or provide cluster_labels_U.")

        print("Cluster Labels for U:", cluster_labels_U, flush=True)

        # Step 4: Construct Empirical Bayes Models
        print("\n=== Step 4: Constructing Empirical Bayes Models ===")
        M_matrices_v = [np.diag(p.feature_aligns) for p in self.pca_model.pca_results.values()]
        S_matrices_v = [np.diag(1 - p.feature_aligns**2) for p in self.pca_model.pca_results.values()]
        M_matrices_u = [np.diag(p.sample_aligns) for p in self.pca_model.pca_results.values()]
        S_matrices_u = [np.diag(1 - p.sample_aligns**2) for p in self.pca_model.pca_results.values()]

        self.cluster_model_u = emp_bayes.ClusterEmpiricalBayes(U_normalized_list, M_matrices_u, S_matrices_u, cluster_labels_U)
        self.cluster_model_v = emp_bayes.ClusterEmpiricalBayes(V_normalized_list, M_matrices_v, S_matrices_v, np.arange(len(X_list)))

        self.cluster_model_u.estimate_cluster_priors()
        self.cluster_model_v.estimate_cluster_priors()

        # Step 5: Run AMP
        print("\n=== Step 5: Running AMP ===")
        self.amp_results = amp.ebamp_multimodal(
            self.pca_model,
            self.cluster_model_v,
            self.cluster_model_u,
            amp_iters=amp_iters,
            muteu=muteu,
            mutev=mutev
        )

        print("\n=== Denoising Complete! ===")
<<<<<<< HEAD
        if hasattr(self, "y_train") and self.y_train is not None:
            # --- Estimate beta from U_denoised and y ---
            print("\n=== Step 6: Estimating Beta via Least Squares ===")
            U_concat = np.hstack([
                self.amp_results["U_denoised"][k][:, :, -1]
                for k in sorted(self.amp_results["U_denoised"].keys())
            ])
            y = self.y_train
            beta_hat = np.linalg.pinv(U_concat) @ y  # least squares solution
            self.amp_results["beta_hat"] = beta_hat
=======

        if hasattr(self, "y_train") and self.y_train is not None:
            # Inline estimate_beta_from_consistent_denoising
            if self.amp_results is None:
                raise RuntimeError("AMP must be run before estimating beta. Please call denoise_amp() first.")
            V_dict = self.amp_results["V_denoised"]
            D_dict = self.amp_results["signal_diag_dict"]
            cluster_model_u = self.cluster_model_u
            cluster_labels_u = cluster_model_u.cluster_labels
            cluster_denoisers = cluster_model_u.cluster_denoisers

            m = len(X_list)
            n = X_list[0].shape[0]
            U_hat_dict = {}

            for k in range(m):
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                A_k = (1.0 / n) * V_k @ D_k
                X_k = X_list[k]
                AtA_inv = np.linalg.inv(A_k.T @ A_k)
                U_hat_k = X_k @ A_k @ AtA_inv
                U_hat_dict[k] = U_hat_k

            U_denoised_dict = {}
            for cluster_id in np.unique(cluster_labels_u):
                modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_id]
                u_denoiser = cluster_denoisers[cluster_id]["denoise"]
                U_concat = np.hstack([U_hat_dict[k] for k in modalities])
                M_cluster = np.eye(U_concat.shape[1])
                S_blocks = []
                for k in modalities:
                    V_k = V_dict[k][:, :, -1]
                    D_k = D_dict[k]
                    D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
                    Sigma_k = (1 / n) * (V_k.T @ V_k)
                    Sigma_k_inv = np.linalg.inv(Sigma_k)
                    S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
                    S_k = S_k @ S_k
                    S_blocks.append(S_k)
                S_cluster = block_diag(*S_blocks)
                U_denoised_concat = u_denoiser(U_concat, M_cluster, S_cluster)
                split_sizes = [U_hat_dict[k].shape[1] for k in modalities]
                starts = np.cumsum([0] + split_sizes[:-1])
                ends = np.cumsum(split_sizes)
                for k, start, end in zip(modalities, starts, ends):
                    U_denoised_dict[k] = U_denoised_concat[:, start:end]

            # Final estimation depending on relation type
            relation_type = getattr(self, "relation", "linear")
            if relation_type == "linear":
                U_concat = np.hstack([U_denoised_dict[k] for k in sorted(U_denoised_dict.keys())])
                y = self.y_train
                beta_hat = np.linalg.pinv(U_concat) @ y
                self.amp_results["beta_hat"] = beta_hat
            if relation_type == "non-linear":
                # train and apply nonlinear regressor
                models, y_pred = train_nonlinear_regressor(self, X_list, self.y_train)
                self.amp_results["cluster_models_nonlinear"] = models
                self.amp_results["y_pred_nonlinear"] = y_pred
>>>>>>> f6fef4f (Added the complete pipeline)
        return self.amp_results
    

class MultimodalPCAPipelineSimulation:
    """
    Implements a full pipeline for multimodal PCA denoising using Gaussian Bayes AMP.

    Steps:
    1. Preprocesses raw modality matrices (normalize observations and PCs).
    2. Runs PCA to extract principal components and estimates noise structure.
    3. Constructs cluster-based empirical Bayes models for U and per-modality denoisers for V.
    4. Runs AMP to obtain denoised U and V matrices.
    5. Optionally, estimate regression coefficients if response vector y is provided via `pipeline.y_train`.

    Attributes
    ----------
    pca_model : pca_pack.MultiModalityPCA
        PCA results after fitting.
    cluster_model_u : emp_bayes.ClusterEmpiricalBayes
        Cluster-based empirical Bayes model for U (shared across modalities in the same cluster).
    cluster_model_v : emp_bayes.ClusterEmpiricalBayes
        Empirical Bayes model where each modality is assigned a unique cluster for V.
    amp_results : dict
        Stores U, V, denoised and raw versions (raw/denoised U and V, estimated regression coefficients if applicable).
    y_train : ndarray
        Response vector for supervised regression (optional).

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
        self.y_train = None  # For supervised regression after AMP
<<<<<<< HEAD
=======
        self.relation = "linear"
>>>>>>> f6fef4f (Added the complete pipeline)

    def denoise_amp(self, X_list, K_list, cluster_labels_U, amp_iters=10, muteu=False, mutev=False, preprocess = False):
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
            - "beta_hat": estimated regression coefficients (if y_train is provided)
        """
        
        if preprocess:
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_preprocessed = diagnostic_tool.normalize_obs(X_list, K_list)

        self.pca_model = pca_pack.MultiModalityPCA()

        if preprocess:
           self.pca_model.fit(X_preprocessed, K_list, plot_residual=False)
        else:
            self.pca_model.fit(X_list, K_list, plot_residual=False)

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

        self.amp_results = amp.ebamp_multimodal(self.pca_model, self.cluster_model_v, self.cluster_model_u,
                                                amp_iters=amp_iters, muteu=muteu, mutev=mutev)

        if hasattr(self, "y_train") and self.y_train is not None:
<<<<<<< HEAD
            # --- Estimate beta from U_denoised and y ---
            print("\n=== Step 6: Estimating Beta via Least Squares ===")
            U_concat = np.hstack([self.amp_results["U_denoised"][k][:, :, -1] for k in sorted(self.amp_results["U_denoised"].keys())])
            y = self.y_train
            beta_hat = np.linalg.pinv(U_concat) @ y  # least squares solution
            self.amp_results["beta_hat"] = beta_hat

=======
            # Inline estimate_beta_from_consistent_denoising
            if self.amp_results is None:
                raise RuntimeError("AMP must be run before estimating beta. Please call denoise_amp() first.")
            V_dict = self.amp_results["V_denoised"]
            D_dict = self.amp_results["signal_diag_dict"]
            cluster_model_u = self.cluster_model_u
            cluster_labels_u = cluster_model_u.cluster_labels
            cluster_denoisers = cluster_model_u.cluster_denoisers

            m = len(X_list)
            n = X_list[0].shape[0]
            U_hat_dict = {}

            for k in range(m):
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                A_k = (1.0 / n) * V_k @ D_k
                X_k = X_list[k]
                AtA_inv = np.linalg.inv(A_k.T @ A_k)
                U_hat_k = X_k @ A_k @ AtA_inv
                U_hat_dict[k] = U_hat_k

            U_denoised_dict = {}
            for cluster_id in np.unique(cluster_labels_u):
                modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_id]
                u_denoiser = cluster_denoisers[cluster_id]["denoise"]
                U_concat = np.hstack([U_hat_dict[k] for k in modalities])
                M_cluster = np.eye(U_concat.shape[1])
                S_blocks = []
                for k in modalities:
                    V_k = V_dict[k][:, :, -1]
                    D_k = D_dict[k]
                    D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
                    Sigma_k = (1 / n) * (V_k.T @ V_k)
                    Sigma_k_inv = np.linalg.inv(Sigma_k)
                    S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
                    S_k = S_k @ S_k
                    S_blocks.append(S_k)
                S_cluster = block_diag(*S_blocks)
                U_denoised_concat = u_denoiser(U_concat, M_cluster, S_cluster)
                split_sizes = [U_hat_dict[k].shape[1] for k in modalities]
                starts = np.cumsum([0] + split_sizes[:-1])
                ends = np.cumsum(split_sizes)
                for k, start, end in zip(modalities, starts, ends):
                    U_denoised_dict[k] = U_denoised_concat[:, start:end]

            # Final estimation depending on relation type
            relation_type = getattr(self, "relation", "linear")
            if relation_type == "linear":
                U_concat = np.hstack([U_denoised_dict[k] for k in sorted(U_denoised_dict.keys())])
                y = self.y_train
                beta_hat = np.linalg.pinv(U_concat) @ y
                self.amp_results["beta_hat"] = beta_hat
            if relation_type == "non-linear":
                # train and apply nonlinear regressor
                models, y_pred = train_nonlinear_regressor(self, X_list, self.y_train)
                self.amp_results["cluster_models_nonlinear"] = models
                self.amp_results["y_pred_nonlinear"] = y_pred
>>>>>>> f6fef4f (Added the complete pipeline)
        return self.amp_results
            


class MultimodalPCAPipelineClusteringSimulation:
    """
    Full pipeline for multimodal PCA denoising using AMP with clustering-based Empirical Bayes for simulation without print statements.

    Steps:
    1. Perform PCA on modality matrices.
    2. Cluster modalities based on normalized sample PCs (U).
    3. Construct empirical Bayes denoisers for U (cluster-based) and V (per-modality).
    4. Run AMP to denoise U and V.
    5. Optionally, estimate regression coefficients if response vector y is provided via `pipeline.y_train`.

    Attributes
    ----------
    pca_model : MultiModalityPCA
        PCA results after fitting.
    cluster_model_u : ClusterEmpiricalBayes
        Empirical Bayes model for U (shared across clusters).
    cluster_model_v : ClusterEmpiricalBayes
        Empirical Bayes model for V (per-modality).
    amp_results : dict
        Dictionary of AMP outputs (raw/denoised U and V).
    y_train : ndarray
        Response vector for supervised regression (optional).
    """

    def __init__(self):
        self.pca_model = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results = None
        self.y_train = None  # For supervised regression after AMP
<<<<<<< HEAD
=======
        self.relation = "linear"
>>>>>>> f6fef4f (Added the complete pipeline)

    def denoise_amp(
        self, X_list, K_list,
        cluster_labels_U=None, compute_clusters=True, num_clusters=None,
        threshold=None, amp_iters=10, muteu=False, mutev=False, preprocess=False, similarity_method="hss"
    ):
        """
        Run the complete AMP denoising pipeline.

        Parameters
        ----------
        X_list : list of np.ndarray
            List of m data matrices of shape (n, r_k) per modality.
        K_list : list of int
            List of number of PCs to retain per modality.
        cluster_labels_U : array-like, optional
            Cluster labels for U. Required if compute_clusters is False.
        compute_clusters : bool
            Whether to compute clusters using similarity of U.
        num_clusters : int
            Number of clusters to compute if compute_clusters is True.
        threshold : float, optional
            Threshold for hierarchical clustering.
        amp_iters : int
            Number of AMP iterations.
        muteu : bool
            If True, disables denoising in U direction.
        mutev : bool
            If True, disables denoising in V direction.
        preprocess : bool
            Whether to normalize input data before PCA.
        similarity_method : string
            Method to compute similarity

        Returns
        -------
        amp_results : dict
            Contains the following structured results:
            - "U_non_denoised": dict of non-denoised U matrices
            - "U_denoised": dict of denoised U matrices
            - "V_non_denoised": dict of non-denoised V matrices
            - "V_denoised": dict of denoised V matrices
            - "beta_hat": estimated regression coefficients (if y_train is provided)
        """
        # Step 1: Preprocessing (optional)
        if preprocess:
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_list = diagnostic_tool.normalize_obs(X_list, K_list)

        # Step 2: PCA
        self.pca_model = pca_pack.MultiModalityPCA()
        self.pca_model.fit(X_list, K_list, plot_residual=False)

        # Step 3: Clustering
        U_normalized_list = extract_normalized_U(self.pca_model, X_list)
        V_normalized_list = extract_normalized_V(self.pca_model, X_list)

        if compute_clusters:
            from hierarchical_clustering_modalities import ModalityClusterer
            clusterer_obj = ModalityClusterer(U_normalized_list)
            similarity_matrix = clusterer_obj.compute_similarity_matrix(similarity_method)
            cluster_labels_U = clusterer_obj.cluster_modalities(similarity_method, num_clusters=num_clusters, threshold=threshold)
            print(cluster_labels_U)
        elif cluster_labels_U is None:
            raise ValueError("Either enable compute_clusters or provide cluster_labels_U.")

        # Step 4: Construct Empirical Bayes Models
        M_matrices_v = [np.diag(p.feature_aligns) for p in self.pca_model.pca_results.values()]
        S_matrices_v = [np.diag(1 - p.feature_aligns**2) for p in self.pca_model.pca_results.values()]
        M_matrices_u = [np.diag(p.sample_aligns) for p in self.pca_model.pca_results.values()]
        S_matrices_u = [np.diag(1 - p.sample_aligns**2) for p in self.pca_model.pca_results.values()]

        self.cluster_model_u = emp_bayes.ClusterEmpiricalBayes(U_normalized_list, M_matrices_u, S_matrices_u, cluster_labels_U)
        self.cluster_model_v = emp_bayes.ClusterEmpiricalBayes(V_normalized_list, M_matrices_v, S_matrices_v, np.arange(len(X_list)))

        self.cluster_model_u.estimate_cluster_priors()
        self.cluster_model_v.estimate_cluster_priors()

        # Step 5: Run AMP
        self.amp_results = amp.ebamp_multimodal(
            self.pca_model,
            self.cluster_model_v,
            self.cluster_model_u,
            amp_iters=amp_iters,
            muteu=muteu,
            mutev=mutev
        )
        if hasattr(self, "y_train") and self.y_train is not None:
<<<<<<< HEAD
            print("\n=== Step 6: Estimating Beta via Least Squares ===")
            U_concat = np.hstack([
                self.amp_results["U_denoised"][k][:, :, -1]
                for k in sorted(self.amp_results["U_denoised"].keys())
            ])
            y = self.y_train
            beta_hat = np.linalg.pinv(U_concat) @ y  # least squares solution
            self.amp_results["beta_hat"] = beta_hat
        return self.amp_results
    

def predict_from_test_data(X_test, amp_results, n):
=======
            # Inline estimate_beta_from_consistent_denoising
            if self.amp_results is None:
                raise RuntimeError("AMP must be run before estimating beta. Please call denoise_amp() first.")
            V_dict = self.amp_results["V_denoised"]
            D_dict = self.amp_results["signal_diag_dict"]
            cluster_model_u = self.cluster_model_u
            cluster_labels_u = cluster_model_u.cluster_labels
            cluster_denoisers = cluster_model_u.cluster_denoisers

            m = len(X_list)
            n = X_list[0].shape[0]
            U_hat_dict = {}

            for k in range(m):
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                A_k = (1.0 / n) * V_k @ D_k
                X_k = X_list[k]
                AtA_inv = np.linalg.inv(A_k.T @ A_k)
                U_hat_k = X_k @ A_k @ AtA_inv
                U_hat_dict[k] = U_hat_k

            U_denoised_dict = {}
            for cluster_id in np.unique(cluster_labels_u):
                modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_id]
                u_denoiser = cluster_denoisers[cluster_id]["denoise"]
                U_concat = np.hstack([U_hat_dict[k] for k in modalities])
                M_cluster = np.eye(U_concat.shape[1])
                S_blocks = []
                for k in modalities:
                    V_k = V_dict[k][:, :, -1]
                    D_k = D_dict[k]
                    D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
                    Sigma_k = (1 / n) * (V_k.T @ V_k)
                    Sigma_k_inv = np.linalg.inv(Sigma_k)
                    S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
                    S_k = S_k @ S_k
                    S_blocks.append(S_k)
                S_cluster = block_diag(*S_blocks)
                U_denoised_concat = u_denoiser(U_concat, M_cluster, S_cluster)
                split_sizes = [U_hat_dict[k].shape[1] for k in modalities]
                starts = np.cumsum([0] + split_sizes[:-1])
                ends = np.cumsum(split_sizes)
                for k, start, end in zip(modalities, starts, ends):
                    U_denoised_dict[k] = U_denoised_concat[:, start:end]

            # Final estimation depending on relation type
            relation_type = getattr(self, "relation", "linear")
            if relation_type == "linear":
                U_concat = np.hstack([U_denoised_dict[k] for k in sorted(U_denoised_dict.keys())])
                y = self.y_train
                beta_hat = np.linalg.pinv(U_concat) @ y
                self.amp_results["beta_hat"] = beta_hat
            if relation_type == "non-linear":
                # train and apply nonlinear regressor
                models, y_pred = train_nonlinear_regressor(self, X_list, self.y_train)
                self.amp_results["cluster_models_nonlinear"] = models
                self.amp_results["y_pred_nonlinear"] = y_pred
        return self.amp_results

def predict_from_test_data(X_test, amp_results, n, relation="linear"):
>>>>>>> f6fef4f (Added the complete pipeline)
    """
    Given test data X_test and AMP results, reconstruct and denoise the U matrices, then generate predicted responses y_hat using the AMP-estimated regression coefficients.

    Parameters
    ----------
    X_test : list of np.ndarray
        List of n x p_k matrices for each modality.
    amp_results : dict
        Output from ebamp_multimodal() containing V_denoised, signal_diag_dict, and cluster models.
    n : int
        Training sample size.
<<<<<<< HEAD
=======
    relation : str
        "linear" or "non-linear"
>>>>>>> f6fef4f (Added the complete pipeline)

    Returns
    -------
    U_denoised_dict : dict
        Dictionary mapping modality index to final denoised U_k matrices.
    y_pred : ndarray or None
        Predicted response vector if beta_hat is available from AMP results; otherwise None.
    """
<<<<<<< HEAD
    
=======
>>>>>>> f6fef4f (Added the complete pipeline)
    V_dict = amp_results["V_denoised"]
    D_dict = amp_results["signal_diag_dict"]
    cluster_model_u = amp_results["cluster_model_u"]
    cluster_labels_u = cluster_model_u.cluster_labels
    cluster_denoisers = cluster_model_u.cluster_denoisers

    m = len(X_test)
    U_hat_dict = {}

    # Step 1: Compute raw U_hat_k
    for k in range(len(X_test)):
<<<<<<< HEAD
        V_k = V_dict[k]             # shape: p_k x r_k
=======
        V_k = V_dict[k][:, :, -1]            # shape: p_k x r_k
>>>>>>> f6fef4f (Added the complete pipeline)
        D_k = D_dict[k]             # shape: r_k x r_k (diagonal matrix)
        A_k = (1.0 / n) * V_k @ D_k # shape: p_k x r_k
        X_k = X_test[k]             # shape: n x p_k

        # Compute U_hat_k = X_k A_k (A_k.T A_k)^{-1}
        AtA_inv = np.linalg.inv(A_k.T @ A_k)
        U_hat_k = X_k @ A_k @ AtA_inv
        U_hat_dict[k] = U_hat_k

    # Step 2: Cluster-wise denoising
    U_denoised_dict = {}
    for cluster_id in np.unique(cluster_labels_u):
        modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_id]
        u_denoiser = cluster_denoisers[cluster_id]["denoise"]

        # Form input matrix for denoiser
        U_concat = np.hstack([U_hat_dict[k] for k in modalities])
        M_cluster = np.eye(U_concat.shape[1])
        S_blocks = []
        for k in modalities:
            V_k = V_dict[k][:, :, -1]
            D_k = D_dict[k]
            D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
            Sigma_k = (1 / n) * (V_k.T @ V_k)  # not the inverse!
            Sigma_k_inv = np.linalg.inv(Sigma_k)
            S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
            S_k = S_k @ S_k  # square to get the covariance
            S_blocks.append(S_k)
        S_cluster = block_diag(*S_blocks)

        # Apply denoiser
        U_denoised_concat = u_denoiser(U_concat, M_cluster, S_cluster)

        # Split and store
        split_sizes = [U_hat_dict[k].shape[1] for k in modalities]
        starts = np.cumsum([0] + split_sizes[:-1])
        ends = np.cumsum(split_sizes)
        for k, start, end in zip(modalities, starts, ends):
            U_denoised_dict[k] = U_denoised_concat[:, start:end]

<<<<<<< HEAD
    # Optional: Predict y using estimated beta
    if "beta_hat" in amp_results:
=======
    # Linear prediction branch
    if relation == "linear" and "beta_hat" in amp_results:
>>>>>>> f6fef4f (Added the complete pipeline)
        beta_hat = amp_results["beta_hat"]
        U_concat_pred = np.hstack([U_denoised_dict[k] for k in sorted(U_denoised_dict.keys())])
        y_pred = U_concat_pred @ beta_hat
        return U_denoised_dict, y_pred
<<<<<<< HEAD
    else:
        return U_denoised_dict, None

=======
    # Non-linear prediction branch
    elif relation == "non-linear" and "cluster_models_nonlinear" in amp_results:
        # Retrieve cluster nonlinear models
        cluster_models = amp_results["cluster_models_nonlinear"]
        # Retrieve cluster prior parameters from cluster_model_u in amp_results
        cluster_model_u = amp_results["cluster_model_u"]
        cluster_priors = cluster_model_u.cluster_priors
        cluster_M = cluster_model_u.cluster_M
        cluster_S = cluster_model_u.cluster_S
        cluster_labels = cluster_model_u.cluster_labels
        # For each cluster, recompute test U_hat and posterior weights P_test
        y_pred_nonlinear = None
        n_samples = X_test[0].shape[0]
        y_pred_sum = torch.zeros((n_samples, 1), dtype=torch.float32)
        # For each cluster
        for c in np.unique(cluster_labels):
            # Find modalities in this cluster
            mods = [k for k in range(m) if cluster_labels[k] == c]
            Uc = np.hstack([U_hat_dict[k] for k in mods])  # (n, d_c)
            Zc, pic = cluster_priors[c]                    # Zc: (m_c, d_c), pic: (m_c,)
            # Compose M and S for this cluster
            # Compose M_cluster (identity) and S_cluster (block_diag of S_k as above)
            M_cluster = np.eye(Uc.shape[1])
            S_blocks = []
            for k in mods:
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                D_inv_sqrt = np.linalg.inv(np.sqrt(D_k))
                Sigma_k = (1 / n) * (V_k.T @ V_k)
                Sigma_k_inv = np.linalg.inv(Sigma_k)
                S_k = D_inv_sqrt @ Sigma_k_inv @ D_inv_sqrt
                S_k = S_k @ S_k
                S_blocks.append(S_k)
            S_cluster = block_diag(*S_blocks)
            covInv = np.linalg.pinv(S_cluster)
            # Compute posterior weights
            P_test = emp_bayes._get_P(Uc, Zc, M_cluster, covInv, pic)
            # Get model for this cluster and Zc as torch tensors
            model = cluster_models[c]
            Zc_tensor = torch.tensor(Zc, dtype=torch.float32)
            P_test_tensor = torch.tensor(P_test, dtype=torch.float32)
            # Predict f(Z) with model (no grad)
            with torch.no_grad():
                fZ = model(Zc_tensor)  # (m_c, 1)
            # Compute E_c = P_test @ fZ
            E_c = P_test_tensor @ fZ   # (n, 1)
            y_pred_sum = y_pred_sum + E_c
        y_pred_nonlinear = y_pred_sum.numpy().flatten()
        return U_denoised_dict, y_pred_nonlinear
>>>>>>> f6fef4f (Added the complete pipeline)








