import numpy as np
import importlib
from emp_bayes import NonparEB, ClusterEmpiricalBayes
from pca_pack import MultiModalityPCA
from scipy.linalg import block_diag

def ebamp_multimodal(pca_model, cluster_model_v, cluster_model_u, amp_iters=10, muteu=False, mutev=False):
    """
    Multimodal Gaussian Bayes AMP with per-modality denoising for V and per-cluster denoising for U.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        PCA model containing PCA results for each modality.
    cluster_model_v : ClusterEmpiricalBayes
        Cluster-based empirical Bayes model providing per-modality denoisers (for V).
    cluster_model_u : ClusterEmpiricalBayes
        Cluster-based empirical Bayes model providing per-cluster denoisers (for U).
    amp_iters : int
        Number of AMP iterations.
    muteu, mutev : bool
        If True, use the identity map as the denoiser in that direction.
    
    Returns
    -------
    U_dict, V_dict : dict
        Denoised U and V matrices for each modality.
    """

    X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
    n_samples, _ = X_list[0].shape  
    m = len(X_list)  # Number of modalities
    gamma_list = [X.shape[1] / n_samples for X in X_list]  # Aspect ratios

    # Initialize storage dictionaries
    U_dict, V_dict, U_dict_denois, V_dict_denois = {}, {}, {}, {}
    mu_dict_u, sigma_sq_dict_u, mu_dict_v, sigma_sq_dict_v = {}, {}, {}, {}
    b_bar_dict_u = {}

    # Store diagonal signal matrices for each modality
    signal_diag_dict = {k: np.diag(pca_model.pca_results[k].signals) for k in range(m)}

    # Initialize storage per modality
    for k in range(m):
        pca_k = pca_model.pca_results[k]
        U_init, V_init = pca_k.U, pca_k.V

        # Normalize U and V
        f_k = U_init / np.linalg.norm(U_init, axis=0) * np.sqrt(n_samples)
        g_k = V_init / np.linalg.norm(V_init, axis=0) * np.sqrt(X_list[k].shape[1])

        # Initialize mu and sigma_sq for V denoising (first step uses cluster model)
        mu_dict_v[k] = np.diag(pca_k.feature_aligns)
        sigma_sq_dict_v[k] = np.diag(1 - pca_k.feature_aligns**2)
        mu_dict_u[k] = np.diag(pca_k.sample_aligns)
        sigma_sq_dict_u[k] = np.diag(1 - pca_k.sample_aligns**2)

        # Store initial values
        U_dict[k] = f_k[:, :, np.newaxis]
        V_dict[k] = g_k[:, :, np.newaxis]
        U_dict_denois[k] = (f_k @ np.sqrt(sigma_sq_dict_v[k]))[:, :, np.newaxis]
        V_dict_denois[k] = g_k[:, :, np.newaxis]

    # Retrieve cluster assignments
    cluster_labels_v = cluster_model_v.cluster_labels  # Per-modality V denoising
    cluster_labels_u = cluster_model_u.cluster_labels  # Per-cluster U denoising

    for t in range(amp_iters):

        #print(f"\n--- AMP Iteration {t + 1} ---")

        # ---- Step 1: Denoising V (PER-MODALITY) ----
        for k in range(m):

            gamma_k = gamma_list[k]

            vdenoiser = cluster_model_v.cluster_denoisers[cluster_labels_v[k]]  # Per-modality denoiser

            g_k = V_dict[k][:, :, -1]  # Latest estimate of V
            mu_k, sigma_sq_k = mu_dict_v[k], sigma_sq_dict_v[k]
            u_k = U_dict_denois[k][:, :, -1]

            if not mutev:
                v_k = vdenoiser["denoise"](g_k, mu_k, sigma_sq_k)
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))

                # Compute correction term
                b_k = gamma_k * np.mean(vdenoiser["ddenoise"](g_k, mu_k, sigma_sq_k), axis=0)
                sigma_bar_sq = v_k.T @ v_k / n_samples
                mu_bar = sigma_bar_sq * pca_model.pca_results[k].signals
            else:
                # Identity denoiser
                mu_inv = np.linalg.pinv(mu_k)
                v_k = g_k @ mu_inv.T
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
                b_k = mu_inv * gamma_k
                mu_bar = np.diag(pca_model.pca_results[k].signals) * gamma_k
                sigma_bar_sq = (np.eye(v_k.shape[1]) + mu_inv @ sigma_sq_k @ mu_inv.T) * gamma_k
            

            # Update f_k using v_k
            f_k = X_list[k] @ v_k - u_k @ b_k.T
            U_dict[k] = np.dstack((U_dict[k], f_k[:, :, np.newaxis]))

            # Store updated mu and sigma_sq for next U denoising
            mu_dict_u[k] = mu_bar
            sigma_sq_dict_u[k] = sigma_bar_sq

        # ---- Step 2: Denoising U (PER-CLUSTER) ----
        for cluster_k in np.unique(cluster_labels_u):
            cluster_modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_k]
            signals_list = [pca_model.pca_results[k].signals for k in cluster_modalities]
            mu_signals = np.concatenate(signals_list)
            udenoiser = cluster_model_u.cluster_denoisers[cluster_k]  # Shared per-cluster denoiser

            # Stack f_k for all modalities in this cluster
            f_cluster = np.hstack([U_dict[k][:, :, -1] for k in cluster_modalities])
            split_sizes = [U_dict[k].shape[1] for k in cluster_modalities]

            # Construct block-diagonal M and S using precomputed values
            M_cluster = block_diag(*[mu_dict_u[k] for k in cluster_modalities])
            S_cluster = block_diag(*[sigma_sq_dict_u[k] for k in cluster_modalities])

            if not muteu:
                u_cluster = udenoiser["denoise"](f_cluster, M_cluster, S_cluster)

                # Compute correction term
                b_bar_cluster = np.mean(udenoiser["ddenoise"](f_cluster, M_cluster, S_cluster), axis=0)
                sigma_sq = u_cluster.T @ u_cluster / n_samples
                mu = sigma_sq * mu_signals

            else:
                # Identity denoiser
                mu_bar_inv = np.linalg.pinv(M_cluster)
                u_cluster = f_cluster @ mu_bar_inv.T
                b_bar_cluster = mu_bar_inv
                mu = np.diag(np.concatenate(signals_list))
                sigma_sq = np.eye(u_cluster.shape[1]) + mu_bar_inv @ S_cluster @ mu_bar_inv.T

            # Compute indices to split results
            start_indices = np.cumsum([0] + split_sizes[:-1])
            end_indices = np.cumsum(split_sizes)

            # Split u_cluster and store in U_dict_denois
            for k, start, end in zip(cluster_modalities, start_indices, end_indices):
                U_dict_denois[k] = np.dstack((U_dict_denois[k], u_cluster[:, start:end, np.newaxis]))

            # Extract block-diagonal submatrices
            for k, start, end in zip(cluster_modalities, start_indices, end_indices):
                b_bar_dict_u[k] = b_bar_cluster[start:end, start:end]
                mu_dict_v[k] = mu[start:end, start:end]
                sigma_sq_dict_v[k] = sigma_sq[start:end, start:end]

            # Update g_k using u_k
            for k in cluster_modalities:
                g_k = X_list[k].T @ U_dict_denois[k][:, :, -1] - V_dict_denois[k][:, :, -1] @ b_bar_dict_u[k].T
                V_dict[k] = np.dstack((V_dict[k], g_k[:, :, np.newaxis]))

        # # --- Log summary statistics ---
        # print("Sum of entries per modality (mu_u, mu_v, sigma_u, sigma_v):")
        # for k in range(m):
        #     mu_u_sum = np.sum(mu_dict_u[k])
        #     mu_v_sum = np.sum(mu_dict_v[k])
        #     sigma_u_sum = np.sum(sigma_sq_dict_u[k])
        #     sigma_v_sum = np.sum(sigma_sq_dict_v[k])
        #     print(f"  Modality {k}: mu_u={mu_u_sum:.4f}, mu_v={mu_v_sum:.4f}, sigma_u={sigma_u_sum:.4f}, sigma_v={sigma_v_sum:.4f}")

        {
        "U_non_denoised": U_dict,
        "U_denoised": U_dict_denois,
        "V_non_denoised": V_dict,
        "V_denoised": V_dict_denois,
        "signal_diag_dict": signal_diag_dict,
        "cluster_model_u": cluster_model_u,
        "cluster_model_v": cluster_model_v
    }