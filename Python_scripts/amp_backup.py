import numpy as np
from emp_bayes import NonparEB, ClusterEmpiricalBayes
from pca_pack import MultiModalityPCA

def ebamp_multimodal(pca_model, cluster_model, amp_iters=5, warm_start=True, muteu=False, mutev=False):
    """
    Multimodal Gaussian Bayes AMP
    
    Parameters
    ----------
    pca_model : MultiModalityPCA
        The PCA model containing PCA results for each modality.
    cluster_model : ClusterEmpiricalBayes
        Cluster-based empirical Bayes model to provide per-cluster denoisers.
    amp_iters : int
        Number of AMP iterations.
    warm_start : bool
        If True, estimate priors only once at the start.
    muteu, mutev : bool
        If True, use the identity map as the denoiser in that direction.
    
    Returns
    -------
    U_dict, V_dict : dict
        Denoised U and V matrices for each modality.
    """
    
    X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
    n_samples, _ = X_list[0].shape  # Assume all modalities have same n

    m = len(X_list)  # Number of modalities
    gamma_list = [X.shape[1] / n_samples for X in X_list]  # Aspect ratios

    U_dict, V_dict = {}, {}

    # Initialize storage for AMP iterations
    for k in range(m):
        X_k = X_list[k]
        pca_k = pca_model.pca_results[k]
        signals = pca_k.signals
        U_init, V_init = pca_k.U, pca_k.V
        
        # Normalize U and V
        f_k = U_init / np.sqrt((U_init**2).sum(axis=0)) * np.sqrt(n_samples)
        g_k = V_init / np.sqrt((V_init**2).sum(axis=0)) * np.sqrt(X_k.shape[1])
        
        # Store initialized U and V
        U_dict[k] = f_k[:, :, np.newaxis]
        V_dict[k] = g_k[:, :, np.newaxis]

    # Initial corrections per cluster
    cluster_labels = cluster_model.cluster_labels
    cluster_priors = cluster_model.cluster_priors
    cluster_denoisers = cluster_model.modality_denoisers

    for t in range(amp_iters):
        for k in range(m):
            X_k = X_list[k]
            gamma_k = gamma_list[k]
            cluster_k = cluster_labels[k]
            
            # Retrieve cluster-based priors and denoisers
            Z_k, pi_k = cluster_priors[cluster_k]
            vdenoiser, udenoiser = cluster_denoisers[k]

            f_k, g_k = U_dict[k][:, :, -1], V_dict[k][:, :, -1]  # Latest estimates
            
            # Initialize mean and variance
            mu = np.diag(pca_model.pca_results[k].feature_aligns)
            sigma_sq = np.diag(1 - pca_model.pca_results[k].feature_aligns**2)

            # Denoise V
            if not mutev:
                v_k = vdenoiser.denoise(g_k, mu, sigma_sq)
                V_dict[k] = np.dstack((V_dict[k], np.reshape(v_k, (-1, v_k.shape[1], 1))))
                
                # Compute correction term
                b_k = gamma_k * np.mean(vdenoiser.ddenoise(g_k, mu, sigma_sq), axis=0)
                sigma_bar_sq = v_k.T @ v_k / n_samples
                mu_bar = sigma_bar_sq * pca_model.pca_results[k].signals
            else:
                # Identity denoiser
                mu_inv = np.linalg.pinv(mu)
                v_k = g_k @ mu_inv.T
                V_dict[k] = np.dstack((V_dict[k], np.reshape(v_k, (-1, v_k.shape[1], 1))))
                b_k = mu_inv * gamma_k
                mu_bar = np.diag(pca_model.pca_results[k].signals) * gamma_k
                sigma_bar_sq = (np.identity(v_k.shape[1]) + mu_inv @ sigma_sq @ mu_inv.T) * gamma_k

            # Update f_k using v_k
            f_k = X_k @ v_k - U_dict[k][:, :, -1] @ b_k.T
            
            # Denoise U
            if not muteu:
                if not warm_start or t == 0:
                    udenoiser.estimate_prior(f_k, mu_bar, sigma_bar_sq)
                u_k = udenoiser.denoise(f_k, mu_bar, sigma_bar_sq)
                U_dict[k] = np.dstack((U_dict[k], np.reshape(u_k, (-1, u_k.shape[1], 1))))
                
                # Compute correction term
                b_bar_k = np.mean(udenoiser.ddenoise(f_k, mu_bar, sigma_bar_sq), axis=0)
                sigma_sq = u_k.T @ u_k / n_samples
                mu = sigma_sq * pca_model.pca_results[k].signals
            else:
                # Identity denoiser
                mu_bar_inv = np.linalg.pinv(mu_bar)
                u_k = f_k @ mu_bar_inv.T
                U_dict[k] = np.dstack((U_dict[k], np.reshape(u_k, (-1, u_k.shape[1], 1))))
                b_bar_k = mu_bar_inv
                mu = np.diag(pca_model.pca_results[k].signals)
                sigma_sq = np.identity(u_k.shape[1]) + mu_bar_inv @ sigma_bar_sq @ mu_bar_inv.T

            # Update g_k using u_k
            g_k = X_k.T @ u_k - v_k @ b_bar_k.T

    return U_dict, V_dict