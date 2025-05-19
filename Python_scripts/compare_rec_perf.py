import sys
import os
import numpy as np
import pandas as pd
import importlib


def load_modules():
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../Python_scripts'))
    global amp, pca_pack, preprocessing, emp_bayes, hierarchical, pipeline
    amp = importlib.import_module("amp")
    pca_pack = importlib.import_module("pca_pack")
    preprocessing = importlib.import_module("preprocessing")
    emp_bayes = importlib.import_module("emp_bayes")
    hierarchical = importlib.import_module("hierarchical_clustering_modalities")
    pipeline = importlib.import_module("complete_pipeline")

    importlib.reload(amp)
    importlib.reload(pca_pack)
    importlib.reload(preprocessing)
    importlib.reload(emp_bayes)
    importlib.reload(hierarchical)
    importlib.reload(pipeline)

    global ebamp_multimodal, MultiModalityPCA, MultiModalityPCADiagnostics
    global ClusterEmpiricalBayes, ModalityClusterer
    global MultimodalPCAPipeline, MultimodalPCAPipelineClustering
    global AJIVEreconstructor, MCCA_denoiser, GCCA_denoiser, DISCO_denoiser, MFA_denoiser, HPCA_denoiser

    ebamp_multimodal = amp.ebamp_multimodal
    MultiModalityPCA = pca_pack.MultiModalityPCA
    MultiModalityPCADiagnostics = preprocessing.MultiModalityPCADiagnostics
    ClusterEmpiricalBayes = emp_bayes.ClusterEmpiricalBayes
    ModalityClusterer = hierarchical.ModalityClusterer
    MultimodalPCAPipeline = pipeline.MultimodalPCAPipelineSimulation
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClusteringSimulation

    import other_multimodal
    importlib.reload(other_multimodal)
    AJIVEreconstructor = other_multimodal.AJIVEReconstructor
    MCCA_denoiser = other_multimodal.MCCAJointIndividual
    GCCA_denoiser = other_multimodal.GCCAJointIndividual
    DISCO_denoiser = other_multimodal.DISCO_SCA
    MFA_denoiser = other_multimodal.MFAJointIndividual
    HPCA_denoiser = other_multimodal.HPCA


def generate_rademacher(shape):
    return np.random.choice([-1, 1], size=shape)


def reconstruction_error(U_est, U_true):
    P_est = U_est @ U_est.T
    P_true = U_true @ U_true.T
    return np.linalg.norm(P_est - P_true, 'fro')**2 / (U_true.shape[0]**2)


def run_amp_rho_experiment(n, p_list, r_list, rho, num_trials, amp_iters, num_clusters=None, threshold=None):
    print(f"\n=== Running for rho = {rho} ===", flush=True)

    errors_clustered = {i: [] for i in range(3)}
    errors_distinct = {i: [] for i in range(3)}
    # prepare storage for other methods: 6 methods × 3 blocks
    errors_ajive  = {i: [] for i in range(3)}
    errors_mcca   = {i: [] for i in range(3)}
    errors_gcca   = {i: [] for i in range(3)}
    errors_disco  = {i: [] for i in range(3)}
    errors_mfa    = {i: [] for i in range(3)}
    errors_hpca   = {i: [] for i in range(3)}

    for trial in range(num_trials):
        print(f"\n=== Current trial for n = {n}, trial = {trial+1} ===", flush=True)

        U1 = generate_rademacher((n, r_list[0]))
        epsilon = generate_rademacher((n, ))

        #U2_col0 = rho * U1[:, 0] + np.sqrt(1 - rho**2) * epsilon
        #U2_col0 = U2_col0.reshape(-1, 1)
        #U2_rest = generate_rademacher((n, r_list[1] - 1))
        #U2 = np.hstack([U2_col0, U2_rest])
        U2 = np.hstack([U1[:, :r_list[0]], generate_rademacher((n, r_list[1] - r_list[0]))])
        U3 = generate_rademacher((n, r_list[2]))
        U_true = [U1, U2, U3]

        V1 = generate_rademacher((p_list[0], r_list[0]))
        V2 = generate_rademacher((p_list[1], r_list[1]))
        V3 = generate_rademacher((p_list[2], r_list[2]))

        D1 = np.diag([5 * (i+1) for i in range(r_list[0])])
        D2 = np.diag([5 * (i+1) for i in range(r_list[1])])
        D3 = np.diag([5 * (i+1) for i in range(r_list[2])])

        Z1 = np.random.randn(n, p_list[0]) / np.sqrt(n)
        Z2 = np.random.randn(n, p_list[1]) / np.sqrt(n)
        Z3 = np.random.randn(n, p_list[2]) / np.sqrt(n)

        X1 = (1/n) * U1 @ D1 @ V1.T + Z1
        X2 = (1/n) * U2 @ D2 @ V2.T + Z2
        X3 = (1/n) * U3 @ D3 @ V3.T + Z3
        X_list = [X1, X2, X3]
        K_list = r_list

        # --- AMP with clustering (HSS) ---
        pipe_cluster = MultimodalPCAPipelineClustering()
        result_cluster = pipe_cluster.denoise_amp(
            X_list, K_list,
            compute_clusters=True,
            amp_iters=amp_iters,
            similarity_method="cca"
        )
        U_cluster = result_cluster["U_denoised"]

        # --- AMP without clustering (distinct) ---
        pipe_distinct = MultimodalPCAPipeline()
        result_distinct = pipe_distinct.denoise_amp(
            X_list, K_list,
            cluster_labels_U=np.array([0, 1, 2]),
            amp_iters=amp_iters
        )
        U_distinct = result_distinct["U_denoised"]

        # === Compare other multimodal methods on all three blocks ===
        # prepare inputs
        views = [X1, X2, X3]
        # AJIVE
        model_ajive = AJIVEreconstructor(rank_list=[r_list[0], r_list[1], r_list[2]], joint_rank=min(r_list[0], r_list[1], r_list[2]))
        U_ajive, _ = model_ajive.fit(views)
        # MCCA
        model_mcca = MCCA_denoiser(individual_ranks=[r_list[0], r_list[1], r_list[2]], joint_rank=min(r_list))
        U_mcca, _ = model_mcca.fit(views)
        # GCCA
        model_gcca = GCCA_denoiser(individual_ranks=[r_list[0], r_list[1], r_list[2]], joint_rank=min(r_list))
        U_gcca, _ = model_gcca.fit(views)
        # DISCO
        model_disco = DISCO_denoiser(individual_ranks=[r_list[0], r_list[1], r_list[2]], n_components=min(r_list))
        U_disco, _ = model_disco.fit(views)
        # MFA
        model_mfa = MFA_denoiser(individual_ranks=[r_list[0], r_list[1], r_list[2]], joint_rank=min(r_list))
        U_mfa, _ = model_mfa.fit(views)
        # HPCA
        model_hpca = HPCA_denoiser(joint_rank=min(r_list), individual_ranks=[r_list[0], r_list[1], r_list[2]])
        hpca_fit = model_hpca.fit(views)
        U_hpca, _ = hpca_fit.get_denoised_factors()

        # accumulate errors for each method on blocks 0, 1, and 2
        errors_ajive[0].append(reconstruction_error(U_ajive[0], U1))
        errors_ajive[1].append(reconstruction_error(U_ajive[1], U2))
        errors_ajive[2].append(reconstruction_error(U_ajive[2], U3))

        errors_mcca[0].append(reconstruction_error(U_mcca[0], U1))
        errors_mcca[1].append(reconstruction_error(U_mcca[1], U2))
        errors_mcca[2].append(reconstruction_error(U_mcca[2], U3))

        errors_gcca[0].append(reconstruction_error(U_gcca[0], U1))
        errors_gcca[1].append(reconstruction_error(U_gcca[1], U2))
        errors_gcca[2].append(reconstruction_error(U_gcca[2], U3))

        errors_disco[0].append(reconstruction_error(U_disco[0], U1))
        errors_disco[1].append(reconstruction_error(U_disco[1], U2))
        errors_disco[2].append(reconstruction_error(U_disco[2], U3))

        errors_mfa[0].append(reconstruction_error(U_mfa[0], U1))
        errors_mfa[1].append(reconstruction_error(U_mfa[1], U2))
        errors_mfa[2].append(reconstruction_error(U_mfa[2], U3))

        errors_hpca[0].append(reconstruction_error(U_hpca[0], U1))
        errors_hpca[1].append(reconstruction_error(U_hpca[1], U2))
        errors_hpca[2].append(reconstruction_error(U_hpca[2], U3))

        # --- Reconstruction Errors ---
        for i in range(3):
            errors_clustered[i].append(reconstruction_error(U_cluster[i][:, :, -1], U_true[i]))
            errors_distinct[i].append(reconstruction_error(U_distinct[i][:, :, -1], U_true[i]))

    # --- Store average errors ---
    results = {
        "clustered": [np.mean(errors_clustered[i]) for i in range(3)],
        "distinct": [np.mean(errors_distinct[i]) for i in range(3)]
    }
    # compute averages for other methods
    results['ajive'] = [np.mean(errors_ajive[i]) for i in range(3)]
    results['mcca']  = [np.mean(errors_mcca[i]) for i in range(3)]
    results['gcca']  = [np.mean(errors_gcca[i]) for i in range(3)]
    results['disco'] = [np.mean(errors_disco[i]) for i in range(3)]
    results['mfa']   = [np.mean(errors_mfa[i]) for i in range(3)]
    results['hpca']  = [np.mean(errors_hpca[i]) for i in range(3)]

    print(f"\n=== For rho = {rho}, the results are {results} ===")
    return results


def main():
    print(">>> Script started", flush=True)
    load_modules()

    if len(sys.argv) != 3:
        print("Usage: python compare_rec_perf.py <rho> <n>", flush=True)
        sys.exit(1)

    rho = float(sys.argv[1])
    n   = int(sys.argv[2])
    gamma_list = [0.25, 0.25, 0.5]
    r_list     = [1, 2, 1]
    p_list     = [int(g * n) for g in gamma_list]

    res = run_amp_rho_experiment(
        n=n,
        p_list=p_list,
        r_list=r_list,
        rho=rho,
        num_trials=30,
        amp_iters=20,
        num_clusters=2,
        threshold=None
    )

    # --- Save results as CSV ---
    rows = []
    for method in ["clustered", "distinct"]:
        for mod in range(3):
            rows.append({
                "rho": rho,
                "Method": method,
                "Modality": mod + 1,
                "Error": res[method][mod]
            })
    # add other methods for blocks 1, 2, and 3 (modality 1, 2, and 3)
    for method in ['ajive', 'mcca', 'gcca', 'disco', 'mfa', 'hpca']:
        for mod in [1, 2, 3]:
            rows.append({
                "rho": rho,
                "Method": method,
                "Modality": mod,
                "Error": res[method][mod-1]
            })

    df = pd.DataFrame(rows)
    os.makedirs("Results/vary_rho_com_meth", exist_ok=True)
    df.to_csv(f"Results/vary_rho_com_meth/rho_{rho}_n_{n}.csv", index=False)
    print(f">>> Results saved to Results/vary_rho_com_meth/rho_{rho}_n_{n}.csv")


if __name__ == "__main__":
    main()