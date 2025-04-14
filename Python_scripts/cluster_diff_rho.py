import sys
import numpy as np
import pandas as pd
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def load_modules():
    sys.path.append('./Python_scripts')
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

    ebamp_multimodal = amp.ebamp_multimodal
    MultiModalityPCA = pca_pack.MultiModalityPCA
    MultiModalityPCADiagnostics = preprocessing.MultiModalityPCADiagnostics
    ClusterEmpiricalBayes = emp_bayes.ClusterEmpiricalBayes
    ModalityClusterer = hierarchical.ModalityClusterer
    MultimodalPCAPipeline = pipeline.MultimodalPCAPipelineSimulation
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClusteringSimulation

def generate_rademacher(shape):
    return np.random.choice([-1, 1], size=shape)

def reconstruction_error(U_est, U_true):
    P_est = U_est @ U_est.T
    P_true = U_true @ U_true.T
    return np.linalg.norm(P_est - P_true, 'fro')**2 / (U_true.shape[0]**2)

def run_amp_rho_experiment(n, p_list, r_list, rho_list, num_trials, amp_iters, num_clusters=None, threshold=None):
    result_by_rho = {}

    for rho in rho_list:
        print(f"\n=== Running for rho = {rho} ===", flush=True)

        errors_clustered = {i: [] for i in range(3)}
        errors_distinct = {i: [] for i in range(3)}

        for trial in range(num_trials):
            print(f"\n=== Current trial for n = {n}, trial = {trial+1} ===", flush=True)

            U1 = generate_rademacher((n, r_list[0]))
            epsilon = generate_rademacher((n, r_list[0]))

            # # Step 2: Create U2 as before
            # Z1 = rho * U1[:, :r_list[0]] + np.sqrt(1 - rho**2) * epsilon
            # Z2 = generate_rademacher((n, r_list[1] - r_list[0]))
            # U2_raw = np.hstack([Z1, Z2])

            # # Step 3: Center and whiten U2 so rows have identity covariance
            # U2_centered = U2_raw - U2_raw.mean(axis=0, keepdims=True)
            # cov = (U2_centered.T @ U2_centered) / n
            # U2 = U2_centered @ np.linalg.inv(np.linalg.cholesky(cov)).T
            
            U2 = np.hstack([rho * U1[:, :r_list[0]] + np.sqrt(1 - rho**2) * epsilon, generate_rademacher((n, r_list[1] - r_list[0]))])
            
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

            # --- Reconstruction Errors ---
            for i in range(3):
                errors_clustered[i].append(reconstruction_error(U_cluster[i][:, :, -1], U_true[i]))
                errors_distinct[i].append(reconstruction_error(U_distinct[i][:, :, -1], U_true[i]))

        # --- Store average errors ---
        result_by_rho[rho] = {
            "clustered": [np.mean(errors_clustered[i]) for i in range(3)],
            "distinct": [np.mean(errors_distinct[i]) for i in range(3)]
        }

        print(f"\n=== For rho = {rho}, the results are {result_by_rho[rho]} ===")

    return result_by_rho

def main():
    print(">>> Script started", flush=True)
    load_modules()

    if len(sys.argv) < 2:
        print("Usage: python cluster_rho_experiment.py <n>", flush=True)
        sys.exit(1)

    n = int(sys.argv[1])
    gamma_list = [0.25, 0.25, 0.5]
    r_list = [1, 2, 1]
    rho_list = [0.8, 0.85, 0.9, 0.95, 1.0]
    p_list = [int(g * n) for g in gamma_list]

    results = run_amp_rho_experiment(
        n=n,
        p_list=p_list,
        r_list=r_list,
        rho_list=rho_list,
        num_trials=50,
        amp_iters=20,
        num_clusters=2,
        threshold=None
    )

    # --- Save results as CSV ---
    rows = []
    for rho, res in results.items():
        for method in ["clustered", "distinct"]:
            for mod in range(3):
                rows.append({
                    "rho": rho,
                    "Method": method,
                    "Modality": mod + 1,
                    "Error": res[method][mod]
                })

    df = pd.DataFrame(rows)
    os.makedirs("Results/vary_rho", exist_ok=True)
    df.to_csv(f"Results/vary_rho/rho_results_n_{n}.csv", index=False)
    print(f">>> Results saved to Results/vary_rho/rho_results_n_{n}.csv")

    # --- Plot ---
    os.makedirs("Plots/vary_rho", exist_ok=True)
    for modality in [1, 2, 3]:
        plt.figure(figsize=(8, 5))
        for method in ["clustered", "distinct"]:
            subset = df[(df["Method"] == method) & (df["Modality"] == modality)]
            subset = subset.sort_values("rho")
            plt.plot(subset["rho"], subset["Error"], marker='o', label=method)

        plt.title(f"Modality {modality} - Error vs Rho")
        plt.xlabel("rho (correlation between U1 and U3)")
        plt.ylabel("Avg Reconstruction Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Plots/vary_rho/modality_{modality}_error_vs_rho.png")
        plt.close()

    print(">>> Plots saved in Plots/vary_rho")

if __name__ == "__main__":
    main()