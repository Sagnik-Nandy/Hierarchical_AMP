import sys
import numpy as np
import pandas as pd
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def run_amp_comparison_experiment(n, p_list, r_list, num_trials, amp_iters):
    errors_clustered_all = {method: {i: [] for i in range(3)} for method in ["hss", "cca", "wasserstein"]}

    for trial in range(num_trials):
        print(f"\n=== Current trial for n = {n}, trial = {trial+1} ===", flush=True)
        
        U1 = generate_rademacher((n, r_list[0]))
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

        for method in ["hss", "cca", "wasserstein"]:
            #print(f"\n ======== Running with method = {method} =============", flush=True)
            pipe_cluster = MultimodalPCAPipelineClustering()
            result_cluster = pipe_cluster.denoise_amp(
                X_list, K_list,
                compute_clusters=True,
                num_clusters=2,
                amp_iters=amp_iters,
                similarity_method=method
            )
            U_cluster = result_cluster["U_denoised"]

            # Compute reconstruction errors
            for i in range(3):
                errors_clustered_all[method][i].append(reconstruction_error(U_cluster[i][:, :, -1], U_true[i]))

    return {
        "n": n,
        "methods": {method: [np.mean(errors_clustered_all[method][i]) for i in range(3)] for method in errors_clustered_all}
    }

def main():
    print(">>> Script started", flush=True)
    load_modules()

    if len(sys.argv) < 2:
        print("Usage: python cluster_ccoef_n.py <n>", flush=True)
        sys.exit(1)

    n = int(sys.argv[1])
    gamma_list = [0.25, 0.25, 0.5]
    r_list = [1, 2, 1]
    p_list = [int(g * n) for g in gamma_list]

    print(f">>> Running AMP comparison experiment with n = {n}", flush=True)
    result = run_amp_comparison_experiment(
        n=n,
        p_list=p_list,
        r_list=r_list,
        num_trials=50,
        amp_iters=10
    )

    print(f">>> Finished experiment for n = {n}", flush=True)
    print(f">>> Result: {result}", flush=True)

    try:
        print(">>> Saving CSV", flush=True)
        df = pd.DataFrame.from_dict(result["methods"], orient='index', columns=['Modality_1', 'Modality_2', 'Modality_3']).reset_index()
        df.columns = ['Method', 'Modality_1', 'Modality_2', 'Modality_3']
        df.to_csv(f"Results/vary_meth/partial_result_{n}.csv", index=False)

        print(">>> Plotting", flush=True)
        for method, errors in result["methods"].items():
            plt.figure(figsize=(6, 4))
            plt.bar(['Mod1', 'Mod2', 'Mod3'], errors)
            plt.title(f"{method.upper()} - n = {n}")
            plt.ylabel("Avg Reconstruction Error")
            plt.savefig(f"Plots/vary_meth/{method}_n_{n}.png")
            plt.close()
        print(">>> Plots saved", flush=True)

    except Exception as e:
        print(f"!!! Error during saving/plotting for n = {n}: {e}", flush=True)

if __name__ == "__main__":
    main()