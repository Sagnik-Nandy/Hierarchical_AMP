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

def run_amp_rho_experiment(n, p_list, r_list, rho_list, num_trials, amp_iters, num_clusters=None, threshold=None, sigma=0.1):
    result_by_rho = {}

    for rho in rho_list:
        print(f"\n=== Running for rho = {rho} ===", flush=True)

        errors_clustered = {i: [] for i in range(3)}
        errors_same = {i: [] for i in range(3)}
        errors_distinct = {i: [] for i in range(3)}
        pred_err_clustered, pred_err_same, pred_err_distinct = [], [], []

        for trial in range(num_trials):
            print(f"\n=== Current trial for n = {n}, trial = {trial+1} ===", flush=True)

            U1 = generate_rademacher((n, r_list[0]))
            epsilon = generate_rademacher((n, r_list[0]))
            
            U2 = np.hstack([rho * U1[:, :r_list[0]] + np.sqrt(1 - rho**2) * epsilon, generate_rademacher((n, r_list[1] - r_list[0]))])
            
            U3 = generate_rademacher((n, r_list[2]))
            U_true = [U1, U2, U3]

            beta = generate_rademacher(sum(r_list))

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

            U_concat = np.hstack(U_true)
            noise = sigma * np.random.randn(n)
            y_train = U_concat @ beta + noise

            n_test = n // 10
            U_test_1 = generate_rademacher((n_test, r_list[0]))
            epsilon_test = generate_rademacher((n_test, r_list[0]))
            U_test_2 = np.hstack([rho * U_test_1[:, :r_list[0]] + np.sqrt(1 - rho**2) * epsilon_test, generate_rademacher((n_test, r_list[1] - r_list[0]))])
            U_test_3 = generate_rademacher((n_test, r_list[2]))
            U_test = [U_test_1, U_test_2, U_test_3]
            Zt = [np.random.randn(n_test, p) / np.sqrt(n) for p in p_list]
            Xt = [(1/n) * U_test[i] @ D @ V.T + Zt[i] for i, (D, V) in enumerate(zip([D1, D2, D3], [V1, V2, V3]))]
            X_test_list = Xt
            y_test = np.hstack(U_test) @ beta + sigma * np.random.randn(n_test)

            # --- AMP with clustering (HSS) ---
            pipe_cluster = MultimodalPCAPipelineClustering()
            pipe_cluster.relation = "linear"
            result_cluster = pipe_cluster.denoise_amp(
                X_list, K_list,
                compute_clusters=True,
                amp_iters=amp_iters,
                similarity_method="cca"
            )
            U_cluster = result_cluster["U_denoised"]

            # --- AMP without clustering (distinct) ---
            pipe_cluster = MultimodalPCAPipelineClustering()
            pipe_cluster.relation = "linear"
            pipe_cluster.y_train = y_train
            U_cluster = pipe_cluster.denoise_amp(X_list, K_list, compute_clusters=True, num_clusters=2, amp_iters=amp_iters)["U_denoised"]
            _, y_pred_cluster = pipeline.predict_from_test_data(X_test_list, pipe_cluster.amp_results, n, relation="linear")
            
            pipe_same = MultimodalPCAPipeline()
            pipe_same.relation = "linear"
            pipe_same.y_train = y_train
            U_same = pipe_same.denoise_amp(X_list, K_list, cluster_labels_U=np.array([0, 0, 0]), amp_iters=amp_iters)["U_denoised"]
            _, y_pred_same = pipeline.predict_from_test_data(X_test_list, pipe_same.amp_results, n, relation="linear")
            
            pipe_distinct = MultimodalPCAPipeline()
            pipe_distinct.relation = "linear"
            pipe_distinct.y_train = y_train
            U_distinct = pipe_distinct.denoise_amp(X_list, K_list, cluster_labels_U=np.array([0, 1, 2]), amp_iters=amp_iters)["U_denoised"]
            _, y_pred_distinct = pipeline.predict_from_test_data(X_test_list, pipe_distinct.amp_results, n, relation="linear")

            # --- Reconstruction Errors ---
            for i in range(3):
                errors_clustered[i].append(reconstruction_error(U_cluster[i][:, :, -1], U_true[i]))
                errors_same[i].append(reconstruction_error(U_same[i][:, :, -1], U_true[i]))
                errors_distinct[i].append(reconstruction_error(U_distinct[i][:, :, -1], U_true[i]))
            
            pred_err_clustered.append(np.mean((y_test - y_pred_cluster) ** 2))
            pred_err_same.append(np.mean((y_test - y_pred_same) ** 2))
            pred_err_distinct.append(np.mean((y_test - y_pred_distinct) ** 2))

        # --- Store average errors ---
        result_by_rho[rho] = {
            "clustered recontruction error": [np.mean(errors_clustered[i]) for i in range(3)],
            "distinct recontruction error": [np.mean(errors_distinct[i]) for i in range(3)],
            "same recontruction error": [np.mean(errors_same[i]) for i in range(3)],
            "pred_clustered": np.mean(pred_err_clustered),
            "pred_same_cluster": np.mean(pred_err_same),
            "pred_distinct_clusters": np.mean(pred_err_distinct)
        }

        print(f"\n=== For rho = {rho}, the results are {result_by_rho[rho]} ===")

    return result_by_rho

def main():
    print(">>> Script started", flush=True)
    load_modules()

    if len(sys.argv) < 2:
        print("Usage: python pred_error_vary_row.py <n>", flush=True)
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

 #- Save reconstruction errors (per modality and method) ---
    rows = []
    for rho, res in results.items():
        for method in ["clustered", "distinct", "same"]:
            for mod in range(3):
                rows.append({
                    "rho": rho,
                    "Method": method,
                    "Modality": mod + 1,
                    "Error": res[f"{method} recontruction error"][mod]
                })

    df_recon = pd.DataFrame(rows)
    os.makedirs("Results/pred_vary_rho", exist_ok=True)
    df_recon.to_csv(f"Results/pred_vary_rho/rho_results_n_{n}.csv", index=False)
    print(f">>> Reconstruction errors saved to Results/pred_vary_rho/rho_results_n_{n}.csv")
    
    # --- Save prediction errors (per rho and method) ---
    pred_rows = []
    for rho, res in results.items():
        pred_rows.append({
            "rho": rho,
            "pred_clustered": res["pred_clustered"],
            "pred_same_cluster": res["pred_same_cluster"],
            "pred_distinct_clusters": res["pred_distinct_clusters"]
        })

    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(f"Results/pred_vary_rho/prediction_results_n_{n}.csv", index=False)
    print(f">>> Prediction errors saved to Results/pred_vary_rho/prediction_results_n_{n}.csv")
    

if __name__ == "__main__":
    main()