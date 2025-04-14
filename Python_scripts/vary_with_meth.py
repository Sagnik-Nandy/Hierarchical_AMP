import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import importlib
from tqdm import tqdm
import pandas as pd

# Dynamically import modules
amp = importlib.import_module("amp")
pca_pack = importlib.import_module("pca_pack")
preprocessing = importlib.import_module("preprocessing")
emp_bayes = importlib.import_module("emp_bayes")
hierarchical = importlib.import_module("hierarchical_clustering_modalities")
pipeline = importlib.import_module("complete_pipeline")

# Reload to reflect any changes made without restarting kernel
importlib.reload(amp)
importlib.reload(pca_pack)
importlib.reload(preprocessing)
importlib.reload(emp_bayes)
importlib.reload(hierarchical)
importlib.reload(pipeline)

# Now access objects from reloaded modules
ebamp_multimodal = amp.ebamp_multimodal
MultiModalityPCA = pca_pack.MultiModalityPCA
MultiModalityPCADiagnostics = preprocessing.MultiModalityPCADiagnostics
ClusterEmpiricalBayes = emp_bayes.ClusterEmpiricalBayes
ModalityClusterer = hierarchical.ModalityClusterer
MultimodalPCAPipeline = pipeline.MultimodalPCAPipelineSimulation
MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClusteringSimulation

# Define gamma for each modality
gamma_list = [0.25, 0.25, 0.5]
r_list = [1, 2, 1]
n_values = [3000, 3500, 4000, 4500, 5000]
num_trials = 50
amp_iters = 10

# --- Utility: Rademacher generator ---
def generate_rademacher(shape):
    return np.random.choice([-1, 1], size=shape)

# --- Utility: Reconstruction error ---
def reconstruction_error(U_est, U_true):
    P_est = U_est @ U_est.T
    P_true = U_true @ U_true.T
    return np.linalg.norm(P_est - P_true, 'fro')**2 / (U_true.shape[0]**2)

# --- Main experiment runner ---
def run_amp_comparison_experiment(n=2000, p_list=None, r_list=None, num_trials=100, num_clusters = None, threshold = None, amp_iters=15):
    errors_clustered_all = {method: {i: [] for i in range(3)} for method in ["hss", "cca", "wasserstein"]}

    for trial in tqdm(range(num_trials), desc="Trials"):
        # Generate U
        U1 = generate_rademacher((n, r_list[0]))
        U2 = np.hstack([U1[:, :r_list[0]], generate_rademacher((n, r_list[1] - r_list[0]))])
        U3 = generate_rademacher((n, r_list[2]))
        U_true = [U1, U2, U3]

        # Generate V
        V1 = generate_rademacher((p_list[0], r_list[0]))
        V2 = generate_rademacher((p_list[1], r_list[1]))
        V3 = generate_rademacher((p_list[2], r_list[2]))

        # D matrices
        D1 = np.diag([5 * (i+1) for i in range(r_list[0])])
        D2 = np.diag([5 * (i+1) for i in range(r_list[1])])
        D3 = np.diag([5 * (i+1) for i in range(r_list[2])])

        # Noise
        Z1 = np.random.randn(n, p_list[0]) / np.sqrt(n)
        Z2 = np.random.randn(n, p_list[1]) / np.sqrt(n)
        Z3 = np.random.randn(n, p_list[2]) / np.sqrt(n)

        # Observed data
        X1 = (1/n) * U1 @ D1 @ V1.T + Z1
        X2 = (1/n) * U2 @ D2 @ V2.T + Z2
        X3 = (1/n) * U3 @ D3 @ V3.T + Z3
        X_list = [X1, X2, X3]
        K_list = r_list

        for method in ["hss", "cca", "wasserstein"]:
            pipe_cluster = MultimodalPCAPipelineClustering()
            result_cluster = pipe_cluster.denoise_amp(
                X_list, K_list,
                compute_clusters=True,
                num_clusters=num_clusters,
                amp_iters=amp_iters,
                similarity_method=method
            )
            U_cluster = result_cluster["U_denoised"]

            # Compute reconstruction errors
            for i in range(3):
                errors_clustered_all[method][i].append(reconstruction_error(U_cluster[i][:, :, -1], U_true[i]))

        print(
            f"[Trial {trial+1}] "
            f"Reconstruction Errors - "
            f"{', '.join([f'{method}: {[round(errors_clustered_all[method][i][-1], 5) for i in range(3)]}' for method in errors_clustered_all])}"
        )

    return errors_clustered_all

# Store results
all_errors = {}

for n in n_values:
    # Compute p_k = γ_k * n
    p_list = [int(gamma * n) for gamma in gamma_list]

    print(f"\n=== Running experiments for n = {n}, p_list = {p_list} ===")

    # Run the comparison experiment with explicit arguments
    errors_clustered_all = run_amp_comparison_experiment(
        n=n,
        p_list=p_list,
        r_list=r_list,
        num_trials=num_trials,
        amp_iters=amp_iters
    )

    # Compute average reconstruction error for each modality
    avg_errors_clustered = {method: [np.mean(errors_clustered_all[method][i]) for i in range(3)] for method in errors_clustered_all}

    # Store in results dict (only averages)
    all_errors[n] = avg_errors_clustered

    print(f"Averages for n = {n}:")
    for method in avg_errors_clustered:
        print(f"  {method}: {avg_errors_clustered[method]}")

print("\n=== All experiments completed! ===")


# Unpack results
n_values_sorted = sorted(all_errors.keys())

# For each modality (0, 1, 2), prepare a plot
for modality_idx in range(3):
    plt.figure(figsize=(8, 5))
    for method in all_errors[n_values_sorted[0]]:
        errors = [all_errors[n][method][modality_idx] for n in n_values_sorted]
        plt.plot(n_values_sorted, errors, marker='o', label=method)

    plt.title(f"Modality {modality_idx + 1} - Avg Reconstruction Error vs Sample Size")
    plt.xlabel("Sample size (n)")
    plt.ylabel("Avg Reconstruction Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"modality_{modality_idx + 1}_rec_error_vs_n.png")
    plt.show()

df = pd.DataFrame(all_errors)
df.to_csv('output_func_n_setup_1.csv', index=False)
