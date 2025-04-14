import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import scipy.cluster.hierarchy as sch
from sklearn.neighbors import NearestNeighbors
import matplotlib as plt
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial import procrustes
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

def compute_kmac(Y_ker, Y_gr, epsilon=0.1, sigma=1.0):
    """
    Compute the kernel measure of association (KMAc) between two modalities.

    Parameters
    ----------
    Y_ker : ndarray of shape (n, d_ker)
        Data samples for the modality used for computing kernel similarity.
    Y_gr : ndarray of shape (n, d_gr)
        Data samples for the modality used for constructing the geometric graph.
    epsilon : float, optional
        Distance threshold for connecting nodes in the geometric graph.
    sigma : float, optional
        Bandwidth parameter for the RBF kernel.

    Returns
    -------
    kmac_value : float
        The computed kernel measure of association.
    """
    n = Y_ker.shape[0]  # Number of samples

    # Step 1: Compute kernel matrix K(Y_ker, Y_ker) using the RBF kernel
    K_matrix = rbf_kernel(Y_ker, Y_ker, gamma=1/(2 * sigma**2))

    # Step 2: Build sparse adjacency matrix for geometric graph
    # Fit Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(Y_gr)
    _, indices = nbrs.kneighbors(Y_gr)

    # Construct symmetric adjacency list (exclude self-loop)
    adj_list = {i: set() for i in range(n)}
    for i in range(n):
        j = indices[i, 1]  # Skip self (index 0)
        adj_list[i].add(j)
        adj_list[j].add(i)  # Make symmetric



    # Step 3: Degree vector (avoid divide-by-zero) and numerator

    degrees = np.array([len(adj_list[i]) for i in range(n)])
    degrees[degrees == 0] = 1  # Avoid divide-by-zero

    local_kernel_sums = []
    for i in range(n):
        degree_i = degrees[i]
        neighbors = adj_list[i]
        if degree_i > 0:
            sum_K = sum(K_matrix[i, j] for j in neighbors)
            local_kernel_sums.append(sum_K / degree_i)

    first_term = np.mean(local_kernel_sums)
    second_term = (np.sum(K_matrix) - np.trace(K_matrix)) / (n * (n - 1)) if n > 1 else 0

    numerator = first_term - second_term

    # Step 4: Compute denominator (kernel variance term)
    # Sum of off-diagonal elements
    total_sum = np.sum(K_matrix) - np.trace(K_matrix)
    kernel_variance = 1 - (total_sum / (n * (n - 1)))


    return numerator / kernel_variance if kernel_variance != 0 else 0.0


# 1. Hilbert-Schmidt Independence Criterion (HSIC)
def kernel_similarity(X, Y, sigma=1.0):
    """Computes similarity using HSIC with RBF kernel."""
    K_X = rbf_kernel(X, X, gamma=1/(2*sigma**2))
    K_Y = rbf_kernel(Y, Y, gamma=1/(2*sigma**2))
    return np.mean(K_X * K_Y)

# 2. PCA-based Correlation Similarity
def pca_correlation_similarity(X, Y, n_components=5):
    """Computes similarity using PCA and correlation."""
    pca_X = PCA(n_components=n_components).fit_transform(X)
    pca_Y = PCA(n_components=n_components).fit_transform(Y)
    correlations = [pearsonr(pca_X[:, i], pca_Y[:, i])[0] for i in range(n_components)]
    return np.mean(correlations)

# 3. Canonical Correlation Analysis (CCA)
def cca_similarity(X, Y, n_components=None):
    """Computes similarity using Canonical Correlation Analysis (CCA)."""
    max_components = min(X.shape[1], Y.shape[1])
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)

    # Compute correlation between corresponding canonical variables
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    mean_corr = np.mean(corrs)

    # Scale from [-1, 1] → [0, 1]
    scaled_similarity = (mean_corr + 1) / 2

    return scaled_similarity

# 4. Procrustes Analysis
def procrustes_similarity(X, Y):
    """Computes similarity using Procrustes analysis (alignment-based)."""
    mtx1, mtx2, disparity = procrustes(X, Y)
    return 1 - disparity  # Similarity is inverse of disparity

# 5. Wasserstein Distance
def wasserstein_similarity(X, Y):
    """Computes similarity using the Wasserstein distance (Earth Mover's Distance)."""
    dist = wasserstein_distance(X.ravel(), Y.ravel())  # Convert to similarity
    return 1 / (1 + dist)

# 6. Graph-Based Diffusion Similarity
def graph_diffusion_similarity(X, Y, k=5, sigma=1):
    """Computes similarity using graph-based diffusion distance."""
    
    A_X = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    A_Y = kneighbors_graph(Y, k, mode='connectivity', include_self=False)

    D_X = shortest_path(A_X, directed=False)
    D_Y = shortest_path(A_Y, directed=False)

    diff = np.linalg.norm(D_X - D_Y)

    if sigma is None:
        sigma = diff  # or some fixed value

    similarity = np.exp(- (diff ** 2) / (sigma ** 2))

    return similarity

# 7. Hilbert-Schmidt Similarity (HSS)
def hilbert_schmidt_similarity(Y1, Y2, epsilon=0.1, sigma=1.0):
    """Computes the Hilbert-Schmidt Similarity (HSS) between two modalities."""
    kmac_1 = compute_kmac(Y1, Y2, epsilon=epsilon, sigma=sigma)
    kmac_2 = compute_kmac(Y2, Y1, epsilon=epsilon, sigma=sigma)
    return 0.5 * (kmac_1 + kmac_2)  # Symmetrization

# 8. Cosine Similarity (Only if Dimensions Match)
def cosine_similarity_if_same_dim(X, Y):
    """Computes cosine similarity only if dimensions match."""
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Cosine similarity requires both modalities to have the same dimension.")
    return np.mean(cosine_similarity(X, Y))  # Average over all samples


class ModalityClusterer:
    """
    A class for computing similarity matrices and clustering modalities.

    Attributes
    ----------
    modalities : list of ndarrays
        A list of modality matrices, where each matrix has shape (n, d_k).
    similarity_matrix : ndarray
        Stores the computed similarity matrix for reuse.
    linkage_matrix : ndarray
        Stores the hierarchical clustering linkage matrix.
    """

    def __init__(self, modalities):
        """
        Initialize the ModalityClusterer with a list of modalities.

        Parameters
        ----------
        modalities : list of ndarrays
            A list of m modality matrices, where each matrix has shape (n, d_k).
        """
        self.modalities = modalities
        self.similarity_matrix = None
        self.linkage_matrix = None  # Stores linkage matrix for dendrograms

    def compute_similarity_matrix(self, similarity_metric="hss", **kwargs):
        """
        Compute the similarity matrix between modalities using the specified similarity metric.

        Parameters
        ----------
        similarity_metric : str, optional
            The similarity function to use. Options: "hss", "hsic", "pca", "cca",
            "procrustes", "wasserstein", "graph_diffusion", "cosine".
        kwargs : dict
            Additional parameters for similarity functions.

        Returns
        -------
        similarity_matrix : ndarray of shape (m, m)
            The computed similarity matrix between modalities.
        """
        m = len(self.modalities)
        self.similarity_matrix = np.zeros((m, m))

        similarity_functions = {
            "hss": hilbert_schmidt_similarity,
            "hsic": kernel_similarity,
            "pca": pca_correlation_similarity,
            "cca": cca_similarity,
            "procrustes": procrustes_similarity,
            "wasserstein": wasserstein_similarity,
            "graph_diffusion": graph_diffusion_similarity,
            "cosine": cosine_similarity_if_same_dim
        }

        if similarity_metric not in similarity_functions:
            raise ValueError(f"Invalid similarity metric. Choose from {list(similarity_functions.keys())}")

        similarity_function = similarity_functions[similarity_metric]

        for i in range(m):
            for j in range(i, m):
                try:
                    sim = similarity_function(self.modalities[i], self.modalities[j], **kwargs)
                    self.similarity_matrix[i, j] = sim
                    self.similarity_matrix[j, i] = sim  # Ensure symmetry
                except ValueError as e:
                    print(f"Skipping similarity computation for ({i}, {j}): {e}")

        return self.similarity_matrix

    def cluster_modalities(self, similarity_metric="hss", num_clusters=None, threshold=None, method="average", **kwargs):
        """
        Perform hierarchical clustering on modalities based on a computed similarity matrix.

        Parameters
        ----------
        similarity_metric : str, optional
            The similarity function to use. Options: "hss", "hsic", "pca", "cca", 
            "procrustes", "wasserstein", "graph_diffusion", "cosine".
        num_clusters : int, optional
            The desired number of clusters. If None, threshold must be provided.
        threshold : float, optional
            The distance threshold for forming clusters. If None and num_clusters is also None, an automatic threshold will be chosen based on largest distance jump.
        method : str, optional
            The linkage method for clustering (default is "average").
        kwargs : dict
            Additional parameters for similarity functions.

        Returns
        -------
        cluster_labels : ndarray of shape (m,)
            Cluster labels for each modality.
        """
        # Compute similarity matrix if not already computed
        if self.similarity_matrix is None:
            self.compute_similarity_matrix(similarity_metric, **kwargs)

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - self.similarity_matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is zero

        # Convert to condensed distance matrix for scipy
        condensed_distance = sch.distance.squareform(distance_matrix)

        # Perform hierarchical clustering
        self.linkage_matrix = sch.linkage(condensed_distance, method=method)

        # Determine cluster labels
        if num_clusters is not None:
            cluster_labels = sch.fcluster(self.linkage_matrix, num_clusters, criterion='maxclust')
        elif threshold is not None:
            cluster_labels = sch.fcluster(self.linkage_matrix, threshold, criterion='distance')
        else:
            # Automatic selection using silhouette score
            from sklearn.metrics import silhouette_score

            best_score = -1
            best_k = 1
            #max_k = min(10, distance_matrix.shape[0])
            max_k = 3

            # Allow one cluster if all distances are effectively zero
            if np.allclose(distance_matrix, 0, atol=1e-3):
                return np.ones(distance_matrix.shape[0], dtype=int)

            for k in range(2, max_k + 1):
                labels = sch.fcluster(self.linkage_matrix, k, criterion='maxclust')
                try:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    if score > best_score:
                        best_score = score
                        best_k = k
                except Exception:
                    continue  # Skip invalid configurations

            cluster_labels = sch.fcluster(self.linkage_matrix, best_k, criterion='maxclust')

        return cluster_labels

    def plot_dendrogram(self, labels=None, title="Dendrogram of Modality Clustering"):
        """
        Plot the hierarchical clustering dendrogram.

        Parameters
        ----------
        labels : list of str, optional
            Labels for each modality. If None, defaults to numerical labels.
        title : str, optional
            Title for the dendrogram plot.
        """
        if self.linkage_matrix is None:
            raise ValueError("Clustering has not been performed. Run `cluster_modalities` first.")

        plt.figure(figsize=(10, 5))
        sch.dendrogram(self.linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=12)
        plt.title(title)
        plt.xlabel("Modalities")
        plt.ylabel("Distance")
        plt.show()