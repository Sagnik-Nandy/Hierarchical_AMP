import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import scipy.cluster.hierarchy as sch
import matplotlib as plt
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial import procrustes
import networkx as nx
from scipy.sparse.csgraph import shortest_path

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

    # Step 2: Construct the geometric graph G_n using Y_gr
    G = nx.Graph()
    
    # Ensure all nodes are in the graph
    G.add_nodes_from(range(n))  

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(Y_gr[i, :] - Y_gr[j, :]) < epsilon:  # Connect if within threshold
                G.add_edge(i, j)

    # Compute degrees safely
    degrees = np.array([G.degree[i] if i in G.nodes else 0 for i in range(n)])  # ✅ FIXED
    degrees[degrees == 0] = 1  # Avoid division by zero

    # Step 3: Compute numerator
    first_term = np.mean([
        (1 / degrees[i]) * sum(K_matrix[i, j] for j in G.neighbors(i))
        for i in range(n) if degrees[i] > 0
    ]) if n > 1 else 0  # Ensure valid mean computation

    second_term = np.sum(K_matrix) / (n * (n - 1)) if n > 1 else 0  # Avoid division by zero

    numerator = first_term - second_term

    # Step 4: Compute denominator (kernel variance term)
    kernel_variance = np.sum([
        np.linalg.norm(K_matrix[i, :] - K_matrix[j, :])**2
        for i in range(n) for j in range(n) if i != j
    ]) / (2 * n * (n - 1)) if n > 1 else 1  # Avoid division by zero

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
def cca_similarity(X, Y, n_components=2):
    """Computes similarity using Canonical Correlation Analysis (CCA)."""
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    return np.mean(np.corrcoef(X_c.T, Y_c.T)[0:n_components, n_components:])

# 4. Procrustes Analysis
def procrustes_similarity(X, Y):
    """Computes similarity using Procrustes analysis (alignment-based)."""
    mtx1, mtx2, disparity = procrustes(X, Y)
    return 1 - disparity  # Similarity is inverse of disparity

# 5. Wasserstein Distance
def wasserstein_similarity(X, Y):
    """Computes similarity using the Wasserstein distance (Earth Mover's Distance)."""
    return -wasserstein_distance(X.ravel(), Y.ravel())  # Convert to similarity

# 6. Graph-Based Diffusion Similarity
def graph_diffusion_similarity(X, Y, k=5):
    """Computes similarity using graph-based diffusion distance."""
    G_X = nx.k_nearest_neighbors(nx.Graph(), X, k)
    G_Y = nx.k_nearest_neighbors(nx.Graph(), Y, k)

    D_X = shortest_path(nx.to_numpy_array(G_X))
    D_Y = shortest_path(nx.to_numpy_array(G_Y))

    return -np.linalg.norm(D_X - D_Y)  # Negative norm as similarity

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
            The distance threshold for forming clusters. If None, num_clusters must be provided.
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
            raise ValueError("Either num_clusters or threshold must be provided.")

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