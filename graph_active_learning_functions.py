# Import necessary packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import graphlearning as gl
from scipy.special import jn
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from itertools import product
from joblib import Parallel, delayed







def build_custom_knn_graph(X, K=50):
    """
    Constructs the graph G using kNN on the nodes X. Cosine similarity is used to 
    construct the weight matrix.

    Parameters:
    X (numpy.ndarray): Data matrix (p, N).
    K (integer): Integer representing number of nearest neighbors (defaulted to 50).

    Returns:
    G (graph), W (scipy.sparse.csr_matrix): The graph G and its symmetric weight matrix W.
    """

    # Initialize the dimensions (N), nearest neighbors (nbrs), distance metric (ang_dist) 
    # and variance (sigma)
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute').fit(X)
    cos_dist, indices = nbrs.kneighbors(X)
    inner_products = np.clip(1.0 - cos_dist, -1.0, 1.0)
    ang_dist = np.arccos(inner_products)
    sigma = np.sqrt(ang_dist[:, K-1])

    row_idx = np.repeat(np.arange(N), K)
    col_idx = indices.flatten()

    sigma_i_expanded = np.repeat(sigma, K)
    sigma_j_expanded = sigma[col_idx]

    denom = sigma_i_expanded * sigma_j_expanded
    denom[denom == 0] = 1e-10

    # Compute weight value
    distances_squared = (ang_dist.flatten())**2
    weights = np.exp(-distances_squared / denom)

    # Construct the similarity matrix W_knn
    W_knn = sps.csr_matrix((weights, (row_idx, col_idx)), shape=(N, N))

    # Make W_knn symmetric
    W = (W_knn + W_knn.T) / 2.0

    # Wrap in a graphlearning graph object
    G = gl.graph(W)

    return G, W








def compute_and_partition_laplacian(W, labeled_indices):
    """
    Computes the unnormalized Graph Laplacian L = D - W and partitions it
    into blocks based on labeled and unlabeled pixel indices.

    Parameters:
    W (scipy.sparse.csr_matrix): The symmetric weight matrix.
    labeled_indices (np.ndarray or list): Indices of the sampled training pixels.

    Returns:
    L, L_ll, L_lu, L_ul, L_uu (scipy.sparse.csr_matrix): The full Laplacian and its blocks.
    """
    N = W.shape[0]

    # 1. Compute the Degree Matrix D
    # The degree of a node is the sum of its edge weights
    # We flatten the matrix to a 1D array to construct the diagonal matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sps.diags(degrees, format='csr')

    # 2. Compute the Unnormalized Graph Laplacian L
    L = D - W

    # 3. Identify unlabeled indices
    all_indices = np.arange(N)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)

    # 4. Partition L into block matrices
    # Because L is a scipy.sparse matrix, we can slice it efficiently using array indexing

    # L_ll: Labeled-to-Labeled
    L_ll = L[labeled_indices, :][:, labeled_indices]

    # L_lu: Labeled-to-Unlabeled
    L_lu = L[labeled_indices, :][:, unlabeled_indices]

    # L_ul: Unlabeled-to-Labeled
    L_ul = L[unlabeled_indices, :][:, labeled_indices]

    # L_uu: Unlabeled-to-Unlabeled
    L_uu = L[unlabeled_indices, :][:, unlabeled_indices]

    return L, L_ll, L_lu, L_ul, L_uu

# --- Example Usage ---
# Assuming W is the sparse matrix from our KNN graph output
# and we randomly selected 50 pixels to act as our "active learning" labels
# N = W.shape[0]
# mock_labeled_indices = np.random.choice(N, size=50, replace=False)
# L, L_ll, L_lu, L_ul, L_uu = compute_and_partition_laplacian(W, mock_labeled_indices)





















def project_onto_simplex(V):
    """
    Projects each column of matrix V onto the probability simplex.
    Based on the fast algorithm by Wang and Carreira-Perpinan (2013).

    Parameters:
    V (numpy.ndarray): Matrix of shape (q, N) where q is the number of
                       end-members and N is the number of pixels.

    Returns:
    numpy.ndarray: Projected matrix of the same shape.
    """
    # Sort each column in descending order
    U = np.sort(V, axis=0)[::-1, :]

    # Cumulative sum down the columns
    cssv = np.cumsum(U, axis=0)

    # Array of 1-based indices for the rows
    indices = np.arange(1, V.shape[0] + 1)[:, np.newaxis]

    # Find the condition threshold
    cond = U - (cssv - 1) / indices > 0

    # rho is the maximum index where the condition holds true
    rho = np.max(cond * indices, axis=0)

    # Calculate theta (the shift value) for each column
    theta = (cssv[rho - 1, np.arange(V.shape[1])] - 1) / rho

    # Apply the projection
    return np.maximum(V - theta, 0)



























def generate_one_hot_labels(A_exact):
    """
    Converts exact abundance vectors into one-hot pseudo-labels.
    This mimics an expert identifying the most significant end-member[cite: 322].

    Parameters:
    A_exact (numpy.ndarray): Exact abundance matrix (q, M).

    Returns:
    numpy.ndarray: One-hot encoded pseudo-label matrix (q, M).
    """
    q, M = A_exact.shape
    A_oh = np.zeros((q, M))

    # Find the index of the maximum abundance for each pixel (column)
    max_indices = np.argmax(A_exact, axis=0)

    # Set the dominant material to 1.0, rest remain 0.0
    A_oh[max_indices, np.arange(M)] = 1.0

    return A_oh





















def compute_vopt_acquisition(V, Lambda_mat, labeled_indices, gamma=0.1):
    """
    Computes the Variance Optimality (VOpt) acquisition function for all nodes.

    Parameters:
    V (numpy.ndarray): Eigenvectors of the Laplacian (N, M_eigs).
    Lambda_mat (numpy.ndarray): Diagonal eigenvalue matrix (M_eigs, M_eigs).
    labeled_indices (list or np.ndarray): Indices of currently labeled nodes.
    gamma (float): Positive constant (paper uses 0.1).

    Returns:
    numpy.ndarray: VOpt scores for all nodes.
    """
    N, M_eigs = V.shape

    # 1. Compute the projection matrix component: (1/gamma^2) * V^T * P^T * P * V
    # Instead of building a massive projection matrix P, we can just slice V!
    # P * V is exactly equivalent to taking the rows of V corresponding to labeled_indices.
    V_labeled = V[labeled_indices, :]

    # Calculate V^T * P^T * P * V
    V_proj_T_V_proj = V_labeled.T @ V_labeled

    # 2. Compute the Gaussian correlation matrix C
    # C = (Lambda + (1/gamma^2) * V^T * P^T * P * V)^-1
    C_inv = Lambda_mat + (1.0 / (gamma**2)) * V_proj_T_V_proj
    C = np.linalg.inv(C_inv) # C is small (M_eigs x M_eigs), so direct inversion is fast

    # 3. Compute VOpt for all nodes
    # A_VOpt(x_k) = ||C v_k||_2^2 / (gamma^2 + v_k^T C v_k)
    # v_k is the k-th row of V

    # We can vectorize this to avoid a slow Python for-loop over N nodes
    # C @ V.T gives a matrix where columns are (C v_k). Shape: (M_eigs, N)
    CV_T = C @ V.T

    # ||C v_k||_2^2 is the squared L2 norm of each column
    numerator = np.sum(CV_T**2, axis=0)

    # v_k^T C v_k can be found by taking the dot product of V with (C V.T).T
    # This is equivalent to row-wise dot product of V and (V @ C.T)
    denominator = (gamma**2) + np.sum(V * (V @ C.T), axis=1)

    vopt_scores = numerator / denominator

    return vopt_scores

















# Spectral Angle Distance (SAD)

def spectral_angle(s_i, s_j):
    """
    Returns spectral angle between s_i and s_j.

    Parameters:
    s_i, s_j (numpy.ndarray): Two vectors of same length.

    Returns:
    float: spectral angle in radians.
    """
    cos_theta = np.dot(s_i, s_j) / (np.linalg.norm(s_i) * np.linalg.norm(s_j))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def SAD(S, S_gt):
    """
    Returns the spectral angle distance between two matrices, S and S_gt.

    Parameters:
    S, S_gt (numpy.ndarray): Two matrices of the same dimension representing 
    the estimated and ground truth endmembers respectively.

    Returns:
    float: spectral angle distance in degrees.
    """
    p = S_gt.shape[0]
    total = sum(spectral_angle(S[i], S_gt[i]) for i in range(p))
    return (total / p) * (180 / np.pi)



# Root-mean-square error (RMSE)

def RMSE(A, A_gt):
    """
    Returns the RMSE between matrices A and A_gt.

    Parameters:
    A, A_gt (numpy.ndarray): Two matrices of the same dimension representing 
    the estimated and ground truth abundances respectively.

    Returns:
    float: The root mean square error.
    """
    m,n = A.shape
    return 100*np.sqrt(np.sum((A-A_gt)**2)/m/n)
    
















def algo_2_glu(X, X_hat, A_hat, alpha=10.0, k=50):
    """
    Executes the Graph Learning Unmixing (GLU) model (Algorithm 2).

    Parameters:
    X (numpy.ndarray): Unlabeled data matrix (p, N).
    X_hat (numpy.ndarray): Labeled data matrix (p, M).
    A_hat (numpy.ndarray): Labeled abundance matrix / pseudo-labels (q, M).
    alpha (float): Weighting parameter for end-member estimation.
    k (int): Number of nearest neighbors for graph construction.

    Returns:
    A_GLU (numpy.ndarray): Estimated abundance map (q, N).
    S_GLU (numpy.ndarray): Estimated end-member spectrum matrix (p, q).
    """
    p, N = X.shape
    q, M = A_hat.shape

    # --- Initialization ---
    # Combine data to match the paper: \tilde{X} = [\hat{X}, X]
    X_tilde = np.concatenate((X_hat, X), axis=1) # Shape: (p, M + N)

    # Scikit-learn expects (samples, features), so we transpose to (M + N, p)
    G, W = build_custom_knn_graph(X_tilde.T, K=k)

    # Since X_hat is the first M columns, our labeled indices are 0 to M-1
    labeled_indices = np.arange(M)
    L, L_ll, L_lu, L_ul, L_uu = compute_and_partition_laplacian(W, labeled_indices)

    # --- Step 1: Graph Learning Step ---
    # Setup the right-hand side of the transposed linear system: -L_lu^T * A_hat^T
    RHS = -(L_lu.T @ A_hat.T) # Shape will be (N, q)

    A_GL_T = np.zeros((N, q))

    # Solve the sparse linear system for each end-member column
    for i in range(q):
        A_GL_T[:, i], _ = spla.cg(L_uu, RHS[:, i])

    A_GL = A_GL_T.T # Transpose back to get shape (q, N)

    # --- Step 2: Projection ---
    # Project the graph Laplace learning solution onto the probability simplex
    A_GLU = project_onto_simplex(A_GL)

    # --- Step 3: Estimate End-member Spectrum Matrix ---
    term1 = (X @ A_GLU.T) + (alpha**2) * (X_hat @ A_hat.T) # Shape: (p, q)
    term2 = (A_GLU @ A_GLU.T) + (alpha**2) * (A_hat @ A_hat.T) # Shape: (q, q)

    S_GLU_0 = term1 @ np.linalg.inv(term2)

    # Apply non-negativity constraint
    S_GLU = np.maximum(S_GLU_0, 0)

    return A_GLU, S_GLU








































def algo_3_grsu(X, X_hat, A_hat, alpha, lam, gamma, rho, max_iters=1000, eps=1e-3, k=50):
    """
    Executes the Graph-Regularized Semi-Supervised Unmixing (GRSU) model (Algorithm 3).

    Parameters:
    X (numpy.ndarray): Unlabeled data matrix (p, N).
    X_hat (numpy.ndarray): Labeled data matrix (p, M).
    A_hat (numpy.ndarray): Labeled abundance matrix / pseudo-labels (q, M).
    alpha (float): Weighting parameter for end-member estimation.
    lam (float): Graph regularization parameter (lambda).
    gamma (float): ADMM penalty parameter for S-T constraint.
    rho (float): ADMM penalty parameter for A-B constraint.
    max_iters (int): Maximum number of ADMM iterations.
    eps (float): Error tolerance for convergence.
    k (int): Number of nearest neighbors for graph construction.

    Returns:
    A_GRSU (numpy.ndarray): Final estimated abundance map (q, N).
    S_GRSU (numpy.ndarray): Final estimated end-member spectrum matrix (p, q).
    """
    p, N = X.shape
    q, M = A_hat.shape

    # --- 1. Graph Construction & Laplacian Partitioning ---
    X_tilde = np.concatenate((X_hat, X), axis=1)
    G, W = build_custom_knn_graph(X_tilde.T, K=k)

    labeled_indices = np.arange(M)
    L, L_ll, L_lu, L_ul, L_uu = compute_and_partition_laplacian(W, labeled_indices)

    # --- 2. Initialization via GLU (Algorithm 2) ---
    A, S = algo_2_glu(X, X_hat, A_hat, alpha, k=k)

    # Initialize ADMM auxiliary and dual variables
    B = np.copy(A)
    B_bar = np.zeros((q, N))
    T_bar = np.zeros((p, q))

    # Pre-compute constants that do not change during iterations
    I_q = np.eye(q)
    I_N_sparse = sps.eye(N, format='csr')

    # Terms in T's update
    X_hat_A_hat_T_alpha2 = (alpha**2) * (X_hat @ A_hat.T)
    A_hat_A_hat_T_alpha2 = (alpha**2) * (A_hat @ A_hat.T)

    # Terms in B's update
    L_B_system = L_uu + (rho / lam) * I_N_sparse
    L_lu_T_A_hat_T = L_lu.T @ A_hat.T # Shape: (N, q)

    Err = 1.0
    i = 0

    # --- 3. ADMM Iteration Loop ---
    while i < max_iters and Err > eps:

        # a) T subproblem update
        term1_T = (X @ A.T) + X_hat_A_hat_T_alpha2 + gamma * (S + T_bar)
        term2_T = (A @ A.T) + A_hat_A_hat_T_alpha2 + gamma * I_q
        T = term1_T @ np.linalg.inv(term2_T)

        # b) S subproblem update
        S_new = np.maximum(T - T_bar, 0)

        # c) A subproblem update
        term1_A = np.linalg.inv((S_new.T @ S_new) + rho * I_q)
        term2_A = (S_new.T @ X) + rho * (B - B_bar)
        A_unproj = term1_A @ term2_A
        A_new = project_onto_simplex(A_unproj)

        # d) B subproblem update (Graph Regularized Laplace Learning)
        # We solve the transposed system: L_B_system * B^T = RHS_B
        RHS_B = -L_lu_T_A_hat_T + (rho / lam) * (A_new.T + B_bar.T)
        B_new_T = np.zeros((N, q))

        for j in range(q):
            B_new_T[:, j], _ = spla.cg(L_B_system, RHS_B[:, j])
        B_new = B_new_T.T

        # e) Update dual variables
        B_bar_new = B_bar + (A_new - B_new)
        T_bar_new = T_bar + (S_new - T)

        # f) Calculate convergence error
        # Protect against division by zero in the denominator
        norm_S = np.linalg.norm(S, 'fro')
        norm_A = np.linalg.norm(A, 'fro')

        err_S = np.linalg.norm(S_new - S, 'fro') / norm_S if norm_S > 0 else 0
        err_A = np.linalg.norm(A_new - A, 'fro') / norm_A if norm_A > 0 else 0

        Err = max(err_S, err_A)

        # g) Advance state for next iteration
        S = S_new
        A = A_new
        B = B_new
        B_bar = B_bar_new
        T_bar = T_bar_new

        i += 1

    return A, S






















def algo_1_active_learning(X, W, m_initial=5, M_total=40, num_eigs=50, gamma=0.1):
    """
    Executes Algorithm 1: Graph-based Active Learning to sample labeled pixels.

    Parameters:
    X (numpy.ndarray): Data matrix (p, N).
    W (scipy.sparse.csr_matrix): KNN weight matrix.
    m_initial (int): Initial number of randomly sampled pixels.
    M_total (int): Total number of labeled pixels to acquire.
    num_eigs (int): Number of eigenvectors to compute for the low-rank approximation.
    gamma (float): Constant for the acquisition function.

    Returns:
    list: Final list of M_total selected pixel indices.
    """
    N = W.shape[0]

    # 1. Compute the Graph Laplacian
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sps.diags(degrees, format='csr')
    L = D - W

    # 2. Compute truncated eigen-decomposition of L
    # We want the smallest algebraic eigenvalues ('SA' or 'SM')
    # Use shift-invert mode (sigma=0) to reliably find eigenvalues near zero
    eigenvalues, V = spla.eigsh(L, k=num_eigs, which='SM')

    # Ensure eigenvalues are non-negative (Laplacian property, but float math can drift)
    eigenvalues = np.maximum(eigenvalues, 0)
    Lambda_mat = np.diag(eigenvalues)

    # 3. Initialize label set randomly
    # Ensure we don't pick duplicate initial indices
    np.random.seed(42) # Set the seed
    labeled_indices = list(np.random.choice(N, size=m_initial, replace=False))

    # 4. Active Learning Iteration
    while len(labeled_indices) < M_total:

        # Calculate acquisition function scores
        scores = compute_vopt_acquisition(V, Lambda_mat, labeled_indices, gamma=gamma)

        # We only want to select from UNLABELED nodes
        # Force the score of already labeled nodes to -infinity so they aren't picked again
        scores[labeled_indices] = -np.inf

        # Sequential Active Learning: Select the single node with the highest score
        best_node_idx = np.argmax(scores)

        # Update the current label set
        labeled_indices.append(best_node_idx)

    return labeled_indices

# --- Note on Pseudo-Labels ---
# Once algo_1_active_learning returns the list of indices, you would extract
# X_hat = X[:, labeled_indices].
# You would then acquire A_hat (the pseudo-labels) by either human inspection
# or by thresholding an exact abundance map for those specific pixels.
















def run_unmixing_pipeline_example(X, A_gt, S_gt, N, alpha = 10.0, lam = 1.0, gamma = 1.0, rho = 1.0, print_bool = True, GRSU_bool = True):
    # ==========================================
    # Phase 0: Load Data (Mocking Jasper Ridge)
    # ==========================================
    #print("Loading data...")
    #N = 10000  # Number of pixels (e.g., 100x100 image)
    #p = 198    # Number of spectral bands [cite: 286]
    #q = 4      # Number of latent end-members (Tree, Water, Dirt, Road) [cite: 287, 288]

    # Mock full hyperspectral image X: shape (p, N)
    #X = np.random.rand(p, N, seed=42)

    # Mock ground-truth abundance map (only used to simulate human labeling)
    #A_gt = np.random.dirichlet(np.ones(q), size=N, seed=42).T

    # ==========================================
    # Phase 1: Active Learning (Algorithm 1)
    # ==========================================
    if print_bool:
        print("Building initial graph for Active Learning...")
    # Scikit-learn expects (samples, features), so we pass X.T
    # K = 50 was picked for a 10000 pixel image, so roughly 0.5%
    G, W = build_custom_knn_graph(X.T, K=int(N*0.005))

    if print_bool:
        print("Running Active Learning...")
    # Start with 1 random pixel per material (m=4), sample up to 0.4% of total pixels (M=40) [cite: 323]
    # num_eigs = 0.05% of the pixels?
    labeled_indices = algo_1_active_learning(X, W, m_initial=2, M_total=int(0.004*N), num_eigs=int(N*0.005))

    # ==========================================
    # Phase 2: Extract Training Data
    # ==========================================
    if print_bool:
        print("Extracting training data and generating pseudo-labels...")
    # Extract the spectral signatures for the selected pixels
    X_hat = X[:, labeled_indices]

    # Extract ground-truth abundances and convert to One-Hot pseudo-labels [cite: 321, 322]
    A_hat_exact = A_gt[:, labeled_indices]
    A_hat_OH = generate_one_hot_labels(A_hat_exact)

    # ==========================================
    # Phase 3: Semi-Supervised Unmixing
    # ==========================================
    if print_bool and GRSU_bool:
        print("Running GRSU (and GLU) Unmixing...")
    elif print_bool:
         print("Running GLU Unmixing...")

    # Hyperparameters based on the Jasper Ridge dataset in Table II [cite: 339, 340]
    # alpha = 10.0
    # lam = 1.0
    # gamma = 1.0
    # rho = 1.0

    # Note: The paper mentions an overlap between X_hat and X, but updates
    # the abundance map for all pixels in X anyway.

    # If we are only running GLU
    if not GRSU_bool:
        A_final, S_final = algo_2_glu(X, X_hat, A_hat_OH, alpha, k=k)
    else:
        A_final, S_final = algo_3_grsu(
            X=X,
            X_hat=X_hat,
            A_hat=A_hat_OH,
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            rho=rho,
            max_iters=1000,
            eps=1e-3,
            k=50
        )

    # Calculate RMSE and SAD
    A_rmse = RMSE(A_final, A_gt)
    S_sad = SAD(S_final, S_gt)

    if print_bool:
        print("Pipeline Complete!\n")
        print(f"Final Abundance Map Shape: {A_final.shape}")
        print(f"Final End-member Matrix Shape: {S_final.shape}")
        print(f"Final Abundance RMSE: {A_rmse}")
        print(f"Final Endmember SAD: {S_sad}")

    return A_final, S_final, A_rmse, S_sad























# Optimizing Parameters
def sum_RMSE_SAD(X, A_gt, S_gt, N, alpha_0, lam_0, gamma_0, rho_0, print_bool = True):
    """
    Returns the sum RMSE + SAD.

    Parameters:
    X (numpy.ndarray): Data matrix (p, N).
    A_gt
    alpha (numpy.ndarray): Regularization parameter for T subproblem.
    lam (numpy.ndarray): Regularization parameter for B subproblem.
    gamma (numpy.ndarray): Regularization parameter for S subproblem.
    rho (numpy.ndarray): Regularization parameter for A subproblem.
    """
    A_f, S_f, A_RMSE, S_SAD = run_unmixing_pipeline_example(X, A_gt, S_gt, N, alpha_0, lam_0, gamma_0, rho_0, print_bool = print_bool)
    return A_RMSE + S_SAD

def parameter_testing(X, A_gt, S_gt, N, alpha, lam, gamma, rho, print_bool = True):
    """
    Performs grid search on the regularization parameters (alpha, lam, gamma, rho) to find
    the optimal combination of the four, minimizing the sum A_RMSE + S_SAD.

    Parameters:
    alpha (numpy.ndarray): Regularization parameter for T subproblem.
    lam (numpy.ndarray): Regularization parameter for B subproblem.
    gamma (numpy.ndarray): Regularization parameter for S subproblem.
    rho (numpy.ndarray): Regularization parameter for A subproblem.

    Returns:
    best (numpy.ndarray): An array with best combination minimizing A_RMSE + S_SAD
    in the format [alpha, lam, gamma, rho].
    """

    # Create a list of each combination
    combos = list(product(alpha, lam, gamma, rho))

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(sum_RMSE_SAD)(X, A_gt, S_gt, N, alpha_0, lam_0, gamma_0, rho_0, print_bool) for alpha_0, lam_0, gamma_0, rho_0 in combos
    )

    # Create a 4D array to match the set order
    results = np.array(results).reshape(len(alpha), len(lam), len(gamma), len(rho))

    # Find min sum value and the corresponding combination
    idx = np.unravel_index(results.argmin(), results.shape)
    min_result = results[idx]
    alpha_idx, lam_idx, gamma_idx, rho_idx = idx

    # Save the best values
    alpha_best = alpha[alpha_idx]
    lam_best = lam[lam_idx]
    gamma_best = gamma[gamma_idx]
    rho_best = rho[rho_idx]

    # Print the best values
    print(f"Best RMSE + SAD: {min_result}")
    print(f"Best alpha: {alpha_best}")
    print(f"Best lambda: {lam_best}")
    print(f"Best gamma: {gamma_best}")
    print(f"Best rho: {rho_best}")

    return [alpha_best, lam_best, gamma_best, rho_best]











### ALMM.ipynb


def vca(X, R = 2, snr_input=0):
    """
    Finds the endmembers in a sample X.

    Parameters:
    X (numpy.ndarray): Unlabeled data matrix (p, N).
    R (integer): Number of endmembers.
    """
    p, N = X.shape

    # SNR estimation and projection method selection
    X_m = np.mean(X, axis=1, keepdims=True)
    X_o = X - X_m

    Ud, Sd, _ = np.linalg.svd(X_o @ X_o.T / N)
    Ud = Ud[:, :R]

    x_p = Ud.T @ X_o

    P_X = np.sum(X ** 2) / N
    P_x = np.sum(x_p ** 2) / N + np.sum(X_m ** 2)

    if snr_input == 0:
        snr_est = 10 * np.log10((P_x - R / p * P_X) / (P_X - P_x + 1e-12))
    else:
        snr_est = snr_input

    snr_th = 15 + 10 * np.log10(R)

    # Project onto R-dimensional subspace
    if snr_est < snr_th:
        d = R - 1
        Ud = Ud[:, :d]
        Xp = Ud @ x_p[:d, :] + X_m
        x = x_p[:d, :]
        c = np.sqrt(np.max(np.sum(x ** 2, axis=0)))
        X = np.vstack([x, c * np.ones((1, N))])
    else:
        d = R
        Ud, Sd, _ = np.linalg.svd(X @ X.T / N)
        Ud = Ud[:, :d]
        x_p = Ud.T @ X
        Xp = Ud @ x_p[:d, :]
        x = Ud.T @ X
        u = np.mean(x, axis=1, keepdims=True)
        X = x / np.sum(x * u, axis=0, keepdims=True)

    indices = np.zeros(R, dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1

    # Iterative vertex search
    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.linalg.norm(f)

        v = f.T @ X
        idx = np.argmax(np.abs(v))

        A[:, i] = X[:, idx]
        indices[i] = idx

    S = Xp[:, indices]

    return S, indices, Xp
















def NNLS(X, S):
    """
    Nonnegative Least Squares to solve for A (abundance matrix).
    
    X (numpy.ndarray): Unlabeled data matrix (p, N).
    S (numpy.ndarray): End-member spectrum matrix (p, q) (can be obtained via VCA, but must average it)
    Returns: phi (N, R) - abundance for each pixel/endmember
    """
    p, N = X.shape
    q = S.shape[1]
    phi = np.zeros((N, q))
    for i in range(N):
        a, _ = nnls(S, X[:, i])
        phi[i, :] = a
    return phi
















def SCLSU(X, S):

    # Run NCLSU
    phi = NNLS(X, S)

    # Get the scaling factor
    psi_hat = phi.sum(axis = 1)

    # Sum to one condition
    A_est = phi / psi_hat[:, None]

    # Return the estimation of A
    return A_est.T
















def algo_2_almm(X, S, alpha, beta, gamma, eta, maxIter):
    """
    Note: X = Y, S = A, A = X, T = S, U = T, and p = D, q = P in paper. 
    A is (q, N). T is (N, N), AT is (q, N). E is (p, L). B is (L, N)
    (L is half the spectral length (p), so L = p/2)

    Parameters:
    X (numpy.ndarray): Unlabeled data matrix (p, N).
    S (numpy.ndarray): End-member spectrum matrix (p, q) (can be obtained via VCA, but must average it)
    alpha (float):
    beta (float):
    gamma (float):
    eta (float):
    max_iters (int):

    Returns:
    E (numpy.ndarray): Spectral variability matrix (p, L).
    A (numpy.ndarray): Abundance matrix (q, N).
    T (numpy.ndarray): Scalar variability matrix (N, N).
    B (numpy.ndarray): Coefficient matrix for E (L, N).
    
    """
    p, N = X.shape
    p, q = S.shape
    L = int(p/2)

    # Initialize variables (A^0 is generated by SCLSU, E^0 is random but switch to S_GLU - vca(X))

    # Initialize G = H = M to 0
    # T = I * originally S
    # B = 0
    # Delta = 0
    # Lambda = V = Omega = 0
    # Q = 0
    # U = 0 * originally T
    # Pi = 0
    # A = SCLSU * originally  X
    # E = S_GLU - vca(X)
    G = np.zeros((q, N))
    H = np.copy(G)
    M = np.copy(G)

    T = np.eye(N)
    B = np.zeros((L, N))

    Delta = np.zeros((N, N)) # associated with T and U
    Lambda = np.zeros((q, N)) # associated with X and G
    Upsilon = np.copy(Lambda) # associated with X and H
    Omega = np.copy(Lambda) # associated with XS and M

    Q = np.zeros((p, L)) # associated with E
    U = np.zeros((N, N)) # associated with T and U
    
    Pi = np.zeros((p, L)) # associaed with E and Q

    # TODO Initialize E using observed - library = S_GLU - vca(X). 
    # E can just also be some random orthogonal matrix
    A = SCLSU(X, S)

    # Generate random orthogonal matrix for E (change later possibly)
    np.random.seed(42)
    random_matrix = np.random.randn(p, L)
    E, _ = np.linalg.qr(random_matrix) # QR decomposition, E guaranteed to be orthogonal


    # Note: this doesn't work because we don't have access to A_gt, unless we call this function using a third party
    # pipeline function, which I'm not sure if it is correct.
    #
    # G, W = build_custom_knn_graph(X.T, K=int(N*0.005))
    # labeled_indices = algo_1_active_learning(X, W, m_initial=2, M_total=int(0.004*N), num_eigs=int(N*0.005))
    # X_hat = X[:, labeled_indices]
    # A_hat_exact = A_gt[:, labeled_indices]
    # A_hat_OH = generate_one_hot_labels(A_hat_exact)
    # _, S_GLU = algo_2_glu(X, X_hat, A_hat_OH, alpha, k=k)

    # Initialize scalars
    t = 0
    xi = 1e-3
    xi_max = 1e6
    rho = 1.5
    epsilon = 1e-6
    converged = False

    # Begin iteration loop
    while (not converged) and t < maxIter:

        # Define constants
        xi_I_q = xi * np.eye(q)   # for M subproblem
        xi_I_N = xi * np.eye(N)   # for A and T subproblems
        xi_I_L = xi * np.eye(L)   # for E subproblem
        xi_I_p = xi * np.eye(p)   # for Q subproblem
        beta_I = beta * np.eye(L) # for B subproblem

        # Update matrices and their auxiliary variables
        
        # M subproblem (fix E, B, A, T)
        term1_M = np.linalg.inv((S.T @ S) + xi_I_q)
        term2_M = (S.T @ X) - (S.T @ E @ B) + (xi * (A @ T)) - Omega
        M_new = term1_M @ term2_M

        # B subproblem (fix E, M_new)
        term1_B = np.linalg.inv((E.T @ E) + beta_I)
        term2_B = (E.T @ X) - (E.T @ S @ M_new)
        B_new = term1_B @ term2_B

        # A subproblem (fix G, H, T, M_new, Lambda, Upsilon)
        term1_A = (xi * G) + Lambda + (xi * H) + Upsilon + (Omega @ T.T) + (xi * M_new @ T.T)
        term2_A = np.linalg.inv((xi * T @ T.T) + (2 * xi_I_N))
        A_new = term1_A @ term2_A
        #print(f"t={t}, xi={xi:.6f}, A_new min/max=({A_new.min():.4f}, {A_new.max():.4f}), RMSE={np.linalg.norm(A_new - A):.4f}")
        A_new = A_new / (A_new.sum(axis=0, keepdims=True) + 1e-10) # Normalize A using Hadamard division

        # T subproblem (fix M_new, A_new, U, Pi, Delta)
        term1_T = np.linalg.inv((xi * A_new.T @ A_new) + xi_I_N)
        term2_T = (xi * A_new.T @ M_new) + (A_new.T @ Omega) + (xi * U) + Delta
        T_new = term1_T @ term2_T
        T_new = np.diag(np.diag(T_new))  # enforce scalar/diagonal structure

        # E subproblem (fix M_new, B_new, Q, Pi)
        term1_E = (X @ B_new.T) - (S @ M_new @ B_new.T) + (xi * Q) + Pi
        term2_E = np.linalg.inv((B_new @ B_new.T) + xi_I_L)
        E_new = term1_E @ term2_E

        # Q subproblem (fix E, Q, E_new, Pi)
        term1_Q = np.linalg.inv((gamma * S @ S.T) + (eta * Q @ Q.T) + xi_I_p)
        term2_Q = (eta * Q) + (xi * E_new) - Pi
        Q_new = term1_Q @ term2_Q

        # G subproblem (fix A_new, Lambda)
        term1_G = np.maximum(0, np.abs(A_new - (Lambda / xi)) - (alpha / xi))
        term2_G = np.sign(A_new - (Lambda / xi))
        G_new = term1_G * term2_G # Double check

        # H subproblem (fix A_new, Upsilon)
        H_new = np.maximum(0, A_new - (Upsilon / xi))

        # U subproblem (fix T_new, Delta)
        U_new = np.maximum(0, T_new - (Delta / xi))

        # AT subproblem (fix A_new, T_new)
        AT_new = A_new @ T_new

        # Update Lagrange multipliers
        Lambda_new = Lambda + xi * (G_new - A_new)
        Upsilon_new = Upsilon + xi * (H_new - A_new)
        Omega_new = Omega + xi * (M_new - AT_new)
        Pi_new = Pi + xi * (Q_new - E_new)
        Delta_new = Delta + xi * (U_new - T_new)


        # Update penalty parameter
        xi_new = min(rho * xi, xi_max)

        # Check convergence conditions
        if ((np.linalg.norm(G_new - A_new) < epsilon) and 
            (np.linalg.norm(H_new - A_new) < epsilon) and
            (np.linalg.norm(M_new - AT_new) < epsilon) and
            (np.linalg.norm(Q_new - E_new) < epsilon) and 
            (np.linalg.norm(U_new - T_new) < epsilon) and
            (np.linalg.norm(E_new - E) < epsilon)):

            converged = True
        
        # Else, start the next iteration
        else:
            # Update iteration count
            t += 1

            # Advance state
            # Normal variables and auxiliary counterparts
            M = M_new
            B = B_new
            A = A_new
            T = T_new
            E = E_new
            Q = Q_new
            G = G_new
            H = H_new
            U = U_new

            # Lagrange multipliers
            Lambda = Lambda_new
            Upsilon = Upsilon_new
            Omega = Omega_new
            Pi = Pi_new
            Delta = Delta_new
            xi = xi_new


        
    return E, A, T, B