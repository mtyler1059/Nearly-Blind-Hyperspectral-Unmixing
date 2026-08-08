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
from scipy.optimize import nnls






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
    norm_i = np.linalg.norm(s_i) + 1e-8
    norm_j = np.linalg.norm(s_j) + 1e-8
    cos_theta = np.dot(s_i, s_j) / (norm_i * norm_j)
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
    q = S_gt.shape[1]
    total = sum(spectral_angle(S[:, i], S_gt[:, i]) for i in range(q))
    return (total / q) * (180 / np.pi)



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








































def algo_3_grsu(X, X_hat, A_hat, alpha, lam, gamma, rho, A_gt, S_gt, max_iters=1000, eps=1e-3, k=50, A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", array_plots = False):
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
    A_error_array = []
    RMSE_array = []
    Energy_array = []

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

        # Calculate error
        if A_error:
            A_error_array.append(np.linalg.norm(A_new - A))
        if RMSE_plot:
            RMSE_array.append(RMSE(A, A_gt))
        if Energy_plot:
            Energy_array.append(GRSU_energy(X = X, S = S, A = A, W = W, M = M, 
                                            X_hat = X_hat, A_hat = A_hat, alpha = alpha, lambda_0 = lam))

        S = S_new
        A = A_new
        B = B_new
        B_bar = B_bar_new
        T_bar = T_bar_new

        i += 1
    
    # Plot
    graph_plotter(i = i, A_error = A_error, RMSE_plot = RMSE_plot, Energy_plot = Energy_plot,
                  A_error_array = A_error_array, RMSE_array = RMSE_array, Energy_array = Energy_array, title_0 = title_0) 

    # Figure out what to return
    if array_plots:
        return RMSE_array, Energy_array
    else:
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
    np.random.seed(42)
    v0 = np.random.rand(L.shape[0]) # to avoid it from randomly picking its own starting vector
    eigenvalues, V = spla.eigsh(L, k=num_eigs, which='SM', v0=v0)

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
















def run_unmixing_pipeline_example(X, A_gt, S_gt, N, iters, alpha = 10.0, lam = 1.0, gamma = 1.0, rho = 1.0, m_0 = 2, print_bool = True, OH_labels = True, GRSU_bool = True, A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", prep = None, array_plots = False):
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

    # Run Phase 1 and Phase 2 if it isn't already given

    if prep is None:
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
        # num_eigs = 0.5% of the pixels (equal to K)?
        labeled_indices = algo_1_active_learning(X, W, m_initial=m_0, M_total=int(0.004*N), num_eigs=int(N*0.005))

        #print("Labeled indices:", labeled_indices)

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

        # Pick between Exact or OH labels
        if OH_labels:
            A_hat_test = A_hat_OH
            label_title = "OH"
        else:
            A_hat_test = A_hat_exact
            label_title = "Exact"

    else:
        # Extract values from array
        X_hat = prep[0]
        A_hat_test = prep[1]
        label_title = prep[2]

    # ==========================================
    # Phase 3: Semi-Supervised Unmixing
    # ==========================================
    if print_bool and GRSU_bool:
        print(f"Running GRSU (and GLU) Unmixing on {label_title}...")
    elif print_bool:
         print(f"Running GLU Unmixing on {label_title}...")

    # Hyperparameters based on the Jasper Ridge dataset in Table II [cite: 339, 340]
    # alpha = 10.0
    # lam = 1.0
    # gamma = 1.0
    # rho = 1.0

    # Note: The paper mentions an overlap between X_hat and X, but updates
    # the abundance map for all pixels in X anyway.

    # If we are only running GLU
    if not GRSU_bool:
        A_final, S_final = algo_2_glu(X, X_hat, A_hat_test, alpha, k=int(N*0.005))
    else:
        A_final, S_final = algo_3_grsu(
            X=X,
            X_hat=X_hat,
            A_hat=A_hat_test,
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            rho=rho,
            max_iters=iters,
            A_gt = A_gt,
            S_gt = S_gt,
            eps=1e-3,
            k=int(N*0.005),
            A_error = A_error,
            RMSE_plot = RMSE_plot,
            Energy_plot = Energy_plot,
            title_0 = title_0,
            array_plots = array_plots
        )

    # If we just want the RMSE and/or Energy arrays
    if array_plots:

        # Pick between GRSU and GLU energy
        if GRSU_bool:
            # Slightly confusing notation, but A_final = RMSE_array and S_final = Energy_array
            return A_final, S_final, None, None
        
        else:
            # Calculate energy
            q, M = A_hat_test.shape
            A_unlabeled = A_gt[:, M:]
            GLU_energy_ = GLU_energy(A = A_unlabeled, W = W, M = M)

            return GLU_energy_, None, None, None
            

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
def sum_RMSE_SAD(X, A_gt, S_gt, N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, print_bool = True, OH_labels = True, GRSU_bool = True, prep = None):
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
    A_f, S_f, A_RMSE, S_SAD = run_unmixing_pipeline_example(X, A_gt, S_gt, 
                                                            N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, 
                                                            print_bool = print_bool, OH_labels = OH_labels, GRSU_bool = GRSU_bool, 
                                                            prep = prep)
    return A_RMSE + S_SAD




def RMSE_GRSU(X, A_gt, S_gt, N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, print_bool = True, OH_labels = True, GRSU_bool = True, prep = None):
    """
    Returns RMSE.

    Parameters:
    X (numpy.ndarray): Data matrix (p, N).
    A_gt
    alpha (numpy.ndarray): Regularization parameter for T subproblem.
    lam (numpy.ndarray): Regularization parameter for B subproblem.
    gamma (numpy.ndarray): Regularization parameter for S subproblem.
    rho (numpy.ndarray): Regularization parameter for A subproblem.
    """
    A_f, S_f, A_RMSE, S_SAD = run_unmixing_pipeline_example(X, A_gt, S_gt, 
                                                            N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, 
                                                            print_bool = print_bool, OH_labels = OH_labels, GRSU_bool = GRSU_bool, 
                                                            prep = prep)
    return A_RMSE






def parameter_testing(X, A_gt, S_gt, N, iters, alpha, lam, gamma, rho, m_0, print_bool = True, GRSU_bool = True, OH_labels = True):
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

    # Precompute the graph and labels

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
    # num_eigs = 0.5% of the pixels (equal to K)?
    labeled_indices = algo_1_active_learning(X, W, m_initial=m_0, M_total=int(0.004*N), num_eigs=int(N*0.005))

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

    # Pick between Exact or OH labels
    if OH_labels:
        A_hat_test = A_hat_OH
        label_title = "OH"
    else:
        A_hat_test = A_hat_exact
        label_title = "Exact"

    prep = [X_hat, A_hat_test, label_title]

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(sum_RMSE_SAD)(X, A_gt, S_gt, N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, print_bool, OH_labels, GRSU_bool, prep) for alpha_0, lam_0, gamma_0, rho_0 in combos
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





def parameter_testing_RMSE(X, A_gt, S_gt, N, iters, alpha, lam, gamma, rho, m_0, print_bool = True, GRSU_bool = True, OH_labels = True):
    """
    Performs grid search on the regularization parameters (alpha, lam, gamma, rho) to find
    the optimal combination of the four, minimizing A_RMSE.

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

    # Precompute the graph and labels

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
    # num_eigs = 0.5% of the pixels (equal to K)?
    labeled_indices = algo_1_active_learning(X, W, m_initial=m_0, M_total=int(0.004*N), num_eigs=int(N*0.005))

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

    # Pick between Exact or OH labels
    if OH_labels:
        A_hat_test = A_hat_OH
        label_title = "OH"
    else:
        A_hat_test = A_hat_exact
        label_title = "Exact"

    prep = [X_hat, A_hat_test, label_title]

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(RMSE_GRSU)(X, A_gt, S_gt, N, iters, alpha_0, lam_0, gamma_0, rho_0, m_0, print_bool, OH_labels, GRSU_bool, prep) for alpha_0, lam_0, gamma_0, rho_0 in combos
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
    print(f"Best RMSE: {min_result}")
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

    # Involves randomness
    np.random.seed(42)

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
















def algo_2_almm_optimized(X, S, A_gt, alpha, beta, gamma, eta, maxIter, A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", array_plots = False):
    """
    Optimized ALMM-Based SVDL.
    Note: Requires passing A_initial (from SCLSU) as an argument since 
    SCLSU is not defined in the scope of this snippet.
    """
    A_initial = SCLSU(X, S)
    p, N = X.shape
    p, q = S.shape
    L = int(p / 2)
    A_error_array = []
    RMSE_array = []
    Energy_array = []

    # 1. Precompute loop-invariant matrices
    StS = S.T @ S
    StX = S.T @ X
    gamma_SSt = gamma * (S @ S.T)

    # Initialize standard variables
    G = np.zeros((q, N))
    H = np.zeros((q, N))
    M = np.zeros((q, N))
    B = np.zeros((L, N))

    # 2. Track N x N diagonal matrices as 1D vectors to save O(N^2) memory
    t_diag = np.ones(N)       # Equivalent to np.eye(N)
    delta_diag = np.zeros(N)  # Equivalent to np.zeros((N, N))
    u_diag = np.zeros(N)      # Equivalent to np.zeros((N, N))

    Lambda = np.zeros((q, N)) 
    Upsilon = np.zeros((q, N)) 
    Omega = np.zeros((q, N)) 

    Q = np.zeros((p, L)) 
    Pi = np.zeros((p, L)) 

    A = np.copy(A_initial)

    # Generate random orthogonal matrix for E
    np.random.seed(42)
    random_matrix = np.random.randn(p, L)
    E, _ = np.linalg.qr(random_matrix)

    # Initialize scalars
    t = 0
    xi = 1e-3
    xi_max = 1e6
    rho = 1.5
    epsilon = 1e-6
    converged = False

    # Identity matrices for subproblems (sizes q, L, p only!)
    I_q = np.eye(q)
    I_L = np.eye(L)
    I_p = np.eye(p)

    while (not converged) and t < maxIter:
        
        # M subproblem (solve instead of inv)
        # S.T @ E @ B evaluated as (S.T @ E) @ B for faster multiplication
        term2_M = StX - (S.T @ E) @ B + (xi * (A * t_diag)) - Omega
        M_new = np.linalg.solve(StS + xi * I_q, term2_M)

        # B subproblem
        term2_B = (E.T @ X) - (E.T @ S) @ M_new
        B_new = np.linalg.solve((E.T @ E) + beta * I_L, term2_B)

        # A subproblem
        # Multiplication by diagonal matrix T is just broadcasting: * t_diag
        term1_A = (xi * G) + Lambda + (xi * H) + Upsilon + (Omega * t_diag) + (xi * M_new * t_diag)
        
        # term2_A was purely diagonal, so we just divide by the diagonal values!
        diag_inv = 1.0 / (xi * (t_diag ** 2) + 2 * xi)
        A_new = term1_A * diag_inv
        A_new = A_new / (A_new.sum(axis=0, keepdims=True) + 1e-10)

        # T subproblem (Woodbury Matrix Identity to avoid N x N operations)
        Y = xi * M_new + Omega
        A_At = A_new @ A_new.T
        
        # K = (I_q + A A^T)^{-1}. Since q is small (e.g. 10), this is instant.
        K = np.linalg.solve(I_q + A_At, I_q) 
        
        # Calculate exactly the diagonal components needed
        Z = Y - (K @ A_At) @ Y
        diag_At_Z = np.sum(A_new * Z, axis=0) # Fast way to get diag of A^T Z
        diag_A_K_A = np.sum(A_new * (K @ A_new), axis=0)
        
        D = xi * u_diag + delta_diag
        T_new_diag = (1.0 / xi) * diag_At_Z + (1.0 / xi) * D * (1.0 - diag_A_K_A)

        # E subproblem (Right-side solve using Transpose)
        term1_E = (X @ B_new.T) - S @ (M_new @ B_new.T) + (xi * Q) + Pi
        term2_E_inv = (B_new @ B_new.T) + xi * I_L
        E_new = np.linalg.solve(term2_E_inv, term1_E.T).T

        # Q subproblem
        term1_Q_inv = gamma_SSt + (eta * Q @ Q.T) + xi * I_p
        term2_Q = (eta * Q) + (xi * E_new) - Pi
        Q_new = np.linalg.solve(term1_Q_inv, term2_Q)

        # G, H, U subproblems
        term1_G = np.maximum(0, np.abs(A_new - (Lambda / xi)) - (alpha / xi))
        term2_G = np.sign(A_new - (Lambda / xi))
        G_new = term1_G * term2_G 
        H_new = np.maximum(0, A_new - (Upsilon / xi))
        U_new_diag = np.maximum(0, T_new_diag - (delta_diag / xi))

        # AT subproblem 
        AT_new = A_new * T_new_diag

        # Update Lagrange multipliers
        Lambda_new = Lambda + xi * (G_new - A_new)
        Upsilon_new = Upsilon + xi * (H_new - A_new)
        Omega_new = Omega + xi * (M_new - AT_new)
        Pi_new = Pi + xi * (Q_new - E_new)
        delta_diag_new = delta_diag + xi * (U_new_diag - T_new_diag)

        xi_new = min(rho * xi, xi_max)

        # Check convergence conditions
        if ((np.linalg.norm(G_new - A_new) < epsilon) and 
            (np.linalg.norm(H_new - A_new) < epsilon) and
            (np.linalg.norm(M_new - AT_new) < epsilon) and
            (np.linalg.norm(Q_new - E_new) < epsilon) and 
            (np.linalg.norm(U_new_diag - T_new_diag) < epsilon) and
            (np.linalg.norm(E_new - E) < epsilon)):

            converged = True
        else:
            t += 1

            # Calculate error
            if A_error:
                A_error_array.append(np.linalg.norm(A_new - A))
            if RMSE_plot:
                # Check for labeling issues
                corr = np.corrcoef(A[0], A_gt[0])[0, 1]
                if corr < 0:
                    A_f_corrected = 1 - A
                else:
                    A_f_corrected = A

                rmse = RMSE(A_f_corrected, A_gt)  # Calculate RMSE
                RMSE_array.append(rmse)
            if Energy_plot:
                Energy_array.append(ALMM_energy(X = X, S = S, A = A, T = t_diag, 
                                                E = E, B = B, alpha = alpha, beta = beta, gamma = gamma, eta = eta))

            M = M_new
            B = B_new
            A = A_new
            t_diag = T_new_diag
            E = E_new
            Q = Q_new
            G = G_new
            H = H_new
            u_diag = U_new_diag

            Lambda = Lambda_new
            Upsilon = Upsilon_new
            Omega = Omega_new
            Pi = Pi_new
            delta_diag = delta_diag_new
            xi = xi_new

    # Reconstruct the N x N diagonal matrix T at the very end to match expected output signature
    T_final = np.diag(t_diag)

    # Plot
    graph_plotter(i = t, A_error = A_error, RMSE_plot = RMSE_plot, Energy_plot = Energy_plot,
                  A_error_array = A_error_array, RMSE_array = RMSE_array, Energy_array = Energy_array, title_0 = title_0) 
    
    # Figure out what to return
    if array_plots:
        return RMSE_array, Energy_array, None, None
    else:
        return E, A, T_final, B
















# Half threshold for L_1_2, generated by Claude, double check
def half_threshold(z, alpha, xi):
    threshold = (54**(1/3) / 4) * (2*alpha/(xi))**(2/3)
    abs_z = np.abs(z)
    
    mask = abs_z > threshold
    result = np.zeros_like(z)
    
    arg = np.clip((alpha / (4*xi)) * (abs_z[mask]/3)**(-1.5), -1, 1)
    phi = np.arccos(arg)
    result[mask] = (2/3) * z[mask] * (1 + np.cos(2*np.pi/3 - 2*phi/3))
    
    return result
















def algo_2_graph_almm_optimized(X, A_0, S, alpha, beta, gamma, eta, maxIter, xi_0, L_uu, L_lu_T_A_hat_T, A_gt, A_hat, W, labeled_indices, A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", array_plots = False):
    """
    Optimized ALMM-Based SVDL.
    Note: Requires passing A_initial (from SCLSU) as an argument since 
    SCLSU is not defined in the scope of this snippet.
    """
    p, N = X.shape
    p, q = S.shape
    q, M_0 = A_hat.shape

    L = int(p / 2)
    A_error_array = []
    RMSE_array = []
    Energy_array = []
    #A_0 = SCLSU(X, S)
    #print("Number of labeled points:", len(labeled_indices))

    # Initialize standard variables
    G = np.zeros((q, N))
    H = np.zeros((q, N))
    M = np.zeros((q, N))
    B = np.zeros((L, N))

    # 2. Track N x N diagonal matrices as 1D vectors to save O(N^2) memory
    t_diag = np.ones(N)       # Equivalent to np.eye(N)
    delta_diag = np.zeros(N)  # Equivalent to np.zeros((N, N))
    u_diag = np.zeros(N)      # Equivalent to np.zeros((N, N))

    Lambda = np.zeros((q, N)) 
    Upsilon = np.zeros((q, N)) 
    Omega = np.zeros((q, N)) 
    Theta = np.zeros((p, q))
    V = np.zeros((p, q))

    Q = np.zeros((p, L)) 
    Pi = np.zeros((p, L)) 

    A = np.copy(A_0)

    # Generate random orthogonal matrix for E
    np.random.seed(42)
    # random_matrix = np.random.randn(p, L)
    # E, _ = np.linalg.qr(random_matrix)

    # E = S - vca(X)
    Ae, _, _ = vca(X, R = q)
    diff = S - Ae


    # Pad with random columns to reach L total, then orthogonalize via QR
    pad = np.random.randn(p, L - diff.shape[1])
    E_init_raw = np.hstack([diff, pad])   # shape (p, L)

    E, _ = np.linalg.qr(E_init_raw)       # (p, L), orthonormal columns, but "seeded" by your real diff


    # Initialize scalars
    t = 0
    xi = xi_0 # originally 1e-3
    xi_max = 1e6
    rho = 1.5
    epsilon = 1e-6
    converged = False

    # Identity matrices for subproblems (sizes q, L, p only!)
    I_q = np.eye(q)
    I_L = np.eye(L)
    I_p = np.eye(p)

    # Derive unlabeled indices from labeled_indices (test)
    all_indices = np.arange(N)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)
    N_unlabeled = len(unlabeled_indices)  # should match L_uu.shape[0]

    while (not converged) and t < maxIter:
        

        # M subproblem (solve instead of inv)
        # S.T @ E @ B evaluated as (S.T @ E) @ B for faster multiplication
        term2_M = (S.T @ X) - (S.T @ E) @ B + (xi * (A * t_diag)) - Omega
        M_new = np.linalg.solve((S.T @ S) + xi * I_q, term2_M)

        # B subproblem
        term2_B = (E.T @ X) - (E.T @ S) @ M_new
        B_new = np.linalg.solve((E.T @ E) + beta * I_L, term2_B)

        # A subproblem
        # Multiplication by diagonal matrix T is just broadcasting: * t_diag
        term1_A = ((xi* 1) * G) + Lambda + ((xi*2)* H) + Upsilon + (Omega * t_diag) + (xi * M_new * t_diag)
        
        # term2_A was purely diagonal, so we just divide by the diagonal values!
        diag_inv = 1.0 / (xi * (t_diag ** 2) + 2 * xi)
        A_new_unproj = term1_A * diag_inv
        A_new_unproj = A_new_unproj / (A_new_unproj.sum(axis=0, keepdims=True) + 1e-10)
        A_new = project_onto_simplex(A_new_unproj) # Project onto the simplex for nonnegativity

        # S subproblem
        term1_S = M_new @ M_new.T + xi * I_q
        term2_S = (X - E @ B_new) @ M_new.T + Theta + xi * V

        S_new = np.linalg.solve(term1_S, term2_S.T).T

        # T subproblem (Woodbury Matrix Identity to avoid N x N operations)
        Y = xi * M_new + Omega
        A_At = A_new @ A_new.T
        
        # K = (I_q + A A^T)^{-1}. Since q is small (e.g. 10), this is instant.
        K = np.linalg.solve(I_q + A_At, I_q) 
        
        # Calculate exactly the diagonal components needed
        Z = Y - (K @ A_At) @ Y
        diag_At_Z = np.sum(A_new * Z, axis=0) # Fast way to get diag of A^T Z
        diag_A_K_A = np.sum(A_new * (K @ A_new), axis=0)
        
        D = xi * u_diag + delta_diag
        T_new_diag = (1.0 / xi) * diag_At_Z + (1.0 / xi) * D * (1.0 - diag_A_K_A)

        # Q subproblem
        gamma_SSt = gamma * (S_new @ S_new.T)
        term1_Q_inv = gamma_SSt + (eta * Q @ Q.T) + xi * I_p
        term2_Q = (eta * Q) + (xi * E) - Pi
        Q_new = np.linalg.solve(term1_Q_inv, term2_Q)

        # E subproblem (Right-side solve using Transpose)
        term1_E = (X @ B_new.T) - S_new @ (M_new @ B_new.T) + (xi * Q_new) + Pi
        term2_E_inv = (B_new @ B_new.T) + xi * I_L
        E_new = np.linalg.solve(term2_E_inv, term1_E.T).T

        # G subproblem
        Z_0 = A_new - (Lambda / xi)
        G_new = half_threshold(Z_0, alpha, xi)


        # H subproblem — solve only on unlabeled portion
        I_N_unlabeled_sparse = sps.eye(N_unlabeled, format='csr')
        L_B_system = L_uu + xi * I_N_unlabeled_sparse

        # Only take unlabeled columns of A_new and Lambda for the RHS
        A_new_unlabeled = A_new[:, unlabeled_indices]      # shape (q, N_unlabeled)
        Upsilon_unlabeled = Upsilon[:, unlabeled_indices]     # shape (q, N_unlabeled)

        RHS_H = -L_lu_T_A_hat_T + xi * A_new_unlabeled.T - Upsilon_unlabeled.T  # (N_unlabeled, q)
        H_unlabeled_T = np.zeros((N_unlabeled, q))

        for j in range(q):
            H_unlabeled_T[:, j], _ = spla.cg(L_B_system, RHS_H[:, j])

        
        H_unlabeled = H_unlabeled_T.T  # (q, N_unlabeled)

        # Reassemble full-size H: labeled portion anchored to A_new, unlabeled portion from graph solve
        H_new = np.zeros((q, N))
        H_new[:, labeled_indices] = A_new[:, labeled_indices]
        H_new[:, unlabeled_indices] = H_unlabeled
        
        # U subproblem
        U_new_diag = np.maximum(0, T_new_diag - (delta_diag / xi))

        # V subproblem
        V_new = np.maximum(0, S_new - (Theta / xi))
        

        # AT subproblem 
        AT_new = A_new * T_new_diag

        # Update Lagrange multipliers
        Lambda_new = Lambda + xi * (G_new - A_new)
        Upsilon_new = Upsilon + xi * (H_new - A_new)
        Omega_new = Omega + xi * (M_new - AT_new)
        Pi_new = Pi + xi * (Q_new - E_new)
        delta_diag_new = delta_diag + xi * (U_new_diag - T_new_diag)
        Theta_new = Theta + xi * (V_new - S_new)

        xi_new = min(rho * xi, xi_max)

        # Check convergence conditions

        # Calculate convergence error
        # Protect against division by zero in the denominator
        norm_S = np.linalg.norm(S, 'fro')
        norm_A = np.linalg.norm(A, 'fro')

        err_S = np.linalg.norm(S_new - S, 'fro') / norm_S if norm_S > 0 else 0
        err_A = np.linalg.norm(A_new - A, 'fro') / norm_A if norm_A > 0 else 0

        Err = max(err_S, err_A)

        # if t == 0:
        #     print("After first update A:", np.sum(A_new))
        #     print("After first update S:", np.sum(S_new))
        #     print("S min/max:", S_new.min(), S_new.max())
        #     print("S norm:", np.linalg.norm(S_new))
        #     print("M norm:", np.linalg.norm(M_new))
        #     print("B norm:", np.linalg.norm(B_new))
        #     print("A norm:", np.linalg.norm(A_new))
        # if t < 5:
        #     print(
        #         t,
        #         np.linalg.norm(A_new),
        #         np.linalg.norm(S_new),
        #         np.linalg.norm(M_new),
        #         np.linalg.norm(B_new),
        #         np.linalg.norm(E_new)
        #     )

        if (((np.linalg.norm(G_new - A_new) < epsilon) and 
            (np.linalg.norm(H_new - A_new) < epsilon) and
            (np.linalg.norm(M_new - AT_new) < epsilon) and
            (np.linalg.norm(Q_new - E_new) < epsilon) and 
            (np.linalg.norm(U_new_diag - T_new_diag) < epsilon) and
            (np.linalg.norm(E_new - E) < epsilon) and
            (np.linalg.norm(V_new - S_new) < epsilon)) or (Err < 1e-4)):

            converged = True
        else:
            t += 1

            # Calculate error
            if A_error:
                A_error_array.append(np.linalg.norm(A_new - A))
            if RMSE_plot:
                # Check for labeling issues
                corr = np.corrcoef(A[0], A_gt[0])[0, 1]
                if corr < 0:
                    A_f_corrected = 1 - A
                else:
                    A_f_corrected = A

                rmse = RMSE(A_f_corrected, A_gt)  # Calculate RMSE
                RMSE_array.append(rmse)
            if Energy_plot:
                Energy_array.append(Graph_ALMM_energy(X = X, S = S, A = A, A_hat = A_hat, 
                                                      T = t_diag, E = E, B = B, W = W, M = M_0, 
                                                      alpha = alpha, beta = beta, gamma = gamma, eta = eta, xi = xi))

            M = M_new
            B = B_new
            A = A_new
            S = S_new
            t_diag = T_new_diag
            E = E_new
            Q = Q_new
            G = G_new
            H = H_new
            u_diag = U_new_diag
            V = V_new

            Lambda = Lambda_new
            Upsilon = Upsilon_new
            Omega = Omega_new
            Pi = Pi_new
            delta_diag = delta_diag_new
            Theta = Theta_new
            xi = xi_new

    # Reconstruct the N x N diagonal matrix T at the very end to match expected output signature
    T_final = np.diag(t_diag)

    # Plot
    graph_plotter(i = t, A_error = A_error, RMSE_plot = RMSE_plot, Energy_plot = Energy_plot,
                  A_error_array = A_error_array, RMSE_array = RMSE_array, Energy_array = Energy_array, title_0 = title_0) 
    
    # Figure out what to return
    #print(f"iterations: {t}")
    if array_plots:
        return RMSE_array, Energy_array, None, None, None
    else:
        return E, A, T_final, B, S















def run_unmixing_pipeline_example2(X, A_gt, S_gt, N, alpha, beta, gamma, eta, maxIter, M_total_0, m_0 = 2, xi_0 = 1e-3, OH_labels = True, print_bool = True, A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", W_0 = None, array_plots = False):

    # ==========================================
    # Phase 1: Active Learning (Algorithm 1)
    # ==========================================

    # Check if we are running the same file
    # import hashlib
    # import inspect

    # source = inspect.getsource(algo_2_almm_optimized)
    # print(hashlib.md5(source.encode()).hexdigest())

    if W_0 is None:
        if print_bool:
            print("Building initial graph for Active Learning...")
        # Scikit-learn expects (samples, features), so we pass X.T
        # K = 50 was picked for a 10000 pixel image, so roughly 0.5%
        G, W = build_custom_knn_graph(X.T, K=int(N*0.005))
    else:
        W = W_0

    if print_bool:
        print("Running Active Learning...")
    # Start with 1 random pixel per material (m=4), sample up to 0.4% of total pixels (M=40) [cite: 323]
    # num_eigs = 0.5% of the pixels (equal to K)?
    #  M_total_0 = int(0.004*N) is the default
    labeled_indices = algo_1_active_learning(X, W, m_initial=m_0, M_total=M_total_0, num_eigs=int(N*0.005))

    # ==========================================
    # Phase 2: Extract Training Data
    # ==========================================
    if print_bool:
        print("Extracting training data and generating pseudo-labels...")
    # Extract the spectral signatures for the selected pixels
    X_hat = X[:, labeled_indices]

    # Extract ground-truth abundances and convert to One-Hot pseudo-labels [cite: 321, 322] or exact labels
    A_hat_exact = A_gt[:, labeled_indices]
    L, L_ll, L_lu, L_ul, L_uu = compute_and_partition_laplacian(W, labeled_indices)

    if OH_labels:
        A_hat_OH = generate_one_hot_labels(A_hat_exact)
        L_lu_T_A_hat_T = L_lu.T @ A_hat_OH.T
        label_title = "OH"
        A_hat_test = A_hat_OH
    else:
        L_lu_T_A_hat_T = L_lu.T @ A_hat_exact.T
        label_title = "Exact"
        A_hat_test = A_hat_exact

    # ==========================================
    # Phase 3: Semi-Supervised Unmixing
    # ==========================================
    if print_bool:
        print(f"Running Graph ALMM Unmixing on {label_title}...")

    # Note: The paper mentions an overlap between X_hat and X, but updates
    # the abundance map for all pixels in X anyway.

    A_GLU, S_GLU = algo_2_glu(X, X_hat, A_hat_test, alpha, k=int(N*0.005))

    # The second condition number is a lot worse than the first
    # print("Condition number of S_gt_chem.T @ S_gt_chem:", np.linalg.cond(S_gt_chem.T @ S_gt_chem))
    # print("Condition number of S_GLU.T @ S_GLU:", np.linalg.cond(S_GLU.T @ S_GLU))

    E_final, A_final, T_final, B_final, S_final = algo_2_graph_almm_optimized(
        X = X, A_0 = A_GLU, S = S_GLU, alpha = alpha, 
        beta = beta, gamma = gamma, eta = eta, 
        maxIter = maxIter, xi_0 = xi_0, 
        L_uu = L_uu, L_lu_T_A_hat_T = L_lu_T_A_hat_T, A_gt = A_gt, A_hat = A_hat_test, W = W, labeled_indices = labeled_indices,
        A_error = A_error, RMSE_plot = RMSE_plot, Energy_plot = Energy_plot, title_0 = title_0, array_plots = array_plots)

    # If we just want the RMSE and/or Energy arrays
    if array_plots:
        # Slightly confusing notation, but E_final = RMSE_array and A_final = Energy_array
        return E_final, A_final, None, None

    # Calculate RMSE and SAD

    # Check for labeling issues
    corr = np.corrcoef(A_final[0], A_gt[0])[0, 1]
    if corr < 0:
        A_f_corrected = 1 - A_final
    else:
        A_f_corrected = A_final

    A_rmse = RMSE(A_f_corrected, A_gt)
    S_sad = SAD(S_final, S_gt)

    if print_bool:
        print("Pipeline Complete!\n")
        print(f"Final Abundance Map Shape: {A_final.shape}")
        print(f"Final End-member Matrix Shape: {S_GLU.shape}")
        print(f"Final Abundance RMSE: {A_rmse}")
        print(f"Final Endmember SAD: {S_sad}")

    return A_final, S_final, A_rmse, S_sad

















def graph_plotter(i, A_error, RMSE_plot, Energy_plot, A_error_array, RMSE_array, Energy_array, title_0):
    x_iterations = [j for j in range(i)]

    plots_to_show = []
    if A_error:
        plots_to_show.append(("A Error (||A_new - A||)", A_error_array))
    if RMSE_plot:
        plots_to_show.append(("RMSE", RMSE_array))
    if Energy_plot:
        plots_to_show.append(("Energy", Energy_array))

    if plots_to_show:
        fig, axes = plt.subplots(1, len(plots_to_show), figsize=(6*len(plots_to_show), 5))
        if len(plots_to_show) == 1:
            axes = [axes]  # make it indexable even with just one subplot

        for ax, (title, data) in zip(axes, plots_to_show):
            ax.plot(x_iterations, data)
            ax.set_xlabel("Iterations", fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.set_title(title, fontsize=16)

        fig.suptitle(title_0, fontsize=16)
        plt.tight_layout()
        plt.show()

    return

















def graph_plotter_multiple(A_error, RMSE_plot, Energy_plot, A_error_data, RMSE_data, Energy_data, title_0):
    # fixed color mapping so each algorithm keeps the same color across all subplots
    color_map = {
        "GLU": "black",
        "GRSU": "tab:blue",
        "ALMM": "tab:orange",
        "Graph-ALMM": "tab:green",
    }

    plots_to_show = []
    if A_error:
        plots_to_show.append(("A Error (||A_new - A||)", A_error_data))
    if RMSE_plot:
        plots_to_show.append(("RMSE", RMSE_data))
    if Energy_plot:
        plots_to_show.append(("Energy", Energy_data))

    if plots_to_show:
        fig, axes = plt.subplots(1, len(plots_to_show), figsize=(6*len(plots_to_show), 5))
        if len(plots_to_show) == 1:
            axes = [axes]

        for ax, (title, series_list) in zip(axes, plots_to_show):
            for label, data in series_list:
                color = color_map.get(label, None)  # falls back to auto-color if label not in map

                if isinstance(data, (int, float)):
                    ax.axhline(y=data, linestyle='--', linewidth=2, label=label, color=color)
                else:
                    x_vals = range(len(data))
                    ax.plot(x_vals, data, label=label, linewidth=2, color=color)

            ax.set_xlabel("Iterations", fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.legend(title="Algorithm", loc="best", fontsize=9)

        fig.suptitle(title_0, fontsize=16)
        plt.tight_layout()

        # Save the image
        filename = title_0.replace(" ", "_")

        plt.savefig(
            f"plot_images/{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    return
















def load_data(name, typename, sample = False, print_bool = True):

    """
    Loads data into X, S, and A.

    Parameters:
    name (string): Name of the dataset. Options include:
        'my_custom_dataset.npy' (nonlinear)
        'synth_chem_data' (Original chem data)
        'synth_CuSO4_data' (CuSO4 data)         
        'synth_FeCl3_data' (FeCl3 data)
        'synth_FeSO4_data' (FeSO4 data)
        'processed_data/processed_data/urban/urban_processed_data.npy' (Urban dataset)

    type (string): Type of dataset loaded. Options include:
        'chem'
        'nonlinear' (additionally flattens the image)
        'HSI'

    sample (bool): For HSI images only, whether to sample a 50x50 patch or not.

    Returns:
    X (numpy.ndarray): Data matrix (p, N).
    S (numpy.ndarray): Endmember library (p, q).
    A (numpy.ndarray): Abundance map (q, N).
    """

    data = np.load(name, allow_pickle = True).item()

    # 1. See what keys are inside
    if print_bool:
        print("Keys in the dataset:", data.keys())

    # Grabbing X, S, and A from the file
    if typename != 'HSI':
        A_gt = data['A_gt']
        S_gt = data['S_gt']
        X_gt = data['X']

    # Printing the shape based on typename
    if typename == 'chem':
        if print_bool:
            # 2. Inspect the exact shapes of the arrays
            print("Shape of X (Chemistry Sample):", data['X'].shape)
            print("Shape of A_gt (Abundance Map):", data['A_gt'].shape)
            print("Shape of S_gt (Endmember Spectra):", data['S_gt'].shape)

            # 3. (Optional) Look at a small slice of the actual numbers
            print("\nFirst chemical's concentration in X (how much of class 1?):\n", data['X'][0, 0])

    elif typename == 'nonlinear':
        if print_bool:
        # 2. Inspect the exact shapes of the arrays
            print("Shape of X (HSI Image):", data['X'].shape)
            print("Shape of A_gt (Abundance Map):", data['A_gt'].shape)
            print("Shape of S_gt (Endmember Spectra):", data['S_gt'].shape)

            # 3. (Optional) Look at a small slice of the actual numbers
            print("\nFirst pixel's spectra in X:\n", data['X'][0, 0, :])

        # Reshaping X from spatial image (10, 1000, 30) to 2D image (30, 10000)
        H, W, p = X_gt.shape  # H=10, W=1000, p=30
        X_gt_flat = X_gt.reshape(-1, p).T  # reshape to (p, N) = (30, 10000)

        if print_bool:
            print(X_gt_flat.shape)  # should print (30, 2500)

    elif typename == 'HSI':
        A_gt = data['A']
        S_gt = data['S']
        X_gt = data['X']

        S_gt_HSI = S_gt['S_ref']
        A_gt_HSI = A_gt['A_ref']

        if print_bool:
            # 1. Let's check X just to be sure it IS an array
            print("Type of X:", type(data['X']))
            if hasattr(data['X'], 'shape'):
                print("Shape of X:", data['X'].shape)

            # 2. Let's peek inside the dictionary 'A'
            print("\nType of A:", type(data['A']))
            if isinstance(data['A'], dict):
                print("Keys inside A:", data['A'].keys())

            # 3. Let's peek inside the dictionary 'S'
            print("\nType of S:", type(data['S']))
            if isinstance(data['S'], dict):
                print("Keys inside S:", data['S'].keys())


    # Printing the shapes
    if typename == 'chem' or typename == 'nonlinear': 
        if print_bool: 
            # Checking the shapes
            print(type(S_gt))
            print(S_gt.shape)
            print(S_gt.dtype)

            print(type(A_gt ))
            print(A_gt .shape)
            print(A_gt .dtype)

    # Returning X, S, and A
    if typename == 'chem':
        if print_bool:
            print("Shape of X:", X_gt.shape)
        return X_gt, S_gt, A_gt 
    
    elif typename == 'nonlinear':
        if print_bool:
            print("Shape of X:", X_gt_flat.shape)
        return X_gt, S_gt, A_gt 
    elif typename == 'HSI':

        # Sample 50x50 patch or return the entire image
        if sample == True:

            # Example: grab a 50x50 spatial patch instead of random pixels
            H, W = 307, 307
            patch_size = 50
            row_start, col_start = 100, 100  # pick wherever

            X_HSI_img = X_gt.T.reshape(H, W, -1)  # reshape to (307, 307, p)
            patch = X_HSI_img[row_start:row_start+patch_size, col_start:col_start+patch_size, :]
            X_HSI_test = patch.reshape(-1, patch.shape[-1]).T  # back to (p, N_patch)
            if print_bool:
                print("X (reshaped):", X_HSI_test.shape)

            A_gt_HSI_img = A_gt_HSI.T.reshape(H, W, -1)
            patch_gt_HSI = A_gt_HSI_img[row_start:row_start+patch_size, col_start:col_start+patch_size, :]
            A_gt_HSI_test = patch_gt_HSI.reshape(-1, patch_gt_HSI.shape[-1]).T

            return X_HSI_test, S_gt_HSI, A_gt_HSI_test

        else:

            return X_gt, S_gt_HSI, A_gt_HSI
        
















def synthetic_linear_data(samples, channels):
    
    # samples = 2000 # N
    # channels = 300 # p
    np.random.seed(42)

    # Create random labels and abundance matrix
    L = np.random.uniform(0,1,samples)
    A = np.array([L,1-L])

    # Smooth out the two spectras
    s_1 = gaussian_filter(np.random.uniform(0,1,channels),2)
    s_2 = gaussian_filter(np.random.uniform(0,1,channels),2)

    S_T = np.array([s_1,s_2])

    S = S_T.T


    # Create the linear mixing model
    X=S@A
    error_std = 0.05

    E = np.random.normal(loc=0.0, scale=error_std, size=X.shape)

    X=X+E

    return X, S, A














def min_RMSE_graph_almm(X, S, A, N, maxIter, alpha, beta, gamma, eta, M_total_0, m_0, xi_0, OH_labels, W_0 = None):
    """
    Returns the RMSE of Graph ALMM (S_sad is assumed to be constant here).
    """
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example2(X = X, A_gt = A, S_gt = S, 
                                                             N = N, alpha = alpha, beta = beta, gamma = gamma, eta = eta, maxIter = maxIter, 
                                                             M_total_0 = M_total_0, m_0 = m_0, xi_0 = xi_0, 
                                                             OH_labels = OH_labels, print_bool = False, 
                                                             A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", W_0 = W_0)
    
    return A_rmse

















def min_RMSE_SAD_graph_almm(X, S, A, N, maxIter, alpha, beta, gamma, eta, M_total_0, m_0, xi_0, OH_labels, W_0 = None):
    """
    Returns the RMSE of Graph ALMM (S_sad is assumed to be constant here).
    """
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example2(X = X, A_gt = A, S_gt = S, 
                                                             N = N, alpha = alpha, beta = beta, gamma = gamma, eta = eta, maxIter = maxIter, 
                                                             M_total_0 = M_total_0, m_0 = m_0, xi_0 = xi_0, 
                                                             OH_labels = OH_labels, print_bool = False, 
                                                             A_error = False, RMSE_plot = False, Energy_plot = False, title_0 = "", W_0 = W_0)
    
    return A_rmse + S_sad
















def best_param_graph_almm(X, A_gt, S_gt, N, maxIter, alpha_0, beta_0, gamma_0, eta_0, m_0):
    """
    Performs grid search on some regularization parameters (alpha, M_total, xi, OH vs. Exact) to find
    the optimal combination of the four, minimizing the sum A_RMSE + S_SAD.

    Parameters:
    alpha_0 (numpy.ndarray): [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    M_total (numpy.ndarray): [j * (N * 0.004) for j in range(1, 5)]


    Returns:
    best (numpy.ndarray): An array with best combination minimizing A_RMSE + S_SAD
    in the format [alpha, lam, gamma, rho].
    """

    # Create a list of each combination
    alpha = [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    M_total = [j * (N * 0.004) for j in range(1, 5)]
    xi = [10**i for i in range(2, -5, -1)]
    OH_labels = [True, False]

    combos = list(product(alpha, M_total, xi, OH_labels))

    # Precompute the graph

    # ==========================================
    # Phase 1: Active Learning (Algorithm 1)
    # ==========================================
    # Scikit-learn expects (samples, features), so we pass X.T
    # K = 50 was picked for a 10000 pixel image, so roughly 0.5%
    G, W = build_custom_knn_graph(X.T, K=int(N*0.005))

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(min_RMSE_graph_almm)(X = X, S = S_gt, A = A_gt, 
                        N = N, alpha = alpha_0, beta = beta_0, gamma = gamma_0, eta = eta_0, maxIter = maxIter, 
                        M_total_0 = M_total_0, m_0 = m_0, xi_0 = xi_0, 
                        OH_labels = label, W_0 = W) for alpha_0, M_total_0, xi_0, label in combos)

    # Create a 4D array to match the set order
    results = np.array(results).reshape(len(alpha), len(M_total), len(xi), len(OH_labels))

    # Find min sum value and the corresponding combination
    idx = np.unravel_index(results.argmin(), results.shape)
    min_result = results[idx]
    alpha_idx, M_total_idx, xi_idx, OH_labels_idx = idx

    # Save the best values
    alpha_best = alpha[alpha_idx]
    M_total_best = M_total[M_total_idx]
    xi_best = xi[xi_idx]
    OH_labels_best = OH_labels[OH_labels_idx]

    # changing label title for readability
    if OH_labels_best == True:
        label_title = "OH"
    else:
        label_title = "Exact"

    # Print the best values
    print(f"Best RMSE: {min_result}")
    print(f"Best alpha: {alpha_best}")
    print(f"Best M_total: {M_total_best}")
    print(f"Best xi: {xi_best}")
    print(f"Best label: {label_title}")

    return [alpha_best, M_total_best, xi_best, OH_labels_best]

















def best_param_graph_almm_RMSE_SAD(X, A_gt, S_gt, N, maxIter, alpha_0, beta_0, gamma_0, eta_0, m_0):
    """
    Performs grid search on some regularization parameters (alpha, M_total, xi, OH vs. Exact) to find
    the optimal combination of the four, minimizing the sum A_RMSE + S_SAD.

    Parameters:
    alpha_0 (numpy.ndarray): [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    M_total (numpy.ndarray): [j * (N * 0.004) for j in range(1, 5)]


    Returns:
    best (numpy.ndarray): An array with best combination minimizing A_RMSE + S_SAD
    in the format [alpha, lam, gamma, rho].
    """

    # Create a list of each combination
    alpha = [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    M_total = [j * (N * 0.004) for j in range(1, 5)]
    xi = [10**i for i in range(2, -5, -1)]
    OH_labels = [True, False]

    combos = list(product(alpha, M_total, xi, OH_labels))

    # Precompute the graph

    # ==========================================
    # Phase 1: Active Learning (Algorithm 1)
    # ==========================================
    # Scikit-learn expects (samples, features), so we pass X.T
    # K = 50 was picked for a 10000 pixel image, so roughly 0.5%
    G, W = build_custom_knn_graph(X.T, K=int(N*0.005))

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(min_RMSE_SAD_graph_almm)(X = X, S = S_gt, A = A_gt, 
                        N = N, alpha = alpha_0, beta = beta_0, gamma = gamma_0, eta = eta_0, maxIter = maxIter, 
                        M_total_0 = M_total_0, m_0 = m_0, xi_0 = xi_0, 
                        OH_labels = label, W_0 = W) for alpha_0, M_total_0, xi_0, label in combos)

    # Create a 4D array to match the set order
    results = np.array(results).reshape(len(alpha), len(M_total), len(xi), len(OH_labels))

    # Find min sum value and the corresponding combination
    idx = np.unravel_index(results.argmin(), results.shape)
    min_result = results[idx]
    alpha_idx, M_total_idx, xi_idx, OH_labels_idx = idx

    # Save the best values
    alpha_best = alpha[alpha_idx]
    M_total_best = M_total[M_total_idx]
    xi_best = xi[xi_idx]
    OH_labels_best = OH_labels[OH_labels_idx]

    # changing label title for readability
    if OH_labels_best == True:
        label_title = "OH"
    else:
        label_title = "Exact"

    # Print the best values
    print(f"Best RMSE + SAD: {min_result}")
    print(f"Best alpha: {alpha_best}")
    print(f"Best M_total: {M_total_best}")
    print(f"Best xi: {xi_best}")
    print(f"Best label: {label_title}")

    return [alpha_best, M_total_best, xi_best, OH_labels_best]
















def best_param_grsu_glu(X, S, A, samples, iters_param, iters_final, m_0, print_bool = True, OH_labels = True, GRSU_bool = True, A_error = False, RMSE_plot = False, title_0 = ""):

    # Set up grid
    alpha_vals = np.array([10, 20, 50, 100])
    lam_vals = np.sort(np.concatenate([10**np.arange(4), 5 * 10**np.arange(4)]))
    gamma_vals = 10.0 ** np.arange(-2, 3)
    rho_vals = 10.0 ** np.arange(-2, 3)

    
    best_params = parameter_testing(X = X, A_gt = A, S_gt = S, N = samples, iters = iters_param, 
                                    alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = m_0, 
                                    print_bool = print_bool, GRSU_bool = GRSU_bool, OH_labels = OH_labels)
    
    # Save best params
    alpha_0 = best_params[0]
    lam_0 = best_params[1]
    gamma_0 = best_params[2]
    rho_0 = best_params[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X, A_gt = A, S_gt = S, N = samples, iters = iters_final, 
                                                                                                    alpha = alpha_0, lam = lam_0, gamma = gamma_0, rho = rho_0, m_0 = m_0, 
                                                                                                    print_bool = print_bool, OH_labels = OH_labels, GRSU_bool = GRSU_bool, 
                                                                                                    A_error = A_error, RMSE_plot = RMSE_plot, title_0 = title_0)
    

    return A_f, S_f, A_rmse, S_sad















def best_alpha_ALMM(X, S, A, iters_param, iters_final, alpha_0 = (1e-3 + 1e-2)/2, beta_0 = (1e-3 + 1e-2)/2, gamma_0 = (1e-3 + 1e-2)/2, eta_0 = (1e-3 + 1e-2)/2, A_error = False, RMSE_plot = False, title_0 = ""):

    # Pick the best alpha
    best_rmse = np.inf
    best_alpha = None

    alpha_vals = [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]

    for a in alpha_vals:

        E_f, A_f, T_f, B_f = algo_2_almm_optimized(X = X, S = S, A_gt = A, 
                                                    alpha = a, beta = beta_0, gamma = gamma_0, eta = eta_0, 
                                                    maxIter = iters_param)

        # Check for labeling issues
        corr = np.corrcoef(A_f[0], A[0])[0, 1]
        if corr < 0:
            A_f_corrected = 1 - A_f
        else:
            A_f_corrected = A_f

        rmse = RMSE(A_f_corrected, A)  # Calculate RMSE
        print(f"alpha={a:.6f} -> RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = a

    print(f"\nBest alpha: {best_alpha}, RMSE: {best_rmse:.4f}")

    # Run on best parameters
    E_f, A_f, T_f, B_f = algo_2_almm_optimized(X = X, S = S, A_gt = A, 
                                                alpha = best_alpha, beta = beta_0, gamma = gamma_0, eta = eta_0, maxIter = iters_final, 
                                                A_error = A_error, RMSE_plot = RMSE_plot, title_0 = title_0)

    return E_f, A_f, T_f, B_f
















def min_RMSE_ALMM(X, S, A, maxIter, alpha, beta, gamma, eta):
    """
    Returns the RMSE of Graph ALMM (S_sad is assumed to be constant here).
    """
    E_f, A_f, T_f, B_f = algo_2_almm_optimized(X = X, S = S, A_gt = A, 
                                                    alpha = alpha, beta = beta, gamma = gamma, eta = eta, 
                                                    maxIter = maxIter)

    # Check for labeling issues
    corr = np.corrcoef(A_f[0], A[0])[0, 1]
    if corr < 0:
        A_f_corrected = 1 - A_f
    else:
        A_f_corrected = A_f

    A_rmse = RMSE(A_f_corrected, A)  # Calculate RMSE

    return A_rmse
















def best_param_almm(X, A_gt, S_gt, maxIter, alpha_0, beta_0, gamma_0, eta_0):
    """
    Performs grid search on some regularization parameters (alpha, beta, xi, OH vs. Exact) to find
    the optimal combination of the four, minimizing the sum A_RMSE + S_SAD.

    Parameters:
    alpha (numpy.ndarray): [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    beta (numpy.ndarray): [beta_0, beta_0/10, beta_0/100, beta_0/1000, beta_0/10000]
    gamma (numpy.ndarray): [gamma_0, gamma_0/10, gamma_0/100, gamma_0/1000, gamma_0/10000]
    eta (numpy.ndarray): [eta_0, eta_0/10, eta_0/100, eta_0/1000, eta_0/10000]


    Returns:
    best (numpy.ndarray): An array with best combination minimizing A_RMSE + S_SAD
    in the format [alpha, beta, gamma, eta].
    """

    # Create a list of each combination
    alpha = [alpha_0, alpha_0/10, alpha_0/100, alpha_0/1000, alpha_0/10000]
    beta = [beta_0, beta_0/10, beta_0/100, beta_0/1000, beta_0/10000]
    gamma = [gamma_0, gamma_0/10, gamma_0/100, gamma_0/1000, gamma_0/10000]
    eta = [eta_0, eta_0/10, eta_0/100, eta_0/1000, eta_0/10000]

    combos = list(product(alpha, beta, gamma, eta))

    # Run the function using combinations
    results = Parallel(n_jobs =-1)(
        delayed(min_RMSE_ALMM)(X = X, S = S_gt, A = A_gt, 
                        alpha = a, beta = b, gamma = g, eta = e, maxIter = maxIter) 
                        for a, b, g, e in combos)

    # Create a 4D array to match the set order
    results = np.array(results).reshape(len(alpha), len(beta), len(gamma), len(eta))

    # Find min sum value and the corresponding combination
    idx = np.unravel_index(results.argmin(), results.shape)
    min_result = results[idx]
    alpha_idx, beta_idx, gamma_idx, eta_idx = idx

    # Save the best values
    alpha_best = alpha[alpha_idx]
    beta_best = beta[beta_idx]
    gamma_best = gamma[gamma_idx]
    eta_best = eta[eta_idx]

    # Print the best values
    print(f"Best RMSE: {min_result}")
    print(f"Best alpha: {alpha_best}")
    print(f"Best beta: {beta_best}")
    print(f"Best gamma: {gamma_best}")
    print(f"Best eta: {eta_best}")

    return [alpha_best, beta_best, gamma_best, eta_best]














def abundance_plotting(X, A_gt, A_f_GLU, A_f_GRSU, A_f_ALMM, A_f_Graph_ALMM, title = 'Abundance Comparison (Class 2)', color_title = 'Abundance (Class 2)'):
    ## Plotting abundance maps

    # Run PCA
    pca1 = PCA(n_components=2)
    X_pca1 = pca1.fit_transform(X.T)

    # Visualizing
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    vmin, vmax = 0, 1  # fixed scale so colors are directly comparable across all 4 plots

    # Top-left: Ground truth
    sc1 = axes[0, 0].scatter(X_pca1[:,0], X_pca1[:,1], c=A_gt[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0, 0].set_title('Ground Truth', fontsize = 14)

    # Top-middle: GLU
    sc2 = axes[0,1].scatter(X_pca1[:,0], X_pca1[:,1], c=A_f_GLU[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0,1].set_title('GLU', fontsize = 14)

    # Bottom-left: GRSU
    sc3 = axes[1,0].scatter(X_pca1[:,0], X_pca1[:,1], c=A_f_GRSU[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1,0].set_title('GRSU', fontsize = 14)

    # Bottom-middle: ALMM
    sc4 = axes[1, 1].scatter(X_pca1[:,0], X_pca1[:,1], c=A_f_ALMM[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1, 1].set_title('ALMM', fontsize = 14)

    # # Top-right: Graph-ALMM
    sc5 = axes[0, 2].scatter(X_pca1[:,0], X_pca1[:,1], c=A_f_Graph_ALMM[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0, 2].set_title('Graph-ALMM', fontsize = 14)

    # Bottom-right: leave empty, or hide it
    axes[1, 2].axis('off')

    for ax in axes.flat:
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)

    fig.suptitle(title, fontsize=16)

    fig.colorbar(sc4, ax=axes, label=color_title, shrink=0.8)

    # Save the image
    filename = title.replace(" ", "_")

    plt.savefig(
        f"plot_images/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()















    
def compute_I1_I2(A, A_hat, W, M):
    W_uu = W[M:, M:] # unlabeled to unlabeled
    W_ul = W[M:, :M] # unlabeled to labeled
    
    A_sq = np.sum(A**2, axis=0)
    dist_sq_uu = A_sq[:, None] + A_sq[None, :] - 2*(A.T @ A)
    I1 = 0.25 * W_uu.multiply(dist_sq_uu).sum()
    
    Ahat_sq = np.sum(A_hat**2, axis=0)
    dist_sq_ul = A_sq[:, None] + Ahat_sq[None, :] - 2*(A.T @ A_hat)
    I2 = 0.25 * W_ul.multiply(dist_sq_ul).sum()
    
    return I1, I2















def GLU_energy(A, W, M):
    W_uu = W[M:, M:] # unlabeled to unlabeled
    
    A_sq = np.sum(A**2, axis=0)
    dist_sq_uu = A_sq[:, None] + A_sq[None, :] - 2*(A.T @ A)
    I1 = 0.25 * W_uu.multiply(dist_sq_uu).sum()

    return I1

















def GRSU_energy(X, S, A, W, M, X_hat, A_hat, alpha, lambda_0):
    
    # Least Squares Problem
    term1 = 0.5 * (np.linalg.norm(X - (S @ A), 'fro') ** 2)

    # Label information
    term2 = ((alpha ** 2) / 2) * (np.linalg.norm(X_hat - (S @ A_hat), 'fro') ** 2)

    # Graph regularization (I1 + I2, equivalent to the inner product)
    I1, I2 = compute_I1_I2(A = A, A_hat = A_hat, W = W, M = M)

    return term1 + term2 + (lambda_0 * I1) + (lambda_0 * I2)
















def ALMM_energy(X, S, A, T, E, B, alpha, beta, gamma, eta):

    L = E.shape[1]
    I_L = np.eye(L)

    # Least Squares Problem
    term1 = 0.5 * (np.linalg.norm(X - ((S @ A) * T) - (E @ B), 'fro') ** 2)

    # Sparsity Regularization (L1)
    term2 = alpha * np.linalg.norm(A, ord = 1)

    # Generalization for Spectral Variability
    term3 = (beta / 2) * (np.linalg.norm(B, 'fro') ** 2)

    # Low-coherence with endmember dictionary and orthogonality
    term4 = (gamma / 2) * (np.linalg.norm(S.T @ E) ** 2) + (eta / 2) * (np.linalg.norm(E.T @ E - I_L, 'fro') ** 2)

    return term1 + term2 + term3 + term4
















def Graph_ALMM_energy(X, S, A, A_hat, T, E, B, W, M, alpha, beta, gamma, eta, xi):

    L = E.shape[1]
    I_L = np.eye(L)

    # Least Squares Problem
    term1 = 0.5 * (np.linalg.norm(X - ((S @ A) * T) - (E @ B), 'fro') ** 2)

    # Sparsity Regularization (L1/2)
    term2 = alpha * np.sum(np.sqrt(np.abs(A)))

    # Generalization for Spectral Variability
    term3 = (beta / 2) * (np.linalg.norm(B, 'fro') ** 2)

    # Low-coherence with endmember dictionary and orthogonality
    term4 = (gamma / 2) * (np.linalg.norm(S.T @ E) ** 2) + (eta / 2) * (np.linalg.norm(E.T @ E - I_L, 'fro') ** 2)

    # Graph Regularization Term
    A_unlabeled = A[:, M:]
    I1, I2 = compute_I1_I2(A = A_unlabeled, A_hat = A_hat, W = W, M = M)

    return term1 + term2 + term3 + term4 + ((1/xi)*I1) + ((1/xi)*I2)
















