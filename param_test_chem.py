# Maintain consistency
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from sklearn.decomposition import PCA

# Import functions from Chen et. al.
from graph_active_learning_functions import *


"""
Parameter testing on GLU, GRSU, ALMM, and Graph ALMM (in this order) on datasets CuSO4, FeCl3, FeSO4, and CuSO4 (exponentially distributed).
Prints out the optimal set of parameters for each dataset and each algorithm.
"""

if __name__ == "__main__":

    # Parameter values 
    # GLU and GRSU
    alpha_vals = np.array([10, 20, 50, 100])
    lam_vals = np.sort(np.concatenate([10**np.arange(4), 5 * 10**np.arange(4)]))
    gamma_vals = 10.0 ** np.arange(-2, 3)
    rho_vals = 10.0 ** np.arange(-2, 3)

    # ALMM and Graph ALMM
    alpha_0 = (1e-3 + 1e-2)/2
    beta_0 = (1e-3 + 1e-2)/2
    gamma_0 = (1e-3 + 1e-2)/2
    eta_0 = (1e-3 + 1e-2)/2

    # Fixed values
    samples = 1000
    iters = 100

    print("==========================================")
    print("Starting Testing")
    print("Minimizing RMSE")

    # ==========================================
    # CuSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4")
    print("==========================================")

    # Load data
    X_chem_CuSO4, S_gt_chem_CuSO4, A_gt_chem_CuSO4 = load_data(name = 'synth_CuSO4_data.npy', typename = 'chem', print_bool = False)
    #print(f"X: {X_chem_CuSO4.shape} \n S: {S_gt_chem_CuSO4.shape} \n A: {A_gt_chem_CuSO4.shape}")


    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_CuSO4 = parameter_testing_RMSE(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_CuSO4[0]
    lam_1 = params_GLU_OH_CuSO4[1]
    gamma_1 = params_GLU_OH_CuSO4[2]
    rho_1 = params_GLU_OH_CuSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_CuSO4 = parameter_testing_RMSE(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_CuSO4[0]
    lam_1 = params_GLU_exact_CuSO4[1]
    gamma_1 = params_GLU_exact_CuSO4[2]
    rho_1 = params_GLU_exact_CuSO4[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_CuSO4 = parameter_testing_RMSE(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_CuSO4[0]
    lam_1 = params_GRSU_OH_CuSO4[1]
    gamma_1 = params_GRSU_OH_CuSO4[2]
    rho_1 = params_GRSU_OH_CuSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_CuSO4 = parameter_testing_RMSE(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_CuSO4[0]
    lam_1 = params_GRSU_exact_CuSO4[1]
    gamma_1 = params_GRSU_exact_CuSO4[2]
    rho_1 = params_GRSU_exact_CuSO4[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### ALMM

    print("\n\nALMM")
    print("==========")

    best_param_almm(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # FeCl3
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeCl3")
    print("==========================================")

    # Load data
    X_chem_FeCl3, S_gt_chem_FeCl3, A_gt_chem_FeCl3 = load_data(name = 'synth_FeCl3_data.npy', typename = 'chem', print_bool = False)
    print(f"X: {X_chem_FeCl3.shape} \n S: {S_gt_chem_FeCl3.shape} \n A: {A_gt_chem_FeCl3.shape}")



    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_FeCl3 = parameter_testing_RMSE(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_FeCl3[0]
    lam_1 = params_GLU_OH_FeCl3[1]
    gamma_1 = params_GLU_OH_FeCl3[2]
    rho_1 = params_GLU_OH_FeCl3[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_FeCl3 = parameter_testing_RMSE(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_FeCl3[0]
    lam_1 = params_GLU_exact_FeCl3[1]
    gamma_1 = params_GLU_exact_FeCl3[2]
    rho_1 = params_GLU_exact_FeCl3[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_FeCl3 = parameter_testing_RMSE(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_FeCl3[0]
    lam_1 = params_GRSU_OH_FeCl3[1]
    gamma_1 = params_GRSU_OH_FeCl3[2]
    rho_1 = params_GRSU_OH_FeCl3[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_FeCl3 = parameter_testing_RMSE(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_FeCl3[0]
    lam_1 = params_GRSU_exact_FeCl3[1]
    gamma_1 = params_GRSU_exact_FeCl3[2]
    rho_1 = params_GRSU_exact_FeCl3[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    ### ALMM

    print("\n\nALMM")
    print("==========")

    best_param_almm(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # FeSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeSO4")
    print("==========================================")

    # Load data
    X_chem_FeSO4, S_gt_chem_FeSO4, A_gt_chem_FeSO4 = load_data(name = 'synth_FeSO4_data.npy', typename = 'chem', print_bool = False)
    #print(f"X: {X_chem_FeSO4.shape} \n S: {S_gt_chem_FeSO4.shape} \n A: {A_gt_chem_FeSO4.shape}")


    ### Checking issue with stuff

    # Generating prep
    G, W = build_custom_knn_graph(X_chem_FeSO4.T, K=int(1000*0.005))
    labeled_indices = algo_1_active_learning(X_chem_FeSO4, W, m_initial=2, M_total=int(0.004*1000), num_eigs=int(1000*0.005))
    X_hat = X_chem_FeSO4[:, labeled_indices]
    A_hat_exact = A_gt_chem_FeSO4[:, labeled_indices]
    A_hat_OH = generate_one_hot_labels(A_hat_exact)
    prep = [X_hat, A_hat_OH, "OH"]

    print("Running consistency check (expect ~14)...")
    _check = Parallel(n_jobs=1)(delayed(sum_RMSE_SAD)(
        X_chem_FeSO4, A_gt_chem_FeSO4, S_gt_chem_FeSO4, 1000, iters, 100.0, 1.0, 0.01, 0.01, 2,
        print_bool=False, OH_labels=True, GRSU_bool=False, prep=prep
    ) for _ in range(1))
    print(f"Sanity check result: {_check[0]}")

    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_FeSO4 = parameter_testing_RMSE(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_FeSO4[0]
    lam_1 = params_GLU_OH_FeSO4[1]
    gamma_1 = params_GLU_OH_FeSO4[2]
    rho_1 = params_GLU_OH_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_FeSO4 = parameter_testing_RMSE(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_FeSO4[0]
    lam_1 = params_GLU_exact_FeSO4[1]
    gamma_1 = params_GLU_exact_FeSO4[2]
    rho_1 = params_GLU_exact_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_FeSO4 = parameter_testing_RMSE(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_FeSO4[0]
    lam_1 = params_GRSU_OH_FeSO4[1]
    gamma_1 = params_GRSU_OH_FeSO4[2]
    rho_1 = params_GRSU_OH_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_FeSO4 = parameter_testing_RMSE(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_FeSO4[0]
    lam_1 = params_GRSU_exact_FeSO4[1]
    gamma_1 = params_GRSU_exact_FeSO4[2]
    rho_1 = params_GRSU_exact_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### ALMM

    print("\n\nALMM")
    print("==========")

    best_param_almm(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # CuSO4 (Exponentially distributed)
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4 (Exponentially distributed)")
    print("==========================================")

    # Load data
    X_chem_CuSO4_exp, S_gt_chem_CuSO4_exp, A_gt_chem_CuSO4_exp = load_data(name = 'synth_chem_data.npy', typename = 'chem', print_bool = False)
    print(f"X: {X_chem_CuSO4_exp.shape} \n S: {S_gt_chem_CuSO4_exp.shape} \n A: {A_gt_chem_CuSO4_exp.shape}")



    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_CuSO4_exp = parameter_testing_RMSE(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_CuSO4_exp[0]
    lam_1 = params_GLU_OH_CuSO4_exp[1]
    gamma_1 = params_GLU_OH_CuSO4_exp[2]
    rho_1 = params_GLU_OH_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_CuSO4_exp = parameter_testing_RMSE(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_CuSO4_exp[0]
    lam_1 = params_GLU_exact_CuSO4_exp[1]
    gamma_1 = params_GLU_exact_CuSO4_exp[2]
    rho_1 = params_GLU_exact_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_CuSO4_exp = parameter_testing_RMSE(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_CuSO4_exp[0]
    lam_1 = params_GRSU_OH_CuSO4_exp[1]
    gamma_1 = params_GRSU_OH_CuSO4_exp[2]
    rho_1 = params_GRSU_OH_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_CuSO4_exp = parameter_testing_RMSE(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_CuSO4_exp[0]
    lam_1 = params_GRSU_exact_CuSO4_exp[1]
    gamma_1 = params_GRSU_exact_CuSO4_exp[2]
    rho_1 = params_GRSU_exact_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### ALMM

    print("\n\nALMM")
    print("==========")

    best_param_almm(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    print("\n\nFinished Testing")
    print("==========================================\n")

    ### RMSE + SAD

    print("==========================================")
    print("Starting Testing")
    print("Minimizing RMSE + SAD")

    # ==========================================
    # CuSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4")
    print("==========================================")

    # Load data
    X_chem_CuSO4, S_gt_chem_CuSO4, A_gt_chem_CuSO4 = load_data(name = 'synth_CuSO4_data.npy', typename = 'chem', print_bool = False)
    #print(f"X: {X_chem_CuSO4.shape} \n S: {S_gt_chem_CuSO4.shape} \n A: {A_gt_chem_CuSO4.shape}")


    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_CuSO4 = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_CuSO4[0]
    lam_1 = params_GLU_OH_CuSO4[1]
    gamma_1 = params_GLU_OH_CuSO4[2]
    rho_1 = params_GLU_OH_CuSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_CuSO4 = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_CuSO4[0]
    lam_1 = params_GLU_exact_CuSO4[1]
    gamma_1 = params_GLU_exact_CuSO4[2]
    rho_1 = params_GLU_exact_CuSO4[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_CuSO4 = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_CuSO4[0]
    lam_1 = params_GRSU_OH_CuSO4[1]
    gamma_1 = params_GRSU_OH_CuSO4[2]
    rho_1 = params_GRSU_OH_CuSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_CuSO4 = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_CuSO4[0]
    lam_1 = params_GRSU_exact_CuSO4[1]
    gamma_1 = params_GRSU_exact_CuSO4[2]
    rho_1 = params_GRSU_exact_CuSO4[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # FeCl3
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeCl3")
    print("==========================================")

    # Load data
    X_chem_FeCl3, S_gt_chem_FeCl3, A_gt_chem_FeCl3 = load_data(name = 'synth_FeCl3_data.npy', typename = 'chem', print_bool = False)
    print(f"X: {X_chem_FeCl3.shape} \n S: {S_gt_chem_FeCl3.shape} \n A: {A_gt_chem_FeCl3.shape}")



    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_FeCl3 = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_FeCl3[0]
    lam_1 = params_GLU_OH_FeCl3[1]
    gamma_1 = params_GLU_OH_FeCl3[2]
    rho_1 = params_GLU_OH_FeCl3[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_FeCl3 = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_FeCl3[0]
    lam_1 = params_GLU_exact_FeCl3[1]
    gamma_1 = params_GLU_exact_FeCl3[2]
    rho_1 = params_GLU_exact_FeCl3[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_FeCl3 = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_FeCl3[0]
    lam_1 = params_GRSU_OH_FeCl3[1]
    gamma_1 = params_GRSU_OH_FeCl3[2]
    rho_1 = params_GRSU_OH_FeCl3[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_FeCl3 = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_FeCl3[0]
    lam_1 = params_GRSU_exact_FeCl3[1]
    gamma_1 = params_GRSU_exact_FeCl3[2]
    rho_1 = params_GRSU_exact_FeCl3[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # FeSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeSO4")
    print("==========================================")

    # Load data
    X_chem_FeSO4, S_gt_chem_FeSO4, A_gt_chem_FeSO4 = load_data(name = 'synth_FeSO4_data.npy', typename = 'chem', print_bool = False)
    #print(f"X: {X_chem_FeSO4.shape} \n S: {S_gt_chem_FeSO4.shape} \n A: {A_gt_chem_FeSO4.shape}")


    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_FeSO4 = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_FeSO4[0]
    lam_1 = params_GLU_OH_FeSO4[1]
    gamma_1 = params_GLU_OH_FeSO4[2]
    rho_1 = params_GLU_OH_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_FeSO4 = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_FeSO4[0]
    lam_1 = params_GLU_exact_FeSO4[1]
    gamma_1 = params_GLU_exact_FeSO4[2]
    rho_1 = params_GLU_exact_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_FeSO4 = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_FeSO4[0]
    lam_1 = params_GRSU_OH_FeSO4[1]
    gamma_1 = params_GRSU_OH_FeSO4[2]
    rho_1 = params_GRSU_OH_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_FeSO4 = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_FeSO4[0]
    lam_1 = params_GRSU_exact_FeSO4[1]
    gamma_1 = params_GRSU_exact_FeSO4[2]
    rho_1 = params_GRSU_exact_FeSO4[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    # ==========================================
    # CuSO4 (Exponentially distributed)
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4 (Exponentially distributed)")
    print("==========================================")

    # Load data
    X_chem_CuSO4_exp, S_gt_chem_CuSO4_exp, A_gt_chem_CuSO4_exp = load_data(name = 'synth_chem_data.npy', typename = 'chem', print_bool = False)
    print(f"X: {X_chem_CuSO4_exp.shape} \n S: {S_gt_chem_CuSO4_exp.shape} \n A: {A_gt_chem_CuSO4_exp.shape}")



    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    params_GLU_OH_CuSO4_exp = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_CuSO4_exp[0]
    lam_1 = params_GLU_OH_CuSO4_exp[1]
    gamma_1 = params_GLU_OH_CuSO4_exp[2]
    rho_1 = params_GLU_OH_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGLU (Exact)")
    print("==========")

    params_GLU_exact_CuSO4_exp = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_CuSO4_exp[0]
    lam_1 = params_GLU_exact_CuSO4_exp[1]
    gamma_1 = params_GLU_exact_CuSO4_exp[2]
    rho_1 = params_GLU_exact_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    params_GRSU_OH_CuSO4_exp = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_CuSO4_exp[0]
    lam_1 = params_GRSU_OH_CuSO4_exp[1]
    gamma_1 = params_GRSU_OH_CuSO4_exp[2]
    rho_1 = params_GRSU_OH_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    
    print("\nGRSU (Exact)")
    print("==========")

    params_GRSU_exact_CuSO4_exp = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_CuSO4_exp[0]
    lam_1 = params_GRSU_exact_CuSO4_exp[1]
    gamma_1 = params_GRSU_exact_CuSO4_exp[2]
    rho_1 = params_GRSU_exact_CuSO4_exp[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    ### Graph ALMM

    print("\n\nGraph ALMM")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    print("\n\nFinished Testing")
    print("==========================================")