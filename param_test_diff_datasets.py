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

    # Set seed
    np.random.seed(42)
    # caffeinate -i python3 -u param_test_diff_datasets.py | tee test_results/param_test_diff_datasets_samson.txt

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
    iters = 60

    print("==========================================")
    print("Starting Testing")

    # ==========================================
    # Urban
    # ==========================================

    print("\n\n==========================================")
    print("Testing on Urban")
    print("==========================================")

    # Load data
    X_0, S_0, A_0 = load_data(name = 'processed_data/processed_data/urban/urban_processed_data.npy', typename = 'HSI', sample = True, H = 307, W = 307)    #print(f"X: {X_0.shape} \n S: {S_0.shape} \n A: {A_0.shape}")
    samples = 2500

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_urban = parameter_testing_RMSE(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_urban[0]
    lam_1 = params_GLU_OH_urban[1]
    gamma_1 = params_GLU_OH_urban[2]
    rho_1 = params_GLU_OH_urban[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_urban = parameter_testing(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_urban[0]
    lam_1 = params_GLU_OH_urban[1]
    gamma_1 = params_GLU_OH_urban[2]
    rho_1 = params_GLU_OH_urban[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_urban = parameter_testing_RMSE(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_urban[0]
    lam_1 = params_GLU_exact_urban[1]
    gamma_1 = params_GLU_exact_urban[2]
    rho_1 = params_GLU_exact_urban[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_urban = parameter_testing(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_urban[0]
    lam_1 = params_GLU_exact_urban[1]
    gamma_1 = params_GLU_exact_urban[2]
    rho_1 = params_GLU_exact_urban[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_urban = parameter_testing_RMSE(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_urban[0]
    lam_1 = params_GRSU_OH_urban[1]
    gamma_1 = params_GRSU_OH_urban[2]
    rho_1 = params_GRSU_OH_urban[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_urban = parameter_testing(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_urban[0]
    lam_1 = params_GRSU_OH_urban[1]
    gamma_1 = params_GRSU_OH_urban[2]
    rho_1 = params_GRSU_OH_urban[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_urban = parameter_testing_RMSE(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_urban[0]
    lam_1 = params_GRSU_exact_urban[1]
    gamma_1 = params_GRSU_exact_urban[2]
    rho_1 = params_GRSU_exact_urban[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_urban = parameter_testing(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_urban[0]
    lam_1 = params_GRSU_exact_urban[1]
    gamma_1 = params_GRSU_exact_urban[2]
    rho_1 = params_GRSU_exact_urban[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_0, A_gt = A_0, S_gt = S_0, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_0, A_gt = A_0, S_gt = S_0,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)


    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_0, A_gt = A_0, S_gt = S_0, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_0, A_gt = A_0, S_gt = S_0, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)



    # ==========================================
    # Samson
    # ==========================================

    print("\n\n==========================================")
    print("Testing on Samson")
    print("==========================================")

    # Load data
    X_2, S_2, A_2 = load_data(name = 'processed_data/processed_data/samson/samson_processed_data.npy', typename = 'HSI', sample = True, H = 95, W = 95)    
    #print(f"X: {X_0.shape} \n S: {S_0.shape} \n A: {A_0.shape}")
    samples = 2500

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_samson = parameter_testing_RMSE(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_samson[0]
    lam_1 = params_GLU_OH_samson[1]
    gamma_1 = params_GLU_OH_samson[2]
    rho_1 = params_GLU_OH_samson[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_samson = parameter_testing(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_samson[0]
    lam_1 = params_GLU_OH_samson[1]
    gamma_1 = params_GLU_OH_samson[2]
    rho_1 = params_GLU_OH_samson[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_samson = parameter_testing_RMSE(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_samson[0]
    lam_1 = params_GLU_exact_samson[1]
    gamma_1 = params_GLU_exact_samson[2]
    rho_1 = params_GLU_exact_samson[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_samson = parameter_testing(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_samson[0]
    lam_1 = params_GLU_exact_samson[1]
    gamma_1 = params_GLU_exact_samson[2]
    rho_1 = params_GLU_exact_samson[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_samson = parameter_testing_RMSE(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_samson[0]
    lam_1 = params_GRSU_OH_samson[1]
    gamma_1 = params_GRSU_OH_samson[2]
    rho_1 = params_GRSU_OH_samson[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_samson = parameter_testing(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_samson[0]
    lam_1 = params_GRSU_OH_samson[1]
    gamma_1 = params_GRSU_OH_samson[2]
    rho_1 = params_GRSU_OH_samson[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_samson = parameter_testing_RMSE(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_samson[0]
    lam_1 = params_GRSU_exact_samson[1]
    gamma_1 = params_GRSU_exact_samson[2]
    rho_1 = params_GRSU_exact_samson[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_samson = parameter_testing(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 3, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_samson[0]
    lam_1 = params_GRSU_exact_samson[1]
    gamma_1 = params_GRSU_exact_samson[2]
    rho_1 = params_GRSU_exact_samson[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_2, A_gt = A_2, S_gt = S_2, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 3, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_2, A_gt = A_2, S_gt = S_2,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)


    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_2, A_gt = A_2, S_gt = S_2, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 3)

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_2, A_gt = A_2, S_gt = S_2, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 3)

    # ==========================================
    # jasper
    # ==========================================

    print("\n\n==========================================")
    print("Testing on jasper")
    print("==========================================")

    # Load data
    X_3, S_3, A_3 = load_data(name = 'processed_data/processed_data/jasper/jasper_processed_data.npy', typename = 'HSI', sample = True, H = 100, W = 100)    
    #print(f"X: {X_0.shape} \n S: {S_0.shape} \n A: {A_0.shape}")
    samples = 2500

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_jasper = parameter_testing_RMSE(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_jasper[0]
    lam_1 = params_GLU_OH_jasper[1]
    gamma_1 = params_GLU_OH_jasper[2]
    rho_1 = params_GLU_OH_jasper[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_jasper = parameter_testing(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_jasper[0]
    lam_1 = params_GLU_OH_jasper[1]
    gamma_1 = params_GLU_OH_jasper[2]
    rho_1 = params_GLU_OH_jasper[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_jasper = parameter_testing_RMSE(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_jasper[0]
    lam_1 = params_GLU_exact_jasper[1]
    gamma_1 = params_GLU_exact_jasper[2]
    rho_1 = params_GLU_exact_jasper[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_jasper = parameter_testing(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_jasper[0]
    lam_1 = params_GLU_exact_jasper[1]
    gamma_1 = params_GLU_exact_jasper[2]
    rho_1 = params_GLU_exact_jasper[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_jasper = parameter_testing_RMSE(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_jasper[0]
    lam_1 = params_GRSU_OH_jasper[1]
    gamma_1 = params_GRSU_OH_jasper[2]
    rho_1 = params_GRSU_OH_jasper[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_jasper = parameter_testing(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_jasper[0]
    lam_1 = params_GRSU_OH_jasper[1]
    gamma_1 = params_GRSU_OH_jasper[2]
    rho_1 = params_GRSU_OH_jasper[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_jasper = parameter_testing_RMSE(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_jasper[0]
    lam_1 = params_GRSU_exact_jasper[1]
    gamma_1 = params_GRSU_exact_jasper[2]
    rho_1 = params_GRSU_exact_jasper[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_jasper = parameter_testing(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_jasper[0]
    lam_1 = params_GRSU_exact_jasper[1]
    gamma_1 = params_GRSU_exact_jasper[2]
    rho_1 = params_GRSU_exact_jasper[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_3, A_gt = A_3, S_gt = S_3, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_3, A_gt = A_3, S_gt = S_3,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)


    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_3, A_gt = A_3, S_gt = S_3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_3, A_gt = A_3, S_gt = S_3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)


    # ==========================================
    # apex
    # ==========================================

    print("\n\n==========================================")
    print("Testing on apex")
    print("==========================================")

    # Load data
    X_4, S_4, A_4 = load_data(name = 'processed_data/processed_data/apex/apex_processed_data.npy', typename = 'HSI', sample = True, H = 111, W = 122)    
    samples = 2500

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_apex = parameter_testing_RMSE(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_apex[0]
    lam_1 = params_GLU_OH_apex[1]
    gamma_1 = params_GLU_OH_apex[2]
    rho_1 = params_GLU_OH_apex[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_apex = parameter_testing(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_apex[0]
    lam_1 = params_GLU_OH_apex[1]
    gamma_1 = params_GLU_OH_apex[2]
    rho_1 = params_GLU_OH_apex[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_apex = parameter_testing_RMSE(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_apex[0]
    lam_1 = params_GLU_exact_apex[1]
    gamma_1 = params_GLU_exact_apex[2]
    rho_1 = params_GLU_exact_apex[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_apex = parameter_testing(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_apex[0]
    lam_1 = params_GLU_exact_apex[1]
    gamma_1 = params_GLU_exact_apex[2]
    rho_1 = params_GLU_exact_apex[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_apex = parameter_testing_RMSE(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_apex[0]
    lam_1 = params_GRSU_OH_apex[1]
    gamma_1 = params_GRSU_OH_apex[2]
    rho_1 = params_GRSU_OH_apex[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_apex = parameter_testing(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_apex[0]
    lam_1 = params_GRSU_OH_apex[1]
    gamma_1 = params_GRSU_OH_apex[2]
    rho_1 = params_GRSU_OH_apex[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_apex = parameter_testing_RMSE(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_apex[0]
    lam_1 = params_GRSU_exact_apex[1]
    gamma_1 = params_GRSU_exact_apex[2]
    rho_1 = params_GRSU_exact_apex[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_apex = parameter_testing(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 4, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_apex[0]
    lam_1 = params_GRSU_exact_apex[1]
    gamma_1 = params_GRSU_exact_apex[2]
    rho_1 = params_GRSU_exact_apex[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_4, A_gt = A_4, S_gt = S_4, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 4, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_4, A_gt = A_4, S_gt = S_4,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)


    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_4, A_gt = A_4, S_gt = S_4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_4, A_gt = A_4, S_gt = S_4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 4)


    # ==========================================
    # Nonlinear
    # ==========================================

    print("\n\n==========================================")
    print("Testing on Nonlinear")
    print("==========================================")

    # Load data
    X_nl_flat, S_gt_nl, A_gt_nl = load_data(name = 'my_custom_dataset.npy', typename = 'nonlinear')
    samples = 2500

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_nl = parameter_testing_RMSE(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_nl[0]
    lam_1 = params_GLU_OH_nl[1]
    gamma_1 = params_GLU_OH_nl[2]
    rho_1 = params_GLU_OH_nl[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_nl = parameter_testing(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_nl[0]
    lam_1 = params_GLU_OH_nl[1]
    gamma_1 = params_GLU_OH_nl[2]
    rho_1 = params_GLU_OH_nl[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_nl = parameter_testing_RMSE(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_nl[0]
    lam_1 = params_GLU_exact_nl[1]
    gamma_1 = params_GLU_exact_nl[2]
    rho_1 = params_GLU_exact_nl[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_nl = parameter_testing(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_nl[0]
    lam_1 = params_GLU_exact_nl[1]
    gamma_1 = params_GLU_exact_nl[2]
    rho_1 = params_GLU_exact_nl[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_nl = parameter_testing_RMSE(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_nl[0]
    lam_1 = params_GRSU_OH_nl[1]
    gamma_1 = params_GRSU_OH_nl[2]
    rho_1 = params_GRSU_OH_nl[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_nl = parameter_testing(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_nl[0]
    lam_1 = params_GRSU_OH_nl[1]
    gamma_1 = params_GRSU_OH_nl[2]
    rho_1 = params_GRSU_OH_nl[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_nl = parameter_testing_RMSE(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_nl[0]
    lam_1 = params_GRSU_exact_nl[1]
    gamma_1 = params_GRSU_exact_nl[2]
    rho_1 = params_GRSU_exact_nl[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_nl = parameter_testing(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_nl[0]
    lam_1 = params_GRSU_exact_nl[1]
    gamma_1 = params_GRSU_exact_nl[2]
    rho_1 = params_GRSU_exact_nl[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)

    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_nl_flat, A_gt = A_gt_nl, S_gt = S_gt_nl, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # ==========================================
    # Linear
    # ==========================================

    print("\n\n==========================================")
    print("Testing on Linear")
    print("==========================================")

    # Load data
    X_linear, S_linear, A_linear = synthetic_linear_data(samples = 2000, channels = 300)
    samples = 2000

    ### GLU

    print("\n\nGLU (OH) (RMSE)")
    print("==========")

    params_GLU_OH_linear = parameter_testing_RMSE(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_linear[0]
    lam_1 = params_GLU_OH_linear[1]
    gamma_1 = params_GLU_OH_linear[2]
    rho_1 = params_GLU_OH_linear[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGLU (OH) (RMSE + SAD)")
    print("==========")

    params_GLU_OH_linear = parameter_testing(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    alpha_1 = params_GLU_OH_linear[0]
    lam_1 = params_GLU_OH_linear[1]
    gamma_1 = params_GLU_OH_linear[2]
    rho_1 = params_GLU_OH_linear[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGLU (Exact) (RMSE)")
    print("==========")

    params_GLU_exact_linear = parameter_testing_RMSE(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_linear[0]
    lam_1 = params_GLU_exact_linear[1]
    gamma_1 = params_GLU_exact_linear[2]
    rho_1 = params_GLU_exact_linear[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGLU (Exact) (RMSE + SAD)")
    print("==========")

    params_GLU_exact_linear = parameter_testing(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    
    alpha_1 = params_GLU_exact_linear[0]
    lam_1 = params_GLU_exact_linear[1]
    gamma_1 = params_GLU_exact_linear[2]
    rho_1 = params_GLU_exact_linear[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = False, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    ### GRSU

    print("\n\nGRSU (OH) (RMSE)")
    print("==========")

    params_GRSU_OH_linear = parameter_testing_RMSE(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_linear[0]
    lam_1 = params_GRSU_OH_linear[1]
    gamma_1 = params_GRSU_OH_linear[2]
    rho_1 = params_GRSU_OH_linear[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\n\nGRSU (OH) (RMSE + SAD)")
    print("==========")

    params_GRSU_OH_linear = parameter_testing(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    alpha_1 = params_GRSU_OH_linear[0]
    lam_1 = params_GRSU_OH_linear[1]
    gamma_1 = params_GRSU_OH_linear[2]
    rho_1 = params_GRSU_OH_linear[3]

    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = True, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")
    
    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")
    


    print("\nGRSU (Exact) (RMSE)")
    print("==========")

    params_GRSU_exact_linear = parameter_testing_RMSE(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_linear[0]
    lam_1 = params_GRSU_exact_linear[1]
    gamma_1 = params_GRSU_exact_linear[2]
    rho_1 = params_GRSU_exact_linear[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")

    # RMSE + SAD

    print("\nGRSU (Exact) (RMSE + SAD)")
    print("==========")

    params_GRSU_exact_linear = parameter_testing(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    
    alpha_1 = params_GRSU_exact_linear[0]
    lam_1 = params_GRSU_exact_linear[1]
    gamma_1 = params_GRSU_exact_linear[2]
    rho_1 = params_GRSU_exact_linear[3]
    
    # Run algorithm on chosen parameters
    A_f, S_f, A_rmse, S_sad = run_unmixing_pipeline_example(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples, iters = iters, 
                                                                                                    alpha = alpha_1, lam = lam_1, gamma = gamma_1, rho = rho_1, m_0 = 2, 
                                                                                                    print_bool = False, OH_labels = False, GRSU_bool = True, 
                                                                                                    A_error = False, RMSE_plot = False, title_0 = "")

    print(f"RMSE: {A_rmse}\n SAD: {S_sad}")



    # ### ALMM

    # print("\n\nALMM")
    # print("==========")

    # best_param_almm(X = X_linear, A_gt = A_linear, S_gt = S_linear,
    #                 maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)

    ### Graph ALMM

    print("\n\nGraph ALMM (RMSE)")
    print("==========")
    
    best_param_graph_almm(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    

    print("\n\nGraph ALMM (RMSE + SAD)")
    print("==========")
    
    best_param_graph_almm_RMSE_SAD(X = X_linear, A_gt = A_linear, S_gt = S_linear, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    

    # ==========================================
    # CuSO4 (Exponentially distributed)
    # ==========================================

    # print("\n\n==========================================")
    # print("Testing on CuSO4 (Exponentially distributed)")
    # print("==========================================")

    # # Load data
    # X_0_exp, S_0_exp, A_0_exp = load_data(name = 'synth_chem_data.npy', typename = 'chem', print_bool = False)
    # print(f"X: {X_0_exp.shape} \n S: {S_0_exp.shape} \n A: {A_0_exp.shape}")


    # ### Graph ALMM

    # print("\n\nGraph ALMM (Ver 1)")
    # print("==========")
    
    # best_param_graph_almm(X = X_0_exp, A_gt = A_0_exp, S_gt = S_0_exp, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    print("\n\nFinished Testing")
    print("==========================================\n")

