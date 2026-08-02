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

# Import functions from Chen et. al.
from graph_active_learning_functions import *


"""
Parameter testing on GLU, GRSU, ALMM, and Graph ALMM (in this order) on datasets CuSO4, FeCl3, FeSO4, and CuSO4 (exponentially distributed).
Prints out the optimal set of parameters for each dataset and each algorithm, and saves them to a CSV file.
"""

def record_params(results_list, dataset, algo, p_type, params):
    """
    Helper function to append parameter results to the main list.
    """
    if p_type == 'glu_grsu':
        results_list.append({
            'Dataset': dataset, 'Algorithm': algo,
            'Alpha': params[0], 'Lambda': params[1], 'Gamma': params[2], 'Rho': params[3],
            'Beta': np.nan, 'Eta': np.nan, 'M_total': np.nan, 'Xi': np.nan, 'OH_labels': np.nan
        })
    elif p_type == 'almm':
        results_list.append({
            'Dataset': dataset, 'Algorithm': algo,
            'Alpha': params[0], 'Lambda': np.nan, 'Gamma': params[2], 'Rho': np.nan,
            'Beta': params[1], 'Eta': params[3], 'M_total': np.nan, 'Xi': np.nan, 'OH_labels': np.nan
        })
    elif p_type == 'graph_almm':
        results_list.append({
            'Dataset': dataset, 'Algorithm': algo,
            'Alpha': params[0], 'Lambda': np.nan, 'Gamma': np.nan, 'Rho': np.nan,
            'Beta': np.nan, 'Eta': np.nan, 'M_total': params[1], 'Xi': params[2], 'OH_labels': params[3]
        })

if __name__ == "__main__":

    # Initialize list to store results for CSV export
    all_results = []

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

    # ==========================================
    # CuSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4")
    print("==========================================")

    # Load data
    X_chem_CuSO4, S_gt_chem_CuSO4, A_gt_chem_CuSO4 = load_data(name = 'synth_CuSO4_data.npy', typename = 'chem', print_bool = False)


    ### GLU
    print("\n\nGLU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    record_params(all_results, 'CuSO4', 'GLU (OH)', 'glu_grsu', p)
    
    print("\nGLU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    record_params(all_results, 'CuSO4', 'GLU (Exact)', 'glu_grsu', p)


    ### GRSU
    print("\n\nGRSU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    record_params(all_results, 'CuSO4', 'GRSU (OH)', 'glu_grsu', p)
    
    print("\nGRSU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    record_params(all_results, 'CuSO4', 'GRSU (Exact)', 'glu_grsu', p)


    ### ALMM
    print("\n\nALMM")
    print("==========")
    p = best_param_almm(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)
    record_params(all_results, 'CuSO4', 'ALMM', 'almm', p)


    ### Graph ALMM
    print("\n\nGraph ALMM")
    print("==========")
    p = best_param_graph_almm(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    record_params(all_results, 'CuSO4', 'Graph ALMM', 'graph_almm', p)


    # ==========================================
    # FeCl3
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeCl3")
    print("==========================================")

    # Load data
    X_chem_FeCl3, S_gt_chem_FeCl3, A_gt_chem_FeCl3 = load_data(name = 'synth_FeCl3_data.npy', typename = 'chem', print_bool = False)

    ### GLU
    print("\n\nGLU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    record_params(all_results, 'FeCl3', 'GLU (OH)', 'glu_grsu', p)
    
    print("\nGLU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    record_params(all_results, 'FeCl3', 'GLU (Exact)', 'glu_grsu', p)


    ### GRSU
    print("\n\nGRSU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    record_params(all_results, 'FeCl3', 'GRSU (OH)', 'glu_grsu', p)
    
    print("\nGRSU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    record_params(all_results, 'FeCl3', 'GRSU (Exact)', 'glu_grsu', p)


    ### ALMM
    print("\n\nALMM")
    print("==========")
    p = best_param_almm(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)
    record_params(all_results, 'FeCl3', 'ALMM', 'almm', p)


    ### Graph ALMM
    print("\n\nGraph ALMM")
    print("==========")
    p = best_param_graph_almm(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    record_params(all_results, 'FeCl3', 'Graph ALMM', 'graph_almm', p)


    # ==========================================
    # FeSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeSO4")
    print("==========================================")

    # Load data
    X_chem_FeSO4, S_gt_chem_FeSO4, A_gt_chem_FeSO4 = load_data(name = 'synth_FeSO4_data.npy', typename = 'chem', print_bool = False)

    ### GLU
    print("\n\nGLU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    record_params(all_results, 'FeSO4', 'GLU (OH)', 'glu_grsu', p)
    
    print("\nGLU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    record_params(all_results, 'FeSO4', 'GLU (Exact)', 'glu_grsu', p)


    ### GRSU
    print("\n\nGRSU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    record_params(all_results, 'FeSO4', 'GRSU (OH)', 'glu_grsu', p)
    
    print("\nGRSU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    record_params(all_results, 'FeSO4', 'GRSU (Exact)', 'glu_grsu', p)


    ### ALMM
    print("\n\nALMM")
    print("==========")
    p = best_param_almm(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)
    record_params(all_results, 'FeSO4', 'ALMM', 'almm', p)


    ### Graph ALMM
    print("\n\nGraph ALMM")
    print("==========")
    p = best_param_graph_almm(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    record_params(all_results, 'FeSO4', 'Graph ALMM', 'graph_almm', p)


    # ==========================================
    # CuSO4 (Exponentially distributed)
    # ==========================================

    print("\n\n==========================================")
    print("Testing on CuSO4 (Exponentially distributed)")
    print("==========================================")

    # Load data
    X_chem_CuSO4_exp, S_gt_chem_CuSO4_exp, A_gt_chem_CuSO4_exp = load_data(name = 'synth_chem_data.npy', typename = 'chem', print_bool = False)


    ### GLU
    print("\n\nGLU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    record_params(all_results, 'CuSO4 (Exp)', 'GLU (OH)', 'glu_grsu', p)
    
    print("\nGLU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    record_params(all_results, 'CuSO4 (Exp)', 'GLU (Exact)', 'glu_grsu', p)


    ### GRSU
    print("\n\nGRSU (OH)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    record_params(all_results, 'CuSO4 (Exp)', 'GRSU (OH)', 'glu_grsu', p)
    
    print("\nGRSU (Exact)")
    print("==========")
    p = parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    record_params(all_results, 'CuSO4 (Exp)', 'GRSU (Exact)', 'glu_grsu', p)


    ### ALMM
    print("\n\nALMM")
    print("==========")
    p = best_param_almm(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp,
                    maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0)
    record_params(all_results, 'CuSO4 (Exp)', 'ALMM', 'almm', p)


    ### Graph ALMM
    print("\n\nGraph ALMM")
    print("==========")
    p = best_param_graph_almm(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    record_params(all_results, 'CuSO4 (Exp)', 'Graph ALMM', 'graph_almm', p)


    print("\n\nFinished Testing")
    print("==========================================")

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_filename = "optimal_parameters.csv"
    df.to_csv(output_filename, index=False)
    print(f"Optimal parameters saved successfully to {output_filename}")