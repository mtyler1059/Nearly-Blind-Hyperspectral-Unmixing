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

    parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    print("\nGLU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    


    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    print("\nGRSU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    


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

    parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    print("\nGLU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    


    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    print("\nGRSU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    


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



    ### GLU

    print("\n\nGLU (OH)")
    print("==========")

    parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    print("\nGLU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    


    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    print("\nGRSU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    


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

    parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = True)
    
    print("\nGLU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = False, OH_labels = False)
    


    ### GRSU

    print("\n\nGRSU (OH)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = True)
    
    print("\nGRSU (Exact)")
    print("==========")

    parameter_testing(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples, 
                  iters = iters, alpha = alpha_vals, lam = lam_vals, gamma = gamma_vals, rho = rho_vals, m_0 = 2, 
                  print_bool = False, GRSU_bool = True, OH_labels = False)
    


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
    print("==========================================")
