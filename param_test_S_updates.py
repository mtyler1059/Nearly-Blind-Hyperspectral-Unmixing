# Maintain consistency
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from graph_almm_experiments import *


"""
Parameter testing on GLU, GRSU, ALMM, and Graph ALMM (in this order) on datasets CuSO4, FeCl3, FeSO4, and CuSO4 (exponentially distributed).
Prints out the optimal set of parameters for each dataset and each algorithm.
"""

if __name__ == "__main__":

    # Set seed
    np.random.seed(42)
    # caffeinate -i python3 -u param_test_S_updates.py | tee param_test_S_updates.txt

    # Parameter values 

    # ALMM and Graph ALMM
    alpha_0 = (1e-3 + 1e-2)/2
    beta_0 = (1e-3 + 1e-2)/2
    gamma_0 = (1e-3 + 1e-2)/2
    eta_0 = (1e-3 + 1e-2)/2

    # Fixed values
    samples = 1000
    iters = 60

    print("==========================================")
    print("Starting Testing")
    print("Ver 1: S update is at the top")
    print("Ver 2: S update is right after M")
    print("Ver 3: S update is right after E")
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


    ### Graph ALMM

    print("\n\nGraph ALMM (Ver 1)")
    print("==========")
    
    best_param_graph_almm_ver1(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)


    print("\n\nGraph ALMM (Ver 2)")
    print("==========")
    
    best_param_graph_almm_ver2(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    
    print("\n\nGraph ALMM (Ver 3)")
    print("==========")
    
    best_param_graph_almm_ver3(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # ==========================================
    # FeCl3
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeCl3")
    print("==========================================")

    # Load data
    X_chem_FeCl3, S_gt_chem_FeCl3, A_gt_chem_FeCl3 = load_data(name = 'synth_FeCl3_data.npy', typename = 'chem', print_bool = False)


    ### Graph ALMM

    print("\n\nGraph ALMM (Ver 1)")
    print("==========")
    
    best_param_graph_almm_ver1(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    
    print("\n\nGraph ALMM (Ver 2)")
    print("==========")
    
    best_param_graph_almm_ver2(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    print("\n\nGraph ALMM (Ver 3)")
    print("==========")
    
    best_param_graph_almm_ver3(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # ==========================================
    # FeSO4
    # ==========================================

    print("\n\n==========================================")
    print("Testing on FeSO4")
    print("==========================================")

    # Load data
    X_chem_FeSO4, S_gt_chem_FeSO4, A_gt_chem_FeSO4 = load_data(name = 'synth_FeSO4_data.npy', typename = 'chem', print_bool = False)


    ### Graph ALMM

    print("\n\nGraph ALMM (Ver 1)")
    print("==========")
    
    best_param_graph_almm_ver1(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    
    print("\n\nGraph ALMM (Ver 2)")
    print("==========")
    
    best_param_graph_almm_ver2(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    print("\n\nGraph ALMM (Ver 3)")
    print("==========")
    
    best_param_graph_almm_ver3(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
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


    ### Graph ALMM

    print("\n\nGraph ALMM (Ver 1)")
    print("==========")
    
    best_param_graph_almm_ver1(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    print("\n\nGraph ALMM (Ver 2)")
    print("==========")
    
    best_param_graph_almm_ver2(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    print("\n\nGraph ALMM (Ver 3)")
    print("==========")
    
    best_param_graph_almm_ver3(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
                        maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)



    print("\n\nFinished Testing")
    print("==========================================\n")

    ### Run this later

    # ### RMSE + SAD

    # print("==========================================")
    # print("Starting Testing")
    # print("Minimizing RMSE + SAD")

    # # ==========================================
    # # CuSO4
    # # ==========================================

    # print("\n\n==========================================")
    # print("Testing on CuSO4")
    # print("==========================================")

    # # Load data
    # X_chem_CuSO4, S_gt_chem_CuSO4, A_gt_chem_CuSO4 = load_data(name = 'synth_CuSO4_data.npy', typename = 'chem', print_bool = False)
    # #print(f"X: {X_chem_CuSO4.shape} \n S: {S_gt_chem_CuSO4.shape} \n A: {A_gt_chem_CuSO4.shape}")


    # ### Graph ALMM

    # print("\n\nGraph ALMM (Ver 1)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver1(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 2)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver2(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 3)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver3(X = X_chem_CuSO4, A_gt = A_gt_chem_CuSO4, S_gt = S_gt_chem_CuSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # # ==========================================
    # # FeCl3
    # # ==========================================

    # print("\n\n==========================================")
    # print("Testing on FeCl3")
    # print("==========================================")

    # # Load data
    # X_chem_FeCl3, S_gt_chem_FeCl3, A_gt_chem_FeCl3 = load_data(name = 'synth_FeCl3_data.npy', typename = 'chem', print_bool = False)
    # print(f"X: {X_chem_FeCl3.shape} \n S: {S_gt_chem_FeCl3.shape} \n A: {A_gt_chem_FeCl3.shape}")


    # ### Graph ALMM

    # print("\n\nGraph ALMM (Ver 1)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver1(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)
    
    # print("\n\nGraph ALMM (Ver 2)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver2(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 3)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver3(X = X_chem_FeCl3, A_gt = A_gt_chem_FeCl3, S_gt = S_gt_chem_FeCl3, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # # ==========================================
    # # FeSO4
    # # ==========================================

    # print("\n\n==========================================")
    # print("Testing on FeSO4")
    # print("==========================================")

    # # Load data
    # X_chem_FeSO4, S_gt_chem_FeSO4, A_gt_chem_FeSO4 = load_data(name = 'synth_FeSO4_data.npy', typename = 'chem', print_bool = False)
    # #print(f"X: {X_chem_FeSO4.shape} \n S: {S_gt_chem_FeSO4.shape} \n A: {A_gt_chem_FeSO4.shape}")


    # ### Graph ALMM

    # print("\n\nGraph ALMM (Ver 1)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver1(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 2)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver2(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 3)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver3(X = X_chem_FeSO4, A_gt = A_gt_chem_FeSO4, S_gt = S_gt_chem_FeSO4, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # # ==========================================
    # # CuSO4 (Exponentially distributed)
    # # ==========================================

    # print("\n\n==========================================")
    # print("Testing on CuSO4 (Exponentially distributed)")
    # print("==========================================")

    # # Load data
    # X_chem_CuSO4_exp, S_gt_chem_CuSO4_exp, A_gt_chem_CuSO4_exp = load_data(name = 'synth_chem_data.npy', typename = 'chem', print_bool = False)
    # print(f"X: {X_chem_CuSO4_exp.shape} \n S: {S_gt_chem_CuSO4_exp.shape} \n A: {A_gt_chem_CuSO4_exp.shape}")


    # ### Graph ALMM

    # print("\n\nGraph ALMM (Ver 1)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver1(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 2)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver2(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nGraph ALMM (Ver 3)")
    # print("==========")
    
    # best_param_graph_almm_RMSE_SAD_ver3(X = X_chem_CuSO4_exp, A_gt = A_gt_chem_CuSO4_exp, S_gt = S_gt_chem_CuSO4_exp, N = samples,
    #                     maxIter = iters, alpha_0 = alpha_0, beta_0 = beta_0, gamma_0 = gamma_0, eta_0 = eta_0, m_0 = 2)

    # print("\n\nFinished Testing")
    # print("==========================================")