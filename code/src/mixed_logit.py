import numpy as np
from scipy import sparse
import numexpr as ne

def likelihood_mixed_logit(theta, X_price, X_ads, X_lag, y, draws_people, person_id):
    """
    Calculate the log-likelihood for the mixed logit model.   
    """
    N = X_price.shape[0] 
    nP = person_id.max() + 1
    R = draws_people.shape[1]
    
    # parameters
    beta_price = theta[0]
    cig_ecig_ad   = theta[1]
    ecig_ecig_ad   = theta[2]
    cess_ecig_ad   = theta[3]
    cess_cess_ad   = theta[4]
    cig_lag = theta[5]
    ecig_lag = theta[6]
    intercept_cig = theta[7]
    intercept_ecig = theta[8]
    intercept_cess = theta[9]

    sigma_cig  = theta[10]
    sigma_ecig = theta[11]

    # Mean utility calculation
    V_mean = np.zeros((N, 4), dtype=np.float32) # (cig, ecig, cess, outside)
    
    # U_cig = beta * price_cig + ads_ecig + lag + intercept_cig
    V_mean[:, 0] = beta_price * X_price[:, 0] + cig_ecig_ad * X_ads[:, 0] + cig_lag * X_lag[:, 0] + intercept_cig 
    
    # U_ecig = beta * price_ecig + ads_ecig + lag + intercept_ecig
    V_mean[:, 1] = beta_price * X_price[:, 1] + ecig_ecig_ad * X_ads[:, 1] + ecig_lag * X_lag[:, 1] + intercept_ecig 
    
    # U_cess = beta * price_cess + ads_ecig + ads_cess + lag + intercept_cess
    V_mean[:, 2] = beta_price * X_price[:, 2] + cess_ecig_ad * X_ads[:, 0] + cess_cess_ad * X_ads[:, 1] + intercept_cess 
    
    # Random utility
    nu = draws_people[person_id, :, :]
    V_rand = np.zeros((N, 4, R), dtype=np.float32)
    V_rand[:, 0, :] = sigma_cig * nu[:, :, 0]
    V_rand[:, 1, :] = sigma_ecig * nu[:, :, 1]

    # Total
    V_total = V_mean[:, :, None] + V_rand # (N, 4, R)
    V_max = np.max(V_total, axis=1, keepdims=True)

    # choice probabilities
    ev = ne.evaluate("exp(V_total - V_max)")
    den = np.sum(ev, axis=1, keepdims=True) # (N, 1, R) 
    prob = ev / den # (N, 4, R) 
    prob_chosen = prob[np.arange(N), y, :]

    data_p = np.ones(N)
    P = sparse.csr_matrix((data_p, (person_id, np.arange(N))), shape=(nP, N))
    
    # individual likelihood
    # L_ir = Product_over_time (prob_chosen_t)
    # log(L_ir) = Sum_over_time (log(prob_chosen_t))
    log_prob_chosen = np.log(prob_chosen) # (N, R)
    log_L_ir = P @ log_prob_chosen # (nP, R)
     
    # Average over Draws
    L_ir = np.exp(log_L_ir) # (nP, R)
    L_i = np.mean(L_ir, axis=1) # (nP,) 
    
    # likelihood
    nll = -np.sum(np.log(np.maximum(L_i, 1e-200)))


    #########################
    # gradient calculation
    #########################

    # Posterior weight: W_ir (nP, R)
    # W_ir = L_ir / Sum_r(L_ir)
    sum_L_ir = np.sum(L_ir, axis=1, keepdims=True)
    W = L_ir / np.maximum(sum_L_ir, 1e-200)
    
    S = np.zeros((nP, len(theta)), dtype=np.float32)
    
    # pre-calculate Residuals = Indicator - Prob (X_chosen-X*P)=X(Ind-P)
    residuals = []
    for j in range(3):
        ind = (y[:, None] == j).astype(np.float32)
        res = ind - prob[:, j, :] 
        residuals.append(res)

    def get_score_fast(X_target, opt_idx):
        """
        X_target: (N,)
        opt_idx: 0, 1, or 2
        """
        # (N, R) * (N, 1) -> (N, R)
        diff = residuals[opt_idx] * X_target[:, None]        
        # P @ diff -> (nP, R) -> sum -> (nP,)
        return -np.sum((P @ diff) * W, axis=1)


    # theta 0: Price
    # diff_price = sum(residual_j * Price_j)
    diff_price = (residuals[0] * X_price[:, 0][:, None] + 
                residuals[1] * X_price[:, 1][:, None] + 
                residuals[2] * X_price[:, 2][:, None])
    S[:, 0] = -np.sum((P @ diff_price) * W, axis=1)

    # theta 1-9: mean coefficients
    ones_vec = np.ones(N, dtype=np.float32)
    S[:, 1] = get_score_fast(X_ads[:, 0],       0) # cig_ecig_ad
    S[:, 2] = get_score_fast(X_ads[:, 1],       1) # ecig_ecig_ad
    S[:, 3] = get_score_fast(X_ads[:, 0],       2) # cess_ecig_ad
    S[:, 4] = get_score_fast(X_ads[:, 1],       2) # cess_cess_ad
    S[:, 5] = get_score_fast(X_lag[:, 0],       0) # cig_lag
    S[:, 6] = get_score_fast(X_lag[:, 1],       1) # ecig_lag
    S[:, 7] = get_score_fast(ones_vec, 0) # intercept_cig
    S[:, 8] = get_score_fast(ones_vec, 1) # intercept_ecig
    S[:, 9] = get_score_fast(ones_vec, 2) # intercept_cess

    # theta 10-11: sigmas 

    # diff = residuals[0] * nu
    diff_sig0 = residuals[0] * nu[:, :, 0]
    S[:, 10] = -np.sum((P @ diff_sig0) * W, axis=1)

    diff_sig1 = residuals[1] * nu[:, :, 1]
    S[:, 11] = -np.sum((P @ diff_sig1) * W, axis=1)

    # sum
    grad = np.sum(S, axis=0)

    return nll, grad, S





