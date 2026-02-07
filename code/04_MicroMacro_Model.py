# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time
import os

DATA_PATH = "C:/Users/liyuz/Dropbox/Replication/DemandEstimationData/"


import torch
from numba import jit

class HomogeneousModel:
    def __init__(self, agg_data, hh_data, t2_data, initial_gamma=None):
        # raw data
        self.agg_data = agg_data
        self.hh_data = hh_data
        self.t2_data = t2_data  
        
        # initialize parameter
        self.J = 3 
        self.gamma = np.array([1.25, 2.0]) if initial_gamma is None else initial_gamma
        self.burn_in_length = 24

        # sort and reshape
        self._prepare_data()

        # after _prepare_data 
        self.delta = np.zeros((self.T, self.M, self.J))


    def _prepare_data(self):
            """
            cleaning data, converting data to 3D arrays for vectorization (equivalent to parallel computing)
            """            
            df = self.agg_data.copy()

            # DMA-border-week
            df['market_key'] = df['dma_code'].astype(str) + "_" + df['_border'].astype(str)
            df = df.sort_values(by=['week', 'market_key'])
            
            self.T = df['week'].nunique()
            self.M = df['market_key'].nunique()
            # check if balanced panel
            if len(df) != self.T * self.M:
                raise ValueError(f"# row ({len(df)}) not equal to T*M ({self.T*self.M})")

            #  (s_obs) -> (T, M, J)
            s_values = df[['share_cigs_norm', 'share_cess_norm', 'share_ecig_norm']].values
            self.s_obs_3d = s_values.reshape(self.T, self.M, self.J)
              
            #  (ecigin) -> (T, M, 1)
            ecig_values = df['ecigin'].values
            self.ecigin_3d = ecig_values.reshape(self.T, self.M, 1)
            
            # initial shares
            df_t2 = self.t2_data.copy()
            df_t2['market_key'] = df_t2['dma_code'].astype(str) + "_" + df_t2['_border'].astype(str)
            df_t2 = df_t2.sort_values(by=['market_key'])
            self.s0_matrix = df_t2[['share_cigs_norm', 'share_cess_norm', 'share_ecig_norm']].values
        

            # connect agg_data and hh_data
            unique_market_keys = sorted(df['market_key'].unique())
            market_map = {key: i for i, key in enumerate(unique_market_keys)}

            self.hh_data['market_key'] = self.hh_data['dma_code'].astype(str) + "_" + self.hh_data['_border'].astype(str)
            self.hh_data['market_idx'] = self.hh_data['market_key'].map(market_map)

            min_week = df['week'].min()
            self.hh_data['week_idx_est'] = self.hh_data['week'] - min_week - self.burn_in_length

            # .copy()
            self.hh_data = self.hh_data[self.hh_data['week_idx_est'] >= 0].copy()
            self.hh_data.reset_index(drop=True, inplace=True)

            self.hh_data['market_idx'] = self.hh_data['market_idx'].astype(int)
            self.hh_data['week_idx_est'] = self.hh_data['week_idx_est'].astype(int)
            


    def _predict_share_t(self, delta_t, last_shares, gamma, ecigin_t):
        """
        delta_t: (M, J) - mean utility this period
        last_shares: (M, J) - market share in the last period
        gamma: (2,) - [gamma_cig, gamma_ecig]
        ecigin_t: (M, 1) - dummy that ecig is in
        """
        M = delta_t.shape[0]
        
        # Case 1: ecigin == 1
        # utility
        v_given_cig = delta_t + np.array([gamma[0], 0, 0]) # cig
        v_given_none = delta_t # none or cess, no state dependence
        v_given_ecig = delta_t + np.array([0, 0, gamma[1]]) # ecig
        
        # choice probability
        def get_probs(v):
            exp_v = np.exp(v)
            prob = exp_v / (1 + np.sum(exp_v, axis=1, keepdims=True))
            return prob

        p_cig = get_probs(v_given_cig)
        p_none = get_probs(v_given_none)
        p_ecig = get_probs(v_given_ecig)
        
        # Total probability
        share_pred_in = (p_cig * last_shares[:, [0]] + 
                        p_none * (1 - last_shares[:, [0]] - last_shares[:, [2]]) + 
                        p_ecig * last_shares[:, [2]])

        # Case 2: ecigin == 0
        v_no_e_cig = delta_t[:, :2] + np.array([gamma[0], 0])
        v_no_e_none = delta_t[:, :2]
        
        def get_probs_no_e(v):
            exp_v = np.exp(v)
            return exp_v / (1 + np.sum(exp_v, axis=1, keepdims=True))

        p_cig_no_e = get_probs_no_e(v_no_e_cig)
        p_none_no_e = get_probs_no_e(v_no_e_none)
        
        share_pred_out_2j = (p_cig_no_e * last_shares[:, [0]] + 
                            p_none_no_e * (1 - last_shares[:, [0]]))
        
        # keep dimension constant (M, 3)
        share_pred_out = np.zeros((M, 3))
        share_pred_out[:, :2] = share_pred_out_2j

        final_share = np.where(ecigin_t == 1, share_pred_in, share_pred_out)
        
        return final_share

    def solve_delta(self, gamma):
        print(f"  > Solving Delta (Vectorized) for gamma={gamma}...")
        
        T, M, J = self.s_obs_3d.shape 
        delta = np.zeros((T, M, J))
        last_share = self.s0_matrix 
        
        # pre-comput log s_obs 
        # log(0) exists, ignore it by suppressing warnings
        with np.errstate(divide='ignore'):
            log_s_obs = np.log(self.s_obs_3d)

        for t in range(T):
            ecigin_t = self.ecigin_3d[t, :, :] # (M, 1)
            log_s_obs_t = log_s_obs[t, :, :]     # (M, J)
            # initial guess: solution of last period
            if t > 0:
                delta_new = delta[t-1, :, :].copy()
            else:
                delta_new = np.zeros((M, J))  

            # Contraction Mapping
            error = 1.0
            while error > 1e-9:
                delta_old = delta_new.copy()
                
                # calculate predicted shares
                s_pred = self._predict_share_t(delta_old, last_share, gamma, ecigin_t)
                                
                # cig and cession
                delta_new[:, :2] = delta_old[:, :2] + log_s_obs_t[:, :2] - np.log(s_pred[:, :2] + 1e-100)
                
                # ecig
                has_ecig = (ecigin_t[:, 0] == 1) 
                
                if np.any(has_ecig):
                    delta_new[has_ecig, 2] = delta_old[has_ecig, 2] + log_s_obs_t[has_ecig, 2] - np.log(s_pred[has_ecig, 2] + 1e-100)
                                
                error = np.max(np.abs(delta_new - delta_old))
            
            # use the last delta_new to compute s_pred, for the next period's last_share
            final_s_pred = self._predict_share_t(delta_new, last_share, gamma, ecigin_t)
            delta[t, :, :] = delta_new
            last_share = final_s_pred
            
        return delta

    def negative_log_likelihood(self, gamma):
            """
            likelihood
            """
            # get utility
            delta_est = self.delta[self.burn_in_length:, :, :]
            h_week = self.hh_data['week_idx_est'].values 
            h_mkt  = self.hh_data['market_idx'].values

            delta_i = delta_est[h_week, h_mkt, :]

            # U = delta + gamma * lagged_choice
            lag_dummies = self.hh_data[['cigflaglag', 'cessflaglag', 'ecigflaglag']].values
            gamma_vec = np.array([gamma[0], 0, gamma[1]])
            U = delta_i + lag_dummies * gamma_vec
            
            ecig_avail = self.hh_data['ecigin'].values[:, None] # (N, 1)
              
            # choice probability
            exp_U = np.exp(U)

            exp_U[:, 2] *= ecig_avail[:, 0]
            sum_exp_U = 1 + np.sum(exp_U, axis=1) # +1 for Outside Option
            
            choice_mask = self.hh_data[['cigflag', 'cessflag', 'ecigflag']].values
            u_chosen = np.sum(U * choice_mask, axis=1)

            log_prob = u_chosen - np.log(sum_exp_U)
            
            return -np.sum(log_prob)

    def fit(self):
        """
        main estimation part
        """
        print("Starting Estimation...")
        dev = 1.0
        iter_count = 0
        
        while dev > 1e-6 and iter_count < 500:
            gamma_old = self.gamma.copy()
            
            # step 1: solve Delta (contraction mapping)
            # update self.delta
            self.delta = self.solve_delta(self.gamma)
            
            # step 2: solve Gamma (maximize likelihood)
            print(f"  > Optimizing LL (Iter {iter_count})...")
            res = minimize(
                self.negative_log_likelihood, 
                self.gamma,
                method='Nelder-Mead', # fminsearch == Nelder-Mead
                options={'disp': False}
            )
            self.gamma = res.x
            
            # step 3: check convergence
            dev = np.max(np.abs(self.gamma - gamma_old))
            print(f"Iter {iter_count}: gamma={self.gamma}, dev={dev:.6f}")
            iter_count += 1
            
        print("Estimation Finished.")
        return self.gamma, self.delta


    def finalize_optimization(self):
            """
            after the estimation, use BFGS to get precise Hessian
            """
            print("\n--- Final Optimization for Hessian ---")
            
            res = minimize(
                self.negative_log_likelihood, 
                self.gamma,
                method='BFGS', 
                options={'disp': True}
            )
            
            self.gamma_final = res.x
            self.hessian_inv = res.hess_inv 
            self.se_gamma = np.sqrt(np.diag(self.hessian_inv))
            
            print(f"Final Gamma: {self.gamma_final}")
            print(f"SE Gamma:    {self.se_gamma}")
            
            # update the final delta
            self.delta_final = self.solve_delta(self.gamma_final)
            
            return self.gamma_final, self.hessian_inv

# prepare data
# agg_data
ic_cols = ['_dma_border', 'dma_code', 'id', 'week', '_border', 'share_cigs_norm', 
           'share_ecig_norm', 'price_pack', 'price_ecig_ct_fill', 'lgrp_ecig_tot', 
           'lgrp_anticig_tot', 'ecigin', 'share_cess_norm', 'price_cess_fill']
df_agg = pd.read_csv(DATA_PATH + 'BorderCountyModelDataIC.csv', header=None, names=ic_cols)


# initial data
t2_cols = ['_dma_border', 'dma_code', 'id', '_border', 'share_cigs_norm', 
           'share_ecig_norm', 'price_pack', 'price_ecig_ct_fill', 'lgrp_ecig_tot', 
           'lgrp_anticig_tot', 'share_cess_norm', 'price_cess_fill']
df_t2 = pd.read_csv(DATA_PATH + 'BorderCountyICt2Data.csv', header=0, names=t2_cols)


# household data
hh_cols = ['hhN', 'dma_code', '_border', 'week', 'cigflag', 'cigflaglag', 
           'ecigflag', 'ecigflaglag', 'nicoflaglag', 'price_pack', 
           'price_ecig_ct_fill', 'lgrp_ecig_tot', 'lgrp_anticig_tot', 
           'ecigin', 'agg_ob_num', 'monthN', 'cessflag', 'cessflaglag', 'price_cess_fill']
df_hh = pd.read_csv(DATA_PATH + 'HHModelData.csv', header=None, names=hh_cols)


df_agg['ecigin'] = df_agg['ecigin'].astype(int)
df_hh['ecigin']  = df_hh['ecigin'].astype(int)

# %%
# instantiation
model = HomogeneousModel(agg_data=df_agg, hh_data=df_hh, t2_data=df_t2)

est_gamma, est_delta = model.fit()

final_gamma, hessian_inv = model.finalize_optimization()


# export gamma
delta_est_3d = model.delta_final[model.burn_in_length:, :, :]
delta_est_flat = delta_est_3d.reshape(-1, 3)

df_export = df_agg.copy()

# drop burn-in period, sorting should be consistent
df_export['market_key'] = df_export['dma_code'].astype(str) + "_" + df_export['_border'].astype(str)
df_export = df_export.sort_values(by=['week', 'market_key'])

est_start_week = df_export['week'].min() + model.burn_in_length
df_export = df_export[df_export['week'] >= est_start_week].copy()

df_export['delta_cig']  = delta_est_flat[:, 0]
df_export['delta_cess'] = delta_est_flat[:, 1]
df_export['delta_ecig'] = delta_est_flat[:, 2]

cols_to_export = ['dma_code', 'week', '_border', 'delta_cig', 'delta_cess', 'delta_ecig']

df_export[cols_to_export].to_csv('./delta_homo.csv', index=False)


# %%
# this part is to used 'numba'. although it is put at the beginning, it was written almost in the end.
@jit(nopython=True, cache=True)
def solve_delta_numba_core(T, M, J, R_total, s_obs_3d, ecigin_3d, eta_stack, 
                        sigma_cig, be_prime, gamma_cig, gamma_ecig):
    """
    Numba-optimized core logic for solving delta (contraction mapping).
    This runs outside the Python interpreter loop for maximum speed.
    """
    
    delta_sol = np.zeros((T, M, J))
    
    # pre-calculate heterogeneity (mu)
    # mu_cig = sigma * eta (1, 2R)
    mu_cig = sigma_cig * eta_stack
    
    # mu_ecig = 0 for low type, be_prime for high type
    mu_ecig = np.zeros((1, R_total))
    half_R = int(R_total / 2)
    mu_ecig[:, half_R:] = be_prime 
    
    # mu_matrix: (1, J, R_total)
    mu_matrix = np.zeros((1, J, R_total))
    mu_matrix[:, 0, :] = mu_cig
    mu_matrix[:, 2, :] = mu_ecig
    
    # initialize state distribution
    # t=0, nobody has ecig. (M, R_total, 3)
    prob_state_lag = np.zeros((M, R_total, 3))
    prob_state_lag[:, :, 0] = 0.5 # cig
    prob_state_lag[:, :, 1] = 0.5 # cess
    prob_state_lag[:, :, 2] = 0.0 # ecig
    
    # pre-compute log observed shares
    log_s_obs = np.log(s_obs_3d + 1e-100)

    # numba is for loop
    for t in range(T):
        
        ecigin_t = ecigin_3d[t, :, :]    # (M, 1)
        log_s_obs_t = log_s_obs[t, :, :] # (M, J)
        
        # initialize delta_t
        if t > 0:
            delta_t = delta_sol[t-1, :, :].copy()
        else:
            delta_t = np.zeros((M, J))
            
        # contraction mapping
        error = 1.0
        iter_count = 0
        
        # calculate lag boost: gamma * lag, (M, R, J)
        lag_boost = np.zeros((M, R_total, J))
        lag_boost[:, :, 0] = prob_state_lag[:, :, 0] * gamma_cig
        lag_boost[:, :, 2] = prob_state_lag[:, :, 2] * gamma_ecig
        
        # lag_boost (M, R, J) -> (M, J, R) ascontiguousarray for better performance in Numba 
        lag_boost_T = np.ascontiguousarray(np.transpose(lag_boost, (0, 2, 1)))

        while error > 1e-9 and iter_count < 2000:
            delta_old = delta_t.copy()
            
            # --- Integration ---
            # V_ijt = delta_jt + mu_ir + gamma * y_it-1
            
            # v_base: (M, J, 1) + (1, J, R) -> (M, J, R)
            delta_old_reshaped = delta_old.reshape((M, J, 1))
            
            v_total = delta_old_reshaped + mu_matrix + lag_boost_T
            
            # choice probabilities
            exp_v = np.exp(v_total)
            
            # apply ecig availability mask
            for m in range(M):
                if ecigin_t[m, 0] == 0:
                    exp_v[m, 2, :] = 0.0
            
            # inclusive value (denominator)
            # sum over j (axis 1) -> (m, r)
            denom = 1.0 + np.sum(exp_v, axis=1) 
            denom_expanded = denom.reshape((M, 1, R_total))
            
            probs = exp_v / denom_expanded # (M, J, R)
            
            # aggregate over R to get market shares
            s_pred = np.sum(probs, axis=2) / R_total # (M, J)
            
            # update delta
            log_s_pred = np.log(s_pred + 1e-100)
            
            delta_new = delta_old + log_s_obs_t - log_s_pred
            
            # if ecig not available
            for m in range(M):
                if ecigin_t[m, 0] == 0:
                    delta_new[m, 2] = delta_old[m, 2] 

            delta_t = delta_new
            
            # check convergence
            error = np.max(np.abs(delta_t - delta_old))
            iter_count += 1
        
        delta_sol[t, :, :] = delta_t
        
        # update lag states for next period t+1
        # probs is (M, J, R), we need (M, R, J)
        prob_state_lag = np.ascontiguousarray(np.transpose(probs, (0, 2, 1)))
        
    return delta_sol



class HeterogeneousModel(HomogeneousModel):
    def __init__(self, agg_data, hh_data, t2_data, R=200, state_dependence=True, initial_params=None):
        """
        R: number of random draw
        state_dependence: True (Scenario 3), False (Scenario 9)
        """
        # initialization
        super().__init__(agg_data, hh_data, t2_data)
        
        self.R = R
        self.state_dependence = state_dependence
        
        # only one random coefficent, cig intercept  (1, R)
        np.random.seed(0)
        self.eta_base = np.random.randn(1, self.R) 
        
        # stack twice: former R -> Low Type, latter R -> High Type, shape (1, 2*R)
        self.eta_stack = np.hstack([self.eta_base, self.eta_base])
        self.total_R = 2 * self.R
        
        # initial guess
        if initial_params is not None:
            self.params = np.array(initial_params)
        else:
            if self.state_dependence:
                # Scenario 3: [sigma_cig, be_prime, pi, rho, gamma_cig, gamma_ecig]
                self.params = np.array([2.31, 3.91, -4.70, -0.63, 0.55, 2.45])
            else:
                # Scenario 9: [sigma_cig, be_prime, pi, rho]
                self.params = np.array([1.69, 5.31, -4.23, -0.22])

        # pre pick data as numpy array, accelurate
        self.lag_dummies_np = self.hh_data[['cigflaglag', 'cessflaglag', 'ecigflaglag']].values # (N, 3)
        self.choice_mask_np = self.hh_data[['cigflag', 'cessflag', 'ecigflag']].values[:, :, None] # (N, 3, 1)
        self.ecig_avail_np = self.hh_data['ecigin'].values[:, None, None] # (N, 1, 1)

        self.h_week_np = self.hh_data['week_idx_est'].values
        self.h_mkt_np  = self.hh_data['market_idx'].values

        raw_hh_ids = self.hh_data['hhN'].values
        unique_hh, self.hh_idx_map = np.unique(raw_hh_ids, return_inverse=True)
        self.num_households = len(unique_hh)

        self.is_outside_np = (np.sum(self.choice_mask_np[:, :, 0], axis=1) == 0)  

        self._init_gpu_tensors()     


    def _init_gpu_tensors(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.t_lag_dummies = torch.tensor(self.lag_dummies_np, device=self.device, dtype=torch.float32)
        self.t_choice_mask = torch.tensor(self.choice_mask_np, device=self.device, dtype=torch.float32)
        self.t_ecig_avail = torch.tensor(self.ecig_avail_np, device=self.device, dtype=torch.float32)
        self.t_h_week = torch.tensor(self.h_week_np, device=self.device, dtype=torch.long)
        self.t_h_mkt = torch.tensor(self.h_mkt_np, device=self.device, dtype=torch.long)
        self.t_hh_idx_map = torch.tensor(self.hh_idx_map, device=self.device, dtype=torch.long)
        self.t_is_outside = torch.tensor(self.is_outside_np, device=self.device) # bool
        self.t_eta_stack = torch.tensor(self.eta_stack, device=self.device, dtype=torch.float32)
        self.t_eta_base = torch.tensor(self.eta_base, device=self.device, dtype=torch.float32)

    def _parse_params(self, params):
        """
        parse parameter, return specific ones
        """
        # random coef std dev (sigma)
        sigma_cig = params[0]
        
        # e-cig high type intercept shift (be_prime)
        be_prime = params[1]
        
        # class probability parameters
        pi_H = params[2]
        rho = params[3]
        
        # state dependence (gamma)
        if self.state_dependence:
            gamma = params[4:6] # [gamma_cig, gamma_ecig]
        else:
            gamma = np.array([0.0, 0.0])
            
        return sigma_cig, be_prime, pi_H, rho, gamma


    def _get_type_weights(self, pi_H, rho):
        """
        probability that each draw belongs to High Type (weights used in integration)
        """
        # eta_base (1, R)
        exponent = rho * self.t_eta_base + pi_H
        pr_H = torch.exp(exponent) / (1 + torch.exp(exponent)) # (1, R)
        
        # [1-PrH, PrH]
        weights_low = 1.0 - pr_H
        weights_high = pr_H

        weights_stack = torch.cat([weights_low, weights_high], dim=1)
        return weights_stack


    def solve_delta(self, params):
        """
        Solves for mean utility (delta) using Numba acceleration.
        """
        # parse parameters
        sigma_cig, be_prime, pi_H, rho, gamma = self._parse_params(params)
        
        # extract dimensions and shapes
        T, M, J = self.s_obs_3d.shape
        
        # Call the Numba compiled function
        delta_sol = solve_delta_numba_core(
            T, M, J, self.total_R,
            self.s_obs_3d,   # Array (T, M, J)
            self.ecigin_3d,  # Array (T, M, 1)
            self.eta_stack,  # Array (1, 2R)
            sigma_cig,       # Scalar
            be_prime,        # Scalar
            gamma[0],        # Scalar (gamma_cig)
            gamma[1]         # Scalar (gamma_ecig)
        )
        
        return delta_sol


    def negative_log_likelihood(self, params):
        """
        likelihood
        """
        if not hasattr(self, 'call_count'):
            self.call_count = 0         
        self.call_count += 1
                
        if self.call_count % 10 == 0:
            print(f"Ev #{self.call_count} | Params sample: {params[:2]}...") 

        # Macro/CPU
        delta_full_np = self.solve_delta(params)


        # Micro/GPU
        delta_est_np = delta_full_np[self.burn_in_length:, :, :]
        t_delta_est = torch.tensor(delta_est_np, device=self.device, dtype=torch.float32)            

        sigma_cig, be_prime, pi_H, rho, gamma = self._parse_params(params)
        
            
        # connect to household data
        # delta_i: (N, J) -> (N, J, 1)
        N = len(self.t_h_week)

        t_delta_i = t_delta_est[self.t_h_week, self.t_h_mkt, :].unsqueeze(2)

        # heterogeneity (mu)
        # mu_matrix: (1, J, 2R)
        # Heterogeneity (Mu): (1, J, 2R)
        t_mu_cig = sigma_cig * self.t_eta_stack
        
        t_mu_ecig = torch.zeros((1, self.total_R), device=self.device)
        t_mu_ecig[:, self.R:] = be_prime # High type get intercept shift
        
        t_mu_matrix = torch.zeros((1, 3, self.total_R), device=self.device)
        t_mu_matrix[:, 0, :] = t_mu_cig
        t_mu_matrix[:, 2, :] = t_mu_ecig
        
        # Lag Boost: (N, J, 1)
        t_lag_boost = torch.zeros((N, 3, 1), device=self.device)
        t_lag_boost[:, 0, 0] = self.t_lag_dummies[:, 0] * gamma[0]
        t_lag_boost[:, 2, 0] = self.t_lag_dummies[:, 2] * gamma[1]
        
        # Total Utility: (N, J, 2R)
        # Broadcasting: (N,J,1) + (1,J,2R) + (N,J,1)
        t_U = t_delta_i + t_mu_matrix + t_lag_boost
        
        t_exp_U = torch.exp(t_U)
        
        t_exp_U[:, 2, :] *= self.t_ecig_avail[:, 0, :]
    
        t_denom = 1 + torch.sum(t_exp_U, dim=1, keepdim=True)
        t_probs = t_exp_U / t_denom # (N, J, 2R)
            
        # choice probability            
        t_prob_chosen = torch.sum(t_probs * self.t_choice_mask, dim=1)  

        t_prob_outside = 1.0 / t_denom[:, 0, :]

        condition = self.t_is_outside.unsqueeze(1)

        t_prob_chosen = torch.where(condition, t_prob_outside, t_prob_chosen)
        
        # likelihood, each household has different obervations so it's hard to vectorize. we can use pandas groupby
        t_log_prob_chosen = torch.log(t_prob_chosen + 1e-30) # (N, 2R)

        t_hh_log_ll = torch.zeros((self.num_households, self.total_R), device=self.device)
        t_hh_log_ll.index_add_(0, self.t_hh_idx_map, t_log_prob_chosen)

        t_hh_ll_draws = torch.exp(t_hh_log_ll) # (N_hh, 2R)
        

        t_weights_stack = self._get_type_weights(pi_H, rho)
        # (N_hh, 2R) * (1, 2R) -> sum over axis 1 -> (N_hh, )
        t_hh_ll_final = torch.sum(t_hh_ll_draws * t_weights_stack, dim=1)
        
        t_ll_final = torch.log(t_hh_ll_final + 1e-30)
        
        final_loss = -torch.sum(t_ll_final).item()

        if self.call_count % 10 == 0:
            print(f"   -> Loss: {final_loss:.4f}")        
        return -torch.sum(t_ll_final).item() # return to python float

    def fit(self):
        print(f"Starting Heterogeneous Estimation (SD={self.state_dependence})...")
        print(f"Initial Params: {self.params}")
        print("开始估计...")

        res = minimize(
            self.negative_log_likelihood, 
            self.params,
            method='Nelder-Mead',
            options={'disp': True, 'maxiter': 500}
        )
        
        self.params = res.x
        self.delta_final = self.solve_delta(self.params)
        
        print("Estimation Finished.")
        print("Final Params:", self.params)
        return self.params, self.delta_final


# %% state_dependence=False
model_no_sd = HeterogeneousModel(
    agg_data=df_agg, 
    hh_data=df_hh, 
    t2_data=df_t2, 
    R=50,               
    state_dependence=False
)

params_no_sd, delta_no_sd = model_no_sd.fit()

print("--- Scenario 9 Results ---")
print(f"Sigma Cig: {params_no_sd[0]}")
print(f"Be Prime:  {params_no_sd[1]}")
print(f"Pi High:   {params_no_sd[2]}")
print(f"Rho:       {params_no_sd[3]}")


delta_full = model_no_sd.delta_final
burn_in = model_no_sd.burn_in_length
delta_est_3d = delta_full[burn_in:, :, :]
delta_est_flat = delta_est_3d.reshape(-1, 3)

df_export = df_agg.copy()
df_export['market_key'] = df_export['dma_code'].astype(str) + "_" + df_export['_border'].astype(str)
df_export = df_export.sort_values(by=['week', 'market_key'])

all_weeks = sorted(df_export['week'].unique())
est_start_week = all_weeks[burn_in] 
df_export = df_export[df_export['week'] >= est_start_week].copy()

df_export['delta_cig']  = delta_est_flat[:, 0]
df_export['delta_cess'] = delta_est_flat[:, 1]
df_export['delta_ecig'] = delta_est_flat[:, 2]

cols_to_export = ['dma_code', 'week', '_border', 'delta_cig', 'delta_cess', 'delta_ecig']
df_export[cols_to_export].to_csv('./delta_heter_woG.csv', index=False)



# %% state_dependence=True
model_sd = HeterogeneousModel(
    agg_data=df_agg, 
    hh_data=df_hh, 
    t2_data=df_t2, 
    R=50, 
    state_dependence=True
)

params_sd, delta_sd = model_sd.fit()

print("--- Scenario 3 Results ---")
print(f"Sigma Cig:  {params_sd[0]}")
print(f"Be Prime:   {params_sd[1]}")
print(f"Pi High:    {params_sd[2]}")
print(f"Rho:        {params_sd[3]}")
print(f"Gamma Cig:  {params_sd[4]}")
print(f"Gamma Ecig: {params_sd[5]}")


delta_full = model_sd.delta_final
burn_in = model_sd.burn_in_length
delta_est_3d = delta_full[burn_in:, :, :]
delta_est_flat = delta_est_3d.reshape(-1, 3)

df_export = df_agg.copy()
df_export['market_key'] = df_export['dma_code'].astype(str) + "_" + df_export['_border'].astype(str)
df_export = df_export.sort_values(by=['week', 'market_key'])

all_weeks = sorted(df_export['week'].unique())
est_start_week = all_weeks[burn_in] 
df_export = df_export[df_export['week'] >= est_start_week].copy()

df_export['delta_cig']  = delta_est_flat[:, 0]
df_export['delta_cess'] = delta_est_flat[:, 1]
df_export['delta_ecig'] = delta_est_flat[:, 2]

cols_to_export = ['dma_code', 'week', '_border', 'delta_cig', 'delta_cess', 'delta_ecig']
df_export[cols_to_export].to_csv('./delta_heter_withG.csv', index=False)
