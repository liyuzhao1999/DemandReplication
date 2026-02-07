import pandas as pd
import numpy as np
import sys
import os
import time
import datetime
import torch

# Setup
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# data preparation
DATA_PATH = "C:/Users/liyuz/Dropbox/Replication/DemandEstimationData/"
HHModelData_path = os.path.join(DATA_PATH, "HHModelData.csv")
HHModelData = pd.read_csv(HHModelData_path)

column_names = [
    'hhN', 'dma_code', '_border', 'week', 'cigflag', 'cigflaglag', 
    'ecigflag', 'ecigflaglag', 'nicoflaglag', 
    'price_pack', 'price_ecig_ct_fill', 'lgrp_ecig_tot', 'lgrp_anticig_tot', 
    'ecigin', 'agg_ob_num', 'monthN', 
    'cessflag', 'cessflaglag', 'price_cess_fill'
]
HHModelData.columns = column_names

# the same as in normal mixed logit
X_price = HHModelData[['price_pack', 'price_ecig_ct_fill', 'price_cess_fill']].values.astype(np.float32)
X_ads = HHModelData[['lgrp_ecig_tot', 'lgrp_anticig_tot']].values.astype(np.float32)
X_lag = HHModelData[['cigflaglag', 'ecigflaglag']].values.astype(np.float32)

conditions = [HHModelData['cigflag'] == 1, HHModelData['ecigflag'] == 1, HHModelData['cessflag'] == 1]
choices = [0, 1, 2]
y = np.select(conditions, choices, default=3)

person_id = HHModelData.groupby(['hhN']).ngroup().values # 确保是 numpy array
nP = person_id.max() + 1
seed = 123
R = 200  
np.random.seed(seed)
draws_people = np.random.randn(nP, R, 2).astype(np.float32)


# convert to tensor data type
def to_tensor(arr, dtype=torch.float32, dev=device):
    return torch.tensor(arr, dtype=dtype, device=dev)

t_X_price = to_tensor(X_price)
t_X_ads = to_tensor(X_ads)
t_X_lag = to_tensor(X_lag)
t_draws_people = to_tensor(draws_people)
t_y = to_tensor(y, dtype=torch.long)
t_person_id = to_tensor(person_id, dtype=torch.long)


# likelihood function
def mixed_logit_likelihood(theta, X_price, X_ads, X_lag, y, person_id, draws, return_vector=False):
    """
    GPU+CPU
    """
    dev = X_price.device
    N = X_price.shape[0]
    nP = person_id.max().item() + 1 if N > 0 else 0
    R = draws.shape[1]

    # parameter
    beta_price = theta[0]
    sigma_cig  = theta[10]
    sigma_ecig = theta[11]

    # mean utility
    V_mean = torch.zeros((N, 4), dtype=torch.float32, device=dev)
    V_mean[:, 0] = (beta_price * X_price[:, 0] + theta[1] * X_ads[:, 0] + theta[5] * X_lag[:, 0] + theta[7]) 
    V_mean[:, 1] = (beta_price * X_price[:, 1] + theta[2] * X_ads[:, 1] + theta[6] * X_lag[:, 1] + theta[8]) 
    V_mean[:, 2] = (beta_price * X_price[:, 2] + theta[3] * X_ads[:, 0] + theta[4] * X_ads[:, 1] + theta[9])
    
    # random utility
    nu = draws[person_id, :, :] 
    
    V_rand = torch.zeros((N, 4, R), dtype=torch.float32, device=dev)
    V_rand[:, 0, :] = sigma_cig * nu[:, :, 0]
    V_rand[:, 1, :] = sigma_ecig * nu[:, :, 1]
    
    # choice probability
    V_total = V_mean[:, :, None] + V_rand # (N, 4, R)
    log_prob_all = V_total - torch.logsumexp(V_total, dim=1, keepdim=True)
    y_expanded = y.view(N, 1, 1).expand(N, 1, R)
    log_prob_chosen = torch.gather(log_prob_all, 1, y_expanded).squeeze(1) # (N, R)
    
    # sum over time t for each person i
    log_L_ir = torch.zeros((nP, R), dtype=torch.float32, device=dev)
    log_L_ir.index_add_(0, person_id, log_prob_chosen)
    
    # L_i = mean(exp(log_L_ir))
    L_ir = torch.exp(log_L_ir)
    L_i = torch.mean(L_ir, dim=1) # (nP,)
    
    nll = -torch.sum(torch.log(L_i + 1e-30))

    if return_vector:
        return torch.log(L_i + 1e-30)
    else:
        return nll

# optimization using gpu
theta_init = np.zeros(12)
theta_init[10] = 0.1 # sigma_cig initial
theta_init[11] = 0.1 # sigma_ecig initial
theta_init_tensor = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)

optimizer = torch.optim.LBFGS([theta_init_tensor], 
                              lr=1, 
                              max_iter=2000, 
                              tolerance_grad=1e-5, 
                              line_search_fn="strong_wolfe")

iteration_count = [0]

def closure():
    optimizer.zero_grad()
    loss = mixed_logit_likelihood(theta_init_tensor, t_X_price, t_X_ads, t_X_lag, t_y, t_person_id, t_draws_people)
    loss.backward()
    
    iteration_count[0] += 1
    grad_norm = torch.nn.utils.clip_grad_norm_(theta_init_tensor, float('inf'))
    
    print(f"Eval {iteration_count[0]:3d} | Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.6f}")
    return loss

if torch.cuda.is_available(): torch.cuda.synchronize()
start_opt = time.time()

optimizer.step(closure)

if torch.cuda.is_available(): torch.cuda.synchronize()
end_opt = time.time()
time_opt = end_opt - start_opt

final_theta = theta_init_tensor.detach().cpu().numpy()
print(f"\nOptimization Finished in {format_time(time_opt)}")
print(f"Optimized Theta: {final_theta}")
