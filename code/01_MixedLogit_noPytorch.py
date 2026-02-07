import pandas as pd
import numpy as np
from scipy.optimize import minimize
import numexpr as ne
import time
import datetime
import os

from src.mixed_logit import likelihood_mixed_logit

# individual level --- mixed logit
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

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

# create price variable
X_price = HHModelData[['price_pack', 'price_ecig_ct_fill', 'price_cess_fill']].values.astype(np.float32)
X_ads = HHModelData[['lgrp_ecig_tot', 'lgrp_anticig_tot']].values.astype(np.float32)
X_lag = HHModelData[['cigflaglag', 'ecigflaglag']].values.astype(np.float32)

# choice variable
conditions = [HHModelData['cigflag'] == 1, HHModelData['ecigflag'] == 1, HHModelData['cessflag'] == 1]
choices = [0, 1, 2]
y = np.select(conditions, choices, default=3)

# random draw
person_id = HHModelData.groupby(['hhN']).ngroup()

seed = 123
R = 200  # number of draws
Kmean = 10  # number of mean coefficients
Krand = 2  # number of random coefficients
np.random.seed(seed)
nP = person_id.max() + 1
draws_people = np.random.randn(nP, R, Krand)
draws_people = draws_people.astype(np.float32)

# initial parameters
theta_init = np.zeros(12)
theta_init[10] = 0.1 # sigma_cig
theta_init[11] = 0.1 # sigma_ecig


# optimize the log-likelihood    
def objective(th):
    return likelihood_mixed_logit(th, X_price, X_ads, X_lag, y, draws_people, person_id)

start_opt = time.time()
# L-BFGS-B method
res = minimize(
    objective, 
    theta_init, 
    method='L-BFGS-B',
    jac=True, # return gradient
    bounds=[(None, None)] * 10 + [(1e-6, None)] * 2,
    options={
        'disp': True, 
        'maxiter': 2000,
        'ftol': 1e-9, 
        'gtol': 1e-6 
    }
)
end_opt = time.time()
time_opt = end_opt - start_opt
print(f"\nOptimization Finished in {format_time(time_opt)}")

# output results
print("\nOptimization Result:")
print(res.message)
print(f"Final NLL: {res.fun:.4f}")

theta_hat = res.x
print("Estimated Parameters:")
print(theta_hat)



# standard errors
def grad(th):
        _, g, _ = likelihood_mixed_logit(th, X_price, X_ads, X_lag, y, draws_people, person_id)
        return g

def get_numerical_hessian(theta, grad_func, eps=1e-6):
    k = len(theta)
    hessian = np.zeros((k, k))
    
    for i in range(k):
        # center finite difference
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad_plus = grad_func(theta_plus)

        theta_minus = theta.copy()
        theta_minus[i] -= eps
        grad_minus = grad_func(theta_minus)

        # derivative (g(θ+e) - g(θ-e)) / (2e)
        hessian[:, i] = (grad_plus - grad_minus) / (2 * eps)
        
    return hessian

# inverse Hessian
start_se = time.time()
H_num = get_numerical_hessian(theta_hat, grad, eps=1e-6)
H = 0.5 * (H_num + H_num.T)
V_oim = np.linalg.inv(H) 

se_oim = np.sqrt(np.diag(V_oim))

# robust se
_, _, S = likelihood_mixed_logit(theta_hat, X_price, X_ads, X_lag, y, draws_people, person_id)
OPG = S.T @ S
V_rob = V_oim @ OPG @ V_oim
se_rob = np.sqrt(np.diag(V_rob))

end_se = time.time()
time_se = end_se - start_se

print(f"SE Time:      {format_time(time_se)}")
print(f"Total Time:        {format_time(time_opt + time_se)}")

# print
param_names = [
    "Price",             # theta[0]
    "E-Cig Ads on Cigs",   # theta[1]
    "E-Cig Ads on E-Cigs",  # theta[2]
    "E-Cig Ads on Cess",  # theta[3]
    "Cessation Ads",  # theta[4]
    "Lag: Cig",          # theta[5]
    "Lag: Ecig",         # theta[6]
    "Intercept: Cig",    # theta[7]
    "Intercept: Ecig",   # theta[8]
    "Intercept: Cess",   # theta[9]
    "Sigma: Cig",        # theta[10]
    "Sigma: Ecig"        # theta[11]
]


df_results = pd.DataFrame({
    "Coeff": theta_hat,
    "Std. Err.": se_rob}, index=param_names)


print("Mixed Logit Estimation Results (Robust SE)")
print(df_results)


