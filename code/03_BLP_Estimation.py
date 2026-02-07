# %%
import pandas as pd
import numpy as np
import pyblp
import os

# Load data
DATA_PATH = "C:/Users/liyuz/Dropbox/Replication/DemandEstimationData/"
df_path = os.path.join(DATA_PATH, "productsample.csv")
df = pd.read_csv(df_path, header=0) 

df = df.rename(columns={
    'mkt_id': 'market_ids',
    'product_id': 'product_ids',
    'cluster_id': 'clustering_ids', 
    'per_price': 'prices',    
    'tot_fat': 'fat',
    'size_main': 'size',
    'share': 'shares',
    'company_id': 'firm_ids'
})

# %%
# scale
cols = ['sodium', 'size', 'calorie', 'sugar', 'fat', 'log_nbflavor']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['sodium']    = df['sodium'] / 100
df['size'] = df['size'] / 100
df['calorie']   = df['calorie'] / 100
df['sugar']        = df['sugar'] / 10
df['fat']      = df['fat'] / 10
df['log_nbflavor'] = df['log_nbflavor'] / 10

# id
df['market_ids'] = df['market_ids'].astype(int).astype(str)
df['product_ids'] = df['product_ids'].astype(int).astype(str)
df['clustering_ids'] = df['clustering_ids'].astype(int).astype(str)

# IV
iv_demo = pd.read_csv(DATA_PATH + 'IV_demo_lasso.csv', header=0)
iv_price = pd.read_csv(DATA_PATH + 'IVphat.csv', header=0) 
iv_out_char = pd.read_csv(DATA_PATH + 'out_iv_chars.csv', header=0)
iv_out_inc = pd.read_csv(DATA_PATH + 'out_iv_income.csv', header=0)
iv_out_kid = pd.read_csv(DATA_PATH + 'out_iv_numkid.csv', header=0)

iv_combined = pd.concat([iv_demo, iv_price, iv_out_char, iv_out_inc, iv_out_kid], axis=1)

for i in range(iv_combined.shape[1]):
    col_name = f'demand_instruments{i}'
    df[col_name] = iv_combined.iloc[:, i].values

# deal with NaN and Inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean = df.dropna()

product_data = df_clean.copy()

# Agent data (demographics)
demo_df = pd.read_csv(DATA_PATH + 'demosample_allyear_weighted_2.csv')

# Deal with fixed effect
market_year = demo_df[['mkt_id', 'year']].drop_duplicates()
market_year.rename(columns={'mkt_id': 'market_ids'}, inplace=True)
market_year['market_ids'] = market_year['market_ids'].astype(int).astype(str)

product_data = pd.merge(product_data, market_year, on='market_ids', how='left')

product_data['fe_brand_retailer_year'] = product_data.groupby(['clustering_ids', 'year']).ngroup()
product_data['fe_product_year'] = product_data.groupby(['product_ids', 'year']).ngroup()


# market list
market_list = product_data['market_ids'].unique()
num_markets = len(market_list)
num_hh = 1000 

# demographics
raw_income = demo_df.iloc[:, 3].values # logincome
raw_kids   = demo_df.iloc[:, 1].values # numkid
raw_edu    = demo_df.iloc[:, 7].values # edu_female
raw_weights = demo_df.iloc[:, 11].values # weight2

# agent_data
agent_data_list = []

start_idx = 0
for m_id in market_list:
    end_idx = start_idx + num_hh
    
    m_data = pd.DataFrame({
        'market_ids': [m_id] * num_hh,
        'weights': raw_weights[start_idx:end_idx],
        'income': raw_income[start_idx:end_idx],
        'kids': raw_kids[start_idx:end_idx],
        'edu': raw_edu[start_idx:end_idx],
        'nodes0': 0 
    })
    
    # normalize weights
    m_data['weights'] = m_data['weights'] / m_data['weights'].sum()
    
    agent_data_list.append(m_data)
    start_idx = end_idx

agent_data = pd.concat(agent_data_list)

# Formulation
# X1: Mean Utility
X1_formulation = pyblp.Formulation(
    '0 + prices + sugar + sodium + fat + size + log_nbflavor + calorie + organic + log_nweekupc_sale',
    absorb='C(fe_product_year) + C(fe_brand_retailer_year)'
)

# X2: Random Coefficients
# Const, Price, Sugar, Sodium, Fat, Size, Flavor
X2_formulation = pyblp.Formulation(
    '0 + 1 + prices + log_nbflavor + size + sodium + sugar + fat'
)
agent_formulation = pyblp.Formulation('0 + income + kids + edu')

# Problem
problem = pyblp.Problem(
    product_formulations=(X1_formulation, X2_formulation),
    product_data=product_data,
    agent_data=agent_data,
    agent_formulation=agent_formulation
)

# Constraints and initial values
# (Demographics x Random Coefficients) matrix
# columns: Income, Kids, Edu
# row: Const, Price, Flavor, Size, Sodium, Sugar, Fat
# Row 1 (Income): Interact with [Price, Sugar, Sodium, Fat, Size, Flavor] 
# Row 2 (Kids):   Interact with [Sodium, Size, Flavor]
# Row 3 (Edu):    Interact with [Const, Price, Fat]. 
initial_pi = np.zeros((3, 7))
initial_pi = np.array([
    [0,   0.1, -2,   -1,  0.5,   -0.1, -0.2], # Income interactions
    [0,    0,   3,   2,   -1,    0,    0], # Kids interactions (Sodium, Size, Flavor)
    [-0.5,  0.5, 0,   0,    0,  0,    0.7]  # Edu interactions
])
initial_pi = initial_pi.T


# Matlab mIndex:
# [0,1,1,1,1,1,1;
#  0,0,0,1,0,1,1;
#  1,1,0,0,1,0,0]

pi_mask = np.array([
    [False, True, True, True, True, True, True],
    [False, False, True, True, True, False, False],
    [True,  True, False, False, False, False, True]
], dtype=bool)
pi_mask = pi_mask.T

pi_lower = np.zeros_like(initial_pi)
pi_upper = np.zeros_like(initial_pi)
pi_lower[pi_mask] = -np.inf
pi_upper[pi_mask] = +np.inf

# Solve
initial_sigma = np.zeros((7, 7)) # no solo random coefficient
results = problem.solve(
    sigma=initial_sigma,
    sigma_bounds=(np.zeros((7, 7)), np.zeros((7, 7))),
    pi=initial_pi,
    pi_bounds=(pi_lower, pi_upper),
    method='2s', # Two-step GMM
    optimization=pyblp.Optimization('l-bfgs-b', {'gtol': 1e-4}),
    se_type='clustered'
)


# Post Estimation
# %% Price Elasticities
elasticities = results.compute_elasticities()


mean_own_elasticity = results.extract_diagonals(elasticities).mean()
print(f"Average own elasticity: {mean_own_elasticity}")

# %% Recover marginal cost 
costs = results.compute_costs()

markups = results.compute_markups(costs=costs)
margins = markups / product_data['prices']

print(f"Margins rate: {np.mean(margins):.2%}")


# %% Merger Simulation
# Ownership changed
product_data_merger = product_data.copy()
product_data_merger.loc[product_data_merger['firm_ids'] == '2', 'firm_ids'] = '1'

# solve new equilibrium
changed_prices = results.compute_approximate_prices(
    firm_ids=product_data_merger['firm_ids'],
    costs=costs
)

# Compare
price_change = (changed_prices - product_data['prices']) / product_data['prices']
print(f"Average price increase after merger: {np.mean(price_change):.2%}")

# %% Consumer Surplus
cs = results.compute_consumer_surpluses()

print(f"CS: {cs.sum()}")

