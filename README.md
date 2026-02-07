# Demand Estimation Replication Package 

## Overview
This repository contains replication code for several structural demand models. The goal is to demonstrate the implementation of these estimators using both standard CPU-based optimization (`scipy`) and modern GPU-accelerated frameworks (`PyTorch`), as well as specialized libraries like `pyblp`.

The package covers three main approaches:
1.  Mixed Logit (Random Coefficient Logit): Implemented via Simulated Maximum Likelihood (SML).
2.  BLP: Implemented using the `pyblp` package, including estimation and counterfactual simulation.
3.  Micro-Macro Integrated Model: A custom implementation combining aggregate market shares with individual panel data using contraction mapping.

I mainly replicate two papers:
1. Tuchman, Anna E. "Advertising and demand for addictive goods: The effects of e-cigarette advertising." Marketing science (2019).  [10.1287/mksc.2019.1195](http://pubsonline.informs.org/doi/10.1287/mksc.2019.1195)
2. Zhu, Xinrong. "Inference and Impact of Category Captaincy." Management Science (2025). [10.1287/mnsc.2023.02039](https://doi.org/10.1287/mnsc.2023.02039)


## Repository Structure

The code is organized by model and implementation method:

| File Name | Description | Key Libraries |
| :--- | :--- | :--- |
| `01_MixedLogit_noPytorch.py` | Baseline Mixed Logit estimation using analytical gradients. Good for understanding the core likelihood logic. | `scipy`, `numexpr` |
| `02_MixedLogit_Pytorch.py` | GPU-accelerated Mixed Logit. Uses PyTorch tensors to parallelize simulation over draws ($R$) and individuals ($N$), offering significant speedups. | `torch` |
| `03_BLP_Estimation.py` | Use package to implement BLP. | `pyblp` |
| `04_MicroMacro_Model.py` | A complex structural model linking Micro (household) and Macro (market share) data. Solves for mean utility ($\delta$) via fixed-point iteration. | `numba`, `torch` |
| `src/mixed_logit.py` | Helper functions for likelihood calculations. | `numpy` |