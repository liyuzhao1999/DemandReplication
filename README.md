# Demand Estimation Replication Package 

## Overview
This repository contains replication code for several structural demand models. The goal is to demonstrate the implementation of these estimators in empirical research. I used some state-of-the-art packages, like both standard CPU-based optimization (`scipy`) and modern GPU-accelerated frameworks (`PyTorch`), as well as specialized libraries like `pyblp`. I also explored an object-oriented programming (OOP) design.

The package covers three main approaches:
1.  Mixed Logit (Random Coefficient Logit): Implemented via Simulated Maximum Likelihood (SML).
2.  BLP: Implemented using the `pyblp` package, including estimation and counterfactual simulation.
3.  Micro-Macro Integrated Model: A custom implementation combining aggregate market shares with individual panel data using contraction mapping.

I mainly replicate these two papers, and their original replication packages can be downloaded from journals' websites:
1. Tuchman, Anna E. "Advertising and demand for addictive goods: The effects of e-cigarette advertising." Marketing science (2019).  [10.1287/mksc.2019.1195](http://pubsonline.informs.org/doi/10.1287/mksc.2019.1195)
2. Zhu, Xinrong. "Inference and Impact of Category Captaincy." Management Science (2025). [10.1287/mnsc.2023.02039](https://doi.org/10.1287/mnsc.2023.02039)


## Repository Structure

The code is organized by model and implementation method:

| File / Folder | Description | Key Libraries |
| :--- | :--- | :--- |
| `01_MixedLogit_noPytorch.py` | Baseline Mixed Logit estimation using analytical gradients. Good for understanding the core likelihood logic. | `scipy`, `numexpr` |
| `02_MixedLogit_Pytorch.py` | GPU-accelerated Mixed Logit. PyTorch tensors offer significant speedups. | `torch` |
| `03_BLP_Estimation.py` | Use package to implement BLP. | `pyblp` |
| `04_MicroMacro_Model.py` | A complex structural model linking micro (household) and macro (market share) data. Solves for mean utility ($\delta$) via contraction mapping. OOP was used. | `numba`, `torch` |
| `src/mixed_logit.py` | Helper functions for the likelihood function. | `numpy` |
| `requirements.txt` | Required Packages | — | 
| `data/` | Input datasets used for estimation and analysis (data link). | — |
| `DemandEstimation.pdf` | Presentation slides summarizing the model, identification, and implementation details. | — |


## Installation
1. Install general dependencies:
```bash
   pip install -r requirements.txt
```
2. Install PyTorch with GPU support matches your CUDA version (e.g., CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```


## Implementation Details

Details can be found in the slides. 

## Computational Performance
Note: Due to the authors' adjustment on the original data, slight discrepancies between the original estimates and this package's output may exist.

Speed comparison between CPU-based `scipy` optimization and GPU-accelerated `PyTorch` (on RTX 4060 Ti):

| Model | Implementation | Time to Convergence | Speedup |
| :--- | :--- | :--- | :--- |
| Mixed Logit | Scipy (CPU) | 1h 22m | 1x |
| Mixed Logit | PyTorch (GPU) | 16m | **~5x** |
| BLP | PyBLP | ~1h 30m | 1x |
| BLP | Matlab (author's code) | ~1h 30m | 1x |
| Micro-Macro | Numba + PyTorch | 40m | N/A |
