# Code base: Similarity-based novelty

This repo accompanies the preprint:

S. Becker, A. Modirshanechi, W. Gerstner (2024) *Representational similarity modulates neural and behavioral signatures of novelty.* bioRxiv 2024.05.01.592002; doi: https://doi.org/10.1101/2024.05.01.592002

In this paper, we propose a computational model of novelty that accounts for the effect of stimulus similarities on novelty computation (`similarity-based novelty'). Using the similarity-based novelty framework on two open data sets, we show that
- low-level feature similarity modulates V1 novelty responses in mice,
- spatial similarity modulates mouse exploration in an unfamiliar maze.

The repo contains 
(i) preprocessed experimental data, i.e. V1 novelty responses by [Homann et al. (2022)](https://doi.org/10.1073/pnas.2108882119) and mouse exploration behavior by [Rosenberg et al. (2021)](https://doi.org/10.7554/eLife.66175), and 
(ii) code files to reproduce data preprocessing, model simulations, analysis and figures contained in the preprint.

## Repository structure

```
sim_nov/
├── ext_data/               # preprocessed experimental data
│   ├── Homann2022/             # V1 novelty responses 
│   └── Rosenberg2021/          # mouse exploration behavior 
├── src/                    # code files for data preprocessing, analysis, modeling, visualization
│   ├── models/                 # novelty models
│   └── fitting_neural/         # functions for neural fitting and analysis
│   └── fitting_behavior/       # functions for behavior fitting and analysis
│   └── scripts/                # scripts for plotting
├── exp/                    # bash files used to run model simulations
├── data/                   # simulation data and analysis results
├── output/                 # visualizations and figures
└── README.md
```

## Reproducing article results

- Toy example (Figure 1, Figure S1): Run python notebook sim_nov/src/scripts/gabor_orientations/toy_example.ipynb

- V1 novelty responses (Figure 2):
    1. Run grid search.
          - For count-based novelty: Run sim_nov/src/fitting_neural/grid_search_cnov.py
          - For similarity-based novelty: Run sim_nov/src/fitting_neural/grid_search_snov.py with config files. Config files created in: sim_nov/src/fitting_neural/make_configs_snov.py
    2. Run fitting to neural data.
          - If grid search results for a given model were saved in separate folders (e.g. to simplify parallelization), combine data with sim_nov/src/fitting_neural/combine_data_snov.py as needed.
          - Run sim_nov/src/fitting_neural/run_fit_withconfig.py with config files. Config files created in sim_nov/src/fitting_neural/make_configs_fitting.py
    4. Create visualizations.
          - Run sim_nov/src/scripts/homann_analysis/plot_crossvalidation.ipynb 
  
- Controls for V1 novelty responses (Figure S3-S4):
    - Parameter robustness (Fig. S3 A-E): Accesses same data as Figure 2. Run python notebook sim_nov/src/scripts/homann_analysis/plot_gridsearch_robustness.ipynb
    - Component width variation (Fig. S3 F): Run grid search (sim_nov/src/fitting_neural/grid_search_snov.py) with config files created in sim_nov/src/fitting_neural/make_configs_robustness.py. To create figures, run python notebook sim_nov/src/homann_analysis/plot_width_variation.ipynb.
    - Comparison with fixed-rate models: Run grid search, fitting and visualization as for Figure 2 but for fixed-rate count-based novelty and fixed-rate similarity-based novelty models.
 
- Exploration behavior (Figure 3, Figure S5): 

- Sum-of-parts protocol (Figure 4, Figure S6):
     - To reproduce Figure 4, run python notebook sim_nov/src/scripts/sum_of_parts/exp_pred1.ipynb
     - To reproduce Figure S6 (robustness under change of leakiness), run simulations using sim_nov/src/scripts/sum_of_parts/exp_pred1_robustness_simulate.py; create plots using sim_nov/src/scripts/sum_of_parts/exp_pred1_robustness_plot.py
