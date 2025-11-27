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
├── src/                    # code files for data preprocessing, analysis, modeling and visualization
│   ├── models/                 # novelty models
│   └── fitting_neural/         # functions for neural fitting and analysis
│   └── fitting_behavior/       # functions for behavior fitting and analysis
│   └── scripts/                # scripts for plotting
├── exp/                    # bash files used to run model simulations
├── data/                   # simulation data and analysis results
├── output/                 # visualizations and figures
└── README.md
```

