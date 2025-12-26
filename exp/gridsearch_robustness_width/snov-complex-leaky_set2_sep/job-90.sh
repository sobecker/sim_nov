#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_job-90"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/gridsearch_robustness_width
mkdir -p ${base_path}/logs/gridsearch_robustness_width/snov-complex-leaky_set2_sep
mkdir -p ${base_path}/logs/gridsearch_robustness_width/snov-complex-leaky_set2_sep/${log_folder}

echo "activating conda environment"
source activate rlnet_cluster

echo "build job-90"
python -u -b ${base_path}/src/fitting_neural/grid_search_snov.py -c ${base_path}/src/fitting_neural/configs_robustness/gridsearch_robustness_width/snov-complex-leaky_set2_sep/job-90.json | tee ${base_path}/logs/gridsearch_robustness_width/snov-complex-leaky_set2_sep/${log_folder}/log.txt
