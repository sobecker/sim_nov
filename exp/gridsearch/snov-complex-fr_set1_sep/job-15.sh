#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_job-15"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/grid_search_results
mkdir -p ${base_path}/logs/grid_search_results/snov-complex-fr_set1_sep
mkdir -p ${base_path}/logs/grid_search_results/snov-complex-fr_set1_sep/${log_folder}

echo "activating conda environment"
source activate rlnet_cluster

echo "build job-15"
python -u -b ${base_path}/src/fitting_neural/grid_search_snov.py -c ${base_path}/src/fitting_neural/configs_gridsearch/snov-complex-fr_set1_sep/job-15.json | tee ${base_path}/logs/grid_search_results/snov-complex-fr_set1_sep/${log_folder}/log.txt
