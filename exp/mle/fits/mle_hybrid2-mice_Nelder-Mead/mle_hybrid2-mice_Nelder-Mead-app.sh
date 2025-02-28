#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_mle_hybrid2-mice_Nelder-Mead-app
"base_path="/lcncluster/becker/sim_nov"echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/mle
mkdir -p ${base_path}/logs/mle/fits/
mkdir -p ${base_path}/logs/mle/fits/${log_folder}

echo "activating conda environment"
source activate rlnet_cluster

echo "build mle_hybrid2-mice_Nelder-Mead-app"
python -u -b ${base_path}/src/fitting_behavior/mle/mle_fit.py -c ${base_path}/src/fitting_behavior/mle/mle_fit_configs/mle_hybrid2-mice_Nelder-Mead-app.json | tee ${base_path}/logs/mle/fits/${log_folder}/log.txt
