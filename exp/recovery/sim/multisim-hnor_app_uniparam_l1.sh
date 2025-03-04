#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_param_recov_multisim-hnor"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/recovery
mkdir -p ${base_path}/logs/recovery/simdata_uniparam/
mkdir -p ${base_path}/logs/recovery/simdata_uniparam/${log_folder}

echo "activating conda environment"
source activate rlnet_cluster

echo "build multisim-nor"
python -u -b ${base_path}/src/fitting_behavior/recovery/sim_p_range.py -c ${base_path}/src/fitting_behavior/recovery/param_recov_range_configs/multisim-hnor_app_uniparam_seed-12345_l1.json | tee ${base_path}/logs/recovery/simdata_uniparam/${log_folder}/log_1.txt
