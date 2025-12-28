#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_leaky_multinov-eps_hnor_notrace-l3-6_app_3"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/crossvalidation/
mkdir -p ${base_path}/logs/crossvalidation/cv-mouseid/
mkdir -p ${base_path}/logs/crossvalidation/cv-mouseid/${log_folder}

echo "build leaky_multinov-eps_hnor_notrace-l3-6_app_3"
python -u -b ${base_path}/src/fitting_behavior/crossvalidation/LL_crossvalidation.py -c ${base_path}/src/fitting_behavior/crossvalidation/configs_cv-mouseid/leaky_multinov-eps_hnor_notrace-l3-6_app_3.json | tee ${base_path}/logs/crossvalidation/cv-mouseid/${log_folder}/log_leaky_multinov-eps_hnor_notrace-l3-6_app_3.txt
