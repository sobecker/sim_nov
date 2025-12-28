#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_leaky_hnor_notrace-l1_app_5"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${log_folder}"
mkdir -p ${base_path}/logs/crossvalidation/
mkdir -p ${base_path}/logs/crossvalidation/cv-mouseid/
mkdir -p ${base_path}/logs/crossvalidation/cv-mouseid/${log_folder}

echo "build leaky_hnor_notrace-l1_app_5"
python -u -b ${base_path}/src/fitting_behavior/crossvalidation/LL_crossvalidation.py -c ${base_path}/src/fitting_behavior/crossvalidation/configs_cv-mouseid/leaky_hnor_notrace-l1_app_5.json | tee ${base_path}/logs/crossvalidation/cv-mouseid/${log_folder}/log_leaky_hnor_notrace-l1_app_5.txt
