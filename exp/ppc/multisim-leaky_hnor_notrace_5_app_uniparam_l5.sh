    #!/bin/bash
    echo "creating directory"
    log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_multisim-leaky_hnor_notrace_5"
    base_path="/lcncluster/becker/sim_nov"
    echo "folder name: ${log_folder}"
    mkdir -p ${base_path}/logs/ppc
    mkdir -p ${base_path}/logs/ppc/sim__uniparam/
    mkdir -p ${base_path}/logs/ppc/sim__uniparam/${log_folder}

    echo "activating conda environment"
    source activate rlnet_cluster

    echo "build multisim-nor"
    python -u -b ${base_path}/src/fitting_behavior/ppc/sim_ppc.py -c ${base_path}/src/fitting_behavior/ppc/configs_ppc/multisim-leaky_hnor_notrace_5_app_uniparam_l5.json | tee ${base_path}/logs/ppc/sim__uniparam/${log_folder}/log_l5.txt
    