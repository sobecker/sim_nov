import numpy as np
import sys

import utils.saveload as sl

path            = f'/Volumes/lcncluster/becker/RL_reward_novelty/exps/ModelRecovery/Fits/'; sl.make_long_dir(path)
comb_type       = 'app' # 'app','sep'
alg_type_sim    = ['hnor','hhybrid2']
alg_type_fit    = ['hnor','hhybrid2']
levels          = [4,5,6]
opt_method      = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'

for i,j in zip(range(len(alg_type_sim)),range(len(alg_type_fit))):
    for ll1 in range(len(levels)):
        for ll2 in range(len(levels)):
            save_name = f'multifit_sim-{alg_type_sim[i]}-l{levels[ll1]}_fit-{alg_type_fit[j]}-l{levels[ll2]}_{comb_type}_{opt_method}'
            with open(path+save_name+'.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name}"
echo "folder name: ${{log_folder}}"
mkdir -p logs/ModelRecovery
mkdir -p logs/ModelRecovery/FitData/
mkdir -p logs/ModelRecovery/FitData/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name}"
python -u -b ./src/scripts/ParameterRecovery/fit_p_rand.py -c ./src/scripts/ParameterRecovery/model_recov_configs/{save_name}.json | tee logs/ParameterRecovery/FitData/${{log_folder}}/log.txt
''')