import numpy as np
import sys

import utils.saveload as sl

exp_sim = True
exp_fit = False

if exp_sim:
    path = '/Volumes/lcncluster/becker/RL_reward_novelty/exps/ParameterRecovery/Sim/'; sl.make_long_dir(path)

    comb_type   = ['app'] # 'app','sep'
    alg_type    = ['hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle','hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box'] #['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace'] #['hnac-gn','nac','nor','hybrid2'] # 'nac','nor','hybrid2','hnac-gn','hnor','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hhybrid2'
    opt_method  = ['Nelder-Mead'] # 'SLSQP','L-BFGS-B'
    levels      = list(range(1,7))

    param_range = 'uniparam' #'fixrange'
    param_str   = '_uniparam' if param_range=='uniparam' else ''

    for i_c in range(len(comb_type)):
        for i_a in range(len(alg_type)):
            if 'hnor' in alg_type[i_a] or 'hnac' in alg_type[i_a] or 'hhybrid' in alg_type[i_a]:
                for i_l in range(len(levels)):
                    save_name = f'multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_l{levels[i_l]}'
                    log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_l{levels[i_l]}'
                    with open(path+save_name+'.sh', 'w') as rsh:
                        rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_param_recov_multisim-{alg_type[i_a]}"
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/ParameterRecovery
mkdir -p ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/
mkdir -p ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build multisim-nor"
python -u -b ${{base_path}}/src/scripts/ParameterRecovery/sim_p_range.py -c ${{base_path}}/src/scripts/ParameterRecovery/param_recov_range_configs/multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_seed-12345_l{levels[i_l]}.json | tee ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/${{log_folder}}/log_{levels[i_l]}.txt
''')
            else:
                save_name = f'multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}'
                log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{param_range}'
                with open(path+save_name+'.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_param_recov_multisim-{alg_type[i_a]}"
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/ParameterRecovery
mkdir -p ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/
mkdir -p ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build multisim-nor"
python -u -b ${{base_path}}/src/scripts/ParameterRecovery/sim_p_range.py -c ${{base_path}}/src/scripts/ParameterRecovery/param_recov_range_configs/multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_seed-12345.json | tee ${{base_path}}/logs/ParameterRecovery/SimData{param_str}/${{log_folder}}/log.txt
''')

if exp_fit:
    path = '/Volumes/lcncluster/becker/RL_reward_novelty/exps/ParameterRecovery/Fits/'; sl.make_long_dir(path)

    comb_type   = ['app','sep'] # 'app','sep'
    alg_type    = ['hybrid','hhybrid'] # 'nac','nor','hnac-gn','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    opt_method  = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'

    for i_c in range(len(comb_type)):
        for i_a in range(len(alg_type)):
            save_name = f'multifit-{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}'
            log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}'
            with open(path+save_name+'.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_param_recov_multifit-{alg_type[i_a]}"
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/ParameterRecovery
mkdir -p ${{base_path}}/logs/ParameterRecovery/FitData/
mkdir -p ${{base_path}}/logs/ParameterRecovery/FitData/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build multifit-{alg_type[i_a]}-{comb_type[i_c]}"
python -u -b ${{base_path}}/src/scripts/ParameterRecovery/fit_p_rand.py -c ${{base_path}}/src/scripts/ParameterRecovery/param_recov_range_configs/multifit-{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}.json | tee ${{base_path}}/logs/ParameterRecovery/FitData/${{log_folder}}/log_{comb_type[i_c]}.txt
''')