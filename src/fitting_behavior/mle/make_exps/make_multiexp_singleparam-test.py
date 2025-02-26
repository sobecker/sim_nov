import numpy as np
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.saveload as sl

alg_type = 'nor' # nac, nor
opt_type = 'opt' # naive

save_name = f'mle_{alg_type}-{opt_type}'
path = '/Volumes/lcncluster/becker/RL_reward_novelty/exps/MLE/'; sl.make_dir(path)
path = path+'SingleParamTests/'; sl.make_dir(path)
path = path+save_name; sl.make_dir(path)

if alg_type=='nor':
    l_var   = ['lambda_N','beta_1','epsilon','k_leak']
elif alg_type=='nac':
    l_var   = ['gamma', 'c_alph', 'a_alph', 'c_lam', 'a_lam', 'temp', 'c_w0', 'a_w0']

for i in range(len(l_var)):
    with open (path+f'/{save_name}_{l_var[i]}.sh', 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name}_{l_var[i]}"
echo "folder name: ${{log_folder}}"
mkdir -p logs/MLE
mkdir -p logs/MLE/SingleParamTests/
mkdir -p logs/MLE/SingleParamTests/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name}"
python -u -b ./src/scripts/MLE/singleparam_tests.py -c ./src/scripts/MLE/singleparam_test_configs/{save_name}_{l_var[i]}.json | tee logs/MLE/SingleParamTests/${{log_folder}}/log.txt
''')