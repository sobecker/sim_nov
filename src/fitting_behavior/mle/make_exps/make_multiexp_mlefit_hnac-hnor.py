import numpy as np
import sys

import utils.saveload as sl

# alg_type    = ['nac','nor']
# opt_type    = 'mice' # opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['sep','app']

# alg_type    = ['nac-nooi','nac-kpop','nac-kpop-t','nac-kmix','nac-kmix-t']
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['']

# alg_type    = ['hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hnor']
# levels      = [1,2,3,4,5,6]
# opt_type    = 'mice' # 'mice','opt','naive'
# opt_alg     = ['Nelder-Mead'] # 'Nelder-Mead','L-BFGS-B','SLSQP'
# comb_type   = [''] # '','sep','app'

alg_type    = ['hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box','hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle'] #['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace']
levels      = [0,1,2,3,4,5,6]
opt_type    = 'mice' # 'mice','opt','naive'
opt_alg     = ['Nelder-Mead'] # 'Nelder-Mead','L-BFGS-B','SLSQP'
comb_type   = [''] # '','sep','app'
maxit       = False

path = '/Volumes/lcncluster/becker/RL_reward_novelty/exps/MLE/'; sl.make_dir(path)
path = path+'Fits/'; sl.make_dir(path)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            str_maxit = '-maxit' if maxit else ''
            save_name1 = f'mle{str_maxit}_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}'
            path_i = path+save_name1; sl.make_dir(path_i)

            for ll in range(len(levels)):
                save_name2 = f'mle{str_maxit}_{alg_type[aa]}-l{levels[ll]}-{opt_type}_{opt_alg[oo]}'
                with open (path_i+f'/{save_name2}{clink}{comb_type[cc]}.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name1}{clink}{comb_type[cc]}"
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/MLE
mkdir -p ${{base_path}}/logs/MLE/Fits/
mkdir -p ${{base_path}}/logs/MLE/Fits/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name2}{clink}{comb_type[cc]}"
python -u -b ${{base_path}}/src/scripts/MLE/mle_fit.py -c ${{base_path}}/src/scripts/MLE/mle_fit_configs/{save_name2}{clink}{comb_type[cc]}.json | tee ${{base_path}}/logs/MLE/Fits/${{log_folder}}/log_l{levels[ll]}.txt
''')