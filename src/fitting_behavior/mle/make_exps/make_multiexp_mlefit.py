import numpy as np
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.saveload as sl

# alg_type    = ['nac','nor']
# opt_type    = 'mice' # opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['sep','app']

# alg_type    = ['nac-nooi','nac-kpop','nac-kpop-t','nac-kmix','nac-kmix-t']
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['']

# alg_type    = ['nac-oi-only']
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['']

# alg_type    = ['nor']
# opt_type    = 'mice' # opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['']

# alg_type    = ['nor']
# opt_type    = 'opt' # mice, opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['']
# multistart  = True

alg_type    = ['hybrid']
opt_type    = 'mice' # mice, opt, naive
opt_alg     = ['Nelder-Mead']
comb_type   = ['']
multistart  = False

path = '/Volumes/lcncluster/becker/RL_reward_novelty/exps/MLE/'; sl.make_dir(path)
path = path+'Fits/'; sl.make_dir(path)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            multi = '_multi' if multistart else ''
            save_name = f'mle-maxit_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'
            path_i = path+save_name; sl.make_dir(path_i)

            with open (path_i+f'/{save_name}{clink}{comb_type[cc]}.sh', 'w') as rsh:
                rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name}{clink}{comb_type[cc]}"
echo "folder name: ${{log_folder}}"
mkdir -p logs/MLE
mkdir -p logs/MLE/Fits/
mkdir -p logs/MLE/Fits/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name}{clink}{comb_type[cc]}"
python -u -b ./src/scripts/MLE/mle_fit.py -c ./src/scripts/MLE/mle_fit_configs/{save_name}{clink}{comb_type[cc]}.json | tee logs/MLE/Fits/${{log_folder}}/log.txt
''')