import numpy as np
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.saveload as sl

leakiness_type = 'leaky'
multinov_type  = '' # '_multinov', '_multinov-eps', '_multinov-alph', 'multinov-eps-alph'

# alg_type    = [
#             #    f'{leakiness_type}{multinov_type}_hnor_notrace_center-box',
#                # f'{leakiness_type}_hnac-gn_notrace_center-box',
#                # f'{leakiness_type}_hhybrid2_notrace_center-box',
#                f'{leakiness_type}{multinov_type}_hnor_notrace',
#                # f'{leakiness_type}_hnac-gn_notrace',
#                # f'{leakiness_type}_hhybrid2_notrace',
#                f'{leakiness_type}{multinov_type}_hnor_triangle',
#                # f'{leakiness_type}_hnac-gn_triangle',
#                # f'{leakiness_type}_hhybrid2_triangle',
#             #    f'{leakiness_type}_hnor_center-triangle'
#                # f'{leakiness_type}_hnac-gn_center-triangle',
#                # f'{leakiness_type}_hhybrid2_center-triangle'
#                ] 
# # levels      = [1,2,3,4,5,6]
# levels   = [[1,6], [2,6], [3,6], [4,6], [5,6], [6,6]] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

alg_type = [f'{leakiness_type}{multinov_type}_hnor_placefields']
levels   = [[1], [2], [3], [4], [5], [1,3,5], [1,2,3,4,5]]

opt_type    = 'mice' # 'mice','opt','naive'
opt_alg     = ['Nelder-Mead'] # 'Nelder-Mead','L-BFGS-B','SLSQP'
comb_type   = ['app'] # '','sep','app'
maxit       = False

name_proj = 'MLE4'

path = f'/Volumes/lcncluster/becker/RL_reward_novelty/exps/{name_proj}/'; sl.make_dir(path)
path = path+'Fits/'; sl.make_dir(path)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            str_maxit = '-maxit' if maxit else ''
            save_name1 = f'{name_proj.lower()}{str_maxit}_{alg_type[aa]}'
            path_i = path; sl.make_dir(path_i)

            for ll in range(len(levels)):
                save_name2 = f'mle_{alg_type[aa]}-l{"-".join(map(str,levels[ll]))}'
                with open (path_i+f'/{save_name2}{clink}{comb_type[cc]}.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name1}{clink}{comb_type[cc]}"
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/{name_proj}
mkdir -p ${{base_path}}/logs/{name_proj}/Fits/
mkdir -p ${{base_path}}/logs/{name_proj}/Fits/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name2}{clink}{comb_type[cc]}"
python -u -b ${{base_path}}/src/scripts/MLE/mle_fit.py -c ${{base_path}}/src/scripts/{name_proj}/mle_fit_configs/{save_name2}{clink}{comb_type[cc]}.json | tee ${{base_path}}/logs/{name_proj}/Fits/${{log_folder}}/log_l{"-".join(map(str,levels[ll]))}.txt
''')