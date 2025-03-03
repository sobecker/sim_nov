import utils.saveload as sl

### Generates shell scripts for fitting RL models with similarity-based novelty using their specific config file. ###

# Requirements:
# Specify the path below. Specify the types of algorithms to fit. Adapt shell script to system needs.

path_exp = sl.get_rootpath() / 'exp' / 'mle' / 'fits'             # path to save the shell scripts
sl.make_long_dir(path_exp)

# Standard setting [reproducing the paper results]
alg_type    = ['hnac-gn-goi','hnor','hhybrid2']                                     # algorithms to fit: ... 'hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi' 
levels      = [1,2,3,4,5,6]                                                     # granularity levels to fit for each algorithm
opt_type    = 'mice' # 'mice','opt','naive'                                     # which data to fit to 
opt_alg     = ['Nelder-Mead'] # 'Nelder-Mead','L-BFGS-B','SLSQP'                # optimization algorithm
comb_type   = ['app'] # '','sep','app'                                          # fitting mode: fit each animals separately (sep) or together (app)
maxit       = False                                                             

# alg_type    = ['hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box','hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle'] #['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace']
# levels      = [0,1,2,3,4,5,6]
# opt_type    = 'mice' # 'mice','opt','naive'
# opt_alg     = ['Nelder-Mead'] # 'Nelder-Mead','L-BFGS-B','SLSQP'
# comb_type   = [''] # '','sep','app'
# maxit       = False

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            str_maxit = '-maxit' if maxit else ''
            save_name1 = f'mle{str_maxit}_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}'
            path_i = path_exp / save_name1 
            sl.make_long_dir(path_i)

            for ll in range(len(levels)):
                save_name2 = f'mle{str_maxit}_{alg_type[aa]}-l{levels[ll]}-{opt_type}_{opt_alg[oo]}'
                with open (path_i / f'{save_name2}{clink}{comb_type[cc]}.sh', 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{save_name1}{clink}{comb_type[cc]}"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/mle
mkdir -p ${{base_path}}/logs/mle/fits/
mkdir -p ${{base_path}}/logs/mle/fits/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {save_name2}{clink}{comb_type[cc]}"
python -u -b ${{base_path}}/src/fitting_behavior/mle/mle_fit.py -c ${{base_path}}/src/fitting_behavior/mle/mle_fit_configs/{save_name2}{clink}{comb_type[cc]}.json | tee ${{base_path}}/logs/mle/fits/${{log_folder}}/log_l{levels[ll]}.txt
''')