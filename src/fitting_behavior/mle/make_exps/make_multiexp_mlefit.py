import utils.saveload as sl

### Generates shell scripts for fitting RL models with count-based novelty using their specific config file. ###

# Requirements:
# Specify the path below. Specify the types of algorithms to fit. Adapt shell script to system needs.

path_exp = sl.get_rootpath() / 'exp' / 'mle' / 'fits'             # path to save the shell scripts
sl.make_long_dir(path_exp)

# Standard settings [reproducing the paper results]
alg_type    = ['nac','nor','hybrid2']
opt_type    = 'mice' # opt, naive
opt_alg     = ['Nelder-Mead']
comb_type   = ['app']
multistart  = False
maxit       = [False, False, True]

# Additional MF tests
# alg_type    = ['nac-nooi','nac-kpop','nac-kpop-t','nac-kmix','nac-kmix-t']
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['app']

# alg_type    = ['nac-oi-only']
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['app']

# Comparing different optimization algorithms, with random starts
# alg_type    = ['nor']
# opt_type    = 'opt' # mice, opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['']
# multistart  = True

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            multi = '_multi' if multistart else ''
            save_name1 = f'mle{"-maxit" if maxit[aa] else ""}_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'
            path_i = path_exp / save_name1 
            sl.make_long_dir(path_i)

            with open (path_i / f'{save_name1}{clink}{comb_type[cc]}.sh', 'w') as rsh:
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

echo "build {save_name1}{clink}{comb_type[cc]}"
python -u -b ${{base_path}}/src/fitting_behavior/mle/mle_fit.py -c ${{base_path}}/src/fitting_behavior/mle/mle_fit_configs/{save_name1}{clink}{comb_type[cc]}.json | tee ${{base_path}}/logs/mle/fits/${{log_folder}}/log.txt
''')