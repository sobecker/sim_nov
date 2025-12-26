import numpy as np
import sys
import os
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
# opt_type    = 'mice' # mice, opt, naive
# opt_alg     = ['Nelder-Mead','L-BFGS-B','SLSQP']
# comb_type   = ['']
# multistart  = True

leakiness_type = 'leaky'
alg_type    = [
               f'{leakiness_type}_nor',
              #  f'{leakiness_type}_hybrid2',
              #  f'{leakiness_type}_nac'
               ]
opt_type    = 'mice' # mice, opt, naive
opt_alg     = ['Nelder-Mead']
comb_type   = ['app']
multistart  = False

num_cpu = 24
name_proj = 'MLE4'
name_set = 'Fits' 

path_yaml = f'/Users/sbecker/runai_cli_files/rlnet/{name_proj}/Fits/'
sl.make_long_dir(path_yaml)

name_yaml_combined = os.path.join(path_yaml,f'start_all_cnov_{leakiness_type}.sh')

for aa in range(len(alg_type)): # iterate over alg types
    
    for oo in range(len(opt_alg)): # iterate over optimizers
        
        for cc in range(len(comb_type)): # iterate over comb types
            
            clink = '-' if len(comb_type[cc])>0 else ''
            multi = '_multi' if multistart else ''

            save_name1 = f'mle_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'
            save_name2 = f'{name_proj.lower()}_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'

            path_yaml_i = path_yaml+save_name1
            sl.make_dir(path_yaml_i)

            # Create individual runai_cli file
            with open (os.path.join(path_yaml_i,f'submit_{save_name1}{clink}{comb_type[cc]}.sh'), 'w') as rsh:
                  rsh.write(f'''\
runai submit \
  --name {save_name2.replace('_','-').lower()}{clink}{comb_type[cc].lower()} \
  --image nvcr.io/nvidia/pytorch:25.03-py3 \
  --gpu 0 \
  --cpu {num_cpu} \
  --cpu-limit {num_cpu} \
  --memory 40Gi \
  --memory-limit 80Gi \
  --large-shm \
  --node-pools default,h100 \
  --environment HOME="/lcncluster/becker/.caas_HOME" \
  --run-as-uid 229361 \
  --run-as-gid 20184 \
  --run-as-user \
  --existing-pvc claimname=lcn1-lcncluster,path=/lcncluster \
  --existing-pvc claimname=lcn1-scratch,path=/scratch \
  --working-dir /lcncluster \
  --command \
  -- /bin/bash /lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/{save_name1}{clink}{comb_type[cc]}.sh
''')
            
            # Create appended runai_cli file
            if aa==0 and oo==0 and cc==0:
                write_mode = 'w'
            else:
                write_mode = 'a'
            with open (name_yaml_combined, write_mode) as rsh:
                  rsh.write(f'''\
runai submit \
  --name {save_name2.replace('_','-').lower()}{clink}{comb_type[cc].lower()} \
  --image nvcr.io/nvidia/pytorch:25.03-py3 \
  --gpu 0 \
  --cpu {num_cpu} \
  --cpu-limit {num_cpu} \
  --memory 40Gi \
  --memory-limit 80Gi \
  --large-shm \
  --node-pools default,h100 \
  --environment HOME="/lcncluster/becker/.caas_HOME" \
  --run-as-uid 229361 \
  --run-as-gid 20184 \
  --run-as-user \
  --existing-pvc claimname=lcn1-lcncluster,path=/lcncluster \
  --existing-pvc claimname=lcn1-scratch,path=/scratch \
  --working-dir /lcncluster \
  --command \
  -- /bin/bash /lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/{save_name1}{clink}{comb_type[cc]}.sh
''')
        
            print(f'Exp file saved as {os.path.join(path_yaml_i,f"submit_{save_name1}{clink}{comb_type[cc]}.sh")}')
