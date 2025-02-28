from pathlib import Path
import utils.saveload as sl

### Generates yaml files for fitting RL models with count-based novelty by running shell script on cluster. ###

# Requirements:
# Specify the paths below. Specify the types of algorithms to fit.

alg_type    = ['nac','nor','hybrid2']
opt_type    = 'mice' # opt, naive
opt_alg     = ['Nelder-Mead']
comb_type   = ['app']
multistart  = False

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

path_yaml = Path(f'/Users/sbecker/yaml_files/yaml_sim_nov/mle/fits')
path_exp  = Path(f'/lcncluster/becker/sim_nov/exps/mle/fits')
sl.make_long_dir(path_yaml)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            multi = '_multi' if multistart else ''
            save_name = f'mle_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'
            path_exps_i = path_exp / save_name
            path_yaml_i = path_yaml / save_name
            sl.make_long_dir(path_yaml_i)
            with open (path_yaml_i / f'{save_name}{clink}{comb_type[cc]}.yaml', 'w') as rsh:
                rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: {save_name.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: {save_name.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: {path_exps_i / f'{save_name}{clink}{comb_type[cc]}.sh'}
  environment:
    items:
      HOME:
        value: /lcncluster/becker/.caas_HOME # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  runAsUser:
    value: true # system automatically fetches and applies UID and GID
  cpu:
    value: "20"
  cpuLimit:
    value: "20"
  gpu:
    value: "0" # up to 4, 0 for CPU only
  memory:
    value: 60Gi
  memoryLimit:
    value: 60Gi
  pvcs:
    items:
      pvc--0: # First is "pvc--0", second is "pvc--1", etc.
        value: 
          claimName: runai-lcn1-sbecker-lcncluster
          existingPvc: true
          path: /lcncluster
  nodePools:
    value: "default" # S8 nodes are now named default, if using GPUs, put "g10 g9"
''')
