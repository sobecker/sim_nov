from pathlib import Path
import utils.saveload as sl

### Generates yaml files for fitting RL models with similarity-based novelty by running shell script on cluster. ###

# Requirements:
# Specify the paths below. Specify the types of algorithms to fit.

alg_type    = ['hnac-gn','hnor','hhybrid2']
levels      = [1,2,3,4,5,6]
opt_type    = 'mice' 
opt_alg     = ['Nelder-Mead']
comb_type   = ['app']
maxit       = False

# alg_type    = ['hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box','hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle']#['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace']
# levels      = [1,2,3,4,5,6]
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['']
# maxit       = False

path_yaml = Path(f'/Users/sbecker/yaml_files/yaml_sim_nov/mle/fits')
path_exp  = Path(f'/lcncluster/becker/sim_nov/exps/mle/fits')
sl.make_long_dir(path_yaml)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            str_maxit = '-maxit' if maxit else ''
            clink = '-' if len(comb_type[cc])>0 else ''
            save_name1 = f'mle{str_maxit}_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}'
            path_exps1 = path_exp / save_name1
            path_yaml1 = path_yaml / save_name1
            sl.make_long_dir(path_yaml1)
            for ll in range(len(levels)):
                save_name2 = f'mle{str_maxit}_{alg_type[aa]}-l{levels[ll]}-{opt_type}_{opt_alg[oo]}'
                with open (path_yaml1 / f'{save_name2}{clink}{comb_type[cc]}.yaml', 'w') as rsh:
                    rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: {save_name2.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: {save_name2.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: {path_exps1 / f'{save_name2}{clink}{comb_type[cc]}.sh'}
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
