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

alg_type    = ['hybrid']
opt_type    = 'mice' # mice, opt, naive
opt_alg     = ['Nelder-Mead']
comb_type   = ['']
multistart  = False


path_yaml = f'/Users/sbecker/yaml_files/yaml_rlnet/MLE/Fits/'
sl.make_long_dir(path_yaml)

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            multi = '_multi' if multistart else ''
            save_name = f'mle-maxit_{alg_type[aa]}-{opt_type}_{opt_alg[oo]}{multi}'
            path_exps_i = f'exps/MLE/Fits/{save_name}' 
            path_yaml_i = path_yaml+save_name
            sl.make_dir(path_yaml_i)
            with open (path_yaml_i+f'/{save_name}{clink}{comb_type[cc]}.yaml', 'w') as rsh:
                rsh.write(f'''\
apiVersion: run.ai/v1
kind: RunaiJob
metadata:
  name: sophia-train-{save_name.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
  labels:
    # leave empty to obtain training job
spec:
  template:
    metadata:
      labels:
        user: sophia.becker # User i.e. firstname.lastname
    spec:
      nodeSelector:
        run.ai/type: "S8"
      hostIPC: true
      securityContext:
        runAsUser: 229361 # insert uid found in people.epfl in admistrative data
        runAsGroup: 20184  # insert gid found in people.epfl in admistrative data
        fsGroup: 0
      containers:
      - name: sophia-train-{save_name.replace('_','-').lower()}{clink}{comb_type[cc].lower()} # e.g. training-pod
        image: nvcr.io/nvidia/pytorch:20.03-py3
        workingDir: /lcncluster/becker/RL_reward_novelty
        volumeMounts: # mount lcncluster shared volume
            - mountPath: /lcncluster
              name: lcncluster

        command: ["/bin/bash"] # bash commands as args below, e.g. using a custom conda installation on the lcncluster
        args: [{path_exps_i}/{save_name}{clink}{comb_type[cc]}.sh]
        resources:
          requests:
            cpu: 5
          limits:
            cpu: 5
        env: # define HOME directory for pod
          - name: HOME
            value: /lcncluster/becker/.caas_HOME # PATH to HOME e.g. /lcncluster/user/.caas_HOME
      volumes: # define shared volume lcncluster
          - name: lcncluster
            persistentVolumeClaim:
              claimName: runai-lcn1-sbecker-lcncluster
      restartPolicy: Never
      schedulerName: runai-scheduler
''')
