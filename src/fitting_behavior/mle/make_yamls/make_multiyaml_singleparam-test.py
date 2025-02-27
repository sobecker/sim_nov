import numpy as np
import sys
import os

import utils.saveload as sl

alg_type = 'nor' # nac
opt_type = 'opt' # naive

save_name = f'mle_{alg_type}-{opt_type}'
path_exps = f'exps/MLE/SingleParamTests/{save_name}' 
path_yaml = f'/Users/sbecker/yaml_files/yaml_rlnet/MLE/SingleParamTests/{save_name}'
sl.make_long_dir(path_yaml)

if alg_type=='nor':
    l_var   = ['lambda_N','beta_1','epsilon','k_leak']
elif alg_type=='nac':
    l_var   = ['gamma', 'c_alph', 'a_alph', 'c_lam', 'a_lam', 'temp', 'c_w0', 'a_w0']

for i in range(len(l_var)):
    with open (path_yaml+f'/{save_name}_{l_var[i]}.yaml', 'w') as rsh:
        rsh.write(f'''\
apiVersion: run.ai/v1
kind: RunaiJob
metadata:
  name: sophia-train-{save_name.replace('_','-')}-{l_var[i].lower().replace('_','')} # e.g. training-pod
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
      - name: sophia-train-{save_name.replace('_','-')}-{l_var[i].lower().replace('_','')} # e.g. training-pod
        image: nvcr.io/nvidia/pytorch:20.03-py3
        workingDir: /lcncluster/becker/RL_reward_novelty
        volumeMounts: # mount lcncluster shared volume
            - mountPath: /lcncluster
              name: lcncluster

        command: ["/bin/bash"] # bash commands as args below, e.g. using a custom conda installation on the lcncluster
        args: [{path_exps}/{save_name}_{l_var[i]}.sh]
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
