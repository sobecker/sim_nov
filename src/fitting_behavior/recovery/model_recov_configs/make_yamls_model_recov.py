import numpy as np
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.saveload as sl

path            = '/Users/sbecker/yaml_files/yaml_rlnet/ModelRecovery/Fits/'; sl.make_long_dir(path)
comb_type       = 'app' # 'app','sep'
alg_type_sim    = ['hhybrid2']
alg_type_fit    = ['hhybrid2']
levels          = [4,5,6]
opt_method      = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'
id              = 5

for i,j in zip(range(len(alg_type_sim)),range(len(alg_type_fit))):
    for ll1 in range(len(levels)):
        for ll2 in range(len(levels)):
            save_name = f'multifit_sim-{alg_type_sim[i]}-l{levels[ll1]}_fit-{alg_type_fit[j]}-l{levels[ll2]}_{comb_type}_{opt_method}'
            with open(path+save_name+'.yaml', 'w') as rsh:
                    rsh.write(f'''\
apiVersion: run.ai/v1
kind: RunaiJob
metadata:
  name: sophia-train-multifit-{alg_type_fit[j]}-l{levels[ll1]}-l{levels[ll2]}-{comb_type}{id} # e.g. training-pod
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
      - name: sophia-train-multifit-{alg_type_fit[j]}-l{levels[ll1]}-l{levels[ll2]}-{comb_type}{id} # e.g. training-pod
        image: nvcr.io/nvidia/pytorch:20.03-py3
        workingDir: /lcncluster/becker/RL_reward_novelty
        volumeMounts: # mount lcncluster shared volume
            - mountPath: /lcncluster
              name: lcncluster

        command: ["/bin/bash"] # bash commands as args below, e.g. using a custom conda installation on the lcncluster
        args: [exps/ModelRecovery/Fits/{save_name}.sh]
        resources:
          requests:
            cpu: 20
          limits:
            cpu: 20
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