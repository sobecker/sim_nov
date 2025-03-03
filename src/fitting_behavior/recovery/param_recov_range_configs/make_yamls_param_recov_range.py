import utils.saveload as sl
from pathlib import Path

### Script to create yaml files for simulation and/or fitting of parameter recovery data ###

yaml_sim = True     # create yaml for simulation
yaml_fit = False    # create yaml for recovery fitting

# Make yaml for simulation (recovery data) ##################################################
if yaml_sim:
    path = Path('/Users/sbecker/yaml_files/yaml_sim_nov/recovery/sim/')
    sl.make_long_dir(path)

    comb_type   = ['app'] # 'app','sep'
    alg_type    = ['hnor','hhybrid2','hnac-gn','nor','nac','hybrid2']
    # ['hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle','hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box']
    # ['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace']
    # ['hnac-gn','nac','nor','hybrid2'] 
    # 'hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    opt_method  = ['Nelder-Mead'] # 'SLSQP','L-BFGS-B'
    param_range = 'uniparam'
    levels      = list(range(1,7))

    for i_c in range(len(comb_type)):
        for i_a in range(len(alg_type)):
            
            # Create yaml files for similarity-based agents
            if 'hnor' in alg_type[i_a] or 'hnac' in alg_type[i_a] or 'hhybrid' in alg_type[i_a]:
                for i_l in range(len(levels)):
                  save_name = f'multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_l{levels[i_l]}'
                  log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_l{levels[i_l]}'
                  with open(path / f'{save_name}.yaml', 'w') as rsh:
                    rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: multisim-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]}-l{levels[i_l]} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: multisim-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]}-l{levels[i_l]} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: /lcncluster/becker/sim_nov/exp/recovery/sim/multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}_l{levels[i_l]}.sh
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
    value: 80Gi
  memoryLimit:
    value: 80Gi
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
            # Create yaml files for count-based agents
            else:
              save_name = f'multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}'
              log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{param_range}'
              with open(path / f'{save_name}.yaml', 'w') as rsh:
                rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: multisim-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: multisim-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: /lcncluster/becker/sim_nov/exp/recovery/sim/multisim-{alg_type[i_a]}_{comb_type[i_c]}_{param_range}.sh
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
    value: 40Gi
  memoryLimit:
    value: 40Gi
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
    
# Make yaml for recovery fitting ############################################################
if yaml_fit:

    path = '/Users/sbecker/yaml_files/yaml_rlnet/ParameterRecovery/Fits/'; sl.make_long_dir(path)

    comb_type   = ['app','sep'] # 'app','sep'
    alg_type    = ['hnor','hhybrid2'] # 'nac','nor','hnac-gn','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    opt_method  = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'
  
    for i_c in range(len(comb_type)):
        for i_a in range(len(alg_type)):
            save_name = f'multifit-{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}'
            log_folder = f'{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}'
            with open(path+save_name+'.yaml', 'w') as rsh:
                    rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: multifit-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: multifit-{alg_type[i_a].replace("_","-")}-{comb_type[i_c]} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: /lcncluster/becker/RL_reward_novelty/exps/ParameterRecovery/Fits/multifit-{alg_type[i_a]}_{comb_type[i_c]}_{opt_method}.sh
  environment:
    items:
      HOME:
        value: /lcncluster/becker/.caas_HOME # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  runAsUser:
    value: true # system automatically fetches and applies UID and GID
  cpu:
    value: "5"
  cpuLimit:
    value: "5"
  gpu:
    value: "0" # up to 4, 0 for CPU only
  memory:
    value: 40Gi
  memoryLimit:
    value: 40Gi
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