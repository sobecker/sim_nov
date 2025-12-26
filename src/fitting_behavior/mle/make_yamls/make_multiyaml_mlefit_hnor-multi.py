import numpy as np
import sys
import os
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.saveload as sl

# alg_type    = ['hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hnor']
# levels      = [1,2,3,4,5,6]
# opt_type    = 'mice' 
# opt_alg     = ['Nelder-Mead']
# comb_type   = ['']

leakiness_type = 'leaky'
multinov_type  = '_multinov-eps' # '_multinov', '_multinov-eps', '_multinov-alph', 'multinov-eps-alph'
alg_type    = [
                # f'{leakiness_type}{multinov_type}_hnor_triangle',
            #    f'{leakiness_type}_hnac-gn_triangle',
            #    f'{leakiness_type}_hhybrid2_triangle',
            #    f'{leakiness_type}_multinov_hnor_center-triangle',
            #    f'{leakiness_type}_hnac-gn_center-triangle',
            #    f'{leakiness_type}_hhybrid2_center-triangle',
               f'{leakiness_type}{multinov_type}_hnor_notrace',
            #    f'{leakiness_type}_hnac-gn_notrace',
            #    f'{leakiness_type}_hhybrid2_notrace',
            #    f'{leakiness_type}_multinov_hnor_notrace_center-box'
            #    f'{leakiness_type}_hnac-gn_notrace_center-box',
            #    f'{leakiness_type}_hhybrid2_notrace_center-box'
            ]
# levels      = [1,2,3,4,5,6]
levels   = [[1,6], [2,6], [3,6], [4,6], [5,6]] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

# alg_type = [f'{leakiness_type}{multinov_type}_hnor_placefields']
# levels   = [[1], [2], [3], [4], [5], [1,3,5], [1,2,3,4,5]]

opt_type    = 'mice' 
opt_alg     = ['Nelder-Mead']
comb_type   = ['app']
single_run_id = '_ri8' # '', '_ri1', '_ri2'

num_cpu = 20
name_proj = 'MLE4'
name_set = 'Fits'

path_yaml = f'/Users/sbecker/runai_cli_files/rlnet/{name_proj}/Fits/'
sl.make_long_dir(path_yaml)

name_yaml_combined = os.path.join(path_yaml,f'start_all_snov_{leakiness_type}.sh')

for aa in range(len(alg_type)):
    for oo in range(len(opt_alg)):
        for cc in range(len(comb_type)):
            clink = '-' if len(comb_type[cc])>0 else ''
            save_name1 = f'{name_proj.lower()}_{alg_type[aa]}'.replace('center','c').replace('notrace','nt').replace('triangle','t').replace('box','b')
            path_yaml1 = path_yaml+save_name1
            sl.make_dir(path_yaml1)
            for ll in range(len(levels)):
                save_name2 = f'mle_{alg_type[aa]}-l{"-".join(map(str,levels[ll]))}'
                full_job_name = f"{save_name1}-l{'-'.join(map(str,levels[ll]))}-{comb_type[cc]}{single_run_id}".replace('_','-').lower()
                if len(full_job_name) > 63:
                    full_job_name = full_job_name[-63:].strip('-').strip('_')
                with open (os.path.join(path_yaml1,f'submit_{save_name1}{clink}{comb_type[cc]}.sh'), 'w') as rsh:
                  rsh.write(f'''\
runai submit \
  --name {full_job_name} \
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
  -- /bin/bash /lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/{save_name2}{clink}{comb_type[cc]}.sh

                            ''')
                  
                # Create runai file
                if aa==0 and oo==0 and cc==0 and ll==0:
                    write_mode = 'w'
                else:
                    write_mode = 'a'

                with open (name_yaml_combined, write_mode) as rsh:
                  rsh.write(f'''\
runai submit \
  --name {full_job_name} \
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
  -- /bin/bash /lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/{save_name2}{clink}{comb_type[cc]}.sh

                            ''')
                  
                # with open (name_yaml_combined, 'a') as rsh:
                #   rsh.write("\n")

            print(f'Exp file saved as {os.path.join(path_yaml1,f"submit_{save_name2}{clink}{comb_type[cc]}.sh")}')