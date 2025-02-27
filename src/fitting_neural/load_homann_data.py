import os
import numpy as np
import pandas as pd

############################################################################################################
#               Function to load experimental data for Homann experiments                                  #
############################################################################################################
def load_exp_data2(h_type,filter_emerge=True,cluster=True,type='mean'):
    # Load experimental data (for fitting measure computation)
    print(f'Current path:{os.getcwd()}')
    if cluster:
        path_cluster = '/lcncluster/becker/RL_reward_novelty/ext_data/Homann2022'
    else:
        path_cluster = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Homann2022'
    if h_type=='tau_memory':
        # Load novelty responses
        path1  = os.path.join(path_cluster,f'Homann2022_{h_type}_{type}.csv')
        edata1 = pd.read_csv(path1)
        edata1 = edata1.sort_values('x')
        edx    = list(map(lambda x: int(np.round(x)),edata1['x']))
        edy1   = np.array(list(map(lambda x: np.round(x,4),edata1[' y'])))
        # Load steady state responses
        path2  = os.path.join(path_cluster,f'Homann2022_steadystate_{type}.csv')
        edata2 = pd.read_csv(path2)
        edata2 = edata2.rename(columns={'y':' y'})
        edy2   = np.array(list(map(lambda x: np.round(x,4),edata2[' y'])))
        # Combine
        edy    = [edy1,edy2]
    else:
        # Load novelty responses
        path = os.path.join(path_cluster,f'Homann2022_{h_type}_{type}.csv')
        edata = pd.read_csv(path)
        edata = edata.sort_values('x')
        if h_type=='tau_emerge' and filter_emerge:
            edata = edata.iloc[::2]
        edx   = list(map(lambda x: int(np.round(x)),edata['x']))
        edy   = np.array(list(map(lambda x: np.round(x,4),edata[' y'])))
    return edx,edy

def load_exp_homann(cluster=True,type='mean'):
    htypes = ['tau_emerge','tau_recovery','tau_memory']
    all_data = []
    for i in range(len(htypes)):
        edx,edy = load_exp_data2(htypes[i],cluster=cluster,type=type)
        if htypes[i] == 'tau_memory':   
            all_data.append((edx,edy[0]))
            all_data.append((edx,edy[1]))
        else:
            all_data.append((edx,edy)) 
    return all_data


