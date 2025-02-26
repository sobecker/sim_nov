import sys
import numpy as np
import pandas as pd
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl
import src.fitting_behavior.optimization.base_params_opt as bpo
from src.fitting_behavior.mle.mle_fit import preprocess_micedata


UnrewNames  = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
RewNames    = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
AllNames    = RewNames+UnrewNames

params = bpo.base_params_nACtree.copy()
P = params['P']

d = []
len_app = 0
len_sep = []
for i in range(len(AllNames)):
    dir=sl.get_datapath().replace('data','ext_data')+'Rosenberg2021/'
    file=f'{AllNames[i]}-stateseq.pickle'
    df_i = preprocess_micedata(dir,file,P,subID=AllNames[i],epi=0)
    d.append(df_i)
    len_app += len(df_i)
    len_sep.append(len(df_i))

all_data = pd.concat(d,ignore_index=True)

print(len_app)
print(AllNames)
print(len_sep)

print({a:l for a,l in zip(AllNames,len_sep)})