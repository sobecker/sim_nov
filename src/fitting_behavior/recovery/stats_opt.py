import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import utils.saveload as sl

base_path   = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'
alg_type    = 'hhybrid2'
comb_type   = 'app'
uniparam    = True
data_path   = f'ParameterRecovery/FitData{"_uniparam" if uniparam else ""}/sim-{alg_type}-{comb_type}_fit-{alg_type}-{comb_type}_Nelder-Mead/'
f_comp      = True

# Get all folders (=comb of levels) that we need to analyze
d         = os.path.join(base_path,data_path)
dl        = glob.glob(os.path.join(d,f'*sim-{alg_type}*'))

if f_comp:
    fitcomb_all = []
    simID_all   = []
    success_all = []
    nit_all     = []
    for i in range(len(dl)):
        # Get all simulation folder that we need to analyze
        ddl = glob.glob(os.path.join(dl[i],'*_sim*'))
        ddl_i = [int(os.path.normpath(f).split('/')[-1].split('_sim')[-1].split('_')[0]) for f in ddl]
        ddl_df = pd.DataFrame({'simID':ddl_i,'folder':ddl})
        ddl_df = ddl_df.sort_values('simID')
        # Get combination of fits
        dl1 = os.path.normpath(dl[i]).split('/')[-1]
        if f'{comb_type}-l' in dl1:
            dl2 = dl1.split('app-l')
            fitcomb = f"l{dl2[1].split('_')[0]}-l{dl2[2].split('_')[0]}"
        else:
            fitcomb = f"{dl1.split('_')[-1]}-{dl1.split('_')[-1]}"

        for ii in range(len(ddl_i)):
            folder = ddl_df.loc[ddl_df['simID']==ddl_i[ii],'folder'].values[0]
            file   = f'{os.path.normpath(folder).split("/")[-1]}_{comb_type}.pickle'
            if os.path.exists(os.path.join(folder,file)):
                data = sl.load_sim_data(folder,file_data=file)
                fitcomb_all.append(fitcomb)
                simID_all.append(ddl_i[ii])
                success_all.append(data.opt_success.values[0])
                nit_all.append(data.opt_it.values[0])
        
    df = pd.DataFrame({'fit_comb':fitcomb_all,'simID':simID_all,'opt_success':success_all,'opt_it':nit_all})
    df.to_csv(os.path.join(d,'opt_stats.csv'))
    df.to_pickle(os.path.join(d,'opt_stats.pickle'))
else:
    df = sl.load_sim_data(d,file_data='opt_stats.pickle')

# Prepare saving
save_path = os.path.join(base_path.replace('data','output'),data_path)
sl.make_long_dir(save_path)

# Plot number of successful fits per fitting combination
df1 = df.groupby('fit_comb').agg(success=('opt_success',np.sum),fail=('opt_success',lambda x: -np.sum(x-1))).reset_index()
df = df.merge(df1,on='fit_comb')
f,ax = plt.subplots(1,1)
a_success = ax.bar(np.arange(len(df1)),df1.success)
a_fail = ax.bar(np.arange(len(df1)),df1.fail,bottom=df1.success)
ax.set_xlabel('Fitted combination')
ax.set_xticks(np.arange(len(df1.fit_comb)))
ax.set_xticklabels(list(df1.fit_comb))
ax.set_ylabel('Counts')
ax.legend([a_success[0],a_fail[0]],['Converged before maxit','Converged after maxit'])
plt.savefig(os.path.join(save_path,'opt_success-vs-fail.svg'))
plt.savefig(os.path.join(save_path,'opt_success-vs-fail.eps'))

# Plot number of opt steps
df2 = df.groupby('fit_comb').agg(mean_it=('opt_it',np.mean),sem_it=('opt_it',stats.sem)).reset_index()
df = df.merge(df2,on='fit_comb')
f,ax = plt.subplots(1,1)
a = ax.bar(np.arange(len(df2)),df2.mean_it,yerr=[df2.mean_it-df2.sem_it,df2.mean_it+df2.sem_it])
xlim = ax.get_xlim(); eps_x = 1
ax.plot([xlim[0]-eps_x,xlim[1]+eps_x],[1000]*2,':',color='k')
ax.set_xlim(xlim)
ax.set_xlabel('Fitted combination')
ax.set_xticks(np.arange(len(df2.fit_comb)))
ax.set_xticklabels(list(df2.fit_comb))
ax.set_ylabel('Mean optimization steps')
plt.savefig(os.path.join(save_path,'opt_mean-steps.svg'))
plt.savefig(os.path.join(save_path,'opt_mean-steps.eps'))



