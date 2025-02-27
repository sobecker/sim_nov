import sys
import glob
import os
import numpy as np
import pandas as pd
import utils.saveload as sl

def make_overview_fits(data_path,sim_subset=[]):
    # Load list of data sets to be fitted
    regex = '*_sim*'
    datasets = glob.glob(data_path / regex)
    datasets.sort(key=lambda x: int(x.split('_')[-2].split('sim')[-1]))
    if len(sim_subset)>0:
        datasets = list(filter(lambda x: int(x.split('_')[-2].split('sim')[-1]) in sim_subset,datasets))

    p_df_list = []
    for i in range(len(datasets)):
        files_app = glob.glob(os.path.join(datasets[i],'*_app.pickle'))
        files_sep = glob.glob(os.path.join(datasets[i],'*_sep.pickle'))
        if len(files_app)>0 and len(files_sep)>0:
            file_app = files_app[0].split('/')[-1]
            file_sep = files_sep[0].split('/')[-1]

            res_app = sl.load_sim_data(datasets[i],file_data=file_app)
            res_sep = sl.load_sim_data(datasets[i],file_data=file_sep)
            
            p_dict = {'simID':int(datasets[i].split('_sim')[-1].split('_')[0])*np.ones(len(res_sep['subID'].unique())+2,dtype=int),
                        'subID':[res_app['subID'].iloc[0]]+list(np.unique(res_sep['subID'].values))+[-1],
                        #'mle_ll':[res_app['mle_ll'].iloc[0]]+list(np.unique(res_sep['mle_ll'].values))+[np.nanmean(np.unique(res_sep['mle_ll'].values))],
                        'mle_ll':[res_app['mle_ll'].iloc[0]]+[res_sep.loc[res_sep.subID==i,'mle_ll'].iloc[0] for i in np.unique(res_sep['subID'].values)]+[np.nanmean(np.unique(res_sep['mle_ll'].values))],
                        'mle_type':['app']+['sep']*len(res_sep['subID'].unique())+['mean_sep']}
            
            p_df = pd.DataFrame(p_dict)

            p_df.loc[p_df['mle_type']=='app',list(res_app['var_name'].unique())]        = list(res_app['mle_var'].values)
            seps = res_sep['mle_var'].values.reshape((-1,len(res_sep['var_name'].unique())))
            p_df.loc[p_df['mle_type']=='sep',list(res_app['var_name'].unique())]        = seps
            p_df.loc[p_df['mle_type']=='mean_sep',list(res_app['var_name'].unique())]   = np.nanmean(seps,axis=0)
            
            p_df_list.append(p_df)

    p_overview = pd.concat(p_df_list)

    p_overview.to_csv(os.path.join(data_path,'fitparam_overview.csv'))
    p_overview.to_pickle(os.path.join(data_path,'fitparam_overview.pickle'))

if __name__=="__main__":
    # Local data (test)
    # data_path = '/Users/sbecker/Projects/RL_reward_novelty/data/ParameterRecovery/FitData/nor_Nelder-Mead/'
    # make_overview_fits(data_path)

    # Cluster data 
    #data_path   = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nor_Nelder-Mead/'
    #data_path   = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nor_allparams_Nelder-Mead/'
    data_path   = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nac_Nelder-Mead/'
    regex       = '*_sim*'
    datasets    = glob.glob(data_path / regex)
    fitIDs      = [int(x.split('_')[-2].split('sim')[-1]) for x in datasets]
    make_overview_fits(data_path,sim_subset=fitIDs)