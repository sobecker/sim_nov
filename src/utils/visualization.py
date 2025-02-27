#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:07:32 2022

@author: sbecker
"""

import numpy as np
import pandas as pd
import os
import pickle
import csv

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm


##############################################################################
#       HELPER FUNCTIONS                                                     #
##############################################################################
def quantile95(data):
    return np.quantile(data,0.95)

def quantile05(data):
    return np.quantile(data,0.05)

def quantile75(data):
    return np.quantile(data,0.75)

def quantile25(data):
    return np.quantile(data,0.25)

def quantile95err(data):
    return abs(np.quantile(data,0.95)-np.quantile(data,0.5))

def quantile05err(data):
    return abs(np.quantile(data,0.05)-np.quantile(data,0.5))

def quantile75err(data):
    return abs(np.quantile(data,0.75)-np.quantile(data,0.5))

def quantile25err(data):
    return abs(np.quantile(data,0.25)-np.quantile(data,0.5))

def ci_median(data):
    bs = stats.bootstrap((data,), np.median, confidence_level=0.95,random_state=1, method='percentile')
    return bs.confidence_interval

def se_median(data):
    bs = stats.bootstrap((data,), np.median, confidence_level=0.95,random_state=1, method='basic')
    return bs.standard_error

def all_zero_x0(trials,epi):
    return np.zeros((trials,epi),dtype=int).tolist()

def seq_per_trial_x0(seq,trials):
    return np.array(seq*12).reshape((trials,len(seq))).tolist()

def auto_seeds(trials):
    return list(range(trials))

##############################################################################
#       COLORMAPS                                                            #
##############################################################################

def prep_cmap(name,num):
    cmap = plt.cm.get_cmap(name)
    vmax = num-1+2
    cnorm = colors.Normalize(vmin=0, vmax=vmax)
    smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    cb = [smap.to_rgba(vmax-i) for i in range(vmax)]
    return cb

def prep_cmap_discrete(name):
    cmap    = plt.cm.get_cmap(name)
    vmin    = 0
    vmax    = cmap.N-1
    cnorm   = colors.Normalize(vmin=vmin, vmax=vmax)
    smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    cb = [smap.to_rgba(i) for i in range(cmap.N)]
    return cb

def prep_cmap_log(name,vmin,vmax):
    cmap  = plt.cm.get_cmap(name)
    cnorm = colors.LogNorm(vmin,vmax)
    smap  = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    return smap

##############################################################################
#       COMPATIBILITY WITH ADOBE ILLUSTRATOR                                 #
##############################################################################

my_ceil = lambda num, prec: (10**(-prec))*np.ceil(num*10**(prec))
my_floor = lambda num, prec: (10**(-prec))*np.floor(num*10**(prec))

def make_shifted_yaxis(ax,shift):
    dt      = int(np.round(ax.get_yticks()[1]-ax.get_yticks()[0])) # get distance between two yticks (rounded to integers)
    rprec   = -(len(str(dt))-1)                                    # get rounding precision, e.g. -2 for yticks with distance ~100
    if shift<0:
        dyt     = my_floor(shift,rprec)-shift
    else:
        dyt     = my_ceil(shift,rprec)-shift                           # get distance between first ytick and first ytick for shifted yaxis
    yl      = ax.get_ylim()+dyt
    yt      = ax.get_yticks()+dyt
    yt      = [ytj for ytj in yt if (ytj>=yl[0] and ytj<=yl[1])]
    ytl     = [f'{int(ytj+shift)}' for ytj in yt]
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)
    ax.set_ylim(yl)

def make_shifted_yaxis_logscale(ax,shift=0):
    # yt_old  = ax.get_yticks()
    # dt      = yt_old[1:]-yt_old[:-1]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels())

##############################################################################
#       DATA SAVING/LOADING                                                  #
##############################################################################
def make_dir_fig(dir_fig):
    if not os.path.isdir(dir_fig):
        os.mkdir(dir_fig) 
    return dir_fig

def make_dir(dir,folder):
    if not folder:
        full_dir = dir
    else:
        if dir[-1]=='/': full_dir = dir+folder
        else: full_dir = dir+'/'+folder

    if not os.path.isdir(full_dir):
        os.mkdir(full_dir) 
    return full_dir

def load_sim_data(dir_data,file_data='all_data.pickle',auto_path=True):
    if auto_path:
        if 'data' in dir_data:
            with open(dir_data / file_data, 'rb') as f:
                all_data = pickle.load(f)  
        else:
            if 'src/' in os.getcwd():
                with open(os.path.join('..','..','data',dir_data,file_data), 'rb') as f:
                    all_data = pickle.load(f)  
            else:
                with open(os.path.join('.','data',dir_data,file_data), 'rb') as f:
                    all_data = pickle.load(f)  
    else:
        with open(dir_data+'/'+file_data, 'rb') as f:
                    all_data = pickle.load(f)  
    return all_data

def load_sim_params(dir_params,file_params='params.pickle'):
    if 'data' in dir_params:
        with open(dir_params / file_params, 'rb') as f:
            params = pickle.load(f)
    else:
        if 'src/' in os.getcwd():
            with open(os.path.join('..','..','data',dir_params,file_params), 'rb') as f:
                params = pickle.load(f)
        else: 
            with open(os.path.join('.','data',dir_params,file_params), 'rb') as f:
                params = pickle.load(f)
    return params

def load_human_data(dir_data='ext_data',file_data='raw_data_behav.csv',map_epi={1:0,2:1,3:2,4:3,5:4},
                    map_states={1:10,2:0,3:7,4:2,5:4,6:1,7:8,8:9,9:3,10:5,11:6}):

    ## Load data
    all_data = pd.read_csv('./'+dir_data+'/'+file_data)      # import behavioural data as dataframe
    all_data = all_data[all_data['env']==3]                         # exclude all data from second block (surprise block)
    all_data['epi'].replace(map_epi,inplace=True)                   # adjust episode data labels
    all_data['state'].replace(map_states,inplace=True)              # adjust state data labels
    all_data['next_state'].replace(map_states,inplace=True)  
    return all_data

def load_df_stateseq(file_name='load_df_stateseq.pickle'):
    path_edata = '../../ext_data/Rosenberg2021/'+file_name
    with open(path_edata,'rb') as f:
        df_stateseq = pickle.load(f)
    return df_stateseq

def convert_dict_to_csv(path,file='params.pickle'):
    params = load_sim_params(path,file)
    with open((path+'/'+file).replace('.pickle','.csv'),'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=params.keys())
        writer.writeheader()
        writer.writerow(params)
            
def convert_df_to_csv(path,file='all_data.pickle'):
    all_data = load_sim_data(path,file)
    all_data.to_csv((path+'/'+file).replace('.pickle','.csv'),sep='\t')
    
def excludeFailedTrials(data):
    data1 = data.groupby(['subID','epi']).last()
    #data2 = data1['reward'].groupby(['subID']).all()
    data2 = data1['reward'].groupby(['subID']).first()
    if not data2.dtype=='bool':
        data2 = data2.apply(lambda x: bool(x))
    subID_noFail    = data2[data2].index.values
    data_noFail     = data[data['subID'].isin(subID_noFail)]
    return data_noFail

##############################################################################
#       STATS FUNCTIONS                                                     #
##############################################################################
# Steps per episode
def get_steps_per_epi(all_data,range_epi=range(0,1)):
    df_it_stats=[]
    flag_dim=True
    # Extract steps per episode from data
    for i in range(len(all_data)):
        # print(range_epi)
        # print(np.unique(all_data[i].epi))
        all_data[i] = all_data[i][all_data[i].epi >= range_epi[0]]
        all_data[i] = all_data[i][all_data[i].epi <= range_epi[-1]]
        # print(np.unique(all_data[i].epi))
        df_it = all_data[i].groupby(['epi','subID']).size().reset_index(name='counts')
        df_it_stats.append(df_it[['epi','counts']].groupby('epi').agg([np.mean, np.std, np.median, quantile25err, quantile75err, stats.sem, ci_median]))
        df_it_stats[i].columns.set_levels(['steps_per_epi'],level=0,inplace=True)
        # Add nan rows for the episodes that are in range_epi but not in data
        for e in list(set(range_epi).symmetric_difference(df_it_stats[i].index.values)):
            df_add = pd.DataFrame(np.zeros((1,len(df_it_stats[i].columns))),index=[e],columns=df_it_stats[i].columns)
            df_it_stats[i] = pd.concat([df_it_stats[i],df_add])
    
    # Check whether dimensions of the data set statistics agree
    if not np.array([np.shape(df_it_stats[0])==np.shape(df_it_stats[i]) for i in range(len(df_it_stats))]).all(): 
        print('No plots were created because the data sets have different number of trials or episodes.')
        flag_dim=False
    
    return df_it_stats, flag_dim

def get_steps_per_epi2(all_data,range_epi=range(0,1)):
    df_it_stats=[]
    flag_dim=True
    # Extract steps per episode from data
    for i in range(len(all_data)):
        # print(range_epi)
        # print(np.unique(all_data[i].epi))
        all_data[i] = all_data[i][all_data[i].epi >= range_epi[0]]
        all_data[i] = all_data[i][all_data[i].epi <= range_epi[-1]]
        # print(np.unique(all_data[i].epi))
        df_it = all_data[i].groupby(['epi','subID']).size().reset_index(name='counts')
        df_it_stats.append(df_it[['epi','counts']].groupby('epi').agg([np.mean, np.std, np.median, quantile25err, quantile75err, stats.sem, ci_median]))
        df_it_stats[i].columns.set_levels(['steps_per_epi'],level=0,inplace=True)
        # Add nan rows for the episodes that are in range_epi but not in data
        for e in list(set(range_epi).symmetric_difference(df_it_stats[i].index.values)):
            df_add = pd.DataFrame(np.zeros((1,len(df_it_stats[i].columns))),index=[e],columns=df_it_stats[i].columns)
            df_it_stats[i] = pd.concat([df_it_stats[i],df_add])
    
    # Check whether dimensions of the data set statistics agree
    if not np.array([np.shape(df_it_stats[0])==np.shape(df_it_stats[i]) for i in range(len(df_it_stats))]).all(): 
        print('No plots were created because the data sets have different number of trials or episodes.')
        flag_dim=False
    
    return df_it, df_it_stats, flag_dim


# Visits per state
def get_visits_per_state(all_data,epi,all_states):
    df_stateit_stats=[]
    df_stateit_statsplot=[]
    flag_dim=True
    # Extract visits to each state per episode
    for i in range(len(all_data)):
        df_stateit = all_data[i].groupby(['epi','state','subID']).size().reset_index(name='counts')  
        df_stateit_stats.append(df_stateit[['epi','state','counts']].groupby(['epi','state']).agg([np.mean, np.std, np.median, quantile25err, quantile75err]))
        df_stateit_stats[i].columns.set_levels(['visits_per_state'],level=0,inplace=True)
        df_stateit_stats[i] = df_stateit_stats[i].reset_index()
    
        # Filter for the input episode
        df_stateit_statsplot.append(df_stateit_stats[i][df_stateit_stats[i][('epi','')]==epi])
    
        # Add zero entries for the states that were not visited
        for s in all_states:
            if not df_stateit_statsplot[i][('state','')].isin([s]).any():
                add_row = pd.Series([epi,s,0,0,0,0,0],index=df_stateit_statsplot[i].columns)
                df_stateit_statsplot[i] = df_stateit_statsplot[i].append(add_row,ignore_index=True)
    
        # Sort by state and exclude goal state (10)
        df_stateit_statsplot[i] = df_stateit_statsplot[i].sort_values(by=['state'])
        df_stateit_statsplot[i] = df_stateit_statsplot[i][df_stateit_statsplot[i][('state','')]!=10]
    
    # Check whether dimensions of the data set statistics agree
    if not np.array([np.shape(df_stateit_statsplot[0])==np.shape(df_stateit_statsplot[i]) for i in range(len(df_stateit_statsplot))]).all(): 
        print('No plots were created because the data sets have different number of trials or episodes.')
        flag_dim=False
    
    return df_stateit_statsplot, flag_dim

# Cumulative reward/novelty signals

def get_cumulativeM(all_data,M):
    cM_all = []
    for i in range(len(all_data)):
        subIDs = all_data[i]['subID'].unique()
        cM = []
        for s in subIDs:
            dfs = all_data[i][all_data[i]['subID']==s]
            cs = [dfs[M][:j].sum() if j>0 else 0 for j in range(len(dfs))]
            cM.append(cs)
        # cM = {}
        # cR = np.array(cR).reshape((len(subIDs),-1))
        cM_all.append(cM)
    return cM_all

# Actions to escape trap states
def get_actions_to_escape(all_data):
    df_stats_ae      = []
    df_stats_av      = []
    df_stats_ve      = []
    
    for i in range(len(all_data)):
        list_epi    = all_data[i]['epi'].unique()
        traps       = [7,8,9]
        
        # Filter for iterations where agent is in trap state, determine 'switch/stay'
        all_data[i] = all_data[i][(all_data[i]['state']==7) | (all_data[i]['state']==8) | (all_data[i]['state']==9)]
        all_data[i]['switch'] = all_data[i]['next_state'].apply(lambda x: not np.isin(x,traps))
        
        # Get number of actions to escape for each visit to trap
        count_stay = 0
        count_stay_list = []
        for j in range(len(all_data[i])):
            count_stay+=1
            count_stay_list.append(count_stay)
            if all_data[i]['switch'].iloc[j]: #reset every time we find a 'switch'
                count_stay=0
        all_data[i]['number_actions'] = count_stay_list
        all_data[i] = all_data[i][all_data[i]['switch']]
        
        # Get number of stats(actions to escape trap) for each episode
        df_stats_ae_i = all_data[i][['epi','number_actions']].groupby(['epi']).agg([np.mean, np.std, np.median, quantile25err, quantile75err])
        df_stats_ae.append(df_stats_ae_i.sort_index())
        
        # Get the index of each visit to trap state per subID
        count_switch = 0
        count_switch_list = [count_switch]
        for j in range(1,len(all_data[i])):
            count_switch += 1
            if all_data[i]['subID'].iloc[j]!=all_data[i]['subID'].iloc[j-1]:
                count_switch = 0
            count_switch_list.append(count_switch) 
        all_data[i]['id_visit'] = count_switch_list
            
        # Get number of stats(actions to escape trap) for each episode
        df_stats_av_i   = all_data[i][['id_visit','number_actions']].groupby(['id_visit']).agg([np.mean, np.std, np.median, quantile25err, quantile75err])
        df_stats_av.append(df_stats_av_i.sort_index())
        
        # Get number of stats(number of visits to trap) for each episode
        visits_trap     = all_data[i][['subID','epi','id_visit']].groupby(['subID','epi']).count()
        df_stats_ve_i   = visits_trap.groupby('epi').agg([np.mean, np.std, np.median, quantile25err, quantile75err])
        # Fill values that were not present
        for e in list_epi:
            if not np.isin(e,all_data[i]['epi']):
                df_add = pd.DataFrame(np.zeros((1,len(df_stats_ve_i.columns))),index=[e],columns=df_stats_ve_i.columns)
                df_stats_ve_i = pd.concat([df_stats_ve_i,df_add])
        
        df_stats_ve.append(df_stats_ve_i.sort_index())
        
    return all_data, df_stats_ae, df_stats_av, df_stats_ve
            
# Action preference (to be completed)
def get_action_pref(data):
    # Get first action of each subject for each state and episode
    first_actions = data.groupby(['epi','subID','state']).first()

    action_pref = first_actions.groupby('action').size().reset_index(name='counts')
    action_pref_individual = first_actions.groupby(['subID','action']).size().reset_index(name='counts')
    action_pref_states = first_actions.groupby(['state','action']).size().reset_index(name='counts')

    return action_pref, action_pref_individual, action_pref_states


def load_human_params(dir_data,file_params):
    return None

def computeRandomAgent():
    return None


##############################################################################
#       BARPLOT FUNCTION (MULTIPLE DATASETS, DIFFERENT STATS                 #
##############################################################################
# Plot cumulative M
def plot_cumulative(dir_data,M,data_type,range_epi=range(0,5),saveFig=True):
    # Make folder and name for saving fig
    if saveFig:
        if dir_fig=='': 
            #dir_fig = make_dir_fig('./output/'+os.path.basename(os.path.normpath(dir_data[0])))
            dir_fig = make_dir_fig('../../output/'+os.path.basename(os.path.normpath(dir_data[0])))
        else: 
            make_dir_fig(dir_fig)
        save_name      = "-".join(data_type)

    # Load data
    all_data = []
    for i in range(len(dir_data)):
        if data_type[i]=='sim':
            all_data.append(load_sim_data(os.path.basename(os.path.normpath(dir_data[i]))))
        elif data_type[i]=='human':
            all_data.append(load_human_data(os.path.basename(os.path.normpath(dir_data[i]))))
        all_data[i] = all_data[i][all_data[i].epi.isin(range_epi)]
    
    cM = get_cumulativeM(all_data,'reward')
   
    fig, ax = plt.subplots()
    for i in range(len(cM)):
        ax.plt(np.arange(len(cM[i])),cM[i])
    
    return None

# Plot statistics of steps until goal state found for multiple experiments
def plot_stats_barplot(dir_data,data_type,data_name,stats_type,dict_min_steps,dict_rand_exp,range_epi=range(0,5),
                       plot_title='',plot_type='median',dir_fig='',stats_params={},col_rgb=[],excludeFails=True,
                       saveFig=True,errors=True,mini=True,rand=True,logscale=True,individual=True,legend=True):
    
    if not col_rgb:
        col_rgb = len(dir_data)*[colors.to_rgba('grey')]
    
    flag_data   = True
    flag_dim    = True

    # Make folder and name for saving fig
    if saveFig:
        if dir_fig=='': 
            if 'src/' in os.getcwd():
                dir_fig = make_dir_fig('../../output/'+os.path.basename(os.path.normpath(dir_data[0])))
            else:
                dir_fig = make_dir_fig('./output/'+os.path.basename(os.path.normpath(dir_data[0])))
        else: 
            make_dir_fig(dir_fig)
        save_name      = "-".join(data_type)+'_'+stats_type
           
    # Init legend string
    legend_labels  = []
    legend_handles = []
    
    # Load data
    all_data = []
    for i in range(len(dir_data)):
        if data_type[i]=='sim':
            all_data.append(load_sim_data(os.path.basename(os.path.normpath(dir_data[i]))))
        elif data_type[i]=='human':
            all_data.append(load_human_data(os.path.basename(os.path.normpath(dir_data[i]))))
        all_data[i] = all_data[i][all_data[i].epi.isin(range_epi)] ## this is where the data disappears
   
        # Exclude fails
        if excludeFails:
            all_data[i] = excludeFailedTrials(all_data[i]) 
            if len(all_data[i])==0: 
                print('No plots were created because data from at least one data set not found.')
                flag_data=False
    
    if flag_data==True:
        if stats_type=='steps_per_epi':
            df_it_stats, flag_dim = get_steps_per_epi(all_data,range_epi)
        elif stats_type=='visits_per_state':
            if 'epi' in stats_params:
                epi=stats_params['epi']
            else:
                epi=0
            set_states=[int(k) for k in list(dict_min_steps[i].keys())]
            df_it_stats, flag_dim = get_visits_per_state(all_data,epi,set_states)
            
    if flag_dim and flag_data:
        fig, ax = plt.subplots()
        
        number_bar  = len(df_it_stats)
        width_bar   = np.round(1/(number_bar+1),4)
        
        x_bar   = [np.round(np.arange(len(df_it_stats[0]))-(number_bar-1)/2*width_bar,4)]
        for i in range(1,number_bar):
            x_bar.append(np.array([x+width_bar for x in x_bar[i-1]]))
        
        # Plot stats for each episode
        for i in range(len(x_bar)):
            p_bar = ax.bar(x_bar[i],df_it_stats[i][(stats_type,plot_type)],width=width_bar,color=col_rgb[i])
            
            if legend: 
                legend_handles  += [p_bar[0]]
                legend_labels   += [f"{data_name[i]}"]
            
            if errors:
                if plot_type=='median':
                    yerr = [df_it_stats[i][stats_type,'quantile25err'],df_it_stats[i][stats_type,'quantile75err']]
                elif plot_type=='mean':
                    yerr = [df_it_stats[i][stats_type,'std'],df_it_stats[i][stats_type,'std']]
                p_err = ax.errorbar(x_bar[i],df_it_stats[i][stats_type,plot_type],yerr,
                                    ecolor='k',elinewidth=1,linestyle='',capsize=4)
            
        if saveFig: 
            save_name += f'_{plot_type}'
            if errors:
                save_name += '_withErrors'
                
        if stats_type=='steps_per_epi':
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Median steps to goal')
        elif stats_type=='visits_per_state':
            ax.set_xlabel('States')
            ax.set_ylabel(f'Median visits in episode {epi+1}')
            
        if (mini or rand) and stats_type=='steps_per_epi':
            starting_states =[]
            for i in range(len(dir_data)):
                data_firsts = all_data[i].groupby(['epi']).first()    
                starting_states.append(data_firsts['state'].values)
            
                # x_combined = np.zeros(len(x_bar_sim)+len(x_bar_behav)+2)
                # x_combined[1:-2:2] = (x_bar_sim-width_bar/2)
                # x_combined[2::2] = (x_bar_behav+width_bar/2)
                # x_combined[0] = x_bar_sim[0]-0.25
                # x_combined[-1] = x_bar_behav[-1]+0.25
            
            if mini:
                min_steps = []
                for i in range(len(starting_states)):
                    min_steps.append([dict_min_steps[i][f'{j}'] for j in starting_states[i]])
                
                    # y_combined = np.zeros(np.size(x_combined))
                    # y_combined[1:-2:2] = min_steps_behav
                    # y_combined[2::2] = min_steps_behav
                    # y_combined[0] = min_steps_behav[0]
                    # y_combined[-1] = min_steps_behav[-1]
                    
                    p_min = [ax.plot([x_bar[i][j]-width_bar/2,x_bar[i][j]+width_bar/2],[min_step,min_step],'k--') for j,min_step in enumerate(min_steps[i])]
                
                if legend:
                    legend_labels += ["minimum steps"]
                    legend_handles += p_min[0]
                
            if rand:
                rand_exp = []
                for i in range(len(starting_states)):
                    rand_exp.append([dict_rand_exp[i][f'{j}'] for j in starting_states[i]])
                    
                    # y_combined_rand = np.zeros(np.size(x_combined))
                    # y_combined_rand[1:-2:2] = rand_exp_behav
                    # y_combined_rand[2::2] = rand_exp_behav
                    # y_combined_rand[0] = rand_exp_behav[0]
                    # y_combined_rand[-1] = rand_exp_behav[-1]
                    
                    p_rand = [ax.plot([x_bar[i][j]-width_bar/2,x_bar[i][j]+width_bar/2],[rand,rand],':',color='k') for j,rand in enumerate(rand_exp[i])]
                
                if legend:
                    legend_labels += ["random exploration"]
                    legend_handles += p_rand[0]
                
        if individual and stats_type=='steps_per_epi':
            for i in range(len(dir_data)):
                df_it = all_data[i].groupby(['epi','subID']).size().reset_index(name='counts')
                for j in range(len(x_bar[i])):   
                    df = df_it.query(f'epi=={j}')
                    p_ind = ax.plot(x_bar[i][j]*np.ones(len(df)),df['counts'],'o',color='k',markersize=1.5)
            
            if legend:
                legend_labels += ["individual data points"]
                legend_handles += [p_ind[0]]
        
        if legend:      
            ax.legend(legend_handles,legend_labels,loc='upper right')
        
        labels=[f'{int(i+1)}' if i>=0 else '' for i in ax.get_xticks()]
        #ax.set_xticks((x_bar_sim+x_bar_behav)/2)
        ax.set_xticklabels(labels)
        #ax.set_xlim([x_bar_sim[0]-0.25,x_bar_behav[-1]+0.25])
        ax.set_title(plot_title)
        
        if logscale:
            ax.set_yscale('log')
            save_name += '_logScale'
            
        if excludeFails:
            save_name += '_exclFail'
        
        if saveFig:
            plt.savefig(dir_fig+'/'+save_name+'.svg')
            plt.savefig(dir_fig+'/'+save_name+'.eps')
            
        #plt.close()

def plot_wrapper(dir_data,data_type,data_name,stats_type,plot_args,plot_title,dir_fig,select_range,
                 saveFig=True,legend=True,excludeFails=True,logscale=True):
    flag_data   = True      # TRUE if there is at least one data point per data set to be plotted
    flag_dim    = True      # TRUE if the data sets to be plotted match in dimension
    limit_epi = 10
    
    # Make folder and name for saving fig
    if saveFig:
        if dir_fig=='': 
            dir_fig = make_dir_fig('./output/'+os.path.basename(os.path.normpath(dir_data[0])))
        else: 
            make_dir_fig(dir_fig)
        save_name      = "-".join(data_type)+'_'+stats_type
        
    # Init legend string
    if legend:
        legend_labels  = []
        legend_handles = []
    
    # Load data
    all_data = []
    for i in range(len(dir_data)):
        if data_type[i]=='sim':
            #all_data.append(load_sim_data(dir_data[i]))
            all_data.append(load_sim_data(os.path.basename(os.path.normpath(dir_data[i]))))
        elif data_type[i]=='human':
            #all_data.append(load_human_data(dir_data[i]))
            all_data.append(load_human_data(os.path.basename(os.path.normpath(dir_data[i]))))
        all_data[i] = all_data[i].iloc[0:min(limit_epi,len(all_data[i]))]
    

        # Exclude fails
        if excludeFails:
            all_data[i] = excludeFailedTrials(all_data[i]) 
            if len(all_data[i])==0: 
                print('No plots were created because data from at least one data set not found.')
                flag_data=False
    
    if flag_data==True:
        fig, ax, legend_handles, legend_labels = funcnames[stats_type](all_data,plot_args,legend_handles,legend_labels,data_name,select_range)
        
        if legend:      
            ax.legend(legend_handles,legend_labels,loc='upper right')
    
        #labels=[f'{i+1}' for i in x_bar]
        #ax.set_xticks((x_bar_sim+x_bar_behav)/2)
        #ax.set_xticklabels(labels)
        #ax.set_xlim([x_bar_sim[0]-0.25,x_bar_behav[-1]+0.25])
        ax.set_title(plot_title)
    
        if logscale:
            ax.set_yscale('log')
            save_name += '_logScale'
        
        if excludeFails:
            save_name += '_exclFail'
    
        if saveFig:
            plt.savefig(dir_fig+'/'+save_name+'.svg')
            plt.savefig(dir_fig+'/'+save_name+'.eps')
            

def plot_traces(dir_data,signal='M',plot_title='',dir_fig='',col_rgb=[],align=False,movmean=False,mov=5,
                plotEpi=True,excludeFails=False,saveFig=True,logscale=False,legend=False):
    # signal = 'M','TD'
   
    if not col_rgb:
        col_rgb=len(dir_data)*[colors.to_rgba('grey')]
    
    flag_data   = True
    flag_dim    = True
    # Make folder and name for saving fig
    if saveFig:
        if dir_fig=='': 
            dir_fig = make_dir_fig('../output/'+os.path.basename(os.path.normpath(dir_data[0])))
        else: 
            make_dir_fig(dir_fig)
        save_name      = 'sim'+'_'+signal
           
    # Init legend string
    legend_labels  = []
    legend_handles = []
    
    # Load data
    all_data = []
    for i in range(len(dir_data)):
        #all_data.append(load_sim_data(dir_data[i]))
        all_data.append(load_sim_data(os.path.basename(os.path.normpath(dir_data[i]))))
            
        # Exclude fails
        if excludeFails:
            all_data[i] = excludeFailedTrials(all_data[i]) 
            if len(all_data[i])==0: 
                print('No plots were created because data from at least one data set not found.')
                flag_data=False
    
    if flag_data==True:
        
        i=0
        
        list_sub = all_data[i]['subID'].unique()
        cols = plt.cm.viridis(np.linspace(0,1,len(list_sub)))
        
        fig, ax = plt.subplots(num=-1)
        ax.plot(np.linspace(-10,250,100),np.zeros(100),'k--')
        
        for j in list_sub:
            
            single_sub          = all_data[i][all_data[i]['subID']==j]
            single_sub          = single_sub.reset_index(drop=True)
            single_sub_index    = single_sub.index.to_numpy()
            
            # Align plots to the first time that the signal is nonzero
            if align:
                single_sub_filtered = single_sub[(~np.isnan(single_sub[f'mod-{i}: {signal}']))&(single_sub[f'mod-{i}: {signal}']!=0)]
                
                # start_index         = single_sub_filtered.index[0]
                # single_sub_index    = single_sub_index[single_sub_index>=start_index]
                # single_sub          = single_sub[single_sub.index>=start_index]
                
                single_sub          = single_sub_filtered.reset_index(drop=True)
                single_sub_index    = single_sub.index.to_numpy()
                
                save_name += '_aligned'   
            #if cumulate:
            
            # Apply moving mean before plotting
            if movmean:
                single_sub_signal = single_sub[f'mod-{i}: {signal}'].rolling(mov).mean()
                save_name += f'_movmean-{mov}'
            else:
                single_sub_signal = single_sub[f'mod-{i}: {signal}']
        
            # Plot into shared figure
            ax.plot(single_sub_index,single_sub_signal,color=cols[j])
            
            # Make single-subject figure
            fig_sub, ax_sub = plt.subplots(num=j)
            ax_sub.plot(np.linspace(-10,50,100),np.zeros(100),'k--')
            ax_sub.plot(single_sub_index,single_sub_signal,color=cols[j])
            
            # Plot episodes
            if plotEpi:
                single_sub  = single_sub.reset_index()
                list_epi    = single_sub[['index','epi']].groupby('epi').first()
                for e in list_epi['index']:
                    ax_sub.plot((e,e),[-1,1],'k-')
                    
            ax_sub.set_title(plot_title+f' for subID {j}')
           
            if saveFig:
                plt.savefig(dir_fig+'/'+save_name+f'_subID-{j}.svg')
                plt.savefig(dir_fig+'/'+save_name+f'_subID-{j}.eps')
                
        ax.set_title(plot_title)
        
        if logscale:
            ax.set_yscale('log')
            save_name += '_logScale'
            
        if excludeFails:
            save_name += '_exclFail'
        
        if saveFig:
            plt.figure(-1)
            plt.savefig(dir_fig+'/'+save_name+'.svg')
            plt.savefig(dir_fig+'/'+save_name+'.eps')
            
def plot_actions_escape_per_epi(all_data,plot_args,legend_handles,legend_labels,data_name,select_range):
    
    processed_data, df_stats_ae, df_stats_av, df_stats_ve = get_actions_to_escape(all_data)
    df_stats = df_stats_ae
    
    ## Plot color
    my_cmap     = cm.get_cmap('Dark2')
    col_behav   = colors.to_rgba('grey')
    my_cols     = [my_cmap.colors[2],my_cmap.colors[1],my_cmap.colors[0],col_behav]  
    
    # Plot number of actions to escape trap (stats across episodes)
    fig, ax = plt.subplots()
        
    for i in range(len(df_stats)):
        
        plt.errorbar(df_stats[i].index.values[select_range[0]:select_range[1]],
                    df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','median')],
                    yerr=[df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','quantile25err')],
                          df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','quantile75err')]],
                    ls=':',capsize=4,
                    color=my_cols[i],ecolor=my_cols[i],mfc=my_cols[i],mec=my_cols[i])
        
        a = ax.plot(df_stats[i].index.values[select_range[0]:select_range[1]],
                    df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','median')],
                    's',color=my_cols[i])
        legend_handles.append(a[0])
        legend_labels.append(data_name[i])
        
    ax.set_xticks(np.arange(0,5))
    ax.set_xticklabels([str(z) for z in np.arange(1,6)])
    ax.set_xlabel('Episodes')
    
    #ax.set_yticks(np.arange(0,9))
    #ax.set_yticklabels([str(z) for z in np.arange(9)])
    ax.set_ylabel('Actions to leave trap state')
        
    return fig, ax, legend_handles, legend_labels

        
def plot_actions_escape_per_visit(all_data,plot_args,legend_handles,legend_labels,data_name,select_range):
    
    processed_data, df_stats_ae, df_stats_av, df_stats_ve = get_actions_to_escape(all_data)
    df_stats = df_stats_av
    
    ## Plot color
    my_cmap     = cm.get_cmap('Dark2')
    col_behav   = colors.to_rgba('grey')
    my_cols     = [my_cmap.colors[2],my_cmap.colors[1],my_cmap.colors[0],col_behav]  
    
    # Plot number of actions to escape trap (stats across episodes)
    fig, ax = plt.subplots()
        
    for i in range(len(df_stats)):
        
        plt.errorbar(df_stats[i].index.values[select_range[0]:select_range[1]],
                    df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','median')],
                    yerr=[df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','quantile25err')],
                          df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','quantile75err')]],
                    ls=':',capsize=4,
                    color=my_cols[i],ecolor=my_cols[i],mfc=my_cols[i],mec=my_cols[i])
        
        a = ax.plot(df_stats[i].index.values[select_range[0]:select_range[1]],
                    df_stats[i].iloc[select_range[0]:select_range[1]][('number_actions','median')],
                    's',color=my_cols[i])
        legend_handles.append(a[0])
        legend_labels.append(data_name[i])
    
    #ax.set_xticks(np.arange(0,5))
    #ax.set_xticklabels([str(z) for z in np.arange(1,6)])
    ax.set_xlabel('Visits to trap states')
    
    #ax.set_yticks(np.arange(0,9))
    #ax.set_yticklabels([str(z) for z in np.arange(9)])
    ax.set_ylabel('Actions to leave trap state')
        
    return fig, ax, legend_handles, legend_labels


def plot_visits_trap_per_epi(all_data,plot_args,legend_handles,legend_labels,data_name,select_range):
    
    processed_data, df_stats_ae, df_stats_av, df_stats_ve = get_actions_to_escape(all_data)
    df_stats = df_stats_ve
    
    ## Plot color
    my_cmap     = cm.get_cmap('Dark2')
    col_behav   = colors.to_rgba('grey')
    my_cols     = [my_cmap.colors[2],my_cmap.colors[1],my_cmap.colors[0],col_behav]  
    
    # Plot number of actions to escape trap (stats across episodes)
    fig, ax = plt.subplots()
        
    for i in range(len(df_stats)):
        
        plt.errorbar(df_stats[i].index.values,
                     df_stats[i][('id_visit','median')],
                     yerr=np.squeeze([df_stats[i][('id_visit','quantile25err')],
                           df_stats[i][('id_visit','quantile75err')]]),
                     ls=':',capsize=4,
                     color=my_cols[i],ecolor=my_cols[i],mfc=my_cols[i],mec=my_cols[i])
        
        a = ax.plot(df_stats[i].index.values,
                    df_stats[i][('id_visit','median')],
                    's',color=my_cols[i])
        
        legend_handles.append(a[0])
        legend_labels.append(data_name[i])  
    
    ax.set_xticks(np.arange(0,5))
    ax.set_xticklabels([str(z) for z in np.arange(1,6)])
    ax.set_xlabel('Episodes')
    
    #ax.set_yticks(np.arange(0,9))
    #ax.set_yticklabels([str(z) for z in np.arange(9)])
    ax.set_ylabel('Visits to trap state')
        
    return fig, ax, legend_handles, legend_labels

funcnames = {'steps_per_epi':plot_stats_barplot, 
             'visit_per_state':plot_stats_barplot,
             'actions_escape_per_epi':plot_actions_escape_per_epi,
             'visits_trap_per_epi':plot_visits_trap_per_epi,
             'actions_escape_per_visit':plot_actions_escape_per_visit}

   


     