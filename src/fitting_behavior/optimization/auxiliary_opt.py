import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from skopt import load
from skopt.plots import plot_evaluations, plot_convergence, plot_objective

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.models.mf_agent.experiment as e

import src.fitting_behavior.optimization.base_params_opt as bpo
from src.models.mb_agent.mb_surnor import *
# import src.hybrid.hybrid_ac_nor_parallel as hyb
import src.models.hybrid_agent.hybrid_ac_nor as hyb

# Load optimization results
def load_opt(folder_data, name_data):
    res = load(folder_data+name_data)
    return res
        
# Plot optimization results: points evaluated, convergence, approximation of the objective function
# Plots single optimization result or multiple optimization results (but in separate plots)
def plot_optstats(res,name,folder_figs):
    plot_evaluations(res, bins=10) 
    plt.savefig(folder_figs+f'{name}_hist.svg',bbox_inches='tight')
    plt.savefig(folder_figs+f'{name}_hist.eps',bbox_inches='tight')
    plt.close()
        
    plot_convergence(res,yscale='log') # if res is a list, all convergence traces and their mean are plotted
    plt.savefig(folder_figs+f'{name}_conv.svg',bbox_inches='tight')
    plt.savefig(folder_figs+f'{name}_conv.eps',bbox_inches='tight')
    plt.close()

    plot_objective(res)
    plt.savefig(folder_figs+f'{name}_obj.svg',bbox_inches='tight')
    plt.savefig(folder_figs+f'{name}_obj.eps',bbox_inches='tight')
    plt.close()

# Multiple optimization results
def plot_multi_optstats(ress,name,folder_figs):

    fig, ax = plt.subplots()

    my_cmap  = cm.get_cmap('Pastel1') #Dark2 

    for i in range(len(ress)):
        plot_convergence(ress[i],ax=ax,color=my_cmap(i),yscale='log') 
    
    plt.savefig(folder_figs+f'{name}_conv.svg',bbox_inches='tight')
    plt.savefig(folder_figs+f'{name}_conv.eps',bbox_inches='tight')
    plt.close()

# Run simulation with optimal params
def run_opt_sim(res,agent_num,epi_num,base_params,sim_name,plot_title,folder_data='',folder_figs='',seeds=None):
    res_params = dict([(n,[p]) for n,p in zip(res.space.dimension_names,res.x)])
    sim_params = base_params.copy() #bpo.base_params_rAC11.copy()
    sim_params.update(res_params)

    # Set seeds
    if not seeds: seeds = list(range(agent_num))

    sim_params['sim_name']      = sim_name
    sim_params['rec_type']      = 'advanced2'
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
        
    exp = e.experiment(sim_params,flag_saveData=True,dataFolder=folder_data)
    exp.runExperiment()

    # Plot simulation 
    # dir_data        = exp.dataFolder
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},dir_fig=folder_figs)
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},dir_fig=folder_figs)
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False,dir_fig=folder_figs)
      
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False,dir_fig=folder_figs)
    
    # plt.close('all')

def run_opt_sim_tree(res,agent_num,epi_num,base_params,sim_name,plot_title,rec_type='advanced2',dir_data='',dir_fig='',seeds=None):
    
    sim_params = base_params.copy()
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    sim_params.update(res_params)

    # Set seeds
    if not seeds: seeds = list(range(agent_num))
    
    sim_params['sim_name']      = sim_name
    sim_params['rec_type']      = rec_type
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
        
    exp = e.experiment(sim_params,flag_saveData=True,dataFolder=dir_data)
    exp.runExperiment()

    # Plot simulation 
    dir_data = exp.dataFolder
    print(dir_data)
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_stepstree6],[dict_rand_exptree6],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={})
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_stepstree6],[dict_rand_exptree6],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={})
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_stepstree6],[dict_rand_exptree6],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False)
      
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_stepstree6],[dict_rand_exptree6],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False)
    
    # plt.close('all')


def run_opt_sim_mbnor(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced',dir_data='',dir_fig='',seeds=None):

    #folder_data,folder_figs
    sim_params_nor = import_params_surnor()
    sim_params_nor['lambda_R']   = 0
    sim_params_nor['beta_N1']    = 1
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    sim_params_nor.update(res_params)
    if 'beta_1r' in res_params.keys():
        sim_params_nor['beta_1'] = beta1r_to_beta1(res_params['beta_1r'])

    # Set seeds
    if not seeds: seeds = list(range(agent_num))

    sim_params_exp = base_params
    sim_params_exp['sim_name']      = sim_name
    sim_params_exp['rec_type']      = rec_type
    sim_params_exp['number_trials'] = agent_num
    sim_params_exp['number_epi']    = epi_num
    sim_params_exp['seeds']         = seeds
    sim_params_exp['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params_exp['max_it']        = 30000
        
    _, _, _, dir_data = run_surnor_exp(sim_params_exp,sim_params_nor,saveData=True,dirData=dir_data,verbose=False)

    # Plot simulation 
    print(dir_data)
    # dir_fig = dir_data.replace('data','output')
    # sl.make_dir(dir_fig)
      
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},dir_fig=dir_fig)
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},dir_fig=dir_fig)
       
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'steps_per_epi',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False,dir_fig=dir_fig)
      
    # vis.plot_stats_barplot([dir_data],['sim'],['Simulation'],'visits_per_state',[dict_min_steps11],[dict_rand_exp11],plot_title=plot_title,plot_type='median',
    #                         col_rgb=[colors.to_rgba('darkred')],stats_params={},excludeFails=False,dir_fig=dir_fig)
    
    # plt.close('all')
    return dir_data

def run_optsim_Nkheuristic(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced2',dir_data='',dir_fig='',seeds=None):
    # Set optimized params
    sim_params = base_params.copy()
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    if 'T' in res_params.keys(): res_params['hT'] = res_params.pop('T')
    sim_params.update(res_params)
    if 'hT' in res_params.keys():
        # if 'n_buffer' in sim_params['h'].keys():
        #     sim_params['h']['n_buffer'] = res_params['T']
        # else:
        #     sim_params['h'] = sim_params['h']|{'n_buffer':res_params['T']}
        if not 'h' in sim_params.keys() or len(sim_params['h'])==0:
            sim_params['h'] = {'n_buffer':res_params['hT']}
        else:
            sim_params['h'] = sim_params['h']|{'n_buffer':res_params['hT']}

    # Correct format
    key_list = ['tauM','RM','c_w0','a_w0','gamma','c_alph','a_alph','c_lam','a_lam','temp']
    for k in key_list:
        if not isinstance(sim_params[k],list):
            sim_params[k]=[sim_params[k]]

    # Set seeds
    if not seeds: seeds = list(range(agent_num))
    
    # Set remaining params
    sim_params['sim_name']      = sim_name
    sim_params['rec_type']      = rec_type
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
        
    exp = e.experiment(sim_params,flag_saveData=True,dataFolder=dir_data)
    exp.runExperiment()

    # Plot simulation 
    dir_data = exp.dataFolder
    print(dir_data)

def run_optsim_Nktemp(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced2',dir_data='',dir_fig='',seeds=None):
    
    sim_params = base_params.copy()
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    sim_params.update(res_params)

    # Set seeds
    if not seeds: seeds = list(range(agent_num))
    
    sim_params['sim_name']      = sim_name
    sim_params['rec_type']      = rec_type
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
    sim_params['ntype']         = 'N-ktemp'
        
    exp = e.experiment(sim_params,flag_saveData=True,dataFolder=dir_data)
    exp.runExperiment()

    # Plot simulation 
    dir_data = exp.dataFolder
    print(dir_data)
       
def run_optsim_nac_tree(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced2',dir_data='',dir_fig='',seeds=None):
    # Set optimized params
    sim_params = base_params.copy()
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    sim_params.update(res_params)

    # Correct format
    key_list = ['tauM','RM','c_w0','a_w0','gamma','c_alph','a_alph','c_lam','a_lam','temp']
    for k in key_list:
        if not isinstance(sim_params[k],list):
            sim_params[k]=[sim_params[k]]

    # Set seeds
    if not seeds: seeds = list(range(agent_num))
    
    # Set remaining params
    sim_params['sim_name']      = sim_name
    sim_params['rec_type']      = rec_type
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
        
    exp = e.experiment(sim_params,flag_saveData=True,dataFolder=dir_data,verbose=False)
    exp.runExperiment()

    # Plot simulation 
    dir_data = exp.dataFolder
    print(dir_data)

    return dir_data

def run_optsim_hybrid_tree(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced2',dir_data='',dir_fig='',seeds=None):
    # Set optimized params
    sim_params = base_params.copy()
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res
    sim_params.update(res_params)

    # Correct format
    key_list = ['tauM','RM','c_w0','a_w0','gamma','c_alph','a_alph','c_lam','a_lam','temp']
    for k in key_list:
        if not isinstance(sim_params[k],list):
            sim_params[k]=[sim_params[k]]

    # Set seeds
    if not seeds: seeds = list(range(agent_num))
    
    # Set remaining params
    sim_params['sim_name']      = sim_name
    sim_params['number_trials'] = agent_num
    sim_params['number_epi']    = epi_num
    sim_params['seeds']         = seeds
    sim_params['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params['max_it']        = 30000
    sim_params['decision_weights'] = [sim_params['decision_weights'][0]]*epi_num
    sim_params['rec_type']      = rec_type
    sim_params['mb_rec_type']   = 'advanced' if rec_type=='advanced2' else 'basic'
    sim_params['mf_rec_type']   = rec_type

    # Separate overlapping params
    overlap = ['rec_type','k','ntype','h','w']
    params_exp = sim_params.copy()
    params_mb = sim_params.copy()
    params_mf = sim_params.copy()
    for i in range(len(overlap)):
        if f'mf_{overlap[i]}' in params_mf.keys(): params_mf[overlap[i]] = params_mf.pop(f'mf_{overlap[i]}')
        if f'mb_{overlap[i]}' in params_mb.keys(): params_mb[overlap[i]] = params_mb.pop(f'mb_{overlap[i]}')
        if f'mb_{overlap[i]}' in params_exp.keys(): params_exp[overlap[i]] = params_exp.pop(f'mb_{overlap[i]}')
        
    _, _, _, _, _, dir_data = hyb.run_hybrid_exp(params_exp,params_mb,params_mf,verbose=False,saveData=True,returnData=False,dirData=dir_data)
    print(dir_data)

    return dir_data
