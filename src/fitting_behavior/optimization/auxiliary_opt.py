import fitting_behavior.optimization.base_params_opt as bpo
from models.mb_agent.mb_surnor import *

def run_opt_sim_mbnor(res,agent_num,epi_num,base_params,sim_name,plot_title='',rec_type='advanced',dir_data='',dir_fig='',seeds=None,path_surnor=None):

    # Convert to dictionary
    if not isinstance(res,dict):
        res_params = dict([(n,p) for n,p in zip(res.space.dimension_names,res.x)])
    else:
        res_params = res

    # Update RL model parameters
    if path_surnor is not None:
        sim_params_nor = import_params_surnor(path=path_surnor)
    else:
        sim_params_nor = import_params_surnor()
    sim_params_nor['lambda_R']   = 0
    sim_params_nor['beta_N1']    = 1

    sim_params_nor.update(res_params)
    if 'beta_1r' in res_params.keys():
        sim_params_nor['beta_1'] = beta1r_to_beta1(res_params['beta_1r'])
    
    # Update experiment parameters 
    if not seeds: seeds = list(range(agent_num))

    sim_params_exp = base_params
    sim_params_exp['sim_name']      = sim_name
    sim_params_exp['rec_type']      = rec_type
    sim_params_exp['number_trials'] = agent_num
    sim_params_exp['number_epi']    = epi_num
    sim_params_exp['seeds']         = seeds
    sim_params_exp['x0']            = bpo.all_zero_x0(agent_num,epi_num)
    sim_params_exp['max_it']        = 10000  

    # Update novelty parameters
    if 'update_type' in base_params['h'].keys():
        sim_params_exp['h']['update_type'] = base_params['h']['update_type']

    nov_params = ['eps_leak','eps_leak1','eps_leak2','alph_leak','alph_leak1','alph_leak2']

    if ('eps_leak1' in sim_params_nor.keys() and 'eps_leak2' in sim_params_nor.keys()):
        sim_params_exp['h']['eps_leak'] = [1, 1]
    if ('alph_leak1' in sim_params_nor.keys() and 'alph_leak2' in sim_params_nor.keys()):
        sim_params_exp['h']['alph_leak'] = [0, 0]

    for kv, vv in sim_params_nor.items():
        if kv in nov_params:
            if 'eps_leak' in kv:
                if kv=='eps_leak':
                    sim_params_exp['h']['eps_leak'] = [vv]
                elif kv=='eps_leak1':
                    sim_params_exp['h']['eps_leak'][0] = vv
                elif kv=='eps_leak2':
                    sim_params_exp['h']['eps_leak'][1] = vv
            elif 'alph_leak' in kv:
                if kv=='alph_leak':
                    sim_params_exp['h']['alph_leak'] = [vv]
                elif kv=='alph_leak1':
                    sim_params_exp['h']['alph_leak'][0] = vv
                elif kv=='alph_leak2':
                    sim_params_exp['h']['alph_leak'][1] = vv
            else:
                sim_params_exp['h'][kv] = vv
        elif kv=='w_cnov':
            sim_params_exp['w'] = [1-vv, vv]
        elif kv in sim_params_exp.keys() and isinstance(sim_params_exp[kv],list): 
            sim_params_exp[kv] = [vv]
        else:                                         
            sim_params_exp[kv] = vv

    # Run experiment
    _, _, _, dir_data = run_surnor_exp(sim_params_exp,sim_params_nor,saveData=True,dirData=dir_data,verbose=False)
    print(f'Simulation data saved in: {dir_data}')

    return dir_data