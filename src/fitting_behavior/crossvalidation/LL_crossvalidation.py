import numpy as np
import pandas as pd
import json
import os
import json
from argparse import ArgumentParser

import utils.saveload as sl
import fitting_behavior.optimization.base_params_opt as bpo
from models.mb_agent.mb_surnor import import_params_surnor
from fitting_behavior.mle.mle_fit_sequential import preprocess_micedata

from fitting_behavior.mle.mle_fit_sequential import mle_fit
from fitting_behavior.mle.mle_fit_parallel import mle_fit_parallel
from fitting_behavior.mle.LL_nor import ll_nor

######################################################################################################################
# Helper function for loading mouse data 
def load_mouse_data(P, base_path):

    UnrewNames  = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
    RewNames    = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
    AllNames    = RewNames+UnrewNames
    d = []
    for i in range(len(AllNames)):
        # dir = sl.get_datapath().replace('data','ext_data')+'Rosenberg2021/'
        dir = f'{base_path}/ext_data/Rosenberg2021/'
        file=f'{AllNames[i]}-stateseq.pickle'
        df_i = preprocess_micedata(dir,file,P,subID=AllNames[i],epi=0)
        d.append(df_i)
    all_data = pd.concat(d,ignore_index=True)
    all_data = all_data.loc[all_data.epi==0]

    subs = np.unique(all_data['subID'])
    all_data_subs = []
    for i in range(len(subs)):
        data_i = all_data[all_data.subID==subs[i]].reset_index(drop=True)
        if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
            data_i = data_i.iloc[:-1]
        if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
            data_i = data_i.iloc[1:]
        all_data_subs.append(data_i)

    return all_data_subs

######################################################################################################################
# Helper function for loading default parameters
def load_default_params(alg_type, level, base_path, kwargs=None):

    notrace     = 'notrace' in alg_type
    center      = 'center' in alg_type
    center_type = alg_type.split('center-')[1].split('_')[0].split('-')[0] if center else ''

    update_type = 'leaky' if ('leaky' in alg_type) else 'fixed' if ('fixed' in alg_type) else 'var'

    if 'multinov' in alg_type:
        notrace         = kwargs['notrace']
        center          = kwargs['center']
        center_type     = kwargs['center_type']
        join_levels     = False
    
    else:
        notrace             = 'notrace' in alg_type
        center              = 'center' in alg_type
        center_type         = alg_type.split('center-')[1].split('_')[0].split('-')[0] if center else ''
        join_levels         = True
    
    filter_duplicates   = 'noduplicates' in alg_type
    maze_norm           = 'mazenorm' in alg_type
    comp_norm           = False if 'nocompnorm' in alg_type else True

    if 'nor' in alg_type: 
        if 'hnor' in alg_type:
            if not isinstance(level,list):
                level = [level]
            params = bpo.baseparams_h1mbnor_eps1(level,join_levels=join_levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type,filter_duplicates=filter_duplicates,maze_norm=maze_norm,comp_norm=comp_norm)
        else:
            params = bpo.base_params_mbnortree_exp.copy()
        params_surnor = import_params_surnor(path=f'{base_path}/src/mbnor/')
        params.update(params_surnor)
        params['h']['update_type'] = update_type
        if 'eps1' in alg_type:
            params['h']['eps_leak'] = [1]
        if 'nonleaky' in alg_type:
            params['h']['alph_leak'] = 0

    elif 'nac' in alg_type:
        if 'hnac' in alg_type:
            params = bpo.baseparams_h1nac_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type=update_type)
        else: 
            params = bpo.base_params_nACtree.copy()
        if 'gv' in alg_type:
            params['agent_types'] = ['gn']
        if 'goi' in alg_type: 
            params['ntype'] = 'hN-k'
        if 'kpop' in alg_type:      params['ntype'] = 'N-kpop'
        elif 'kmix' in alg_type:    params['ntype'] = 'N-kmix'
        elif 'oi-only' in alg_type: params['agent_type'] = ['oi']
        params['h']['update_type'] = update_type
        if 'eps1' in alg_type:
            params['h']['eps_leak'] = [1]
        if 'nonleaky' in alg_type:
            params['h']['alph_leak'] = 0

    elif 'hybrid' in alg_type:
        if 'hhybrid' in alg_type:
            mb_type = 'hnor'
            mf_type = 'hnac-gn'
            params = bpo.baseparams_all_hhybrid_comb(mb_type,mf_type,levels=[level],notrace=notrace,center=center,center_type=center_type,path_surnor=f'{base_path}/src/mbnor/',update_type=update_type)
        else:
            params = bpo.baseparams_hybrid_comb(path_surnor=f'{base_path}/src/mbnor/')
        params['mb_h']['update_type'] = update_type
        params['mf_h']['update_type'] = update_type
        if 'eps1' in alg_type:
            params['mb_h']['eps_leak'] = [1]
            params['mf_h']['eps_leak'] = [1]
        if 'nonleaky' in alg_type:
            params['mb_h']['alph_leak'] = 0
            params['mf_h']['alph_leak'] = 0

    return params

######################################################################################################################
# Helper function to update the default parameters with fitted params
def update_default_params(alg_type, default_params, fitted_params):

    var_name    = list(fitted_params['var_name'].values)
    var_value   = list(fitted_params['mle_var'].values)

    h_vars      = ['eps_leak','eps_leak1','eps_leak2','alph_leak','alph_leak1','alph_leak2']

    params      = default_params.copy()

    if ('eps_leak1' in var_name and 'eps_leak2' in var_name):
        params['h']['eps_leak'] = [1, 1]
    if ('alph_leak1' in var_name and 'alph_leak2' in var_name):
        params['h']['alph_leak'] = [0, 0]


    if 'hybrid' in alg_type:
        for vn, vv in zip(var_name, var_value):
            if vn in h_vars:
                if vn=='eps_leak':
                    params['mb_h'][vn] = [vv]
                    params['mf_h'][vn] = [vv]
                else:
                    params['mb_h'][vn] = vv
                    params['mf_h'][vn] = vv
            elif vn=='w_cnov':
                params['w'] = [1-vv, vv]
            elif isinstance(params[vn],list): 
                params[vn] = [vv]
            else:                                         
                params[vn] = vv

    else:
        for vn, vv in zip(var_name, var_value):
            if vn in h_vars:

                if 'eps_leak' in vn:
                    if vn=='eps_leak':
                        params['h']['eps_leak'] = [vv]
                    elif vn=='eps_leak1':
                        params['h']['eps_leak'][0] = vv
                    elif vn=='eps_leak2':
                        params['h']['eps_leak'][1] = vv
                elif 'alph_leak' in vn:
                    if vn=='alph_leak':
                        params['h']['alph_leak'] = [vv]
                    elif vn=='alph_leak1':
                        params['h']['alph_leak'][0] = vv
                    elif vn=='alph_leak2':
                        params['h']['alph_leak'][1] = vv
                else:
                    params['h'][vn] = vv
            elif vn=='w_cnov':
                params['w'] = [1-vv, vv]
            elif isinstance(params[vn],list): 
                params[vn] = [vv]
            else:                                         
                params[vn] = vv

    return params

######################################################################################################################
# Helper function for splitting mouse data 
def get_splits_mouseid(all_data, kfold=5, seed=98765):
    mouse_id = np.arange(len(all_data))
    mouse_name = np.array([data['subID'][0] for data in all_data])

    # Split mouse into sets of size 1/kfold * num_mice
    rng = np.random.default_rng(seed)
    test_set_size = int(len(mouse_id) * 1/kfold)
    test_set_num  = kfold
    k_sets = rng.choice(mouse_id, size=(test_set_num, test_set_size), replace=False)

    test_set_all  = []
    train_set_all = []
    info_set_all  = []
    for i in range(kfold):
        test_set_i  = k_sets[i]
        train_set_i = np.where(~np.isin(mouse_id, test_set_i))[0]
        test_set_all.append(test_set_i) 
        train_set_all.append(train_set_i)

        info_i = pd.DataFrame({'mouse_id': mouse_id, 
                               'mouse_name': mouse_name, 
                               'set_id': [i]*len(mouse_id),
                               'test_set': [mouse in test_set_i for mouse in mouse_id],
                               'train_set': [mouse in train_set_i for mouse in mouse_id]})
        info_set_all.append(info_i)

    info_set_all = pd.concat(info_set_all, ignore_index=True)

    return train_set_all, test_set_all, info_set_all

def get_random_splits_mouseid(all_data, num_splits=1, test_ratio=0.5, seed=98765):
    mouse_id = np.arange(len(all_data))
    mouse_name = np.array([data['subID'][0] for data in all_data])

    rng = np.random.default_rng(seed)
    test_set_all  = []
    train_set_all = []
    info_set_all  = []
    for i in range(num_splits):
        test_set_i  = rng.choice(mouse_id, size=int(len(mouse_id) * test_ratio), replace=False)
        train_set_i = np.where(~np.isin(mouse_id, test_set_i))[0]
        test_set_all.append(test_set_i) 
        train_set_all.append(train_set_i)

        info_i = pd.DataFrame({'mouse_id': mouse_id, 
                               'mouse_name': mouse_name, 
                               'set_id': [i]*len(mouse_id),
                               'test_set': [mouse in test_set_i for mouse in mouse_id],
                               'train_set': [mouse in train_set_i for mouse in mouse_id]})
        info_set_all.append(info_i)

    info_set_all = pd.concat(info_set_all, ignore_index=True)

    return train_set_all, test_set_all, info_set_all

def get_splits_pathlen(all_data, test_order='testset_last', test_ratio_list=[0.5]):
    num_splits = len(test_ratio_list)
    mouse_id   = np.arange(len(all_data))
    mouse_name = np.array([data['subID'][0] for data in all_data])
    path_len   = np.array([len(data) for data in all_data])

    test_set_all  = []
    train_set_all = []
    info_set_all  = []
    for i in range(num_splits):
        test_len_i = np.round(test_ratio_list[i]*path_len).astype(int) # get length of test set
        train_len_i = path_len - test_len_i

        if test_order=='testset_last':
            test_set_i  = [np.arange(tlj, plj) for tlj, plj in zip(train_len_i, path_len)] #[all_data[tlj:] for tlj in test_len_i]
            train_set_i = [np.arange(0, tlj) for tlj in train_len_i] #[all_data[:tlj] for tlj in test_len_i]

        elif test_order=='testset_first':
            test_set_i  = [np.arange(0, tlj) for tlj in test_len_i] #[all_data[:tlj] for tlj in test_len_i]
            train_set_i = [np.arange(tlj, plj) for tlj, plj in zip(test_len_i, path_len)] #[all_data[tlj:] for tlj in test_len_i]

        test_set_all.append(test_set_i) 
        train_set_all.append(train_set_i)

        info_i = pd.DataFrame({'mouse_id':      mouse_id, 
                               'mouse_name':    mouse_name, 
                               'set_id':        [i]*len(mouse_id),
                               'test_len':      test_len_i,
                               'train_len':     train_len_i,
                               'path_len':      path_len,
                               'test_order':    [test_order]*len(mouse_id)})
        info_set_all.append(info_i)

    info_set_all = pd.concat(info_set_all, ignore_index=True)

    return train_set_all, test_set_all, info_set_all

######################################################################################################################
# Main function to run crossvalidation
def run_crossvalidation(config):

    leakiness_type  = config['leakiness_type']
    model_type      = config['model_type']
    level           = config['level'] if 'level' in config else None
    kernel_type     = config['kernel_type'] if 'kernel_type' in config else None
    comb_type       = config['comb_type'] if 'comb_type' in config else 'app'
    single_run_id   = config['single_run_id'] if 'single_run_id' in config else '' # '', '_1', '_2', ... to distinguish multiple single runs with different initial conditions
    alg_type_fit    = f'{leakiness_type}_{model_type}_{kernel_type}' if kernel_type is not None else f'{leakiness_type}_{model_type}'

    split_type      = config['split_type'] if 'split_type' in config else 'mouseid'
    kfold           = config['kfold'] if 'kfold' in config else 5
    seed_crossval   = config['seed_crossval'] if 'seed_crossval' in config else np.random.default_rng().integers(0, 100000)
    test_order      = config['test_order'] if 'test_order' in config else 'testset_last'
    test_ratio_list = config['test_ratio_list'] if 'test_ratio_list' in config else [0.5]

    var_name        = config['var_name']
    kwargs          = config['kwargs'] 
    run_local       = config['run_local'] if 'run_local' in config else False
    parallel        = config['parallel'] if 'parallel' in config else False

    # Define MLE function function 
    mle_fun         = mle_fit_parallel if parallel else mle_fit

    # Set paths
    base_path       = sl.get_rootpath()
    base_path_data  = base_path / 'data' / 'mle_results' / f'fits_cv-{split_type}/'
    sl.make_long_dir(base_path_data)

    snov = 'hnor' in model_type or 'hhyb' in model_type or 'hnac' in model_type
    if snov:
        if isinstance(level,list):
            level_str = '-'.join([str(l) for l in level])
        else:
            level_str = str(level)
        dir_save  = base_path_data / alg_type_fit / f'{leakiness_type}_{model_type}_{kernel_type}-l{level_str}-{comb_type}{single_run_id}'
    else:
        dir_save  = base_path_data / alg_type_fit
    sl.make_long_dir(dir_save)

    # Save log file and config for reproducibility
    if split_type=='mouseid':
        name_log = f'{comb_type}_{kfold}-fold_seed-{seed_crossval}'
    elif split_type=='pathlen':
        if len(test_ratio_list)>1:
            name_log = f'{comb_type}_{test_order}_pathsplits_multi-{len(test_ratio_list)}'
        else:
            name_log = f'{comb_type}_{test_order}_pathsplits_single-{str(test_ratio_list[0]).replace(".","-")}'
    log_file = open(dir_save / f'log_{name_log}.txt', 'w')
    with open(dir_save / f'config_{name_log}.json', 'w') as fc:
        json.dump(config, fc)

    # Load default parameters model
    default_params = load_default_params(alg_type=alg_type_fit, level=level, base_path=base_path, kwargs=kwargs)

    # Load mouse data
    P = default_params['P']
    mouse_data = load_mouse_data(P=P, base_path=base_path)
    # mouse_data = mouse_data[:2] # only for debugging

    # Create train / test splits
    if split_type=='mouseid':
        train_set, test_set, info_set = get_splits_mouseid(mouse_data, kfold=kfold, seed=seed_crossval)
    elif split_type=='pathlen':
        train_set, test_set, info_set = get_splits_pathlen(mouse_data, test_order=test_order, test_ratio_list=test_ratio_list)

    # Save info about splits
    info_set.to_csv(dir_save / f'info-splits_{name_log}.csv', index=False)

    # Run crossvalidation
    res_train = []
    res_test  = []
    for i, (train_set_i, test_set_i, setid_i) in enumerate(zip(train_set, test_set, info_set.set_id.unique())):

        # Run fitting on train set, save parameters + train error (LL)
        if split_type=='mouseid':
            train_data_i = pd.concat([mouse_data[j] for j in train_set_i], ignore_index=True)
            start_ll = [None] * len(train_set_i)
            stop_ll  = [None] * len(train_set_i)
        
        elif split_type=='pathlen':
            if test_order=='testset_first':
                train_data_i = pd.concat(mouse_data, ignore_index=True)
                start_ll     = [j_idx[0] for j_idx in train_set_i if len(j_idx)>0]
                stop_ll      = [j_idx[-1]+1 for j_idx in train_set_i if len(j_idx)>0]
                
            elif test_order=='testset_last':
                train_data_i = pd.concat([mouse_data[j].iloc[j_idx] for j, j_idx in zip(np.arange(len(train_set_i)),train_set_i)], ignore_index=True)
                train_subIDs_i = np.unique(train_data_i['subID'])
                start_ll = [None] * len(train_subIDs_i)
                stop_ll  = [None] * len(train_subIDs_i)

        kwargs['start_ll'] = start_ll
        kwargs['stop_ll']  = stop_ll
        # kwargs['maxit'] = 2 # only for debugging!!
        res = mle_fun(train_data_i,default_params,var_name,alg_type_fit,comb_type,kwargs,verbose=True,log_file=log_file)
        res['cvID'] = [i]*len(res)
        info_set_i = info_set.loc[info_set.set_id==setid_i]

        if split_type=='mouseid':
            res['ll_per_mouse'] = res['mle_ll'] / np.sum(info_set_i['train_set'].values)
        elif split_type=='pathlen':
            res['ll_per_step'] = res['mle_ll'] / (np.sum(info_set_i['path_len'].values) - np.sum(info_set_i['test_len'].values))
        res_train.append(pd.DataFrame(res))

        # Get fitted parameters for LL evaluation on test set
        fitted_params = update_default_params(model_type, default_params, res)

        # Set LL function for evaluation on test set
        fun_ll = ll_nor if 'nor' in model_type else None
        if fun_ll is None:
            raise ValueError(f"Unknown model type: {model_type}.")
        
        # Compute LL on test set
        if split_type=='mouseid':
            test_data_i = pd.concat([mouse_data[j] for j in test_set_i], ignore_index=True)
            start_ll = [None] * len(test_set_i) # no start / stop: use entire mouse trajectory
            stop_ll  = [None] * len(test_set_i) # no start / stop: use entire mouse trajectory
            subs_test = np.unique(test_data_i['subID']) # subjects in test set

        elif split_type=='pathlen':
            if test_order=='testset_first':
                test_data_i = pd.concat([mouse_data[j].iloc[j_idx] for j, j_idx in zip(np.arange(len(test_set_i)),test_set_i)], ignore_index=True)
                subs_test = np.unique(test_data_i['subID']) # subjects in test set: the ones with non-empty test set
                start_ll    = [None] * len(subs_test) # no start / stop: already filtered in test_data_i
                stop_ll     = [None] * len(subs_test) # no start / stop: already filtered in test_data_i

            elif test_order=='testset_last':
                test_data_i = pd.concat(mouse_data, ignore_index=True)
                start_ll    = [j_idx[0] for j_idx in test_set_i if len(j_idx)>0]
                stop_ll     = [j_idx[-1]+1 for j_idx in test_set_i if len(j_idx)>0]
                subs_test   = [sub_id for sub_id, j_idx in zip(np.unique(test_data_i['subID']),test_set_i) if len(j_idx)>0] # subjects in test set: the ones with non-empty test set

        LL_i = 0
        for j in range(len(subs_test)):  
            test_data_ij = test_data_i[test_data_i.subID==subs_test[j]].reset_index(drop=True)
            if test_data_ij.state.iloc[-1]==test_data_ij.next_state.iloc[-1]:
                test_data_ij = test_data_ij.iloc[:-1]
            if test_data_ij.state.iloc[0]==test_data_ij.next_state.iloc[0]:
                test_data_ij = test_data_ij.iloc[1:]
            LL_ij, _ = fun_ll(params=fitted_params, data=test_data_ij, start_ll=start_ll[j], stop_ll=stop_ll[j])
            LL_i += LL_ij
        if split_type=='mouseid':
            res_test.append(pd.DataFrame({'cvID': i, 'LL': -LL_i, 'LL_per_mouse': -LL_i / np.sum(info_set_i['test_set'].values)}, index=[0]))
        elif split_type=='pathlen':
            res_test.append(pd.DataFrame({'cvID': i, 'LL': -LL_i, 'LL_per_step': -LL_i / np.sum(info_set_i['test_len'].values)}, index=[0]))

        print(f'done with training set {i}/{len(train_set)}\n')

    # Save train and test results
    res_train = pd.concat(res_train, ignore_index=True)
    res_test = pd.concat(res_test, ignore_index=True)
    res_train.to_csv(dir_save / f'results_train_{name_log}.csv', index=False)
    res_test.to_csv(dir_save / f'results_test_{name_log}.csv', index=False)
    
    return res_train, res_test

    
if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument(
            '-c',
            '--config',
            dest='config_file',
            type=str,
            default=None,
            help='config file',
        )
    args = parser.parse_args()
    print('Successfully loaded config file.\n')

    if args.config_file:
        config = json.load(open(args.config_file))
    else: 
        config = json.load(open(sl.get_rootpath() / 'src' / 'fitting_behavior' / 'crossvalidation' / 'configs_cv-mouseid' / 'leaky_multinov-eps_hnor_notrace-l5-6_app_1.json'))

    print(config)

    res_train, res_test = run_crossvalidation(config)






