import sys

import utils.saveload as sl
import fitting_behavior.optimization.base_params_opt as bpo

from fitting_behavior.mle.LL_nor import ll_nor
from fitting_behavior.mle.LL_nAC import ll_nac
from fitting_behavior.mle.LL_hybrid import ll_hybrid
from fitting_behavior.mle.LL_hybrid2 import ll_hybrid2

test_ll_nor = False
test_ll_nac = False
test_ll_hyb = False
test_ll_ghyb = True

# Test ll_nor function with NoR simulations
if test_ll_nor:
    dir_data = sl.get_datapath()+'2022_12_09_13-31-14_sim_nor-tree_naive-nov' # nor sim with naive parameters
    subID    = 0

    all_data = sl.load_sim_data(dir_data)
    all_data = all_data[all_data.subID==subID].reset_index(drop=True)
    all_data = all_data.iloc[1:-1]

    params = sl.load_sim_params(dir_data)

    qvals = sl.load_sim_data(dir_data,file_data='qvals.pickle')
    qvals = qvals[qvals.subID==subID].reset_index(drop=True)
    qvals = qvals.iloc[1:-1]

    LL, qvals_equal = ll_nor(params,all_data,qvals)

    print(f'Loglikelihood of true parameters: {LL}\n')
    if qvals_equal: print('All qvals are equal.\n')
    else:           print('Qvals do not match.\n')

## Test LL function with nAC simulations ##
if test_ll_nac:
    dir_data = sl.get_datapath()+'2022_11_17_10-57-08_nAC_debug' # nAC sim with naive parameters
    subID    = 0

    # Load data and params
    all_data = sl.load_sim_data(dir_data)
    all_data = all_data[all_data.subID==subID].reset_index(drop=True)
    params = sl.load_sim_params(dir_data)

    # Load qvals (and convert to nor format if necessary)
    # if os.path.exists(dir_data+'/qvals.pickle'):
    #     qvals = sl.load_sim_data(dir_data,file_data='qvals.pickle')
    # else:
    #     qvals_nac = sl.load_sim_data(dir_data,file_data='wa.pickle')
    #     qvals = sl.qvals_nac2nor(all_data,qvals_nac,params,dir_data)
    qvals_nac = sl.load_sim_data(dir_data,file_data='wa.pickle')
    qvals_cols = qvals_nac.columns
    qvals_nac['subID'] = all_data['subID']     
    qvals = qvals_nac[qvals_nac.subID==subID].reset_index(drop=True)
    qvals = qvals[qvals_cols].to_numpy()

    # Compute LL for true params and check whether qvals are equal
    LL, qvals_equal = ll_nac(params,all_data,qvals)

    print(f'Loglikelihood of true parameters: {LL}\n')
    if qvals_equal: print('All qvals are equal.\n')
    else:           print('Qvals do not match.\n')

## Test LL function with hybrid simulations ##
if test_ll_hyb:
    dir_data = sl.get_datapath()+'TestCode/TestHybrid_BasicNov/2023_01_26_14-28-42_hybrid_balanced' 
    subID    = 0

    # Load data + params
    mf_data     = sl.load_sim_data(dir_data,file_data='mf_data_basic.pickle')
    mf_params   = sl.load_sim_params(dir_data,file_params='mf_params.pickle')
    mb_params   = sl.load_sim_params(dir_data,file_params='mb_params.pickle')

    # Combine params
    overlap = ['rec_type','k','ntype']
    for i in range(len(overlap)):
        mf_params[f'mf_{overlap[i]}'] = mf_params.pop(overlap[i])
        mb_params[f'mb_{overlap[i]}'] = mb_params.pop(overlap[i])
    mf_params.update(mb_params)

    # Load qvals
    qvals_nac           = sl.load_sim_data(dir_data,file_data='wa.pickle')
    qvals_cols          = qvals_nac.columns
    qvals_nac['subID']  = mf_data['subID']   
    mb_qvals            = sl.load_sim_data(dir_data,file_data='qvals.pickle')  

    # Restrict subIDs
    mf_data     = mf_data[mf_data.subID==subID].reset_index(drop=True)
    mf_data     = mf_data[1:]
    mf_qvals    = qvals_nac[qvals_nac.subID==subID].reset_index(drop=True)
    mf_qvals    = mf_qvals[qvals_cols].to_numpy()
    mf_qvals    = mf_qvals[1:]
    mb_qvals = mb_qvals[mb_qvals.subID==subID].reset_index(drop=True)
    mb_qvals = mb_qvals[1:]
    
    # Compute LL for true params and check whether qvals are equal
    LL, [mb_qvals_equal, mf_qvals_equal] = ll_hybrid(mf_params,mf_data,mb_qvals,mf_qvals,verbose=True)

    print(f'Loglikelihood of true parameters: {LL}\n')
    if mf_qvals_equal: print('All MF qvals are equal.\n')
    else:              print('MF qvals do not match.\n')
    if mb_qvals_equal: print('All MB qvals are equal.\n')
    else:              print('MB qvals do not match.\n')

## Test LL function with granular-hybrid simulations ##
if test_ll_ghyb:
    dir_data = sl.get_datapath()+'TestCode/TestHybrid_GranularNov/2023_01_30_14-15-06_hybrid_balanced' 
    subID    = 0

    # Load data + params
    mf_data     = sl.load_sim_data(dir_data,file_data='mf_data_basic.pickle')
    mf_params   = sl.load_sim_params(dir_data,file_params='mf_params.pickle')
    mb_params   = sl.load_sim_params(dir_data,file_params='mb_params.pickle')

    # Combine params
    all_params = bpo.comb_params(mf_params,mb_params,mf_params)
    # overlap = ['rec_type','k','ntype']
    # for i in range(len(overlap)):
    #     mf_params[f'mf_{overlap[i]}'] = mf_params.pop(overlap[i])
    #     mb_params[f'mb_{overlap[i]}'] = mb_params.pop(overlap[i])
    # mf_params.update(mb_params)

    # Load qvals
    qvals_nac           = sl.load_sim_data(dir_data,file_data='wa.pickle')
    qvals_cols          = qvals_nac.columns
    qvals_nac['subID']  = mf_data['subID']   
    mb_qvals            = sl.load_sim_data(dir_data,file_data='qvals.pickle')  

    # Restrict subIDs
    mf_data     = mf_data[mf_data.subID==subID].reset_index(drop=True)
    mf_data     = mf_data[1:]
    mf_qvals    = qvals_nac[qvals_nac.subID==subID].reset_index(drop=True)
    mf_qvals    = mf_qvals[qvals_cols].to_numpy()
    mf_qvals    = mf_qvals[1:]
    mb_qvals = mb_qvals[mb_qvals.subID==subID].reset_index(drop=True)
    mb_qvals = mb_qvals[1:]
    
    # Compute LL for true params and check whether qvals are equal
    LL, [mb_qvals_equal, mf_qvals_equal] = ll_hybrid(mf_params,mf_data,mb_qvals,mf_qvals,verbose=True)
    LL2, [mb_qvals_equal2, mf_qvals_equal2] = ll_hybrid2(mf_params,mf_data,mb_qvals,mf_qvals,verbose=True)

    print(f'Loglikelihood of true parameters: {LL}\n')
    if mf_qvals_equal: print('All MF qvals are equal.\n')
    else:              print('MF qvals do not match.\n')
    if mb_qvals_equal: print('All MB qvals are equal.\n')
    else:              print('MB qvals do not match.\n')

    print(f'Loglikelihood of true parameters (hybrid2): {LL2}\n')
    if mf_qvals_equal2: print('All MF qvals are equal.\n')
    else:              print('MF qvals do not match.\n')
    if mb_qvals_equal2: print('All MB qvals are equal.\n')
    else:              print('MB qvals do not match.\n')