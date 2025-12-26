import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import os
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

from src.mbnor.mb_surnor import import_params_surnor
from src.scripts.OffPolicy.offpolicy_nor import offpolicy_nor
from src.scripts.MLE.LL_nor import ll_nor

from src.scripts.OffPolicy.offpolicy_nac import offpolicy_nac
from src.scripts.MLE.LL_nAC import ll_nac

from src.scripts.OffPolicy.offpolicy_hyb2 import offpolicy_hyb2
from src.scripts.MLE.LL_hybrid import ll_hybrid
from src.scripts.MLE.LL_hybrid2 import ll_hybrid2

import src.optimization.base_params_opt as bpo
import src.optimization.auxiliary_opt as auxo
import src.utils.saveload as sl

##########################################################################################
def plot_nov(rec_all,labels,colors,algname,plot_field):
    f,ax = plt.subplots(1,1,figsize=(5,5))
    for ii, rec in enumerate(rec_all):
        ax.scatter(rec['time'],rec[plot_field],label=labels[ii],c=colors[ii])
        ax.plot(rec['time'],rec[plot_field],'--',c=colors[ii])
    ax.axvline(x=np.where(rec['state']==64)[0],c='k',ls=':')
    ax.set_xlabel('Time steps in maze')
    ax.set_ylabel('Novelty of next state')
    ax.legend()
    ax.set_title(f'Novelty signals of {algname} along test path')

def plot_fam(rec_all,labels,colors,algname,plot_field):
    f,ax = plt.subplots(1,1,figsize=(5,5))
    for ii, rec in enumerate(rec_all):
        ax.scatter(rec['time'],rec[plot_field],label=labels[ii],c=colors[ii])
        ax.plot(rec['time'],rec[plot_field],'--',c=colors[ii])
    ax.axvline(x=np.where(rec['state']==64)[0],c='k',ls=':')
    ax.set_xlabel('Time steps in maze')
    ax.set_ylabel('Familiarity of next state')
    ax.legend()
    ax.set_title(f'Novelty signals of {algname} along test path')

##########################################################################################
test_op     = False
test_sim_ll = True

##########################################################################################
# Set based path to project directory
base_path = '/Users/sbecker/Projects/RL_reward_novelty/'

# Construct toy example data
sequence = pd.DataFrame({'state':       np.array([0,1,2,4,8,16,32,64,32,16,33,16],dtype=int), 
                         'action':      np.array([1,2,2,2,2,2,2,0,0,3,0,0],dtype=int),
                         'next_state':  np.array([1,2,4,8,16,32,64,32,16,33,16,8],dtype=int)
                         })

##########################################################################################
if test_op:

    ####### nor/nac/hyb model: off-policy (comp. with nor) ###################
    comp_algs = ['nor','nac','hyb2']
    alph_leak = [0,0.1,0.5]
    eps_leak  = 1
    labels = ['non-leaky','leaky (0.1)','leaky (0.5)']
    colors = ['C0','C1','C2']

    # params sim-nov
    level       = 2
    notrace     = False
    center      = False
    center_type = 'triangle'

    rec_all = []
    for ii, cc in enumerate(comp_algs):
        if cc=='nor':
            # Load default parameters for nor model
            params = bpo.base_params_mbnortree_exp.copy()
            params_surnor = import_params_surnor(path=f'{base_path}src/mbnor/')
            params.update(params_surnor)

            rec_all_nor = []
            for ka in alph_leak:
                # Simulate leaky nor model (off-policy) + extract novelty signals
                params['alph_leak'] = ka
                params['eps_leak']  = eps_leak
                print(f'Leakiness of counts: {params["alph_leak"]}.')
                rec_nor, _ = offpolicy_nor(params,sequence,rec_counts=True)
                rec_all_nor.append(rec_nor)

            # Plot novelty/familiarity signals of leaky vs. non-leaky along path
            plot_nov(rec_all_nor,labels,colors,'nor',plot_field='nov_s_new')
            plot_fam(rec_all_nor,labels,colors,'nor',plot_field='counts_s_new')

        if cc=='hnor':
            # Load default parameters for nor model
            params = bpo.baseparams_h1mbnor_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type='leaky')
            params_surnor = import_params_surnor(path=f'{base_path}src/mbnor/')
            params.update(params_surnor)

            rec_all_nor = []
            for ka in alph_leak:
                # Simulate leaky nor model (off-policy) + extract novelty signals
                params['h']['alph_leak'] = ka
                params['h']['eps_leak']  = eps_leak
                print(f'Leakiness of counts: {params["h"]["alph_leak"]}.')
                rec_nor, _ = offpolicy_nor(params,sequence,rec_counts=True)
                rec_all_nor.append(rec_nor)

            # Plot novelty/familiarity signals of leaky vs. non-leaky along path
            plot_nov(rec_all_nor,labels,colors,'nor',plot_field='nov_s_new')
            # plot_fam(rec_all_nor,labels,colors,'nor',plot_field='counts_s_new')

        if cc=='nac':
            # Load parameters for nac model
            params_nac = bpo.base_params_nACtree.copy()

            rec_all_nac = []
            for jj, ka in enumerate(alph_leak):
                # Simulate leaky nor model (off-policy) + extract novelty signals
                params_nac['h']['alph_leak'] = ka
                params_nac['h']['eps_leak']  = eps_leak
                print(f'Leakiness of counts: {params_nac["h"]["alph_leak"]}.')
                rec_nac, _ = offpolicy_nac(params_nac,sequence,rec_counts=True)
                rec_all_nac.append(rec_nac)

                # Compare with nor model
                match_n = (rec_nac['nov_s_post'].values==np.round(rec_all_nor[jj]['nov_s'].values[1:],4)).all() and (rec_nac['nov_s_new_post'].values==np.round(rec_all_nor[jj]['nov_s_new'].values[1:],4)).all()
                match_c = (rec_nac['counts_s_post'].values==rec_all_nor[jj]['counts_s'].values[1:]).all() and (rec_nac['counts_s_new_post'].values==rec_all_nor[jj]['counts_s_new'].values[1:]).all()
                print(f'{cc} (leaky={ka}): novelty signals {"match" if match_n else "do not match"}.\n')
                print(f'{cc} (leaky={ka}): familiarity signals {"match" if match_c else "do not match"}.\n')
                
            # Plot novelty/familiarity signals of leaky vs. non-leaky along path
            plot_nov(rec_all_nac,labels,colors,cc,plot_field='nov_s_new_pre')
            plot_fam(rec_all_nac,labels,colors,cc,plot_field='counts_s_new_pre')

        elif cc=='hyb' or cc=='hyb2':
            # Load parameters for hybrid model
            params_hyb = bpo.baseparams_hybrid_comb()

            rec_all_hyb = []
            for jj, ka in enumerate(alph_leak):
                # Simulate leaky nor model (off-policy) + extract novelty signals
                params_hyb['alph_leak'] = ka
                params_hyb['eps_leak']  = eps_leak
                params_hyb['mf_h']['alph_leak'] = ka
                params_hyb['mf_h']['eps_leak']  = eps_leak
                print(f'Leakiness of counts: {params_hyb["alph_leak"]}.')
                rec_hyb, _ = offpolicy_hyb2(params_hyb,sequence,rec_counts=True)
                rec_all_hyb.append(rec_hyb)

                # Compare MB-part with nor model
                match_n_mb = (rec_hyb['mb_nov_s_new'].values==rec_all_nor[jj]['nov_s_new'].values).all()
                match_c_mb = (rec_hyb['mb_counts_s_new'].values==rec_all_nor[jj]['counts_s_new'].values).all()
                print(f'MB-part {cc} (leaky={ka}): novelty signals {"match" if match_n else "do not match"}.\n')
                print(f'MB-part {cc} (leaky={ka}): familiarity signals {"match" if match_c else "do not match"}.\n')

                # Compare MB-part with nor model
                match_n_mb = (rec_hyb['mf_nov_s_new_post'].values[1:]==rec_all_nac[jj]['nov_s_new_post'].values).all()
                match_c_mb = (rec_hyb['mf_counts_s_new_post'].values[1:]==rec_all_nac[jj]['counts_s_new_post'].values).all()
                print(f'MB-part {cc} (leaky={ka}): novelty signals {"match" if match_n else "do not match"}.\n')
                print(f'MB-part {cc} (leaky={ka}): familiarity signals {"match" if match_c else "do not match"}.\n')

##########################################################################################
if test_sim_ll:

    comp_algs = ['hyb2']

    ll_all = []
    for ii, cc in enumerate(comp_algs):

        # Leakiness values to test
        alph_leak = [0,0.1,0.5]
        eps_leak  = 1

        # Create folder to save simulation data
        save_path = f'{base_path}data/2025-07_test2_leaky_{cc}'
        sl.make_long_dir(save_path)

        if cc=='nor':
            # Load default parameters for nor model
            params = bpo.base_params_mbnortree_exp.copy()
            params_surnor = import_params_surnor(path=f'{base_path}src/mbnor/')
            params.update(params_surnor)

            # Set simulation function and off-policy function
            sim_fun_cc = auxo.run_opt_sim_mbnor
            op_fun_cc = offpolicy_nor
            ll_fun_cc = ll_nor

            # Set test field
            test_field_sim = 'nov_s_new'
            test_field_op = 'nov_s_new'

        elif cc=='nac':
            params = bpo.base_params_nACtree.copy()
            sim_fun_cc = auxo.run_optsim_nac_tree
            op_fun_cc  = offpolicy_nac
            ll_fun_cc = ll_nac
            test_field_sim = 'nov_s_new'
            test_field_op = 'nov_s_new_post'

        elif cc=='hyb' or cc=='hyb2':
            params = bpo.baseparams_hybrid_comb()
            sim_fun_cc = auxo.run_optsim_hybrid_tree
            op_fun_cc = offpolicy_hyb2
            ll_fun_cc = ll_hybrid2
            test_field_sim = 'nov_s_new'
            test_field_op_mf = 'mf_nov_s_new_post'
            test_field_op_mb = 'mb_nov_s_new'

        ll_all_cc = []
        for ka in alph_leak:
            # Simulate leaky nor model + save data
            if cc=='nor':
                params['alph_leak'] = ka
                params['eps_leak']  = eps_leak
            elif cc=='nac':
                params['h']['alph_leak'] = ka
                params['h']['eps_leak']  = eps_leak
            elif cc=='hyb' or cc=='hyb2':
                params['alph_leak'] = ka
                params['eps_leak']  = eps_leak
                params['mf_h']['alph_leak'] = ka
                params['mf_h']['eps_leak']  = eps_leak
            params['alph_leak'] = ka
            print(f'Leakiness of counts: {params["alph_leak"]}.')
            sim_name = f'leaky-{params["alph_leak"]}'.replace('.','_')
            dir_data = sim_fun_cc(params,1,1,params,sim_name,dir_data=save_path)

            # Extract state-action sequence from data for leaky nor model
            if cc=='nor':
                all_data = sl.load_sim_data(dir_data)
                all_data = all_data.iloc[1:]
                all_data[test_field_sim] = all_data.apply(lambda x: x['novelty'][x['next_state']],axis=1)
                qvals = sl.load_sim_data(dir_data,file_data='qvals.pickle')
                qvals = qvals.iloc[1:]

                # Verify match of off-policy novelty signals for leaky nor model 
                rec, _ = op_fun_cc(params,all_data,rec_counts=True)

            elif cc=='nac':
                all_data = sl.load_sim_data(dir_data)
                all_data[test_field_sim] = all_data.apply(lambda x: x['mod-0: nov_post'][x['next_state']],axis=1)
                qvals = sl.load_sim_data(dir_data,file_data='wa.pickle')
                qvals_cols = qvals.columns
                qvals = qvals[qvals_cols].to_numpy()

                # Verify match of off-policy novelty signals for leaky nor model 
                rec, _ = op_fun_cc(params,all_data,rec_counts=True)

            elif cc=='hyb' or cc=='hyb2':
                # Load MF data part
                all_data_mf = sl.load_sim_data(dir_data,file_data='mf_data_basic.pickle')
                all_data_mf = all_data_mf.iloc[1:]
                all_data_mf[test_field_sim] = all_data_mf.apply(lambda x: np.round(x['mod-0: nov_post'][x['next_state']],4),axis=1)
                qvals_mf = sl.load_sim_data(dir_data,file_data='wa.pickle')
                qvals_mf = qvals_mf.iloc[1:]
                qvals_cols = qvals_mf.columns
                qvals_mf = qvals_mf[qvals_cols].to_numpy()

                # Load MB data part
                all_data_mb = sl.load_sim_data(dir_data,file_data='mb_data_basic.pickle')
                all_data_mb = all_data_mb.iloc[1:]
                all_data_mb[test_field_sim] = all_data_mb.apply(lambda x: x['novelty'][x['next_state']],axis=1)
                qvals_mb = sl.load_sim_data(dir_data,file_data='qvals.pickle')
                qvals_mb = qvals_mb.iloc[1:]
                
                # Verify match of off-policy novelty signals for leaky nor model 
                rec, _ = op_fun_cc(params,all_data_mf,rec_counts=True)

            if cc=='nor' or cc=='hyb' or cc=='hyb2': 
                rec = rec.iloc[1:]
            if cc=='hyb' or cc=='hyb2':
                nov_equal_mb = (rec[test_field_op_mb].values == all_data_mb[test_field_sim].values).all()
                nov_equal_mf = (rec[test_field_op_mf].values == all_data_mf[test_field_sim].values).all()
                nov_equal = nov_equal_mb and nov_equal_mf   
            else:
                nov_equal = (rec[test_field_op].values == all_data[test_field_sim].values).all()
            if nov_equal: print('All novelty values are equal.\n')
            else:         print('Novelty values do not match.\n')

            # Verify match of LL and qvals for leaky nor model 
            if cc=='nor' or cc=='nac':
                LL, qvals_equal = ll_fun_cc(params,all_data,qvals=qvals)
            elif cc=='hyb' or cc=='hyb2': 
                LL, qvals_equal = ll_fun_cc(params,all_data_mf,mb_qvals=qvals_mb,mf_qvals=qvals_mf)
                qvals_equal = np.array(qvals_equal).all()
                print(f'Loglikelihood of true parameters: {LL}\n')
            if qvals_equal: print('All qvals are equal.\n')
            else:           print('Qvals do not match.\n')
            ll_all_cc.append(LL)
        ll_all.append(ll_all_cc)

print('done')

