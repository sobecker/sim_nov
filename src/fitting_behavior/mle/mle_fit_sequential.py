import numpy as np
import pandas as pd
from scipy.optimize import minimize
import timeit

import sys
import utils.saveload as sl
from fitting_behavior.mle.LL_nor import ll_nor
from fitting_behavior.mle.LL_nAC import ll_nac
from fitting_behavior.mle.LL_hybrid import ll_hybrid
from fitting_behavior.mle.LL_hybrid2 import ll_hybrid2

### HELPER FUNCTIONS ###
def mle_opt_summary(res,var_names,log_file=None):
    print(f'LL maximization successful: {res.success}. Number of iterations performed: {res.nit}.')
    print(f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res.x)]}')
    print(f'LL value for estimated parameter values: {-res.fun}\n')

    if not log_file==None:
        sl.write_log(log_file,f'LL maximization successful: {res.success}. Number of iterations performed: {res.nit}.')
        sl.write_log(log_file,f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res.x)]}')
        sl.write_log(log_file,f'LL value for estimated parameter values: {-res.fun}.\n')

# Preprocess data (add action) -> tested in LL_nor.ipynb
def add_actions(s_seq,ns_seq,P):
    a_seq = []
    for i in range(len(s_seq)):
        s = s_seq[i]
        s_new = ns_seq[i]
        a = (P[s]==s_new).nonzero()[0][0]
        a_seq.append(a)
    return np.array(a_seq)

# Preprocess mice data (add action) -> tested in LL_nor.ipynb
def preprocess_micedata(dir,file,P,subID='',epi=0):
    data    = sl.load_df_stateseq(dir=dir,file=file) 
    s_seq   = data[:-1,1]
    ns_seq  = data[1:,1]
    a_seq   = add_actions(s_seq,ns_seq,P)

    dict_df = {'epi':epi*np.ones(len(s_seq),dtype=int),'it':data[:-1,0],'state':s_seq,'action':a_seq,'next_state':ns_seq}
    if len('subID')>0: dict_df['subID'] = [subID]*len(dict_df['it'])
    df = pd.DataFrame(dict_df)
    return df
    
### Test function LL and MLE (single parameter) ###
def mle_fit(all_data,params,var_name,alg_type,comb_type,kwargs,verbose=False,log_file=None,plot_state=False,write_state=False):
    # Init variables
    subs        = np.unique(all_data['subID'])
    params_lb   = params.copy()
    # save_name = f'{save_folder}_{comb_type}'

    if 'nor' in alg_type:     ll_fun = ll_nor
    elif 'nac' in alg_type:   ll_fun = ll_nac
    elif 'hybrid2' in alg_type:  ll_fun = ll_hybrid2
    elif 'hybrid' in alg_type:   ll_fun = ll_hybrid
    else:                   raise ValueError('Please specify a valid algorithm type ("nor" or "nac").')
    
    if 'opt_method' in kwargs.keys(): opt_method = kwargs['opt_method'] # Nelder-Mead, L-BFGS-B, SLSQP
    else:                             opt_method = '' 
    
    if 'x0' in kwargs.keys(): x0 = kwargs['x0']
    else:                     raise ValueError('Please specify initial values for the parameter estimates.')
    
    if 'bounds' in kwargs.keys(): bounds = kwargs['bounds']
    else:                         raise ValueError('Please specify bounds for the parameter estimates.')

    if 'hT' in var_name:
        if not 'h' in params_lb.keys() or len(params_lb['h'])==0:
            params_lb['h'] = {'n_buffer':0}
        else:
            params_lb['h'] = params_lb['h']|{'n_buffer':0}
    print(params_lb.keys())

    if 'maxit' in kwargs.keys():
        solver_options = {'maxiter':kwargs['maxit']}
    else:
        solver_options = {}

    # Compute MLE (opt) for separate or appended data sets
    opt_success = True
    num_opt = 0
    start_opt = timeit.default_timer()
    if comb_type=='sep':
        if verbose: 
            print(f'Computing MLE ({opt_method}) for separate data sets.\n')
            if not log_file==None: sl.write_log(log_file,f'Computing MLE ({opt_method}) for separate data sets.\n')
        l_mle_ll    = []
        l_mle_var   = []
        l_subs      = []
        l_vars      = []
        l_success   = []
        l_it        = []
        for i in range(len(subs)):  
            data_i = all_data[all_data.subID==subs[i]].reset_index(drop=True)
            if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
                data_i = data_i.iloc[:-1]
            if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
                data_i = data_i.iloc[1:]
            def apply_ll_fun(opt_var):
                for j in range(len(opt_var)):
                    if var_name[j]=='hT':                         params_lb['h']['n_buffer'] = opt_var[j] 
                    elif isinstance(params_lb[var_name[j]],list): params_lb[var_name[j]] = [opt_var[j]]
                    else:                                         params_lb[var_name[j]] = opt_var[j]
                LL, _ = ll_fun(params=params_lb,data=data_i)
                return -LL
            res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 
            l_subs.extend([subs[i]]*len(res_i.x))
            l_vars.extend(var_name)
            l_mle_ll.extend([res_i.fun]*len(res_i.x))
            l_mle_var.extend(res_i.x) 
            l_success.extend([res_i.success]*len(res_i.x))
            l_it.extend([res_i.nit]*len(res_i.x))
            if not res_i.success: opt_success = False
            num_opt += res_i.nit
            if verbose: 
                mle_opt_summary(res_i,var_name,log_file) 
                print(f'Subject {i} done.\n')
                if not log_file==None: sl.write_log(log_file,f'Subject {i} done.\n')
        end_opt = timeit.default_timer()
        res = pd.DataFrame({'subID':l_subs,
                            'var_name':l_vars, 
                            'mle_ll':l_mle_ll,
                            'mle_var':l_mle_var,
                            'opt_success':l_success,
                            'opt_it':l_it,
                            'opt_time':[end_opt-start_opt]*len(l_subs)
                            })
    elif comb_type=='app':
        if verbose: 
            print(f'Computing MLE ({opt_method}) for combined data set.\n')
            if not log_file==None: sl.write_log(log_file,f'Computing MLE ({opt_method}) for combined data set.\n')
        def apply_ll_fun(opt_var):
            LL = 0
            for j in range(len(opt_var)):
                if var_name[j]=='hT':                       params_lb['h']['n_buffer'] = opt_var[j] 
                if isinstance(params_lb[var_name[j]],list): params_lb[var_name[j]] = [opt_var[j]]
                else:                                       params_lb[var_name[j]] = opt_var[j]
            for i in range(len(subs)):  
                data_i = all_data[all_data.subID==subs[i]].reset_index(drop=True)
                if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
                    data_i = data_i.iloc[:-1]
                if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
                    data_i = data_i.iloc[1:]
                ll, _  = ll_fun(params=params_lb,data=data_i)
                LL+=ll
            return -LL
        res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 
        if not res_i.success: opt_success = False
        num_opt += res_i.nit
        if verbose: mle_opt_summary(res_i,var_name,log_file)
        end_opt = timeit.default_timer()
        res = pd.DataFrame({'subID':[-1]*len(res_i.x), 
                            'var_name':var_name,
                            'mle_ll':[res_i.fun]*len(res_i.x),
                            'mle_var':res_i.x,
                            'opt_success':[res_i.success]*len(res_i.x),
                            'opt_it':[res_i.nit]*len(res_i.x),
                            'opt_time':[end_opt-start_opt]*len(res_i.x)
                            })
    print(f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')
    if not log_file==None: 
            sl.write_log(log_file,f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')

    return res

