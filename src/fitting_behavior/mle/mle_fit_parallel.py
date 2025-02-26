import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import timeit
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl
from src.fitting_behavior.mle.LL_nor import ll_nor
from src.fitting_behavior.mle.LL_nAC import ll_nac
from src.fitting_behavior.mle.LL_hybrid import ll_hybrid
from src.fitting_behavior.mle.LL_hybrid2 import ll_hybrid2
    
### Helper ###
def mle_opt_summary(res,var_names,log_file=None):
    print(f'LL maximization successful: {res[3]}. Number of iterations performed: {res[4]}.')
    print(f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res[1])]}')
    print(f'LL value for estimated parameter values: {-res[2]}\n')

    if not log_file==None:
        sl.write_log(log_file,f'LL maximization successful: {res[3]}. Number of iterations performed: {res[4]}.')
        sl.write_log(log_file,f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res[1])]}')
        sl.write_log(log_file,f'LL value for estimated parameter values: {-res[2]}.\n')

### Function to apply in parallel
def fit_sub_sep(psub,all_data,var_name,params_lb,ll_fun,x0,bounds,opt_method,solver_options): 
    print(f'Fitting subject {psub}') 
    data_i = all_data[all_data.subID==psub].reset_index(drop=True)
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
    return [psub,res_i.x,res_i.fun,res_i.success,res_i.nit]

def fit_sub_app(psub,all_data,params_lb,ll_fun):
    data_i = all_data[all_data.subID==psub].reset_index(drop=True)
    if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
        data_i = data_i.iloc[:-1]
    if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
        data_i = data_i.iloc[1:]
    ll, _  = ll_fun(params=params_lb,data=data_i)
    return ll

### Function to run MLE fits ###
def mle_fit_parallel(all_data,params,var_name,alg_type,comb_type,kwargs,verbose=False,log_file=None,plot_state=False,write_state=False):
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
        solver_options = {'maxiter':kwargs['maxit'],'disp':True}
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
        
        # def fit_sub_sep(psub):  
        #     data_i = all_data[all_data.subID==psub].reset_index(drop=True)
        #     if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
        #         data_i = data_i.iloc[:-1]
        #     if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
        #         data_i = data_i.iloc[1:]
        #     def apply_ll_fun(opt_var):
        #         for j in range(len(opt_var)):
        #             if var_name[j]=='hT':                         params_lb['h']['n_buffer'] = opt_var[j] 
        #             elif isinstance(params_lb[var_name[j]],list): params_lb[var_name[j]] = [opt_var[j]]
        #             else:                                         params_lb[var_name[j]] = opt_var[j]
        #         LL, _ = ll_fun(params=params_lb,data=data_i)
        #         return -LL
        #     res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 
        #     return [psub,res_i.x,res_i.fun,res_i.success,res_i.nit]

        # res_all = []
        # def collect_result(result):
        #     global res_all
        #     res_all.append(result)

        #res_all = pool.map_async(fit_sub_sep,list(subs)).get()
        # for i in range(len(subs)):
        #     pool.apply_async(fit_sub_sep2,args=(subs[i],all_data,var_name,params_lb,ll_fun,x0,bounds,opt_method,solver_options),callback=collect_result)
        num_pool = min(mp.cpu_count(),len(subs))
        pool = mp.Pool(num_pool)
        res_obj = [pool.apply_async(fit_sub_sep,args=(subs[i],all_data,var_name,params_lb,ll_fun,x0,bounds,opt_method,solver_options)) for i in range(len(subs))]
        res_all = [r.get() for r in res_obj]
        
        pool.close()
        pool.join()
        end_opt = timeit.default_timer()

        l_mle_ll    = []
        l_mle_var   = []
        l_subs      = []
        l_vars      = []
        l_success   = []
        l_it        = []
        l_time      = []
        for i in range(len(res_all)):
            res_i = res_all[i]
            l_subs.extend([res_i[0]]*len(res_i[1]))
            l_vars.extend(var_name)
            l_mle_ll.extend([res_i[2]]*len(res_i[1]))
            l_mle_var.extend(res_i[1]) 
            l_success.extend([res_i[3]]*len(res_i[1]))
            l_it.extend([res_i[4]]*len(res_i[1]))
            l_time.extend([end_opt-start_opt]*len(res_i[1]))
            if not res_i[3]: opt_success = False
            num_opt += res_i[4]
            if verbose: 
                mle_opt_summary(res_i,var_name,log_file) 
                print(f'Subject {i} done.\n')
                if not log_file==None: sl.write_log(log_file,f'Subject {i} done.\n')
        res = pd.DataFrame({'subID':l_subs,
                            'var_name':l_vars, 
                            'mle_ll':l_mle_ll,
                            'mle_var':l_mle_var,
                            'opt_success':l_success,
                            'opt_it': l_it,
                            'opt_time': l_time
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
            pool = mp.Pool(mp.cpu_count())
            res_obj = [pool.apply_async(fit_sub_app,args=(subs[i],all_data,params_lb,ll_fun)) for i in range(len(subs))]
            res_all = [r.get() for r in res_obj]
            pool.close()
            pool.join()
            LL = np.sum(res_all)
            print(f'LL={LL}')
            return -LL
        res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 
        end_opt = timeit.default_timer()
        if not res_i.success: opt_success = False
        num_opt += res_i.nit
        if verbose: mle_opt_summary([-1,res_i.x,res_i.fun,res_i.success,res_i.nit],var_name,log_file)
        
        res = pd.DataFrame({'subID':[-1]*len(res_i.x), 
                            'var_name':var_name,
                            'mle_ll':[res_i.fun]*len(res_i.x),
                            'mle_var':res_i.x,
                            'opt_success':[res_i.success]*len(res_i.x),
                            'opt_it': [res_i.nit]*len(res_i.x),
                            'opt_time': [end_opt-start_opt]*len(res_i.x)
                            })
    print(f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')
    if not log_file==None: 
            sl.write_log(log_file,f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')

    return res