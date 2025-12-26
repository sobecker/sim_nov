import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import timeit
import utils.saveload as sl
from fitting_behavior.mle.LL_nor import ll_nor
from fitting_behavior.mle.LL_nAC import ll_nac
from fitting_behavior.mle.LL_hybrid import ll_hybrid
from fitting_behavior.mle.LL_hybrid2 import ll_hybrid2
    
### HELPER FUNCTIONS ###
def mle_opt_summary(res,var_names,log_file=None):
    print(f'LL maximization successful: {res[3]}. Number of iterations performed: {res[4]}.')
    print(f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res[1])]}')
    print(f'LL value for estimated parameter values: {-res[2]}\n')

    if not log_file==None:
        sl.write_log(log_file,f'LL maximization successful: {res[3]}. Number of iterations performed: {res[4]}.')
        sl.write_log(log_file,f'Estimated values for parameters:\n {[i+": "+str(j) for i,j in zip(var_names,res[1])]}')
        sl.write_log(log_file,f'LL value for estimated parameter values: {-res[2]}.\n')

# Softplus function for transformed optimization
def softplus(x):
    return np.log(1+np.exp(x))

def softplus_inv(x):
    return np.log(np.exp(x)-1)

def logit(x):
    return np.log(x/(1-x))

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# Fitting function to be run in parallel (separate fit)
def fit_sub_sep(psub,alg_type,all_data,var_name,params_lb,ll_fun,x0,bounds,transformed,l_transfun,opt_method,solver_options,start_ll,stop_ll): 

    print(f'Fitting subject {psub}') 

    data_i = all_data[all_data.subID==psub].reset_index(drop=True)
    if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
        data_i = data_i.iloc[:-1]
    if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
        data_i = data_i.iloc[1:]

    l_transfun = [eval(fun) if fun is not None else None for fun in l_transfun]

    # Define optimization wrapper
    h_vars = ['hT','eps_leak','eps_leak1','eps_leak2','alph_leak','alph_leak1','alph_leak2','k_alph']

    if 'hybrid' in alg_type: # hybrid RL agent

        def apply_ll_fun(opt_var):
            for j in range(len(opt_var)):

                # Apply transformation (if applicable)
                if transformed[j]:
                    ovj = l_transfun[j](opt_var[j])
                else:
                    ovj = opt_var[j]
                
                # Update parameter dictionary
                if var_name[j] in h_vars:
                    if var_name[j]=='hT': 
                        params_lb['mb_h']['n_buffer'] = ovj 
                        params_lb['mf_h']['n_buffer'] = ovj 
                    elif var_name[j]=='eps_leak':
                        params_lb['mb_h']['eps_leak'] = [ovj]
                        params_lb['mf_h']['eps_leak'] = [ovj]
                    else:
                        params_lb['mb_h'][var_name[j]] = ovj
                        params_lb['mf_h'][var_name[j]] = ovj
                elif var_name[j]=='w_cnov':
                    params_lb['w'] = [1-ovj, ovj]
                elif isinstance(params_lb[var_name[j]],list): 
                    params_lb[var_name[j]] = [ovj]
                else:                                         
                    params_lb[var_name[j]] = ovj

            # Compute loglikelihood
            LL, _ = ll_fun(params=params_lb,data=data_i,start_ll=start_ll,stop_ll=stop_ll)
            return -LL
                
    else: # non-hybrid RL agent

        def apply_ll_fun(opt_var):

            if ('eps_leak1' in var_name and 'eps_leak2' in var_name):
                params_lb['h']['eps_leak'] = [1, 1]
            if ('alph_leak1' in var_name and 'alph_leak2' in var_name):
                params_lb['h']['alph_leak'] = [0, 0]

            for j in range(len(opt_var)):

                # Apply transformation (if applicable)
                if transformed[j]:
                    ovj = l_transfun[j](opt_var[j])
                else:
                    ovj = opt_var[j]

                # Update parameter dictionary
                if var_name[j] in h_vars:
                    if var_name[j]=='hT': 
                        params_lb['h']['n_buffer'] = ovj 
                    elif 'eps_leak' in var_name[j]:
                        if var_name[j]=='eps_leak':
                            params_lb['h']['eps_leak'] = [ovj]
                        elif var_name[j]=='eps_leak1':
                            params_lb['h']['eps_leak'][0] = ovj
                        elif var_name[j]=='eps_leak2':
                            params_lb['h']['eps_leak'][1] = ovj
                    elif 'alph_leak' in var_name[j]:
                        if var_name[j]=='alph_leak':
                            params_lb['h']['alph_leak'] = [ovj]
                        elif var_name[j]=='alph_leak1':
                            params_lb['h']['alph_leak'][0] = ovj
                        elif var_name[j]=='alph_leak2':
                            params_lb['h']['alph_leak'][1] = ovj
                    else:
                        params_lb['h'][var_name[j]] = ovj
                elif var_name[j]=='w_cnov':
                    params_lb['w'] = [1-ovj, ovj]
                elif isinstance(params_lb[var_name[j]],list): 
                    params_lb[var_name[j]] = [ovj]
                else:                                         
                    params_lb[var_name[j]] = ovj

            # Compute loglikelihood
            LL, _ = ll_fun(params=params_lb,data=data_i,start_ll=start_ll,stop_ll=stop_ll)
            return -LL
            
    # Run optimization
    res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 

    # Transform parameters back (if applicable)
    res_ix = [funix(rix) if transformed[i_rix] else rix for i_rix, (rix, funix) in enumerate(zip(res_i.x,l_transfun))]

    return [psub,res_ix,res_i.fun,res_i.success,res_i.nit]

# Fitting function to be run in parallel (joint fit)
def fit_sub_app(psub,all_data,params_lb,ll_fun,start_ll,stop_ll):

    data_i = all_data[all_data.subID==psub].reset_index(drop=True)
    if data_i.state.iloc[-1]==data_i.next_state.iloc[-1]:
        data_i = data_i.iloc[:-1]
    if data_i.state.iloc[0]==data_i.next_state.iloc[0]:
        data_i = data_i.iloc[1:]

    # Compute loglikelihood
    ll, _  = ll_fun(params=params_lb,data=data_i,start_ll=start_ll,stop_ll=stop_ll)
    return ll

### Function to run MLE fits ###
def mle_fit_parallel(all_data,params,var_name,alg_type,comb_type,kwargs,verbose=False,log_file=None,plot_state=False,write_state=False):

    # Initialize variables #################################################################################################################
    subs        = np.unique(all_data['subID'])
    params_lb   = params.copy()
    # save_name = f'{save_folder}_{comb_type}'

    ## Allow to fit only parts of the mouse trajectory (between start_ll and stop_ll, if specified) ########################################
    start_ll = kwargs['start_ll'] if 'start_ll' in kwargs.keys() else [None] *len(subs)
    stop_ll  = kwargs['stop_ll'] if 'stop_ll' in kwargs.keys() else [None] *len(subs)
    assert len(start_ll) == len(subs) and len(stop_ll) == len(subs)

    ## Specify likelihood function, optimization method, initial values and bounds #########################################################
    if 'nor' in alg_type.replace('mazenorm','').replace('nocompnorm',''):     
        ll_fun = ll_nor
    elif 'nac' in alg_type:   
        ll_fun = ll_nac
    elif 'hybrid2' in alg_type:  
        ll_fun = ll_hybrid2
    elif 'hybrid' in alg_type:   
        ll_fun = ll_hybrid
    else:                   
        raise ValueError('Please specify a valid algorithm type ("nor" or "nac").')
    
    # Optimization method
    if 'opt_method' in kwargs.keys(): 
        opt_method = kwargs['opt_method'] # Nelder-Mead, L-BFGS-B, SLSQP
    else:                             
        opt_method = '' 
    
    # Initial values
    if 'x0' in kwargs.keys(): 
        x0 = kwargs['x0']
    else:                     
        raise ValueError('Please specify initial values for the parameter estimates.')
    
    # Bounds
    if 'bounds' in kwargs.keys(): 
        bounds = kwargs['bounds']
    else:                         
        raise ValueError('Please specify bounds for the parameter estimates.')

    # Transformed optimization
    if 'transformed' in kwargs.keys(): 
        transformed = kwargs['transformed']
    else:                         
        transformed = [False]*len(var_name)
    
    try:
        if 'transfun' in kwargs.keys(): 
            l_transfun = [fun if fun is not None else None for fun in kwargs['transfun']]
    except:
        raise ValueError('Please specify a valid transformation function ("softplus" or "log").')

    # Specify hT variable (DEPRECATED, not used anymore)
    if 'hT' in var_name:
        if not 'h' in params_lb.keys() or len(params_lb['h'])==0:
            params_lb['h'] = {'n_buffer':0}
        else:
            params_lb['h'] = params_lb['h']|{'n_buffer':0}
    print(params_lb.keys())

    # Solver options
    if 'maxit' in kwargs.keys():
        solver_options = {'maxiter':kwargs['maxit'],'disp':True}
    else:
        solver_options = {}
    
    ## Compute MLE (opt) for separate or appended data sets ################################################################################
    opt_success = True
    num_opt = 0
    start_opt = timeit.default_timer()

    if comb_type=='sep':

        ## Fit separately ##################################################################################################################
        if verbose: 
            print(f'Computing MLE ({opt_method}) for separate data sets.\n')
            if not log_file==None: 
                sl.write_log(log_file,f'Computing MLE ({opt_method}) for separate data sets.\n')
        
        # Run parallel pool
        num_pool = min(mp.cpu_count(),len(subs))
        pool     = mp.Pool(num_pool)
        res_obj  = [pool.apply_async(fit_sub_sep,args=(subs[i],alg_type,all_data,var_name,params_lb,ll_fun,x0,bounds,transformed,l_transfun,opt_method,solver_options,start_ll[i],stop_ll[i])) for i in range(len(subs))]
        res_all  = [r.get() for r in res_obj]
        pool.close()
        pool.join()

        end_opt = timeit.default_timer()

        # Collect results
        l_mle_ll    = []
        l_mle_var   = []
        l_subs      = []
        l_vars      = []
        l_success   = []
        l_it        = []
        l_time      = []
        for i in range(len(res_all)):
            res_i = res_all[i]
            l_subs.extend([res_i[0]]*len(res_i[1])) # subject ID
            l_vars.extend(var_name) # parameter names
            l_mle_ll.extend([res_i[2]]*len(res_i[1])) # negative loglikelihood (loss)
            l_mle_var.extend(res_i[1]) # parameter estimates (already transformed back if applicable)
            l_success.extend([res_i[3]]*len(res_i[1])) # whether optimization was successful
            l_it.extend([res_i[4]]*len(res_i[1])) # number of iterations
            l_time.extend([end_opt-start_opt]*len(res_i[1])) # time taken

            if not res_i[3]: 
                opt_success = False
            num_opt += res_i[4]

            # Print summary + write to log file
            if verbose: 
                mle_opt_summary(res_i,var_name,log_file) 
                print(f'Subject {i} done.\n')
                if not log_file==None: 
                    sl.write_log(log_file,f'Subject {i} done.\n')

        res = pd.DataFrame({'subID':l_subs,
                            'var_name':l_vars, 
                            'mle_ll':l_mle_ll,
                            'mle_var':l_mle_var,
                            'opt_success':l_success,
                            'opt_it': l_it,
                            'opt_time': l_time
                            })
        
    elif comb_type=='app':

        ## Fit jointly #####################################################################################################################
        if verbose: 
            print(f'Computing MLE ({opt_method}) for combined data set.\n')
            if not log_file==None: 
                sl.write_log(log_file,f'Computing MLE ({opt_method}) for combined data set.\n')

        # Define optimization wrapper
        h_vars = ['hT','eps_leak','eps_leak1','eps_leak2','alph_leak','alph_leak1','alph_leak2','k_alph']

        l_transfun = [eval(fun) if fun is not None else None for fun in l_transfun]

        if 'hybrid' in alg_type: # hybrid RL agent

            def apply_ll_fun(opt_var):
                LL = 0
                for j in range(len(opt_var)):

                    # Apply transformation (if applicable)
                    if transformed[j]:
                        ovj = l_transfun[j](opt_var[j])
                    else:
                        ovj = opt_var[j]

                    # Update parameter dictionary
                    if var_name[j] in h_vars:
                        if var_name[j]=='hT': 
                            params_lb['mb_h']['n_buffer'] = ovj 
                            params_lb['mf_h']['n_buffer'] = ovj 
                        elif var_name[j]=='eps_leak':
                            params_lb['mb_h']['eps_leak'] = [ovj]
                            params_lb['mf_h']['eps_leak'] = [ovj]
                        else:
                            params_lb['mb_h'][var_name[j]] = ovj
                            params_lb['mf_h'][var_name[j]] = ovj
                    elif var_name[j]=='w_cnov':
                        params_lb['w'] = [1-ovj, ovj]
                    elif isinstance(params_lb[var_name[j]],list): 
                        params_lb[var_name[j]] = [ovj]
                    else:                                         
                        params_lb[var_name[j]] = ovj

                # Compute loglikelihood (parallelized)
                pool = mp.Pool(mp.cpu_count())
                res_obj = [pool.apply_async(fit_sub_app,args=(subs[i],all_data,params_lb,ll_fun,start_ll[i],stop_ll[i])) for i in range(len(subs))]
                res_all = [r.get() for r in res_obj]
                pool.close()
                pool.join()
                LL = np.sum(res_all)
                return -LL
                
        else: # non-hybrid RL agent
            
            def apply_ll_fun(opt_var):

                LL = 0
                print(dict(zip(var_name,[l_transfun[j](opt_var[j]) if transformed[j] else opt_var[j] for j in range(len(opt_var))])))

                if ('eps_leak1' in var_name and 'eps_leak2' in var_name):
                    params_lb['h']['eps_leak'] = [1, 1]
                if ('alph_leak1' in var_name and 'alph_leak2' in var_name):
                    params_lb['h']['alph_leak'] = [0, 0]

                for j in range(len(opt_var)):

                    # Apply transformation (if applicable)
                    if transformed[j]:
                        ovj = l_transfun[j](opt_var[j])
                    else:
                        ovj = opt_var[j]

                    if var_name[j] in h_vars:
                        if var_name[j]=='hT': 
                            params_lb['h']['n_buffer'] = ovj 
                        elif 'eps_leak' in var_name[j]:
                            if var_name[j]=='eps_leak':
                                params_lb['h']['eps_leak'] = [ovj]
                            elif var_name[j]=='eps_leak1':
                                params_lb['h']['eps_leak'][0] = ovj
                            elif var_name[j]=='eps_leak2':
                                params_lb['h']['eps_leak'][1] = ovj
                        elif 'alph_leak' in var_name[j]:
                            if var_name[j]=='alph_leak':
                                params_lb['h']['alph_leak'] = [ovj]
                            elif var_name[j]=='alph_leak1':
                                params_lb['h']['alph_leak'][0] = ovj
                            elif var_name[j]=='alph_leak2':
                                params_lb['h']['alph_leak'][1] = ovj
                        else:
                            params_lb['h'][var_name[j]] = ovj
                    elif var_name[j]=='w_cnov':
                            params_lb['w'] = [1-ovj, ovj]
                    elif isinstance(params_lb[var_name[j]],list): 
                        params_lb[var_name[j]] = [ovj]
                    else:                                         
                        params_lb[var_name[j]] = ovj

                # Compute loglikelihood (parallelized)
                pool = mp.Pool(mp.cpu_count())
                res_obj = [pool.apply_async(fit_sub_app,args=(subs[i],all_data,params_lb,ll_fun,start_ll[i],stop_ll[i])) for i in range(len(subs))]
                res_all = [r.get() for r in res_obj]
                pool.close()
                pool.join()
                LL = np.sum(res_all)
                return -LL
        
        # Run optimization
        res_i = minimize(fun=apply_ll_fun,x0=x0,bounds=bounds,method=opt_method,options=solver_options) 

        end_opt = timeit.default_timer()
        if not res_i.success: 
            opt_success = False
        num_opt += res_i.nit

        # Print summary
        if verbose: 
            mle_opt_summary([-1,res_i.x,res_i.fun,res_i.success,res_i.nit],var_name,log_file)
        
        res = pd.DataFrame({'subID':        [-1]*len(res_i.x), # subject ID -1 indicates joint fit
                            'var_name':     var_name, # parameter names
                            'mle_ll':       [res_i.fun]*len(res_i.x), # negative loglikelihood (loss)
                            'mle_var':      [funix(rix) if transformed[i_rix] else rix for i_rix, (rix, funix) in enumerate(zip(res_i.x,l_transfun))], # parameter estimates (already transformed back if applicable)
                            'opt_success':  [res_i.success]*len(res_i.x), # whether optimization was successful
                            'opt_it':       [res_i.nit]*len(res_i.x), # number of iterations
                            'opt_time':     [end_opt-start_opt]*len(res_i.x) # time taken
                            })
        
    print(f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')
    
    if not log_file==None: 
        sl.write_log(log_file,f'Optimization sucessful for all data samples: {opt_success}. Time taken: {end_opt-start_opt} s. Number of iterations: {num_opt}.\n')

    return res