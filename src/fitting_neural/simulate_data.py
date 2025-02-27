import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy import stats
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
import src.utils.saveload as sl
import src.models.snov.run_gabor_knov as gknov
import src.models.snov.run_gabor_knov2 as gknov2
import src.fitting_neural.load_homann_data as load_homann

# def load_exp_data(h_type,path_load_exp,filter_emerge=True):
#     emu = pd.read_csv(os.path.join(path_load_exp,f'Homann2022_{h_type}_mean.csv'))
#     lstd = pd.read_csv(os.path.join(path_load_exp,f'Homann2022_{h_type}_lowerstd.csv'))
#     ustd = pd.read_csv(os.path.join(path_load_exp,f'Homann2022_{h_type}_upperstd.csv'))
#     if ' y' in emu.columns: emu = emu.rename(columns={' y':'y'})
#     if ' y' in lstd.columns: lstd = lstd.rename(columns={' y':'y'})
#     if ' y' in ustd.columns: ustd = ustd.rename(columns={' y':'y'})
#     edata = emu.rename(columns={'y':'y_mean'})
#     edata['lstd'] = np.round(lstd['y'],4)
#     edata['ustd'] = np.round(ustd['y'],4)
#     if h_type=='steadystate':
#         edata['x']   = [3,6,9,12]
#     edata = edata.sort_values('x')
#     if h_type=='tau_emerge' and filter_emerge:
#         edata = edata.iloc[::2]
#     edata['x'] = np.round(edata['x'].values)
#     edata['y_mean'] = np.round(edata['y_mean'].values,4)
#     return edata

# def load_exp_data2(h_type,filter_emerge=True):
#     # Load experimental data (for fitting measure computation)
#     if h_type=='tau_memory':
#         # Load novelty responses
#         path1  = os.path.join(sl.get_datapath().replace('data','ext_data'),f'Homann2022/Homann2022_{h_type}_mean.csv')
#         edata1 = pd.read_csv(path1)
#         edata1 = edata1.sort_values('x')
#         edx    = list(map(lambda x: int(np.round(x)),edata1['x']))
#         edy1   = np.array(list(map(lambda x: np.round(x,4),edata1[' y'])))
#         # Load steady state responses
#         path2  = os.path.join(sl.get_datapath().replace('data','ext_data'),f'Homann2022/Homann2022_steadystate_mean.csv')
#         edata2 = pd.read_csv(path2)
#         edata2 = edata2.rename(columns={'y':' y'})
#         edy2   = np.array(list(map(lambda x: np.round(x,4),edata2[' y'])))
#         # Combine
#         edy    = [edy1,edy2]
#     else:
#         # Load novelty responses
#         path = os.path.join(sl.get_datapath().replace('data','ext_data'),f'Homann2022/Homann2022_{h_type}_mean.csv')
#         edata = pd.read_csv(path)
#         edata = edata.sort_values('x')
#         if h_type=='tau_emerge' and filter_emerge:
#             edata = edata.iloc[::2]
#         edx   = list(map(lambda x: int(np.round(x)),edata['x']))
#         edy   = np.array(list(map(lambda x: np.round(x,4),edata[' y'])))
#     return edx,edy

def plot_fits(edx,stats_nov,tr_name,edata,mu,xl,yl,title,savename,savepath,plot_fitted=True,pfit=[]):
        f,ax = plt.subplots(1,1)
        y       = stats_nov[tr_name,'mean']
        yerr    = stats_nov[tr_name,'sem']
        rf      = np.nanmax(y)-np.nanmin(y)
        y_rf    = y/rf
        yerr_rf = yerr/rf
        ax.scatter(edx,y_rf,c=['k']*len(edx))
        ax.errorbar(x=edx,y=y_rf,yerr=yerr_rf,c='k')
        ye      = edata
        rfe     = np.nanmax(ye)-np.nanmin(ye)
        ye_rf   = ye/rfe
        ax.scatter(edx,ye_rf,c=['r']*len(edx))
        if plot_fitted:
            xfitted = np.linspace(np.min(edx),np.max(edx),1000)
            yfitted = pfit[0]*np.exp(-1/pfit[1] * xfitted) + pfit[2]
            ax.plot(xfitted,yfitted,'r--')
        ax.set_title(f'Correlation: {mu}')
        ax.set_xticks(edx)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        # savename = f"fit_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','') if 'reset' in t_type else str(np.round(t_alph,4)).replace('.','')}"
        # title = f"J={k_num}, $\\epsilon$={np.round(t_eps,4)}" if 'reset' in t_type else f"J={k_num}, $\\alpha$={np.round(t_alph,4)}" 
        f.suptitle(title)
        f.tight_layout()
        plt.savefig(os.path.join(savepath,savename+'.svg'))
        plt.savefig(os.path.join(savepath,savename+'.eps'))
    
def get_nov_response(data,aggname,window_size=3,steady=False,get_stats=True,vary_with_m=False,cycle_m=5):
        loc_nov     = np.where(data.stim_type.values=='nov')[0]
        if vary_with_m:
            steady_nov = [np.mean(data.nt.values[max(i-cycle_m*data.n_im.values[i],0):i]) for i in loc_nov] 
        else:
            steady_nov = [np.mean(data.nt.values[max(i-window_size,0):i]) for i in loc_nov]           # steady state activity before each novelty response
        data_nov    = data.loc[data.stim_type=='nov']                                       # extract novelty responses
        data_nov['steady']  = steady_nov                                                    # add steady states
        data_nov['nt_norm'] = data_nov.nt-data_nov.steady                                   # compute normalized novelty responses
        if get_stats:
            scols       = [aggname,'nt','nt_norm']
            scols       = scols+['steady'] if steady else scols
            stats_nov   = data_nov[scols].groupby([aggname]).agg([np.mean,np.std,stats.sem])  
        else:
            stats_nov   = None
            data_nov    = data_nov.reset_index(drop=True)
        return data_nov, stats_nov       

def get_trans_response(data,aggname,len_seq=3,window_size=3,get_stats=True,steady=False):
        data    = data.loc[data.stim_type!='nov']
        loc_nov = list(set(np.where(data.stim_type.values=='fam_r')[0]) & set(np.where(data.stim_type.values=='fam')[0]+1)) # get indices where familiar sequence is 'reshown'
        loc_nov.sort()
        steady_nov  = [np.mean(data.nt.values[max(i-window_size,0):i]) for i in loc_nov]           # steady state activity in the end of adaptation period
        tr_nov      = [np.mean(data.nt.values[i:i+len_seq]) for i in loc_nov]                # transient response to repeated familiar stimulus
        data['steady']  = [np.NaN]*len(data)
        data['tr']      = [np.NaN]*len(data) 
        for i in range(0,len(loc_nov)):                                                     # add steady state
            if i==len(loc_nov)-1:   
                data['steady'].iloc[loc_nov[i]:] = steady_nov[i]
                data['tr'].iloc[loc_nov[i]:]     = tr_nov[i]      
            else:                   
                data['steady'].iloc[loc_nov[i]:loc_nov[i+1]] = steady_nov[i] 
                data['tr'].iloc[loc_nov[i]:loc_nov[i+1]]     = tr_nov[i] 
        data_nov            = data.loc[data.stim_type=='fam_r']                             # extract data for familiar sequences after delay
        data_nov['nt_norm'] = data_nov['nt']-data_nov['steady']                             # compute normalized responses
        data_nov['tr_norm'] = data_nov['tr']-data_nov['steady']                             # compute normalized transient responses
        aggvals = ['tr','tr_norm','steady'] if steady else ['tr','tr_norm']
        if get_stats:
            stats_nov = data_nov[[aggname] + aggvals].groupby([aggname]).agg([np.mean,np.std,stats.sem])      # stats of novelty, steady state and normalized novelty responses 
        else:
            data_nov = data_nov[[aggname] + aggvals].drop_duplicates().reset_index(drop=True)
            stats_nov = None
        return data_nov, stats_nov

def comp_tau_emerge(t_type,t_alph,t_eps,k_type,k_num,k_params,gabor_seed,ksig=1,num_gabor=1,edx=[1,3,8,18,38],n_images=3,n_nov=20,edata=[],norm=True,hp=False,hp_alph=0.5,plot_fits=False,savepath='',file_data='',file_stats='',input_fs=[],input_fb=None,full_update=False,return_data=False,return_stats=False,input_seed=54321): ##n_nov=20
    # Run variable repetition experiment
    n_fam = edx
    n_im  = n_images
    k_params['alph_k'] = t_alph
    k_params['eps_k']  = t_eps
    if hp:
        sim_fun = gknov2.run_gabor_knov_withparams_flr
    else:
        sim_fun = gknov2.run_gabor_knov_withparams
    all_data = []
    input_seed_all = gknov.get_random_seed(5,n_nov,input_seed)
    for i in range(n_nov):
        ivec, iseed, itype, iparams = gknov.create_tau_emerge_input_gabor(n_fam=n_fam,len_fam=n_images,num_gabor=num_gabor,input_seed=input_seed_all[i]) # Create inputs 
        # ivec, iseed, itype, iparams = gknov.create_input_gabor(n_nov,n_fam[i],n_images,num_gabor=num_gabor)  # Create inputs 
        for j in range(len(ivec)):
            idata,_,_ = sim_fun(ivec[j],k_params)                  # Simulate experiment
            idata['n_fam'] = [n_fam[j]]*len(idata) 
            idata['input_seed']  = [iseed]*len(idata)  
            idata['stim_type'] = itype[j]   
            idata['sample_id'] = [i]*len(idata)               
            all_data.append(idata)
            print(f'Done with experimental condition {j}/{len(n_fam)}.')
        print(f'Done with sample {i}/{n_nov}.')
    data = pd.concat(all_data)

    # Process data
    data_nov,stats_nov = get_nov_response(data,'n_fam')          
    if norm:    tr_name = 'nt_norm'
    else:       tr_name = 'nt'
    if not return_data:
        data = None

    # Save data
    dsave = data_nov[['n_fam','sample_id','input_seed','steady','nt','nt_norm']].drop_duplicates().reset_index(drop=True)
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_im','n_nov','ksig']:
        dsave[fi] = [eval(fi)]*len(dsave)
    ssave = stats_nov.reset_index()
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_im','n_nov','ksig']:
        ssave[fi] = [eval(fi)]*len(ssave)
    if len(file_data)>0:
        dsave.to_csv(file_data,mode='a',header=not os.path.exists(file_data),index=False)
    if len(file_stats)>0:
        ssave.to_csv(file_stats,mode='a',header=not os.path.exists(file_stats),index=False)
    if return_stats:
        all_stats = [dsave.copy(), ssave.copy()]
    else:
        all_stats = [None, None]
    
    # Compute correlation between experimental and simulated data
    if len(edata)>0:
        mu = np.round(np.corrcoef(edata,stats_nov[tr_name,'mean'].values)[0,1],4)
        mu_std   = []
        if plot_fits:
            savename = f"fit_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','') if 'reset' in t_type else str(np.round(t_alph,4)).replace('.','')}"
            title    = f"J={k_num}, $\\epsilon$={np.round(t_eps,4)}" if 'reset' in t_type else f"J={k_num}, $\\alpha$={np.round(t_alph,4)}" 
            xl   = 'Number of repetitions (L)'
            yl   = 'Novelty response'
            pfit = [-1.04,5.45,1.3]
            plot_fits(edx,stats_nov,tr_name,edata,mu,xl,yl,title,savename,savepath,pfit=pfit)
    else:
        mu = None; mu_std = []

    return [mu, mu_std], data, all_stats

def comp_tau_recovery(t_type,t_alph,t_eps,k_type,k_num,k_params,gabor_seed,ksig=1,num_gabor=1,n_fam=22,n_images=3,n_nov=20,edx=[0,21,42,63,84,108,144],edata=[],norm=True,hp=False,hp_alph=0.5,plot_fits=False,savepath='',file_data='',file_stats='',input_fb=None,input_fs=[],full_update=False,return_data=False,return_stats=False,input_seed=54321):
    dT = edx
    dN = [int(dT[i]/0.3) for i in range(len(dT))] 
    n_im = n_images
    k_params['alph_k'] = t_alph
    k_params['eps_k']  = t_eps
    if hp:
        sim_fun = gknov2.run_gabor_knov_withparams_flr
    else:
        sim_fun = gknov2.run_gabor_knov_withparams
    all_data = []
    input_seed_all = gknov.get_random_seed(5,n_nov,input_seed)
    for i in range(n_nov):
        ivec, iseed, itype, iparams = gknov.create_tau_recovery_input_gabor(n_fam=n_fam,len_fam=n_images,dN=dN,num_gabor=num_gabor,input_seed=input_seed_all[i]) # dN=[0,70,140,210,280,360,480]
        # ivec, iseed, itype, iparams = gknov.create_repeated_input_gabor(n_nov,n_fam,n_images,dN[i],num_gabor=num_gabor)  # Create inputs
        for j in range(len(ivec)):
            idata,_,_ = sim_fun(ivec[j],k_params)                  # Simulate experiment
            idata['dN']    = [dN[j]]*len(idata) 
            idata['dT']    = [dT[j]]*len(idata)
            idata['input_seed']  = [iseed]*len(idata)  
            idata['stim_type'] = itype[j]   
            idata['sample_id'] = [i]*len(idata)                                 
            all_data.append(idata)
            print(f'Done with experimental condition {j}/{len(dN)}.')
        print(f'Done with sample {i}/{n_nov}.')
    data = pd.concat(all_data)

    # Process data
    data_nov,stats_nov = get_trans_response(data,'dN')    
    if norm:    tr_name = 'tr_norm'
    else:       tr_name = 'tr'
    if not return_data: 
        data = None

    # Save data
    dsave = data_nov[['dN','sample_id','input_seed','dT','steady','tr','tr_norm']].drop_duplicates().reset_index(drop=True)
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_im','n_nov','n_fam','ksig']:
        dsave[fi] = [eval(fi)]*len(dsave)
    ssave = stats_nov.reset_index()
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_im','n_nov','n_fam','ksig']:
        ssave[fi] = [eval(fi)]*len(ssave)
    if len(file_data)>0:
        dsave.to_csv(file_data,mode='a',header=not os.path.exists(file_data),index=False)
    if len(file_stats)>0:
        ssave.to_csv(file_stats,mode='a',header=not os.path.exists(file_stats),index=False)
    if return_stats:
        all_stats = [dsave.copy(), ssave.copy()]
    else:
        all_stats = [None, None]
  
    # Compute correlation between experimental and simulated data
    if len(edata)>0:
        mu = np.round(np.corrcoef(edata,stats_nov[tr_name,'mean'].values)[0,1],4)
        mu_std   = []
        if plot_fits:
            savename = f"fit_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','') if 'reset' in t_type else str(np.round(t_alph,4)).replace('.','')}"
            title    = f"J={k_num}, $\\epsilon$={np.round(t_eps,4)}" if 'reset' in t_type else f"J={k_num}, $\\alpha$={np.round(t_alph,4)}" 
            xl   = 'Number of images (dN)'
            yl   = 'Transient response'
            pfit = [-1,123,1.2]
            plot_fits(edx,stats_nov,tr_name,edata,mu,xl,yl,title,savename,savepath,pfit=pfit)
    else:
        mu = None; mu_std = []

    return [mu, mu_std], data, all_stats

def comp_tau_memory(t_type,t_alph,t_eps,k_type,k_num,k_params,gabor_seed,ksig=1,num_gabor=1,edx=[3,6,9,12],n_fam=17,n_nov=20,edata=[],norm=True,hp=False,hp_alph=0.5,plot_fits=False,savepath=['',''],file_data=['',''],file_stats=['',''],input_fb=None,input_fs=[],full_update=False,return_data=False,return_stats=False,input_seed=54321):
    # Run variable repetition experiment
    n_images = edx
    k_params['alph_k'] = t_alph
    k_params['eps_k']  = t_eps
    if hp:
        sim_fun = gknov2.run_gabor_knov_withparams_flr
    else:
        sim_fun = gknov2.run_gabor_knov_withparams
    all_data = []
    input_seed_all = gknov.get_random_seed(5,n_nov,input_seed)
    for i in range(n_nov):
        ivec, iseed, itype, iparams = gknov.create_tau_memory_input_gabor(n_fam=n_fam,len_fam=n_images,num_gabor=num_gabor,idx=True,seed=input_seed_all[i]) 
        # ivec, iseed, itype, iparams = gknov.create_input_gabor(n_nov,n_fam,n_images[i],num_gabor=num_gabor)  # Create inputs
        for j in range(len(ivec)):
            idata,_,_ = sim_fun(ivec[j],k_params)                  # Simulate experiment
            idata['n_im'] = [n_images[j]]*len(idata) 
            idata['input_seed']  = [iseed]*len(idata)  
            idata['stim_type'] = itype[j]   
            idata['sample_id'] = [i]*len(idata)                                 
            all_data.append(idata)
            print(f'Done with experimental condition {j}/{len(n_images)}.')
        print(f'Done with sample {i}/{n_nov}.')
    data = pd.concat(all_data)

    # Process data (novelty responses)
    data_nov,stats_nov = get_nov_response(data,'n_im') 
    if norm:    tr_name = 'nt_norm'
    else:       tr_name = 'nt'

    # Save data (novelty responses)
    dsave = data_nov[['n_im','sample_id','input_seed','steady','nt','nt_norm']].drop_duplicates().reset_index(drop=True)
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_nov','n_fam','ksig']:
        dsave[fi] = [eval(fi)]*len(dsave)
    ssave = stats_nov.reset_index()
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_nov','n_fam','ksig']:
        ssave[fi] = [eval(fi)]*len(ssave)
    if len(file_data[0])>0:
        dsave.to_csv(file_data[0],mode='a',header=not os.path.exists(file_data[0]),index=False)
    if len(file_stats[0])>0:
        ssave.to_csv(file_stats[0],mode='a',header=not os.path.exists(file_stats[0]),index=False)
    if return_stats:
        all_stats = [dsave.copy(), ssave.copy()]
    else:
        all_stats = [None, None]

    # Compute fit measure (novelty responses)
    if len(edata)>0:
        edata_nov = edata[0]
        mu = np.round(np.corrcoef(edata_nov,stats_nov[tr_name,'mean'].values)[0,1],4)
        mu_std   = []
        if plot_fits:
            savename = f"fit_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','') if 'reset' in t_type else str(np.round(t_alph,4)).replace('.','')}"
            title    = f"J={k_num}, $\\epsilon$={np.round(t_eps,4)}" if 'reset' in t_type else f"J={k_num}, $\\alpha$={np.round(t_alph,4)}" 
            xl   = 'Number of images (S)'
            yl   = 'Novelty response'
            pfit = [2.2,10,0.5]
            plot_fits(edx,stats_nov,tr_name,edata_nov,mu,xl,yl,title,savename,savepath[0],pfit=pfit)
    else:
        mu = None; mu_std = []

    # Process data (steady state novelty)
    data_steady,stats_steady = get_nov_response(data,'n_im',steady=True) 
    if not return_data: 
        data = None
    
    # Save data (steady state novelty)
    dsave = data_steady[['n_im','sample_id','input_seed','steady','nt','nt_norm']].drop_duplicates().reset_index(drop=True)
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_nov','n_fam','ksig']:
        dsave[fi] = [eval(fi)]*len(dsave)
    ssave = stats_steady.reset_index()
    for fi in ['t_eps','t_alph','k_num','gabor_seed','n_nov','n_fam','ksig']:
        ssave[fi] = [eval(fi)]*len(ssave)
    if len(file_data[1])>0:
        dsave.to_csv(file_data[1],mode='a',header=not os.path.exists(file_data[1]),index=False)
    if len(file_stats[1])>0:
        ssave.to_csv(file_stats[1],mode='a',header=not os.path.exists(file_stats[1]),index=False)
    if return_stats:
        all_stats.append(dsave)
        all_stats.append(ssave)
    else:
        all_stats.extend([None,None])

    # Compute fit measure (steady state novelty)
    if len(edata)>0:
        edata_steady = edata[1]
        mu_steady = np.corrcoef(edata_steady,stats_steady['steady','mean'])[1,1]
        mu_std_steady   = []
        if plot_fits:
            savename = f"fit_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','') if 'reset' in t_type else str(np.round(t_alph,4)).replace('.','')}"
            title    = f"J={k_num}, $\\epsilon$={np.round(t_eps,4)}" if 'reset' in t_type else f"J={k_num}, $\\alpha$={np.round(t_alph,4)}" 
            xl   = 'Number of images (S)'
            yl   = 'Steady state novelty'
            tr_name = 'steady'
            plot_fits(edx,stats_steady,tr_name,edata_steady,mu_steady,xl,yl,title,savename,savepath[1],plot_fits=False)
    else:
        mu_steady = None; mu_std_steady = []
        
    return [mu, mu_std, mu_steady, mu_std_steady], data, all_stats

if __name__=='__main__':

    ## Data saving ###################################################################################################################
    f_overwrite = False     # if True, overwrite the data/stats files; if False, add to existing data/stats file
    save_data   = True
    save_stats  = True
    plot_fits   = False

    ## Input sampling ################################################################################################################
        # fixed_boxes:      specify fixed number of sampling boxes that is used for all experiments (default: None)
        # fixed_stimuli:    specify fixed set of stimuli from which all experiments sample their input (default: None)
        # if neither of the two are specified, on-demand sampling is applied, i.e. #(boxes) = #(familiar stimuli) + #(novel stimuli) for each experiment
    fixed_boxes         = None  # 13   
    fixed_stimuli       = []    # np.linspace(0,180,100)   
    if fixed_boxes and len(fixed_stimuli)>0:
        fixed_stimuli   = []
        fixed_boxes     = None
        print("Two input sampling methods specified. Default method (on-demand sampling with flexible number of boxes) will be used.")
    input_str = "_sampling-fb" if fixed_boxes else "_sampling-fs" if len(fixed_stimuli)>0 else ""

    ## Novelty type ##################################################################################################################
    full_update   = False       # True: apply full EM update; False: apply incremental EM update (default: False)
    update_str    = "_full-update" if full_update else ""
    hp            = True       # True: apply leaky novelty ('homeostatic plasticity') with leak parameter hp_alph; False: apply non-leaky novelty (default)
    hp_alph       = 0.1
    hp_str        = f'_hp{str(hp_alph).replace(".","")}' if hp else ''
    t_type        = 'reset_t'   # Specifies the novelty learning rate    
    # Options: 
        # 'reset_t':     learning rate 1/(t+eps) with prior eps (free parameter); reset the time counter after each experiment run (default)
        # 'fixed_t':     fixed learning rate alph (free parameter)
    k_type        = 'box'       # Specifies the kernel shape to be applied (available options: 'box','triangle','sigmoid')
    n_type        = 'conv'     # Specifies the novelty type used (see below for different options)
    n_nov         = 50           # Number of simulations per data set
    # Options:
        # 'kernel':     one-dimensional kernel-based novelty, kernels are specified for the single feature dimension
        # 'gabor':      kernel-based novelty in similarity spaces of N randomly chosen Gabor filters 
    if n_type=='gabor':
        num_gabor = 100        # Number of reference Gabor filters
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_sampling = 'basic'
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        ksig = 1
        kcenter = 1
        input_seed = 54321
    if n_type=='gabor2':
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 10
        gabor_sampling = 'basic'
        multigabor_str = f'_gaborstim-{num_gabor}' if num_gabor>1 else ''
        ksig = 1
        kcenter = 1
        input_seed = 54321
    if n_type=='gabor3': ## Simulations with random sampling (box, triangle) and more realistic (50-Gabor) images ##
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 50
        gabor_sampling = 'basic'
        multigabor_str = f'_gaborstim-{num_gabor}' if num_gabor>1 else ''
        ksig = 0.5
        kcenter = 1
        input_seed = 54321
    if n_type=='gabor4': ## Simulations with Sobol sampling (triangle kernels)
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 50
        gabor_sampling = 'sobol_4d'
        multigabor_str = f'_gaborstim-{num_gabor}_{gabor_sampling}' if num_gabor>1 else ''
        ksig = 1
        kcenter = 1
        input_seed = 54321
    if n_type=='gabor5': ## Simulations with box kernels of half the size ##
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 50
        gabor_sampling = 'basic'
        multigabor_str = f'_gaborstim-{num_gabor}' if num_gabor>1 else ''
        ksig = 0.5
        kcenter = 1
        input_seed = 54321
    if n_type=='gabor6': ## Simulations with sigmoid activation function ##
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 5
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 50
        gabor_sampling = 'basic'
        multigabor_str = f'_gaborstim-{num_gabor}' if num_gabor>1 else ''
        ksig = 0.5 # for sigmoid: this means the shift of the sigmoid (the larger, the more rightward-shifted), ksig should be between 0 and 1
        kcenter = 20 # for sigmoid: this means the rescaling of the sigmoid (the larger, the more steep), kcenter should be larger than 5
        input_seed = 54321
    if n_type=='conv': ## Simulations with conv. kernel functions
        init_seed = 12345      # Specifies the random seed for the generation of the reference Gabor filters
        num_seeds = 1
        gabor_seeds = gknov.get_random_seed(5,num_seeds,init_seed)
        num_gabor = 50
        gabor_sampling = 'equidistant'
        cdens = 8 # how many pixels to skip in the convolution, e.g. cdens=4 takes only every 4th pixel of the convolution
        conv = True
        parallel = False # computes activations of conv. kernels in parallel during initialization of each experiment  
        multigabor_str = f'_gaborstim-{num_gabor}' if num_gabor>1 else ''
        ksig = 0.98 
        kcenter = 1 
        input_seed = 54321


    ## Fitting measure #############################################################################################################
    norm    = True              # ???
    
    ## Grid search parameters #####################################################################################################

    # batch_name = 'xxlarge'
    # t_alpha_ll = np.linspace(0.001,0.999,10)
    # t_eps_ll   = np.linspace(0.2,2,10)
    # k_num_ll   = np.arange(3,150,1) 

    # batch_name = 'xlarge'
    # t_alpha_ll = np.linspace(0.001,0.999,10)
    # t_eps_ll   = np.linspace(0.2,2,10)
    # k_num_ll   = np.arange(6,150,5) 

    # batch_name = 'large'
    # t_alpha_ll = np.linspace(0.001,0.999,10)
    # t_eps_ll   = np.linspace(0.2,2,10)
    # k_num_ll   = np.arange(5,15,10) 

    batch_name = 'test_tmem'
    t_alpha_ll = [0.1]
    t_eps_ll   = [1]
    k_num_ll   = [4] 

    h_types  = ['tau_memory']#,'tau_memory','tau_recovery'] # Specifies list of experiments to run ,'tau_memory','tau_recovery'
    # Options: 
        # 'tau_emerge':     Homann et al. (2022) experiment where the number of familiar sequence repetitions (L) is varied
        # 'tau_memory':     Homann et al. (2022) experiment where the number of images in the familiar sequence (M) is varied
        # 'steadystate':    same experiment as 'tau_memory' but the steady state before the novel image presentation is extracted
        # 'tau_recovery':   Homann et al. (2022) experiment where the novelty response after 'recovery interval' (L' repetitions) is measured
    
    ## Simulate ##################################################################################################################
    for i in range(len(h_types)):
            
        h_type = h_types[i]
        h_fun  = eval(f'comp_{h_type}')
        print(f'Simulating experiment {h_type}.')

        # Specify save paths
        path_save_data      = os.path.join(sl.get_datapath(),f'GaborPredictions_{n_type}/heatmaps_{batch_name}_{h_type}_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}/')
        path_save_figs      = path_save_data.replace('data','output')
        sl.make_long_dir(path_save_data)
        sl.make_long_dir(path_save_figs)
        path_save_fits      = os.path.join(path_save_figs,f'fits_{k_type}_{t_type}{hp_str}{input_str}{update_str}')
        if plot_fits: sl.make_long_dir(path_save_fits)
        name_save = f'{h_type}_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}'
        if h_type=='tau_memory':
            path_save_data_steady     = os.path.join(sl.get_datapath(),f'GaborPredictions_{n_type}/heatmaps_{batch_name}_steadystate_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}/')
            path_save_figs_steady     = path_save_data.replace('data','output')
            sl.make_long_dir(path_save_data_steady)
            sl.make_long_dir(path_save_figs_steady)
            path_save_fits_steady     = os.path.join(path_save_figs_steady,f'fits_{k_type}_{t_type}{hp_str}{input_str}{update_str}')
            path_save_fits            = [path_save_fits,path_save_fits_steady]
            if plot_fits: sl.make_long_dir(path_save_fits_steady)
            name_save_steady = f'steadystate_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}'
                
        # Specify plotting options
        title   = f"{'Triangle' if 'triangle' in k_type else 'Box'} kernels ({'decay + reset' if t_type=='reset_t' else 'fixed rate'}{', with homeost. plast.' if hp else ''})"
        yl      = 'Prior $\\epsilon$' if t_type=='reset_t' else 'Learning rate $\\alpha$'
        ytl     = [np.round(t_eps_ll[i],2) for i in range(len(t_eps_ll))] if t_type=='reset_t' else [np.round(t_alpha_ll[i],2) for i in range(len(t_alpha_ll))]

        # Specify file names
        if save_data: # full data for all simulations of each parameter combination
            file_data = os.path.join(path_save_data,f'data_{h_type}_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}.csv')
            if f_overwrite and os.path.exists(file_data): os.remove(file_data)
            if h_type=='tau_memory':
                file_data_steady = os.path.join(path_save_data_steady,f'data_steadystate_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}.csv')
                if f_overwrite and os.path.exists(file_data_steady): os.remove(file_data_steady)
        else:
            file_data = ''
            if h_type=='tau_memory':
                file_data_steady = ''
        if save_stats: # statistics of simulation for each parameter combination
            file_stats = os.path.join(path_save_data,f'stats_{h_type}_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}.csv')
            if f_overwrite and os.path.exists(file_stats): os.remove(file_stats)
            if h_type=='tau_memory':
                file_stats_steady = os.path.join(path_save_data_steady,f'stats_steadystate_{k_type}_{t_type}{hp_str}{input_str}{update_str}{multigabor_str}.csv')
                if f_overwrite and os.path.exists(file_stats_steady): os.remove(file_stats_steady)
        else:
            file_stats = ''
            if h_type=='tau_memory':
                file_stats_steady = ''  
        if h_type=='tau_memory':
            file_data = [file_data,file_data_steady]
            file_stats = [file_stats,file_stats_steady]    
        
        # Load experimental data (for fitting measure computation)
        edx,edy = load_homann.load_exp_data2(h_type,cluster=False)

        # Run grid search simulations and compute fitting measure
        tau         = [] 
        tau_s       = []
        if h_type=='tau_memory':
            tau_steady   = []
            tau_s_steady = []
        ll_k_num    = []
        itv         = t_eps_ll if t_type=='reset_t' else t_alpha_ll 
        itv_name    = 'eps' if t_type=='reset_t' else 'alpha'
        ll_itv      = []
        ll_gseed   = []
        c_i         = 0                                         # number of simulated parameter sets
        c_T         = len(itv)*len(k_num_ll)*len(gabor_seeds)      # total number of parameter sets to be simulated
        for j in range(len(k_num_ll)):                             # iterate over number of kernels
            for s in range(len(gabor_seeds)):                   # iterate over random seeds for reference filters
                if 'conv' in n_type:
                    k_params = gknov2.init_gabor_knov(gnum=k_num_ll[j],
                                                    k_type=k_type,
                                                    ksig=ksig,
                                                    kcenter=kcenter,
                                                    seed=gabor_seeds[s],
                                                    rng=None,
                                                    sampling=gabor_sampling,
                                                    conv=conv,
                                                    cdens=cdens,
                                                    parallel=parallel,
                                                    eps_k=t_eps_ll[0],
                                                    alph_k=t_alpha_ll[0]
                                                    )
                else:
                    k_params = gknov2.init_gabor_knov(gnum=k_num_ll[j],
                                                    k_type=k_type,
                                                    ksig=ksig,
                                                    kcenter=kcenter,
                                                    seed=gabor_seeds[s],
                                                    rng=None,
                                                    sampling=gabor_sampling,
                                                    eps_k=t_eps_ll[0],
                                                    alph_k=t_alpha_ll[0]
                                                    )
                for i in range(len(itv)):                       # iterate over prior/learning rate
                    ll_itv.append(itv[i])
                    ll_k_num.append(k_num_ll[j])
                    ll_gseed.append(gabor_seeds[s])
                    c_i+=1
                    if t_type=='reset_t':
                        t_eps   = itv[i]
                        t_alpha = t_alpha_ll[0]
                    else:
                        t_eps   = t_eps_ll[0]
                        t_alpha = itv[i]
                    meas, _, _ = h_fun(t_type=t_type,t_alph=t_alpha,t_eps=t_eps,k_type=k_type,k_num=k_num_ll[j],k_params=k_params,ksig=ksig,
                                edx=edx,edata=edy,norm=norm,hp=hp,hp_alph=hp_alph,n_nov=n_nov,num_gabor=num_gabor,gabor_seed=gabor_seeds[s],
                                plot_fits=plot_fits,savepath=path_save_fits,file_data=file_data,file_stats=file_stats,
                                input_fb=fixed_boxes,input_fs=fixed_stimuli,full_update=full_update,input_seed=input_seed)
                    tau_ij = meas[0]
                    tau_s_ij = meas[1]
                    tau.append(tau_ij); tau_s.append(tau_s_ij)
                    if h_type=='tau_memory': 
                        tau_ij_steady = meas[2]
                        tau_s_ij_steady = meas[3]
                        tau_steady.append(tau_ij_steady)
                        tau_s_steady.append(tau_s_ij_steady)
                    print(f'Done with parameter set {c_i}/{c_T} ({h_type}).')
        # Save fit measures for novelty responses
        df_tau = pd.DataFrame({f'{itv_name}':ll_itv,'k_num':ll_k_num,'gabor_seed':ll_gseed,'tau':tau})
        if len(tau_s)>0: 
            df_tau['tau_s'] = tau_s
        df_tau.to_csv(os.path.join(path_save_data,name_save+'.csv'))   
        # Save fit measures for steady state novelty
        if h_type=='tau_memory':
            df_tau_steady = pd.DataFrame({f'{itv_name}':ll_itv,'k_num':ll_k_num,'gabor_seed':ll_gseed,'tau':tau_steady})
            if len(tau_s_steady)>0: 
                df_tau_steady['tau_s'] = tau_s_steady
            df_tau_steady.to_csv(os.path.join(path_save_data_steady,name_save_steady+'.csv'))   
        print(f'Done simulating experiment {h_type}.')
