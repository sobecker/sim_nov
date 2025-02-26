import numpy as np
import pandas as pd

####### Novelty model ########################################################
# Initialize novelty variables
def init_nov(states,counts0=np.array([])):
    if len(counts0)==1:
        counts = counts0[0]*np.ones(len(states))
    elif len(counts0)>1:
        counts = counts0.copy()
    else:
        counts = np.zeros(len(states))
    return counts

# Evaluate novelty for all states
def comp_nov(counts,states,t,eps=1,c_norm=0):
    pc = (counts+eps)/(len(states)*eps+t)
    nc = -np.log(pc)+c_norm
    return pc,nc

# Update novelty 
def update_nov(counts,s):
    counts[s] += 1
    return counts

def update_nov_leaky(counts,times,s,k_alph=0.1):
    counts = counts*k_alph
    counts[s] += 1
    times = times*k_alph + 1
    return counts, times

# Plot familiarity
def plot_fam(ax,pc,states):
    ax.plot(states,pc,'o',c='orange')
    ax.set_xlabel('States')
    ax.set_ylabel('Familiarity')

####### Generate Homann inputs ###############################################
def generate_tau_emerge(n_fam,len_fam=3,input_corrected=True):
    s_fam = np.arange(len_fam)  # Create familiar sequence
    s_nov = np.array([len_fam]) # Create novel image

    s_list = []
    t_list = []
    for i in range(len(n_fam)):
        if input_corrected:
            stim = (list(s_fam)*(n_fam[i]+2))[:-1] + list(s_nov) + list(s_fam)*2
            type = (['fam']*len(s_fam)*(n_fam[i]+2))[:-1] + ['nov'] + ['fam']*len(s_fam)*2
        else:
            stim = list(s_fam)*n_fam[i] + list(s_nov) + list(s_fam)*2
            type = ['fam']*len(s_fam)*n_fam[i] + ['nov'] + ['fam']*len(s_fam)*2
        s_list.append(np.array(stim))
        t_list.append(np.array(type))

    return s_list,t_list

def generate_tau_recovery(dN,n_fam=22,len_fam=3,input_corrected=True):
    s_fam   = np.arange(len_fam)          # Create familiar sequence
    s_recov = len_fam + np.arange(len_fam) # Create recovery sequence

    s_list = []
    t_list = []
    for i in range(len(dN)):
        rep_nov = int(np.round((dN[i]/len_fam)))
        if input_corrected:
            stim = list(s_fam)*n_fam + list(s_recov)*rep_nov + list(s_fam)*5
            type = ['fam']*len(s_fam)*n_fam + ['recov']*len(s_recov)*rep_nov + ['fam_r']*len(s_fam)*5
        else:
            stim = list(s_fam)*n_fam + list(s_recov)*rep_nov + list(s_fam)*10
            type = ['fam']*len(s_fam)*n_fam + ['recov']*len(s_recov)*rep_nov + ['fam_r']*len(s_fam)*10
        s_list.append(np.array(stim))
        t_list.append(np.array(type))

    return s_list,t_list

def generate_tau_memory(len_fam,n_fam=17,input_corrected=True):
    s_list = []
    t_list = []
    for i in range(len(len_fam)):
        s_fam = np.arange(len_fam[i])  # Create familiar sequence
        s_nov = np.array([len_fam[i]]) # Create novel image
        if input_corrected:
            stim = (list(s_fam)*(n_fam+1))[:-1] + list(s_nov) + list(s_fam)*2
            type = (['fam']*len(s_fam)*(n_fam+1))[:-1] + ['nov'] + ['fam']*len(s_fam)*2
        else:
            stim = list(s_fam)*n_fam + list(s_nov) + list(s_fam)*2
            type = ['fam']*len(s_fam)*n_fam + ['nov'] + ['fam']*len(s_fam)*2
        s_list.append(np.array(stim))
        t_list.append(np.array(type))

    return s_list,t_list

def generate_input(params_input,input_corrected=True):
    # Create tau_emerge input 
    n_fam   = params_input['n_fam'] # Number of sequence repetitions (L)
    len_fam = 3                     # Length of familiar sequence (M) 
    s_tem, t_tem = generate_tau_emerge(n_fam,len_fam,input_corrected=input_corrected)

    # Create tau_recovery input
    n_fam   = 22                    # Number of sequence repetitions (L)
    len_fam = 3                     # Length of familiar sequence (M) 
    dN      = params_input['dN']    # Number of recovery repetitions (L')
    s_trec, t_trec = generate_tau_recovery(dN,n_fam,len_fam,input_corrected=input_corrected)

    # Create tau_memory input
    n_fam   = 17                    # Number of sequence repetitions (L)
    len_fam = params_input['n_im']  # Length of familiar sequence (M) 
    s_tmem, t_tmem = generate_tau_memory(len_fam,n_fam,input_corrected=input_corrected)

    stim_unique = np.sort(np.array(list(set(np.concatenate(s_tem)).union(set(np.concatenate(s_trec))).union(set(np.concatenate(s_tmem))))))
    count_min   = len(stim_unique)
    
    return [[s_tem,t_tem],[s_trec,t_trec],[s_tmem,t_tmem]], count_min, stim_unique

####### Simulate Homann experiments ##########################################
def sim_experiment(states,input_exp,eps=1,c_norm=0):
    # Initialize novelty model
    counts = init_nov(states)

    s_list,t_list = input_exp[0], input_exp[1]

    # Simulate experiment
    pcl = []; ncl = []
    for t in range(len(s_list)):
        pc,nc  = comp_nov(counts,states,t,eps=eps,c_norm=c_norm) # Compute novelty
        pcl.append(pc[s_list[t]]); ncl.append(nc[s_list[t]])                    # Store familiarity + novelty    
        counts = update_nov(counts,s_list[t])             # Update novelty

    data = pd.DataFrame({'time_step': np.arange(len(s_list)),
                         'stimulus': s_list,
                         'type': t_list,
                         'familiarity': pcl,
                         'novelty': ncl
                         })
    
    return data

def sim_experiment_leaky(states,input_exp,k_alph=0.1,eps=1,c_norm=0):
    # Initialize novelty model
    counts = init_nov(states)
    times = 0

    s_list,t_list = input_exp[0], input_exp[1]

    # Simulate experiment
    pcl = []; ncl = []
    for t in range(len(s_list)):
        pc,nc  = comp_nov(counts,states,times,eps=eps,c_norm=c_norm)                    # Compute novelty
        pcl.append(pc[s_list[t]]); ncl.append(nc[s_list[t]])                            # Store familiarity + novelty    
        counts, times = update_nov_leaky(counts,times,s_list[t],k_alph=k_alph)          # Update novelty

    data = pd.DataFrame({'time_step': np.arange(len(s_list)),
                         'stimulus': s_list,
                         'type': t_list,
                         'familiarity': pcl,
                         'novelty': ncl
                         })
    
    return data

def sim_tau_emerge(states,input_exp,var_exp,eps=1,c_norm=0,k_alph=0.1,leaky=False,steady=False):
    nrl = []
    data_all = []
    if steady: steadyl = []
    for i in range(len(var_exp)):
        # Run experiment 
        if leaky:
            data = sim_experiment_leaky(states,[input_exp[0][i],input_exp[1][i]],k_alph=k_alph,eps=eps,c_norm=c_norm) 
        else:
            data = sim_experiment(states,[input_exp[0][i],input_exp[1][i]],eps=eps,c_norm=c_norm)
        data['n_fam'] = [var_exp[i]]*len(data)

        # Extract normalized novelty response
        nov_idx = np.where(data['type']=='nov')[0][0]
        nov_resp = data['novelty'].values[nov_idx]
        steady_resp = np.mean(data['novelty'].values[max(0,nov_idx-3):nov_idx])
        if steady:
            data['steady'] = [steady_resp]*len(data)
            steadyl.append(steady_resp)
        nrl.append(nov_resp-steady_resp)
        data_all.append(data)

    stats = pd.DataFrame({'n_fam': var_exp,
                          'nt_norm': nrl})
    if steady: stats['steady'] = steadyl

    data_all = pd.concat(data_all)

    return stats, data_all

def sim_tau_memory(states,input_exp,var_exp,eps=1,c_norm=0,k_alph=0.1,leaky=False):
    nrl = []; srl = []
    data_all = []
    for i in range(len(var_exp)):
        # Run experiment 
        if leaky:
            data = sim_experiment_leaky(states,[input_exp[0][i],input_exp[1][i]],k_alph=k_alph,eps=eps,c_norm=c_norm) 
        else:
            data = sim_experiment(states,[input_exp[0][i],input_exp[1][i]],eps=eps,c_norm=c_norm)
        data['n_im'] = [var_exp[i]]*len(data)
        data_all.append(data)

        # Extract normalized novelty response
        nov_idx = np.where(data['type']=='nov')[0][0]
        nov_resp = data['novelty'].values[nov_idx]
        steady_resp = np.mean(data['novelty'].values[max(0,nov_idx-5*var_exp[i]):nov_idx])
        srl.append(steady_resp)
        nrl.append(nov_resp-steady_resp)

    stats_nov = pd.DataFrame({'n_im': var_exp,'nt_norm': nrl})
    stats_steady = pd.DataFrame({'n_im': var_exp,'steady': srl})
    
    data_all = pd.concat(data_all)

    return stats_nov, stats_steady, data_all

def sim_tau_recovery(states,input_exp,var_exp,eps=1,c_norm=0,k_alph=0.1,leaky=False,steady=False):
    nrl = []
    data_all = []
    if steady: steadyl = []
    for i in range(len(var_exp)):
        # Run experiment 
        if leaky:
            data = sim_experiment_leaky(states,[input_exp[0][i],input_exp[1][i]],k_alph=k_alph,eps=eps,c_norm=c_norm) 
        else:
            data = sim_experiment(states,[input_exp[0][i],input_exp[1][i]],eps=eps,c_norm=c_norm)
        data['dN'] = [var_exp[i]]*len(data)

        # Extract normalized novelty response
        nov_idx = np.where(data['type']=='fam_r')[0][0]
        nov_resp = np.mean(data['novelty'].values[nov_idx:min(nov_idx+3,len(data))])
        steady_resp = np.mean(data['novelty'].values[max(0,nov_idx-3):nov_idx])
        if steady:
            data['steady'] = [steady_resp]*len(data)
            steadyl.append(steady_resp)
        nrl.append(nov_resp-steady_resp)
        data_all.append(data)

    stats = pd.DataFrame({'dN': var_exp,
                          'tr_norm': nrl})
    if steady: stats['steady'] = steadyl

    data_all = pd.concat(data_all)

    return stats, data_all



        


