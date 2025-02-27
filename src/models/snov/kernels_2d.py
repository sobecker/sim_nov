import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy import stats 
from scipy.optimize import curve_fit
import os
import sys
import utils.saveload as sl
from scipy.stats import multivariate_normal 

#####################################################################################################################
#           Kernel-related functions                                                                                #
#####################################################################################################################
k_gauss_2d     = lambda x,loc,scale,k: multivariate_normal(mean=loc[k,:],cov=scale[k,:]).pdf(x)
k_triangle_2d  = lambda x,loc,scale,k: (1-np.abs(x-loc)/scale)*np.heaviside(1-np.abs(x-loc)/scale,0)
k_box_2d       = lambda x,loc,scale,k: 1/(2*scale[k,0]*2*scale[k,1])*(np.heaviside(scale[k,0]+loc[k,0]-x[:,0],1)-np.heaviside(-scale[k,0]+loc[k,0]-x[:,0],1))*(np.heaviside(scale[k,1]+loc[k,1]-x[:,1],1)-np.heaviside(loc[k,1]-scale[k,1]-x[:,1],1))

def k_wrapper(f,x,loc,scale,lb=None,rb=None,fuse_boundary=False):
    if not fuse_boundary:
        if lb!=None:    k = 2*f(x,lb,scale)*np.heaviside(x-lb,1)  
        elif rb!=None:  k = 2*f(x,rb,scale)*np.heaviside(rb-x,1)
        else:           k = f(x,loc,scale) 
    else:
        if lb!=None and rb!=None:    k = f(x,lb,scale)*np.heaviside(x-lb,1)+f(x,rb,scale)*np.heaviside(rb-x,0)
        else:                        k = f(x,loc,scale)
    return k

def comp_kwidth_2d(k_num,space):
    if 'box'==k_type:                  
        k_width = (space[1,:]-space[0,:])/(2*k_num)
    elif 'triangle_overlap'==k_type:   
        k_width = (space[1]-space[0])/(k_num-1)
    elif 'triangle_overlap_fuse-bound'==k_type:
        k_width = (space[1]-space[0])/k_num
    return k_width

def get_kernel_params_2d(k_type,k_num,k_width,space):
    fuse_boundary = 'fuse-bound' in k_type
    # Set kernel function shape
    if 'gauss' in k_type:        k_fun  = k_gauss_2d
    elif 'triangle' in k_type:   k_fun  = k_triangle_2d
    elif 'box' in k_type:        k_fun  = k_box_2d
    # Set kernel parameters
    lb = np.array([None]*k_num)
    [lb1,lb2] = np.meshgrid(lb,lb)
    rb = np.array([None]*k_num)
    [rb1,rb2] = np.meshgrid(rb,rb)
    lb = np.concatenate([lb1.flatten().reshape((-1,1)),lb2.flatten().reshape((-1,1))],axis=1)
    rb = np.concatenate([rb1.flatten().reshape((-1,1)),rb2.flatten().reshape((-1,1))],axis=1)
    k_widths  = k_width*np.ones((k_num**2,2))   
    k_weights = 1/(k_num**2)*np.ones((k_num**2,1))
    k_centers = np.linspace(space[0],space[1],k_num+1)
    k_centers = k_centers[:-1]+k_width  
    [kc1,kc2] = np.meshgrid(k_centers[:,0],k_centers[:,1])
    k_centers = np.concatenate([kc1.flatten().reshape((-1,1)),kc2.flatten().reshape((-1,1))],axis=1)
    #print(f'Initial mixture weights sum to {np.sum(k_weights)}.')
    #print(k_centers)
    return k_fun, k_centers, k_widths, lb, rb, k_weights

def k_wrapper_2d(f,x,loc,scale,lb=None,rb=None,fuse_boundary=False):
    # if not fuse_boundary:
    #     if lb!=None:    k = 2*f(x,lb,scale)*np.heaviside(x-lb,1)  
    #     elif rb!=None:  k = 2*f(x,rb,scale)*np.heaviside(rb-x,1)
    #     else:           k = f(x,loc,scale) 
    # else:
    #     if lb!=None and rb!=None:    k = f(x,lb,scale)*np.heaviside(x-lb,1)+f(x,rb,scale)*np.heaviside(rb-x,0)
    #     else:                        k = f(x,loc,scale)
    k_num = loc.shape[0]
    kk = [f(x,loc,scale,k).reshape((-1,1)) for k in range(k_num)]
    kk = np.concatenate(kk,axis=1)
    return kk

def plot_kernel_structure(space,k_fun,k_centers,k_widths,lb,rb,ax1,col=None,lt='--',fuse_boundary=False):
    if not col:
         # Set plot color
        cmap = plt.cm.get_cmap('tab20c')
        cnorm = colors.Normalize(vmin=0, vmax=19)
        smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        col  = smap.to_rgba(0)
    x = np.linspace(space[0],space[1],5000)  
    k_num = len(k_centers)
    kk = np.array([k_wrapper(k_fun,x,k_centers[i],k_widths[i],lb[i],rb[i],fuse_boundary=fuse_boundary) for i in range(k_num)])
    for i in range(k_num):
        ax1.plot(x,kk[i,:],lt,c=col)
        ax1.fill_between(x,kk[i,:],color=col,alpha=0.4)
    ax1.set_xlabel('$\\alpha$ (bar orientation)')
    ax1.set_ylabel('$\kappa_i$($\\alpha$)')
    ax1.set_xlim(space)
    # ymax = np.max(kk.flatten())
    # ax1.set_ylim([0,ymax])
    ax1.set_ylim([0,ax1.get_ylim()[-1]])
    #ax1.spines[['top','right','left']].set_visible(False)
        
def plot_kernel_structure_2d(space,k_fun,k_centers,k_widths,lb,rb,ax1,col=None,lt='--',fuse_boundary=False):
    if not col:
         # Set plot color
        cmap = plt.cm.get_cmap('tab20c')
        cnorm = colors.Normalize(vmin=0, vmax=19)
        smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        col  = smap.to_rgba(0)
    xprec = 1000
    [x1,x2] = np.meshgrid(np.linspace(space[0,0],space[1,0],xprec),np.linspace(space[0,1],space[1,1],xprec))
    x = np.concatenate([x1.flatten().reshape((-1,1)),x2.flatten().reshape((-1,1))],axis=1)
    kk = k_wrapper_2d(k_fun,x,k_centers,k_widths,lb,rb,fuse_boundary=fuse_boundary)
    k_num = int(np.sqrt(kk.shape[1]))
    f,ax = plt.subplots(k_num,k_num,sharex=True,sharey=True)
    k_count=0
    for i in range(k_num):
        for j in range(k_num):
            ax[i][j].imshow(kk[:,k_count].reshape((xprec,xprec)),cmap='Blues',extent=(space[0,0],space[1,0],space[0,1],space[1,1]))
            ax[i][j].set_xlim(space[:,0])
            ax[i][j].set_ylim(space[:,1])
            k_count+=1
    f.supxlabel('Orientation (bar 1)')
    f.supylabel('Orientation (bar 2)')
    f.tight_layout()

#####################################################################################################################
#           Input sampling                                                                                          #
#####################################################################################################################
def sample_delta(n_nov=50,len_fam=3,seed=12345,len_nov=1):
    rng = np.random.default_rng(seed=seed)  
    dim = int(space.size/2)
    b = np.linspace(space[0,0],space[1,0],len_fam+len_nov+1) 
    if dim==2:       
        b = np.meshgrid(b[:-1],b[:-1])
        b = np.concatenate([b[i].flatten().reshape((-1,1)) for i in range(len(b))],axis=1)
    draw_delta = lambda size, width: width*rng.random(size) # draw <size> delta from interval [0,width]
    delta = draw_delta((n_nov,dim),(space[1,0]-space[0,0])/(len_fam+1))
    return b, delta, rng

def create_input(n_nov=50,n_fam=17,len_fam=3,seed=12345,space=np.array([-90,90]).reshape((-1,1))):
    # Sample random delta for each equidistant box
    b,delta,rng = sample_delta(n_nov=n_nov,len_fam=len_fam,seed=seed)

    # Generate input sequence
    seq         = []
    seq_type    = []
    sample_id   = []
    len_fam_id  = []
    for i in range(n_nov):
        # Choose familiar/novel stimuli from intervals in random order
        order = list(np.arange(len(b)))
        rng.shuffle(order)
        # Append initial familiar sequence (n_fam repeats)
        b_i_fam = b[order[:-1],:]+delta[i,:]
        seq_i = np.concatenate([b_i_fam]*(n_fam+1))
        # Append novel image
        b_i_nov = b[order[-1],:]+delta[i,:]
        seq_i[-1,:] = b_i_nov
        # Append familiar stimulus sequence after novel stimulus
        seq_i_post = np.concatenate([b_i_fam]*2)
        # Append to overall lists
        seq.append(seq_i)
        seq.append(seq_i_post)
        seq_type.extend(['fam']*(len(seq_i)-1))
        seq_type.append('nov')
        seq_type.extend(['fam']*len(seq_i_post))
        sample_id.extend([i]*(len(seq_i)+len(seq_i_post)))
    seq = np.concatenate(seq)
    len_fam_id = [len_fam]*len(seq)
    seq_df = pd.DataFrame({'len_fam':len_fam_id,'sample_id':sample_id,'stim_angle1':seq[:,0],'stim_angle2':seq[:,1],'stim_type':seq_type})
    return seq_df

def create_repeated_input(n_nov=50,n_fam=17,len_fam=3,dN=0,seed=12345,space=[-90,90]):
    # Sample random delta for each equidistant box
    b,delta,rng = sample_delta(n_nov=n_nov,len_fam=len_fam,seed=seed)
    # Generate input sequence
    seq         = []
    seq_type    = []
    sample_id   = []
    len_fam_id  = []
    for i in range(len(delta)):
        # Choose familiar/novel stimuli from intervals in random order
        order = list(np.arange(len(b)))
        rng.shuffle(order)
        # Append initial familiar sequence (n_fam repeats)
        b_i_fam = list(b[order[:-1],:]+delta[i,:])
        seq_i = np.concatenate([b_i_fam]*n_fam)
        seq_type_i = ['fam']*len(b_i_fam)*n_fam
        # Append delay images (currently: dN repeats of the same image)
        b_i_nov = b[order[-1],:]+delta[i,:]
        seq_i = np.concatenate([seq_i,np.concatenate([b_i_nov]*dN)])
        seq_type_i.extend(['nov']*dN)
        # Append repeated familiar sequence (n_fam repeats)
        seq_i = np.concatenate([seq_i,np.concatenate([b_i_fam]*n_fam)])
        seq_type_i.extend(['fam_r']*len(b_i_fam)*n_fam)
        # Append to overall lists
        seq.append(seq_i)
        seq_type.extend(seq_type_i)
        sample_id.extend([i]*(len(seq_i)))
    seq = np.concatenate(seq)
    dN_id = [dN]*len(seq)
    len_fam_id = [len_fam]*len(seq)
    seq_df = pd.DataFrame({'len_fam':len_fam_id,'dN':dN_id,'sample_id':sample_id,'stim_angle1':seq[:,0],'stim_angle2':seq[:,1],'stim_type':seq_type})
    return seq_df

#####################################################################################################################
#           Novelty-related functions                                                                               #
#####################################################################################################################
def update_nov_step(s,t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph=None):
    # Compute current empirical distribution and novelty at the current state
    kk_s = kw*k_wrapper_2d(k_fun,s.reshape((-1,2)),k_centers,k_widths,lb,rb).transpose()
    pk_s = np.sum(kk_s) 
    nk_s = -np.log(pk_s)
    # Update kernel mixture weights
    rk   = kk_s/pk_s 
    if t_type=='fixed_t':   kw = kw+t_alph*(rk.reshape((-1,1))-kw)
    else:                   kw = kw+1/(t+t_eps*k_num)*(rk.reshape((-1,1))-kw)
    if hp:                  kw = (1-hp_alph)*kw + hp_alph*1/k_num # homeostatic plasticity
    return nk_s, pk_s, kw

def update_nov_step_full(t,kw,rk,s_all,seq_j,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph=None):
    # Compute current empirical distribution and novelty at the current state
    k_num  = len(kw)
    kmat   = k_wrapper_2d(k_fun,s_all.reshape((-1,2)),k_centers,k_widths,lb,rb).transpose()
    kk_all = kw*kmat
    pk_all = np.sum(kk_all,axis=0)  # sum over rows (=kernels)
    pk_all[(pk_all==0).nonzero()[0]] = np.NaN
    kk_all[(pk_all==0).nonzero()[0]] = np.NaN
    rk_all = kk_all/pk_all
    pk_s   = pk_all[seq_j[t,0]]
    nk_s   = -np.log(pk_s)
    # Update kernel mixture weights
    if t_type=='fixed_t':   kw = kw+t_alph*(rk_all[:,seq_j[t,0]].reshape((-1,1))-kw)+np.sum(rk[:,seq_j[:t+1,0]]-rk_all[:,seq_j[:t+1,0]])
    else:                   kw = kw+1/(t+t_eps*k_num)*(rk_all[:,seq_j[t,0]].reshape((-1,1))-kw)+np.sum(rk[:,seq_j[:t+1,0]]-rk_all[:,seq_j[:t+1,0]])
    if hp:                  kw = (1-hp_alph)*kw + hp_alph*1/k_num # homeostatic plasticity
    return nk_s, pk_s, kw, rk

def plot_nov_traces(exp_param, k_fun, k_centers, k_widths, lb, rb, path_save, name_save, space, dim, ax_traces, i, seq_df, ids, save_plot, exp_type):
    f_ti, ax_ti = plt.subplots(len(ids),2,figsize=(6,1*len(ids)))
    for j in range(len(ids)):
        if exp_type=='recovery':
            assign_col  = lambda typestr: 'r' if 'fam' in typestr else 'k'
            assign_s    = lambda typestr: 3 if 'fam' in typestr else 1
        else:
            assign_col  = lambda typestr: 'k' if typestr=='fam' else 'r'
            assign_s    = lambda typestr: 1 if typestr=='fam' else 3
        cols    = list(map(assign_col,seq_df.loc[seq_df.sample_id==ids[j],'stim_type'].values))
        s       = list(map(assign_s,seq_df.loc[seq_df.sample_id==ids[j],'stim_type'].values))
        nt_i    = seq_df.loc[seq_df.sample_id==ids[j],'nt'].values
        # Plot separate plot (novelty traces per S)
        if dim==2: plot_kernel_structure_2d(space,k_fun,k_centers,k_widths,lb,rb,ax_ti[j][0])
        else:      plot_kernel_structure(space,k_fun,k_centers,k_widths,lb,rb,ax_ti[j][0]) # space=[-90,90]
        if j<(len(ids)-1): 
            ax_ti[j][0].set_xlabel('')
            ax_ti[j][0].set_xticks([])
        if exp_type=='recovery':        stim_ij = seq_df.loc[seq_df.sample_id==ids[j],['dN','sample_id','stim_angle','stim_type','dT']].drop_duplicates()
        elif exp_type=='image':         stim_ij = seq_df.loc[seq_df.sample_id==ids[j],['len_fam','sample_id','stim_angle','stim_type','n_im']].drop_duplicates()
        elif exp_type=='repetition':    stim_ij = seq_df.loc[seq_df.sample_id==ids[j],['len_fam','sample_id','stim_angle','stim_type','n_fam']].drop_duplicates()
        ylim_ij = ax_ti[j][0].get_ylim()
        col_ij = list(map(assign_col,stim_ij.stim_type.values))
        for k in range(len(stim_ij)):
            ax_ti[j][0].plot([stim_ij.stim_angle.values[k]]*2,ylim_ij,'-',c=col_ij[k])
        ax_ti[j][1].plot(np.arange(len(nt_i)),nt_i,'k-',lw=1) 
        ax_ti[j][1].scatter(np.arange(len(nt_i)),nt_i,s=s,c=cols)
        ax_ti[j][1].set_xlim([0,len(nt_i)])
        if j==len(ids)-1:
            ax_ti[j][1].set_xlabel('Time steps')
        # Plot into overview plot (nov traces for all S)
        ax_traces[i].plot(np.arange(len(nt_i)),nt_i,'k-',lw=0.5) 
        ax_traces[i].scatter(np.arange(len(nt_i)),nt_i,s=s,c=cols)
    # Plot mean into overview plot (nov traces for all S)
    mean_i = np.mean(seq_df.nt.values.reshape(-1,(len(nt_i))),axis=0)
    ax_traces[i].plot(np.arange(len(nt_i)),mean_i,'-',c='grey',lw=2,alpha=0.5)
    ax_traces[i].set_xlim([0,len(nt_i)])
    ax_traces[i].set_ylabel('Novelty')
    # Save separate plot (nov traces per S)
    f_ti.tight_layout()
    if save_plot:  
        if exp_type=='recovery':        name_save1 = f'{"2d_" if dim==2 else ""}nov-traces_dT-{exp_param[i]}_{name_save}'
        elif exp_type=='image':         name_save1 = f'{"2d_" if dim==2 else ""}nov-traces_S-{exp_param[i]}_{name_save}'
        elif exp_type=='repetition':    name_save1 = f'{"2d_" if dim==2 else ""}nov-traces_L-{exp_param[i]}_{name_save}'
        yl_ti = [np.min([ax_ti[j][1].get_ylim()[0] for j in range(len(ids))]),np.max([ax_ti[j][1].get_ylim()[1] for j in range(len(ids))])]
        [ax_ti[j][1].set_ylim(yl_ti) for j in range(len(ids))]
        f_ti.savefig(os.path.join(path_save,name_save1+'.svg'),bbox_inches='tight')
        f_ti.savefig(os.path.join(path_save,name_save1+'.eps'),bbox_inches='tight')
                
def format_nov_traces(path_save, name_save, f_traces, ax_traces, save_plot):
    f_traces.tight_layout()
    if save_plot:
        name_save1 = f'nov-traces_{name_save}'
        sl.make_long_dir(path_save)
        yl_traces = [np.min([ax_traces[i].get_ylim()[0] for i in range(len(ax_traces))]),np.max([ax_traces[i].get_ylim()[1] for i in range(len(ax_traces))])]
        [ax_traces[i].set_ylim(yl_traces) for i in range(len(ax_traces))]
        ax_traces[-1].set_xlabel('Time steps')
        f_traces.savefig(os.path.join(path_save,name_save1+'.svg'),bbox_inches='tight')
        f_traces.savefig(os.path.join(path_save,name_save1+'.eps'),bbox_inches='tight')

def format_hist(path_save, name_save, f_hist, ax_hist, save_plot):
    if save_plot:
        name_save1 = f'stim-input-hist_{name_save}'
        sl.make_long_dir(path_save)
        yl_hist = [0,np.max([ax_hist[i].get_ylim()[1] for i in range(len(ax_hist))])]
        for i in range(len(ax_hist)):
            ax_hist[i].set_ylim(yl_hist)
            ax_hist[i].set_xlim([-90,90])
            ax_hist[i].set_xticks([])
        ax_hist[-1].set_xticks(np.linspace(-90,90,5))
        ax_hist[-1].set_xlabel('Stimulus angle in degree')
        f_hist.savefig(os.path.join(path_save,name_save1+'.svg'),bbox_inches='tight')
        f_hist.savefig(os.path.join(path_save,name_save1+'.eps'),bbox_inches='tight')

#####################################################################################################################
#           Experiments                                                                                             #
#####################################################################################################################
### Variable repetition experiment ###
def run_variable_repetition_exp_2d(n_images,n_fam,n_nov,k_fun,k_centers,k_widths,lb,rb,k_weights,plot_response=False,r_type='exp_dec',k_type='',path_save='',name_save='',t_type='global_t',t_alph=1,t_eps=1,hp=False,hp_alph=0.5,plot=True,save_plot=True,input_fb=None,input_fs=[],space=np.array([[-90]*2,[90]*2]),full_update=False):
    dim     = 2
    k_num   = len(k_centers)
    seq_df_all = []
    if plot:
        f_hist,ax_hist = plt.subplots(len(n_fam),1,figsize=(3,1.5*len(n_fam)),constrained_layout=True)
        f_traces,ax_traces = plt.subplots(len(n_fam),1,figsize=(3+4,2*len(n_fam)),constrained_layout=True)
    # Run experiment for each image number
    for i in range(len(n_fam)):
        # Create input stimuli and plot 
        seq_df  = create_input(n_nov,n_fam[i],n_images,space=space) 
        seq = seq_df[['stim_angle1','stim_angle2']].values
        ids     = np.unique(seq_df.sample_id)
        # if plot:
        #     plot_input_hist(seq_df,ax=ax_hist[i],legend=i==0)
        #     plot_input(seq_df,path_save=path_save,name_save=name_save)
        # Init kernel weights for novelty experiment
        kw = k_weights.copy() 
        nt = []; pt = []
        if full_update: ######### Full update #####################################################################################################
            rk = 1/k_num*np.ones((1,k_num))
            s_all = np.unique(seq)
            if t_type=='reset_t': 
                for j in range(len(ids)):
                    seq_j = seq_df.loc[seq_df.sample_id==ids[j],['stim_angle1','stim_angle2']].values 
                    idx_seq_j = np.array([np.where(seq_j[t]==s_all) for t in range(len(seq_j))])
                    for t in range(len(idx_seq_j)):
                        nk_s, pk_s, kw, rk = update_nov_step_full(t,kw,rk,seq_j,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph=None)
                        nt.append(nk_s)
                        pt.append(pk_s)
            else:
                for t in range(len(seq)):
                    nk_s, pk_s, kw = update_nov_step(seq[t],t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph,'repetition')
                    nt.append(nk_s)
                    pt.append(pk_s)
        else: ######### Approximate update #########################################################################################################
            if t_type=='reset_t': 
                for j in range(len(ids)):
                    seq_j = seq_df.loc[seq_df.sample_id==ids[j],['stim_angle1','stim_angle2']].values 
                    for t in range(len(seq_j)):
                        nk_s, pk_s, kw = update_nov_step(seq_j[t],t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph)
                        nt.append(nk_s)
                        pt.append(pk_s)
            else:
                for t in range(len(seq)):
                    nk_s, pk_s, kw = update_nov_step(seq[t],t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph,'repetition')
                    nt.append(nk_s)
                    pt.append(pk_s)

        # Save novelty and frequency traces
        seq_df['n_fam'] = [n_fam[i]]*len(seq_df)
        seq_df['nt'] = nt
        seq_df['pt'] = pt
        seq_df_all.append(seq_df)
        if plot: plot_nov_traces(n_fam, k_fun, k_centers, k_widths, lb, rb, path_save, name_save, space, dim, ax_traces, i, seq_df, ids, save_plot,'image')
    # Concatenate data + format/save figs
    df_data = pd.concat(seq_df_all)
    if plot: 
        format_nov_traces(path_save, name_save, f_hist, ax_hist, f_traces, ax_traces, save_plot)
        format_hist(path_save, name_save, f_hist, ax_hist, save_plot)
    return df_data

### Variable image number experiment ###
def run_variable_image_number_exp_2d(n_images,n_fam,n_nov,k_fun,k_centers,k_widths,lb,rb,k_weights,plot_response=False,r_type='exp_dec',k_type='',path_save='',name_save='',t_type='global_t',t_alph=0.5,t_eps=1,hp=False,hp_alph=0.5,plot=True,input_fb=None,input_fs=[],space=np.array([[-90]*2,[90]*2]),full_update=False):
    dim = 2
    k_num = len(k_centers)
    seq_df_all = []
    if plot: f_hist,ax_hist = plt.subplots(len(n_images),1,figsize=(3,1.5*len(n_images)),constrained_layout=True)
    if plot: f_traces,ax_traces = plt.subplots(len(n_images),1,figsize=(3+4,2*len(n_images)),constrained_layout=True)
    # Run experiment for each image number
    for i in range(len(n_images)):
        # Create input stimuli and plot 
        seq_df  = create_input(n_nov,n_fam,n_images[i],space=space) 
        seq = seq_df[['stim_angle1','stim_angle2']].values
        ids     = np.unique(seq_df.sample_id)
        # if plot: plot_input_hist(seq_df,ax=ax_hist[i],legend=i==0)
        # if plot: plot_input(seq_df,path_save=path_save,name_save=name_save)
        kw = k_weights.copy() 
        nt = []; pt = []
        if full_update:
            s_all = np.unique(seq,axis=0)
            rk = 1/k_num*np.ones((k_num,s_all.shape[0]))
            s_all1 = pd.DataFrame({'stim_angle1':s_all[:,0],'stim_angle2':s_all[:,1]}).reset_index().rename(columns={'index':'state_idx'})
            seq_df = seq_df.merge(s_all1,how='left',on=['stim_angle1','stim_angle2'])
            if t_type=='reset_t': 
                for j in range(len(ids)):
                    seq_j = seq_df.loc[seq_df.sample_id==ids[j],['stim_angle1','stim_angle2']].values
                    idx_seq_j = seq_df.loc[seq_df.sample_id==ids[j],['state_idx']].values 
                    for t in range(len(seq_j)):
                        nk_s, pk_s, kw, rk = update_nov_step_full(t,kw,rk,s_all,idx_seq_j,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph=None)
                        nt.append(nk_s)
                        pt.append(pk_s)
            else:
                idx_seq = seq_df['state_idx'].values
                for t in range(len(seq)):
                    nk_s, pk_s, kw, rk = update_nov_step_full(t,kw,rk,s_all,idx_seq,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph=None)
                    nt.append(nk_s)
                    pt.append(pk_s)
        else:
            if t_type=='reset_t': 
                for j in range(len(ids)):
                    seq_j = seq_df.loc[seq_df.sample_id==ids[j],['stim_angle1','stim_angle2']].values 
                    for t in range(len(seq_j)):
                        nk_s, pk_s, kw = update_nov_step(seq_j[t],t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph)
                        nt.append(nk_s)
                        pt.append(pk_s)
            else:
                for t in range(len(seq)):
                    nk_s, pk_s, kw = update_nov_step(seq[t],t,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph)
                    nt.append(nk_s)
                    pt.append(pk_s)
        # Save novelty and frequency traces
        seq_df['n_im'] = [n_images[i]]*len(seq_df)
        seq_df['nt'] = nt
        seq_df['pt'] = pt
        seq_df_all.append(seq_df)
        if plot: plot_nov_traces(n_images, k_fun, k_centers, k_widths, lb, rb, path_save, name_save, space, dim, ax_traces, i, seq_df, ids, save_plot,'image')
    # Concatenate data + format/save figs
    df_data = pd.concat(seq_df_all)
    if plot: 
        format_nov_traces(path_save, name_save, f_hist, ax_hist, f_traces, ax_traces, save_plot)
        format_hist(path_save, name_save, f_hist, ax_hist, save_plot)
    return df_data

### Repeated image set experiment ###
def run_repeated_imageset_exp_2d(n_images,n_fam,n_nov,dT,k_fun,k_centers,k_widths,lb,rb,k_weights,plot_response=False,r_type='exp_dec',k_type='',path_save='',name_save='',t_type='global_t',t_alph=0.5,t_eps=1,hp=False,hp_alph=0.5,plot=True,input_fb=None,input_fs=None,space=np.array([[-90]*2,[90]*2])):
    dim     = 2
    k_num   = len(k_centers)                                   # extract number of kernels
    dN      = [int(dT[i]/0.3) for i in range(len(dT))]         # convert dT into number of non-familiar images (each presented 300 ms)
    if plot:
        f_hist,ax_hist      = plt.subplots(len(dN),1,figsize=(3,1.5*len(dN)),constrained_layout=True)
        f_traces,ax_traces  = plt.subplots(len(dN),1,figsize=(3+4,2*len(dN)),constrained_layout=True)
    seq_df_all  = []
    # Run experiment for recovery period length
    for i in range(len(dN)):
        # Create input stimuli and plot 
        seq_df  = create_repeated_input(n_nov,n_fam,n_images,dN[i],space=space) 
        ids     = np.unique(seq_df.sample_id)
        # if plot:
        #     plot_input_hist(seq_df,ax=ax_hist[i],legend=i==0)
        #     plot_input(seq_df,path_save=path_save,name_save=name_save)
        # Run each trial 
        for j in range(len(ids)):
            seq     = seq_df.loc[seq_df.sample_id==ids[j],['stim_angle1','stim_angle2']].values
            kw = k_weights.copy() 
            nt = []; pt = []
            tlr = 0
            for t in range(len(seq)):
                if seq_df.stim_type[t]=='nov' and seq_df.stim_type[t+1]=='fam_r': tlr = 0
                nk_s,pk_s,kw = update_nov_step(seq[t],tlr,kw,k_fun,k_centers,k_widths,lb,rb,t_eps,k_num,t_type,hp,hp_alph)    
                nt.append(nk_s)
                pt.append(pk_s)
                tlr += 1
            # Save novelty and frequency traces
            seq_df.loc[seq_df.sample_id==ids[j],'nt'] = nt
            seq_df.loc[seq_df.sample_id==ids[j],'pt'] = pt
            seq_df.loc[seq_df.sample_id==ids[j],'dT'] = dT[i]
        seq_df_all.append(seq_df)
        if plot: plot_nov_traces(dT, k_fun, k_centers, k_widths, lb, rb, path_save, name_save, space, dim, ax_traces, i, seq_df, ids, save_plot,'recovery')
    # Concatenate data + format/save figs
    df_data = pd.concat(seq_df_all)
    if plot: 
        format_nov_traces(path_save, name_save, f_hist, ax_hist, f_traces, ax_traces, save_plot)
        format_hist(path_save, name_save, f_hist, ax_hist, save_plot)
    return df_data

#####################################################################################################################
#           Run main                                                                                                #
#####################################################################################################################
if __name__=="__main__":

    ### Parameters ###
    dim = 2
    space = np.array([[-90]*dim,[90]*dim])

    k_type  = 'box' # gauss_overlap, triangle_overlap, box_overlap, box
    k_num   = 4     # number of kernels per stimulus dimension
    k_width = comp_kwidth_2d(k_num,space) 

    t_type      = 'reset_t'
    t_alph      = 0.5
    t_eps       = 0.4
    save_plot   = True
    plot        = False

    full_update = True
    input_fb    = None #13 # default: None
    input_fs    = np.linspace(-90,90,100) # default: []
        
    ### Create title, path and folder for plots ###
    title_kernels = f'{"Triangle" if "triangle" in k_type else "Gaussian" if "gauss" in k_type else "Box"} kernels (n={k_num}, width={k_width})'
    title_plot    = f'{"Triangle" if "triangle" in k_type else "Gaussian" if "gauss" in k_type else "Box"} kernels (n={k_num}, width={k_width}, learning rate: {"constant" if "fixed" in t_type else "decay + reset" if "reset" in t_type else "decay"}, update: {"full" if full_update else "approx."})'
    if save_plot:
        name_save = f'2d_{"full-update_" if full_update else ""}{k_type}_n-{k_num}_w-{k_width}_{t_type}{t_alph if "fixed" in t_type else t_eps}{"_input-fb" if input_fb else ""}{"_input-fs" if len(input_fs)>0 else ""}'
        path_save   = os.path.join(sl.get_datapath().replace('data','output'),f'GaborPredictions_2d/variable_sequence_length/{name_save}/')
        sl.make_long_dir(path_save)
        
    ### Create equidistant kernels ###
    k_fun,k_centers,k_widths,lb,rb,k_weights = get_kernel_params_2d(k_type,k_num,k_width,space)

    ### Plot kernel structure ###
    plot_kernel_structure_2d(space,k_fun,k_centers,k_widths,lb,rb,None,fuse_boundary=('fuse-bound' in k_type))

    ### Run variable image number experiment (parameters) ###
    n_images = [3,6,9,12]
    n_fam = 17
    n_nov = 10
    data = run_variable_image_number_exp_2d(n_images,n_fam,n_nov,k_fun,k_centers,k_widths,lb,rb,k_weights,k_type=k_type,path_save=path_save,name_save=name_save,t_type=t_type,t_alph=t_alph,t_eps=t_eps,plot=plot,input_fb=input_fb,input_fs=input_fs,full_update=full_update)

    ### Set plot color ###
    cmap = plt.cm.get_cmap('tab20c')
    cnorm = colors.Normalize(vmin=0, vmax=19)
    smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    col  = smap.to_rgba(0)
    col1 = smap.to_rgba(4)

    ### Plot novelty response amplitude as function of n_image ###
    f,ax2      = plt.subplots(1,2)
    loc_nov     = np.where(data.stim_type.values=='nov')[0]
    steady_nov  = [np.mean(data.nt.values[i-5:i]) for i in np.where(data.stim_type.values=='nov')[0]] # steady state activity before each novelty response
    data_nov    = data.loc[data.stim_type=='nov']       # extract novelty responses
    data_nov['steady']  = steady_nov                     # add steady states
    data_nov['nt_norm'] = data_nov.nt-data_nov.steady   # compute normalized novelty responses
    stats_nov           = data_nov.groupby(['n_im']).agg([np.mean,np.std,stats.sem]) # stats of novelty, steady state and normalized novelty responses 
    #stats_fam = data.loc[data.stim_type=='fam'].groupby(['n_im']).agg([np.mean,np.std,stats.sem])
    # steady = []
    # for i in range(len(n_images)):
    #     nt_i = data.loc[data.n_im==n_images[i],'nt'].values
    #     dnt_i = nt_i[1:]-nt_i[:-1]
    #     data.loc[data.n_im==n_images[i],'steady'] = [False] + list(np.abs(dnt_i)<=np.mean(np.abs(dnt_i)))
    #     snt_i = nt_i[np.where(np.abs(dnt_i)<=np.mean(np.abs(dnt_i)))[0]]
    #     steady.append(np.mean(snt_i))
    ax2[0].scatter(n_images,stats_nov['nt','mean'],c=[col]*len(n_images))
    ax2[0].errorbar(x=n_images,y=stats_nov['nt','mean'],yerr=stats_nov['nt','sem'],c=col)
    ax2[0].scatter(n_images,stats_nov['nt_norm','mean'].values,c=[col1]*len(n_images))
    ax2[0].errorbar(x=n_images,y=stats_nov['nt_norm','mean'].values,yerr=stats_nov['nt_norm','sem'],c=col1)
    # Fit exponential to novelty response per number of images
    try:
        popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(-1/b * t) + c, np.array(n_images,dtype=float), stats_nov['nt','mean'].values)
        popt1, pcov1 = curve_fit(lambda t, a, b, c: a * np.exp(-1/b * t) + c, np.array(n_images,dtype=float), stats_nov['nt_norm','mean'].values)
    except:
        print('No exponential fit possible.')
    else:
        x_fitted = np.linspace(np.min(n_images), np.max(n_images), 1000)
        y_fitted = popt[0] * np.exp(-1/popt[1] * x_fitted) + popt[2]
        y_fitted1 = popt1[0] * np.exp(-1/popt1[1] * x_fitted) + popt1[2]
        ax2[0].plot(x_fitted,y_fitted,':',c=col)
        ax2[0].plot(x_fitted,y_fitted1,':',c=col1)
        ax2[0].annotate(f'$\\tau_{{memory}}$ = {np.round(popt[1],2)}$\pm$ {np.round(np.sqrt(pcov[1,1]),2)}',(5,4.5),c=col)
        ax2[0].annotate(f'$\\tau_{{memory}}$ = {np.round(popt1[1],2)}$\pm$ {np.round(np.sqrt(pcov1[1,1]),2)}',(5,2),c=col1)
    ax2[0].set_xticks(n_images)
    ax2[0].set_xlabel('Sequence length (S)')
    ax2[0].set_ylabel('Novelty response')
    ax2[1].bar(x=np.array(n_images),height=stats_nov['steady','mean'],yerr=[stats_nov['steady','sem'],stats_nov['steady','sem']],color=col)
    #eps = 0.5
    #ax2[1].bar(x=np.array(n_images)-eps,height=stats_fam['nt','mean'],yerr=[stats_fam['nt','sem'],stats_fam['nt','sem']],color=col,width=2*eps)
    #ax2[1].bar(x=np.array(n_images)+eps,height=steady,yerr=[stats_fam['nt','sem'],stats_fam['nt','sem']],color=col1,width=2*eps)
    ax2[1].set_xticks(n_images)
    ax2[1].set_xlabel('Sequence length (S)')
    ax2[1].set_ylabel('Steady-state response')
    # Save figure
    if save_plot:
        name_save = f'nov-response_{name_save}'
        f.tight_layout()
        f.savefig(os.path.join(path_save,name_save+'.svg'),bbox_inches='tight')
        f.savefig(os.path.join(path_save,name_save+'.eps'),bbox_inches='tight')
        
    print('done')

        