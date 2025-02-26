import numpy as np
rng = np.random.default_rng(seed=12345)
import pandas as pd
import os
import matplotlib.pyplot as plt
import src.models.snov.kernel_nov as knov
import src.models.snov.kernel_nov_2d as knov2d
import src.models.snov.kernel_nov_multidim as knov_md
import src.models.snov.gabor_stimuli as gs

####################################################################################################################################################################
# Compute similarity of stimulus with given Gabor kernel
def comp_overlap(stim,k,contrast='off'):
    if contrast=='off':
        m = np.dot(stim.flatten(),k.flatten()) / (np.linalg.norm(stim.flatten()) * np.linalg.norm(k.flatten())) # cosine similarity (contrast-invariant)
    elif contrast=='on':
        m = np.tanh(np.dot(stim.flatten(),k.flatten()) / np.linalg.norm(k.flatten())) # bounded but contrast-variant similarity
    return m

# Compute similarity of stimulus (masked) with given Gabor kernel
def comp_overlap_mask(stim,mask,k,contrast='off'):
    if contrast=='off':
        mm = np.dot((np.diag(mask)*stim).flatten(),k.flatten())
        m = mm / np.linalg.norm(mm.flatten()) # cosine similarity (contrast-invariant)
    elif contrast=='on':
        m = np.tanh(np.dot(np.diag(mask)*stim.flatten(),k.flatten()) / np.linalg.norm(k.flatten())) # bounded but contrast-variant similarity
    return m

# Compute similarity of stimulus vector with given Gabor kernel
def comp_overlap_vec(stim_vec,k,contrast='off',axis=0):
    ml = []
    for i in range(stim_vec.shape[axis]):
        m = comp_overlap(stim_vec[i],k,contrast=contrast)
        ml.append(m)
    m = np.stack(ml)
    return m

# Compute similarity of stimulus vector (masked) with given Gabor kernel
def comp_overlap_mask_vec(stim_vec,mask_vec,k,contrast='off',axis=0):
    ml = []
    for i in range(stim_vec.shape[axis]):
        m = comp_overlap_mask(stim_vec[i],mask_vec,k,contrast=contrast)
        ml.append(m)
    m = np.stack(ml)
    return m

####################################################################################################################################################################
# Simulate kernel-based novelty with given Gabor kernels
def sim_knov_gabor(stim,knum,k,kmu0,ksig0,maps,dwl,eps_k,t0_update=0,plot=False,figsize=None,splot=np.array([]),dim_names=[],rec_time=100,ylim=[0,0.05],dim_types=[],dim_ranges=[],save_plot=False,save_plot_dir='',save_plot_name=''):
    d = len(k)
    if not figsize:
        if d>10:
            a1 = 10 # num columns
            a2 = int(np.ceil(d/10)) # num rows
        else:
            a1 = d
            a2 = 1
        figsize=(4*a1,3*a2)

    # Initialize novelty
    kwl, _, kmat_seql, _, _, _ = knov_md.init_nov(k,kmu0,ksig0,maps,dim_types=dim_types,seq=stim)
    kmatl = [knov.compute_kmat(k[j],kmu0[j],ksig0[j],splot) for j in range(d)]

    if plot:
        # Plot initial novelty
        kkl,pkl,_,_,_ = knov_md.comp_nov(dwl,kwl,kmatl,dim_types)
        width_ratios = np.array([1 if dim_types[i]=='single' else 2.5 for i in range(len(dim_types))]).reshape((a1,a2))
        f,ax_x = plt.subplots(a2,a1,figsize=figsize) #gridspec_kw={'width_ratios':width_ratios} #,constrained_layout=True
        ax = ax_x.flatten()
        for i in range(d):
            if dim_types[i]=='single':
                knov.plot_fam(ax[i],pkl[i],knum,splot,kkl[i])
                # ax[i].set_title(f'{dim_names[i]}')
                # ax[i].set_xlim(dim_ranges[i])
                # ax[i].set_ylim(ylim)
            elif dim_types[i]=='shared':
                # splot_shared = np.meshgrid(maps[i](splot)[0,:],maps[i](splot)[1,:])
                # knov2d.plot_knov2d_gauss_kk(splot_shared,k[i],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                # knov2d.plot_knov2d_gauss_single(kkl[i][0,:,:],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                knov2d.plot_knov2d_gauss(pkl[i],kmu0[i],ksig0[i],knum,dim_ranges[4:],fig_res=5,f=f,ax=ax[i],cax_pos=[0.683, 0.92, 0.315, 0.07])
        f.suptitle('Latent kernels (t=0)')
        # f.set_tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot_dir,f'{save_plot_name}_step-0.svg'))

    # Initialize recording
    pklist = []
    nklist = []
    plist = []
    nlist = []

    # Run task
    for t in range(stim.shape[0]):
        # Compute bandit values
        # curr_kmat_seql = [kmat_seql[j][:,t].reshape((-1,1)) if dim_types[j]=='single' else kmat_seql[j][:,t,t].reshape((-1,1)) for j in range(d)]
        curr_kmat_seql = [kmat_seql[j][:,t].reshape((-1,1)) if dim_types[j]=='single' else kmat_seql[j][:,t,t].reshape((-1,1,1)) for j in range(d)]
        kkl,pkl,nkl,pkd,nkd = knov_md.comp_nov(dwl,kwl,curr_kmat_seql,dim_types)
        pklist.append(np.squeeze(np.stack(pkl)))
        nklist.append(np.squeeze(np.stack(nkl)))
        # pkd = np.sum(dwl*np.squeeze(np.stack(pkl)))
        # nkd = -np.log(pkd)
        plist.append(pkd)
        nlist.append(nkd)

        # Update novelty parameters in response to stimulus
        rk_newl    = knov_md.update_rk_approx(kwl,kmat_seql,t,dim_types)                             # Update responsibilities
        kwl        = knov_md.update_nov_approx(kwl,t+t0_update,rk_newl,knum,eps=eps_k)               # Update kernel weights
        # kw0        = nov.update_nov_approx_flr(kw0,rk_new0,alph):        

        if plot and t%rec_time==0:
            # Plot initial novelty
            kkl,pkl,_,_,_ = knov_md.comp_nov(dwl,kwl,kmatl,dim_types)
            f,ax_x = plt.subplots(a2,a1,figsize=figsize) #,constrained_layout=True,gridspec_kw={'width_ratios':width_ratios})
            ax = ax_x.flatten()
            for i in range(d):
                if dim_types[i]=='single':
                    knov.plot_fam(ax[i],pkl[i],knum,splot,kkl[i])
                    # ax[i].set_title(f'{dim_names[i]} (t={t+1})')
                    # ax[i].set_xlim(dim_ranges[i])
                    # ax[i].set_ylim(ylim)
                elif dim_types[i]=='shared':
                    # splot_shared = np.meshgrid(maps[i](splot)[0,:],maps[i](splot)[1,:])
                    # knov2d.plot_knov2d_gauss_kk(splot_shared,k[i],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                    # plot_knov2d_gauss_single(kkl[i][0,:,:],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                    knov2d.plot_knov2d_gauss(pkl[i],kmu0[i],ksig0[i],knum,dim_ranges[4:],fig_res=5,f=f,ax=ax[i],cax_pos=[0.683, 0.92, 0.315, 0.07])
            f.suptitle(f'Latent kernels (t={t})')
            if save_plot:
                plt.savefig(os.path.join(save_plot_dir,f'{save_plot_name}_step-{t}.svg'))

    return pklist,nklist,plist,nlist

# Simulate kernel-based novelty with given Gabor kernels (fixed learning rate)
def sim_knov_gabor_flr(stim,knum,k,kmu0,ksig0,maps,dwl,alph_k,plot=False,figsize=None,splot=np.array([]),dim_names=[],rec_time=100,ylim=[0,0.05],dim_types=[],dim_ranges=[],save_plot=False,save_plot_dir='',save_plot_name=''):
    d = len(k)
    if not figsize:
        if d>10:
            a1 = 10 # num columns
            a2 = int(np.ceil(d/10)) # num rows
        else:
            a1 = d
            a2 = 1
        figsize=(4*a1,3*a2)

    # Initialize novelty
    kwl, _, kmat_seql, _, _, _ = knov_md.init_nov(k,kmu0,ksig0,maps,dim_types=dim_types,seq=stim)
    kmatl = [knov.compute_kmat(k[j],kmu0[j],ksig0[j],splot) for j in range(d)]

    if plot:
        # Plot initial novelty
        kkl,pkl,_,_,_ = knov_md.comp_nov(dwl,kwl,kmatl,dim_types)
        width_ratios = np.array([1 if dim_types[i]=='single' else 2.5 for i in range(len(dim_types))]).reshape((a1,a2))
        f,ax_x = plt.subplots(a2,a1,figsize=figsize) #gridspec_kw={'width_ratios':width_ratios} #,constrained_layout=True
        ax = ax_x.flatten()
        for i in range(d):
            if dim_types[i]=='single':
                knov.plot_fam(ax[i],pkl[i],knum,splot,kkl[i])
                # ax[i].set_title(f'{dim_names[i]}')
                # ax[i].set_xlim(dim_ranges[i])
                # ax[i].set_ylim(ylim)
            elif dim_types[i]=='shared':
                # splot_shared = np.meshgrid(maps[i](splot)[0,:],maps[i](splot)[1,:])
                # knov2d.plot_knov2d_gauss_kk(splot_shared,k[i],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                # knov2d.plot_knov2d_gauss_single(kkl[i][0,:,:],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                knov2d.plot_knov2d_gauss(pkl[i],kmu0[i],ksig0[i],knum,dim_ranges[4:],fig_res=5,f=f,ax=ax[i],cax_pos=[0.683, 0.92, 0.315, 0.07])
        f.suptitle('Latent kernels (t=0)')
        # f.set_tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot_dir,f'{save_plot_name}_step-0.svg'))

    # Initialize recording
    pklist = []
    nklist = []
    plist = []
    nlist = []

    # Run task
    for t in range(stim.shape[0]):
        # Compute bandit values
        # curr_kmat_seql = [kmat_seql[j][:,t].reshape((-1,1)) if dim_types[j]=='single' else kmat_seql[j][:,t,t].reshape((-1,1)) for j in range(d)]
        curr_kmat_seql = [kmat_seql[j][:,t].reshape((-1,1)) if dim_types[j]=='single' else kmat_seql[j][:,t,t].reshape((-1,1,1)) for j in range(d)]
        kkl,pkl,nkl,pkd,nkd = knov_md.comp_nov(dwl,kwl,curr_kmat_seql,dim_types)
        pklist.append(np.squeeze(np.stack(pkl)))
        nklist.append(np.squeeze(np.stack(nkl)))
        # pkd = np.sum(dwl*np.squeeze(np.stack(pkl)))
        # nkd = -np.log(pkd)
        plist.append(pkd)
        nlist.append(nkd)

        # Update novelty parameters in response to stimulus
        rk_newl    = knov_md.update_rk_approx(kwl,kmat_seql,t,dim_types)                             # Update responsibilities
        kwl        = knov_md.update_nov_approx_flr(kwl,rk_newl,alphl=alph_k)                         # Update kernel weights
        # kw0        = nov.update_nov_approx_flr(kw0,rk_new0,alph):        

        if plot and t%rec_time==0:
            # Plot initial novelty
            kkl,pkl,_,_,_ = knov_md.comp_nov(dwl,kwl,kmatl,dim_types)
            f,ax_x = plt.subplots(a2,a1,figsize=figsize) #,constrained_layout=True,gridspec_kw={'width_ratios':width_ratios})
            ax = ax_x.flatten()
            for i in range(d):
                if dim_types[i]=='single':
                    knov.plot_fam(ax[i],pkl[i],knum,splot,kkl[i])
                    # ax[i].set_title(f'{dim_names[i]} (t={t+1})')
                    # ax[i].set_xlim(dim_ranges[i])
                    # ax[i].set_ylim(ylim)
                elif dim_types[i]=='shared':
                    # splot_shared = np.meshgrid(maps[i](splot)[0,:],maps[i](splot)[1,:])
                    # knov2d.plot_knov2d_gauss_kk(splot_shared,k[i],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                    # plot_knov2d_gauss_single(kkl[i][0,:,:],kmu[i],ksig[i],1,dim_ranges[4:],fig_res=5)
                    knov2d.plot_knov2d_gauss(pkl[i],kmu0[i],ksig0[i],knum,dim_ranges[4:],fig_res=5,f=f,ax=ax[i],cax_pos=[0.683, 0.92, 0.315, 0.07])
            f.suptitle(f'Latent kernels (t={t})')
            if save_plot:
                plt.savefig(os.path.join(save_plot_dir,f'{save_plot_name}_step-{t}.svg'))

    return pklist,nklist,plist,nlist


####################################################################################################################################################################
# Simulate Gabor novelty with given stimulus set and parameters
def run_gabor_knov(gfam,gnov,gnum=16,knum=10,krange=[-1,1],kspace='line',plot_dims=True,plot_kernels=False,nplot=4,rng=np.random.default_rng(seed=12345),mask=True,adj_w=True):

    dim_ranges_gabor = gs.dim_ranges_rad.copy()
    dim_names_gabor = gs.dim_names.copy()

    # Make gnum Gabor dimensions randomly
    kcl    = gs.generate_stim(dim_ranges_gabor,gnum,rng,adj_w=adj_w)
    kcl_df = pd.DataFrame(dict(zip(dim_names_gabor,kcl)))
    kgabor = []
    if mask:
        kmasks = []
    for i in range(len(kcl_df)):
        gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1) 
        kgabor.append(gi)
        if mask:
            mi,_ = gs.comp_gauss(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1)
            kmasks.append(mi)

    # Plot N Gabor dimensions
    if plot_dims:
        res_image = 2*nplot
        f,ax = plt.subplots(nplot,nplot,figsize=(res_image,res_image),constrained_layout=True)
        axf = ax.flatten()  
        for i in range(len(axf)):
            gi = kgabor[i]
            im = axf[i].imshow(gi,cmap='binary',vmin=-1,vmax=1,origin='lower')
            axf[i].set_xticks([])
            axf[i].set_yticks([])
        cax = f.add_axes([1, 0.01, 0.02, 0.96])
        f.colorbar(im,cax=cax)
        f.suptitle('Gabor kernel dimensions')

    # Define kernels for each Gabor dimension
    if kspace=='torus':
        k = [lambda x,loc,scale: knov.fusebounds(knov.k_triangle,krange[0],krange[1],x,loc,scale)]*gnum
    elif kspace=='line':
        k = [knov.k_triangle]*gnum

    # Create centers for each Gabor dimensions
    kmu_i,ksig_i = knov_md.choose_centers([krange],knum,rng,mode='equidistant',type='centered')
    kmu  = [kmu_i[0]]*gnum
    ksig = [ksig_i*np.ones(knum)]*gnum

    # Define dimension parameters
    if mask:
        maps     = [lambda x,kg=kg: comp_overlap_mask(x,km,kg) for kg, km in zip(kgabor,kmasks)]
        maps_vec = [lambda x,kg=kg: comp_overlap_mask_vec(x,km,kg) for kg, km in zip(kgabor,kmasks)]
    else:
        maps = [lambda x,kg=kg: comp_overlap(x,kg) for kg in kgabor]
        maps_vec = [lambda x,kg=kg: comp_overlap_vec(x,kg) for kg in kgabor]
    dim_names = [f'Gabor filter {i}' for i in range(len(maps))]
    dim_types = ['single']*len(maps)
    dim_ranges = [krange]*len(maps)
    
    # stim_gabor = [[m1[0]]*10+[m1[1]]]
    stim_gabor_vec = np.squeeze(np.stack([[gfam]*10+[gnov]]))

    # Run novelty experiment
    pklist_gabor,nklist_gabor,plist_gabor,nlist_gabor = sim_knov_gabor(stim_gabor_vec,knum,k,kmu,ksig,maps_vec,dwl=[1]*len(k),eps_k=[1],t0_update=0,plot=plot_kernels,splot=np.linspace(krange[0],krange[1],100),dim_names=dim_names,rec_time=5,dim_types=dim_types,dim_ranges=dim_ranges)

    pkl_gabor = np.squeeze(np.stack(pklist_gabor))
    nkl_gabor = np.squeeze(np.stack(nklist_gabor))
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    return pkl_gabor, nkl_gabor, pl_gabor, nl_gabor


####################################################################################################################################################################
# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment
def run_gabor_knov_withparams(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name=''):
    # Run novelty experiment
    _,_,plist_gabor,nlist_gabor = sim_knov_gabor(stim,
                                                knum=k_params['knum'],
                                                k=k_params['k'],
                                                kmu0=k_params['kmu'],
                                                ksig0=k_params['ksig'],
                                                maps=k_params['maps_vec'],
                                                dwl=k_params['dwl'],
                                                eps_k=k_params['eps_k'],
                                                t0_update=k_params['t0_update'],
                                                plot=plot_kernels,
                                                splot=k_params['splot'],
                                                dim_names=k_params['dim_names'],
                                                rec_time=20,
                                                dim_types=k_params['dim_types'],
                                                dim_ranges=k_params['dim_ranges'],
                                                save_plot=save_plot,
                                                save_plot_dir=save_plot_dir,
                                                save_plot_name=save_plot_name)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    return df_all

# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment (fixed learning rate)
def run_gabor_knov_withparams_flr(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name=''):
    # Run novelty experiment
    _,_,plist_gabor,nlist_gabor = sim_knov_gabor_flr(stim,
                                                knum=k_params['knum'],
                                                k=k_params['k'],
                                                kmu0=k_params['kmu'],
                                                ksig0=k_params['ksig'],
                                                maps=k_params['maps_vec'],
                                                dwl=k_params['dwl'],
                                                alph_k=k_params['alph_k'],
                                                plot=plot_kernels,
                                                splot=k_params['splot'],
                                                dim_names=k_params['dim_names'],
                                                rec_time=20,
                                                dim_types=k_params['dim_types'],
                                                dim_ranges=k_params['dim_ranges'],
                                                save_plot=save_plot,
                                                save_plot_dir=save_plot_dir,
                                                save_plot_name=save_plot_name)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    return df_all


####################################################################################################################################################################
# Initialize Gabor novelty and return dictionary with parameters
def init_gabor_knov(gnum=16,knum=10,k_type='triangle',krange=[-1,1],kspace='line',seed=12345,rng=None,mask=True,adj_w=True):
    
    # Define random generator and stimulus dimensions
    if not rng: rng = np.random.default_rng(seed)
    dim_ranges_gabor = gs.dim_ranges_rad.copy()
    dim_names_gabor = gs.dim_names.copy()

    # Make gnum Gabor dimensions randomly
    kcl    = gs.generate_stim(dim_ranges_gabor,gnum,rng,adj_w=adj_w)
    kcl_df = pd.DataFrame(dict(zip(dim_names_gabor,kcl)))
    kgabor = []
    if mask:
        kmasks = []
    for i in range(len(kcl_df)):
        gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
        kgabor.append(gi)
        if mask:
            mi,_ = gs.comp_gauss(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1)
            kmasks.append(mi)

    # Define kernels for each Gabor dimension
    if kspace=='torus':
        k = [lambda x,loc,scale: knov.fusebounds(eval(f'knov.k_{k_type}'),krange[0],krange[1],x,loc,scale)]*gnum
    elif kspace=='line':
        k = [eval(f'knov.k_{k_type}')]*gnum

    # Create centers for each Gabor dimensions
    kmu_i,ksig_i = knov_md.choose_centers([krange],knum,rng,mode='equidistant',type='centered')
    kmu  = [kmu_i[0]]*gnum
    ksig = [ksig_i*np.ones(knum)]*gnum

    # Define dimension parameters
    if mask:
        maps     = [lambda x,kg=kg: comp_overlap_mask(x,km,kg) for kg, km in zip(kgabor,kmasks)]
        maps_vec = [lambda x,kg=kg: comp_overlap_mask_vec(x,km,kg) for kg, km in zip(kgabor,kmasks)]
    else:
        maps = [lambda x,kg=kg: comp_overlap(x,kg) for kg in kgabor]
        maps_vec = [lambda x,kg=kg: comp_overlap_vec(x,kg) for kg in kgabor]
    dim_names = [f'Gabor filter {i}' for i in range(len(maps))]
    dim_types = ['single']*len(maps)
    dim_ranges = [krange]*len(maps)

    # Format parameters
    k_params = {'gnum':gnum,
                'knum':knum,
                'k':k,
                'kmu':kmu,
                'ksig':ksig,
                'maps_vec':maps_vec,
                'dwl':[1]*len(k),
                'eps_k':[1],
                't0_update':0,
                'splot':np.linspace(krange[0],krange[1],100),
                'dim_names':dim_names,
                'dim_types':dim_types,
                'dim_ranges':dim_ranges,
                'ref_gabors':kcl_df
    }
    return k_params

####################################################################################################################################################################
# Create input sequences for Homann experiments
def get_random_seed(length,n,init_seed=None):
    if not init_seed: 
        np.random.seed()
    else:
        np.random.seed(init_seed)
    min = 10**(length-1)
    max = 9*min + (min-1)
    return np.random.randint(min, max, n)

def create_single_input_gabor(n_fam=17,len_fam=3,num_gabor=1,seed=12345,plot=False,plotlabel='',adj_w=True,idx=False):
    rng = np.random.default_rng(seed=seed)
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,adj_w=adj_w)
    nov = gs.generate_stim(dim_ranges,num_gabor*1,rng,adj_w=adj_w)
    gfam = []
    for i in range(len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
    gnov,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov.reshape((-1,num_gabor)),resolution=100,magn=1)

    # Create sequence
    if len_fam==1: gfam = [gfam]
    if idx:
        stim_unique = np.stack(gfam + [gnov])
        stim_idx    = list(np.arange(len(gfam)))*n_fam + [len(gfam)] + list(np.arange(len(gfam)))*2
        seq_vec     = (stim_unique,stim_idx)
    else:
        seq = gfam*n_fam + [gnov] + gfam*2
        seq_vec = np.squeeze(np.stack(seq))
    stim_type = ['fam']*len_fam*n_fam + ['nov'] + ['fam']*len_fam*2

    if plot:
        # Plot familiar images
        res_image = 4
        f,ax = plt.subplots(1,len_fam,figsize=(res_image*len_fam,res_image),sharex=True,sharey=True)
        for i in range(len_fam):
            im_fam = ax[i].imshow(gfam[i],cmap='binary',vmin=-1,vmax=1,origin='lower')
        # cax = f.add_axes([0.9, 0.13, 0.02, 0.75])
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        f.suptitle(f'Familiar images ({plotlabel}, seed: {seed})')
        f.patch.set_facecolor('white')
        f.tight_layout()
        # f.colorbar(im_fam,cax=cax)
        # Plot novel image
        f,ax = plt.subplots(1,1,figsize=(res_image,res_image))
        im_nov = ax.imshow(gnov,cmap='binary',vmin=-1,vmax=1,origin='lower')
        # cax = f.add_axes([0.9, 0.13, 0.02, 0.75])
        ax.set_xticks([])
        ax.set_yticks([])
        f.suptitle(f'Novel image ({plotlabel}, seed: {seed})')
        f.patch.set_facecolor('white')
        f.tight_layout()
        # f.colorbar(im_nov,cax=cax)
    
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    return seq_vec, stim_type, [df_fam, df_nov]

def create_input_gabor(n_nov=50,n_fam=17,len_fam=3,num_gabor=1,init_seed=None,seed_list=[],plot=False,plotlabel='',adj_w=True,idx=False):
    if len(seed_list)==0:
        seed_list = list(get_random_seed(5,n_nov,init_seed=init_seed))
    vec_list  = []
    type_list = []
    params_list = []
    for i in range(n_nov):
        vec, stim_type, params = create_single_input_gabor(n_fam,len_fam,num_gabor,seed_list[i],plot,plotlabel,adj_w=adj_w,idx=idx)
        vec_list.append(vec)
        type_list.append(stim_type)
        params_list.append(params)
    return vec_list, seed_list, type_list, params_list

def create_single_repeated_input_gabor(n_fam=17,len_fam=3,dN=0,num_gabor=1,seed=12345,plot=False,plotlabel='',adj_w=True,idx=False):
    rng = np.random.default_rng(seed=seed)
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,adj_w=adj_w)
    nov = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,adj_w=adj_w)
    gfam = []
    gnov = []
    for i in range(len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
        gnov_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gnov.append(gnov_i)

    # Create sequence
    if len_fam==1: 
        gfam = [gfam]
        gnov = [gnov]
    rep_nov = int(np.round((dN/len_fam)))
    if idx:
        stim_unique = np.stack(gfam + gnov)
        stim_idx    = list(np.arange(len(gfam)))*n_fam + list(len(gfam)+np.arange(len(gnov)))*rep_nov + list(np.arange(len(gfam)))*10
        seq_vec     = (stim_unique,stim_idx)
    else:
        seq = gfam*n_fam + gnov*rep_nov+ gfam*10
        seq_vec = np.squeeze(np.stack(seq))
    stim_type = ['fam']*len_fam*n_fam + ['nov']*len_fam*rep_nov + ['fam_r']*len_fam*10

    if plot:
        # Plot familiar images
        res_image = 4
        f,ax = plt.subplots(1,len_fam,figsize=(res_image,res_image*len_fam),sharex=True,sharey=True)
        for i in range(len_fam):
            im_fam = ax[i].imshow(gfam[i],cmap='binary',vmin=-1,vmax=1,origin='lower')
        # cax = f.add_axes([0.9, 0.13, 0.02, 0.75])
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        f.suptitle(f'Familiar images ({plotlabel}, seed: {seed})')
        f.patch.set_facecolor('white')
        # f.colorbar(im_fam,cax=cax)
        # Plot novel image
        f,ax = plt.subplots(1,len_fam,figsize=(res_image,res_image*len_fam),sharex=True,sharey=True)
        for i in range(len_fam):
            im_nov = ax[i].imshow(gnov[i],cmap='binary',vmin=-1,vmax=1,origin='lower')
        # cax = f.add_axes([0.9, 0.13, 0.02, 0.75])
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        ax[-1].set_title(f'Novel images ({plotlabel}, seed: {seed})')
        f.patch.set_facecolor('white')
        # f.colorbar(im_nov,cax=cax)

    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    return seq_vec, stim_type, [df_fam, df_nov]

def create_repeated_input_gabor(n_nov=50,n_fam=17,len_fam=3,dN=0,num_gabor=1,init_seed=None,seed_list=[],plot=False,plotlabel='',adj_w=True,idx=False):
    if len(seed_list)==0:
        seed_list = list(get_random_seed(5,n_nov,init_seed=init_seed))
    vec_list = []
    type_list = []
    params_list = []
    for i in range(n_nov):
        vec, stim_type, params = create_single_repeated_input_gabor(n_fam,len_fam,dN,num_gabor,seed_list[i],plot,plotlabel,adj_w=adj_w,idx=idx)
        vec_list.append(vec)
        type_list.append(stim_type)
        params_list.append(params)
    return vec_list, seed_list, type_list, params_list

####################################################################################################################################################################
def create_tau_emerge_input_gabor(n_fam=[1,3,8,18,38],len_fam=3,num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None):
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)
    
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,num_gabor*1,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    for i in range(len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
    gnov,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov.reshape((-1,num_gabor)),resolution=100,magn=1)

    # Create sequences
    if len_fam==1: gfam = [gfam]
    vec_list  = []
    type_list = []
    for i in range(len(n_fam)):
        if idx:
            stim_unique = np.stack(gfam + [gnov])
            stim_idx    = list(np.arange(len(gfam)))*n_fam[i] + [len(gfam)] + list(np.arange(len(gfam)))*2
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = gfam*n_fam[i] + [gnov] + gfam*2
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = ['fam']*len_fam*n_fam[i] + ['nov'] + ['fam']*len_fam*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, seed, type_list, params

def create_tau_memory_input_gabor(n_fam=17,len_fam=[3,6,9,12],num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None):
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)
    
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,num_gabor*np.max(len_fam),rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,num_gabor*1,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    for i in range(np.max(len_fam)):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
    gnov,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov.reshape((-1,num_gabor)),resolution=100,magn=1)

    # Create sequences
    vec_list  = []
    type_list = []
    for i in range(len(len_fam)):
        if len_fam[i]==1: gfam_i = [gfam[0]]
        else: gfam_i = gfam[:len_fam[i]]
        if idx:
            stim_unique = np.stack(gfam_i + [gnov])
            stim_idx    = list(np.arange(len(gfam_i)))*n_fam + [len(gfam_i)] + list(np.arange(len(gfam_i)))*2
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = gfam_i*n_fam + [gnov] + gfam_i*2
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = ['fam']*len_fam[i]*n_fam + ['nov'] + ['fam']*len_fam[i]*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, seed, type_list, params

def create_tau_recovery_input_gabor(n_fam=22,len_fam=3,dN=[0,70,140,210,280,360,480],num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None):
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)

    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    gnov = []
    for i in range(len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
        gnov_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gnov.append(gnov_i)

    # Create sequence
    if len_fam==1: 
        gfam = [gfam]
        gnov = [gnov]
    vec_list = []
    type_list = []
    for i in range(len(dN)):
        rep_nov = int(np.round((dN[i]/len_fam)))
        if idx:
            stim_unique = np.stack(gfam + gnov)
            stim_idx    = list(np.arange(len(gfam)))*n_fam + list(len(gfam)+np.arange(len(gnov)))*rep_nov + list(np.arange(len(gfam)))*10
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = gfam*n_fam + gnov*rep_nov+ gfam*10
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = ['fam']*len_fam*n_fam + ['nov']*len_fam*rep_nov + ['fam_r']*len_fam*10
        vec_list.append(seq_vec)
        type_list.append(stim_type)
    
    return vec_list, seed, type_list, params
    
