import numpy as np
import matplotlib.pyplot as plt
import models.snov.kernel_nov_multidim as knov_md

#############################################################################################################################################################################################################
# Define kernels
def k_gauss_2d(x1,x2,loc1,loc2,scale1,scale2): 
    return 1/(2*np.pi*scale1*scale2)*np.exp(-1/2*(((x1-loc1)/(scale1))**2+((x2-loc2)/(scale2))**2))
    # return multivariate_normal(mean=loc[k,:],cov=scale[k,:]).pdf(x)

def k_triangle_2d(x1,x2,loc1,loc2,scale1,scale2): 
    h = 3/(np.pi*scale1*scale2)
    z1 = 1-(np.sqrt(((x1-loc1)/scale1)**2+((x2-loc2)/scale2)**2))
    z2 = h*z1*np.heaviside(z1,0)
    return z2

def k_box_2d(x1,x2,loc1,loc2,scale1,scale2): 
    scale1 = scale1/2
    scale2 = scale2/2
    return 1/(2*scale1*2*scale2)*(np.heaviside(scale1+loc1-x1,1)-np.heaviside(-scale1+loc1-x1,1))*(np.heaviside(scale2+loc2-x2,1)-np.heaviside(loc2-scale2-x2,1))

# Choose kernel centers
def choose_centers(dim_ranges,knum,rng,mode='equidistant',type='torus'): #mode={'equidistant','random'}, type={'random','centered'}
    kcl, kwl = knov_md.choose_centers(dim_ranges,knum,rng,mode,type)
    kc_list = np.meshgrid(np.array(kcl[0]),np.array(kcl[1]))
    kc_list = [kc_list[i].flatten() for i in range(len(kc_list))]
    kw_list = [kwl[0]*np.ones(knum**2),kwl[1]*np.ones(knum**2)]
    return kc_list, kw_list

# Compute widths
def comp_kwidth_2d(k_type,k_num,space):
    if 'box'==k_type:                  
        k_width = (space[1,:]-space[0,:])/(2*k_num)
    elif 'triangle_overlap'==k_type:   
        k_width = (space[1]-space[0])/(k_num-1)
    elif 'triangle_overlap_fuse-bound'==k_type:
        k_width = (space[1]-space[0])/k_num
    return k_width

#############################################################################################################################################################################################################
# Compute kernel values for array of states 
def compute_kmat_2d(k,kc1,kc2,ksig1,ksig2,x1=np.array([]),x2=np.array([])):
    if len(x1)>0: 
        kmat = np.array([k(x1,x2,kc1[i],kc2[i],ksig1[i],ksig2[i]) for i in range(len(kc1))])
    else:        
        kmat = None
    return kmat

# Initialize novelty variables
def init_nov_2d(k,kc1,kc2,ksig1,ksig2,x1=np.array([]),x2=np.array([]),seq1=np.array([]),seq2=np.array([]),update_means=False,update_widths=False,full_update=False): 
    # k:kernel function, kc:kernel centers, ksig:kernel widths, seq:sequence of stimuli presented
    kw   = 1/(len(kc1))*np.ones(len(kc1)).reshape((-1,1,1))
    if len(x1)>0 and len(x2)>0: kmat = compute_kmat_2d(k,kc1,kc2,ksig1,ksig2,x1,x2)
    else:                       kmat = None
    if len(seq1)>0 and len(seq2)>0: kmat_seq = compute_kmat_2d(k,kc1,kc2,ksig1,ksig2,seq1,seq2) 
    else:                           kmat_seq = None

    rksum = None; rkmat = None; kmumat = None
    if update_means: 
        rksum = 1/(len(kc1))*np.ones((len(kc1),1))                            
    if full_update or update_widths:
        rkmat = 1/(len(kc1))*np.ones((len(kc1),len(seq1)))  
    if update_widths: 
        kmumat = kc1.reshape((-1,1))*np.ones((len(kc1),2)) 
    return kw, kmat, kmat_seq, rksum, rkmat, kmumat

#############################################################################################################################################################################################################
# Update responsibilities
def update_rk_approx(kw,kmat,t):
    kk_s  = kw*kmat[:,t,t].reshape((-1,1,1))
    rk    = (kk_s/np.sum(kk_s)).reshape((-1,1,1))
    return rk

def update_rkmat_approx(kw,kmat,rkmat,t): # NOT YET TESTED
    kk_s  = kw*kmat[:,t,t].reshape((-1,1,1))
    rk    = (kk_s/np.sum(kk_s)).reshape((-1,1,1))
    rkmat[:,t,t] = rk.flatten()
    return rkmat

#############################################################################################################################################################################################################
# Plot kernels
def plot_kernels_2d(ax,knum,k,kmu,ksig,splot,title=''): 
    # [s1,s2] = np.meshgrid(splot[0],splot[1])
    # s = np.concatenate([s1.flatten().reshape((-1,1)),s2.flatten().reshape((-1,1))],axis=1)
    kk = np.array([k(splot[0],splot[1],kmu[0][i],kmu[1][i],ksig[0][i],ksig[1][i]) for i in range(knum)])
    for i in range(knum):
        ax.imshow(kk[i,:].reshape(splot[0].shape),cmap='Blues')

# Plot familiarity
def plot_fam(ax,pk,knum,splot,kk=[]):
    if len(kk)>0:
        for i in range(knum):
            ax.plot(splot,kk[i,:],'--',c='orange')
            ax.fill_between(splot,kk[i,:],color='orange',alpha=0.4)
    ax.plot(splot,pk,'-',c='orange',lw=2)
    ax.set_xlabel('State space')
    ax.set_ylabel('Familiarity')

#############################################################################################################################################################################################################
# Additional plotting
def plot_knov2d_gauss_kk(splot_shared,k,kmu,ksig,knum,dim_ranges,fig_res=5,f=None,ax=None,cax_pos=[0.97, 0.055, 0.05, 0.925]):
    if not f and not ax: 
        f,ax = plt.subplots(figsize=(fig_res*260/90,fig_res),constrained_layout=True)
    kkl = [k(splot_shared[0],splot_shared[1],kmu[0][i],kmu[1][i],ksig[0][i],ksig[1][i]) for i in range(knum**2)]
    kkla = np.stack(kkl)
    kksum = np.sum(kkla,axis=0)

    im = ax.imshow(kksum,cmap='Blues',extent=[dim_ranges[0][0],dim_ranges[0][1],dim_ranges[1][0],dim_ranges[1][1]],alpha=1,origin='lower') #,vmin=0,vmax=0.005) 
    cax = f.add_axes(cax_pos)
    for i in range(knum**2):
        # Plot centers
        ax.plot(kmu[0][i],kmu[1][i],'ko')
        # Plot elliptic support
        t = np.linspace(0,2*np.pi)
        xell = kmu[0][i]+ksig[0][i]*np.cos(t)
        yell = kmu[1][i]+ksig[1][i]*np.sin(t)
        ax.plot(xell,yell,'k--')
    ax.set_xlim(dim_ranges[0])
    ax.set_ylim(dim_ranges[1])
    f.colorbar(im,cax=cax)
    ax.set_title('Single kernel (not weighted)')

def plot_knov2d_gauss_single(pkl,kmu,ksig,knum,dim_ranges,fig_res=5,f=None,ax=None,cax_pos=[0.97, 0.055, 0.05, 0.925]):
    if not f and not ax: 
        f,ax = plt.subplots(figsize=(fig_res*260/90,fig_res),constrained_layout=True)
    # im = ax.imshow(pkl,cmap='Blues',extent=[dim_ranges[0][0],dim_ranges[0][1],dim_ranges[1][0],dim_ranges[1][1]],alpha=1,origin='lower',vmin=0,vmax=0.005) 
    im = ax.imshow(pkl,cmap='Blues',extent=[dim_ranges[0][0],dim_ranges[0][1],dim_ranges[1][0],dim_ranges[1][1]],alpha=1,origin='lower') 
    cax = f.add_axes(cax_pos)
    for i in range(knum**2):
        # Plot centers
        ax.plot(kmu[0][i],kmu[1][i],'ko')
        # Plot elliptic support
        # t = np.linspace(0,2*np.pi)
        # xell = kmu[0][i]+ksig[0][i]*np.cos(t)
        # yell = kmu[1][i]+ksig[1][i]*np.sin(t)
        # ax.plot(xell,yell,'k--')
    ax.set_xlim(dim_ranges[0])
    ax.set_ylim(dim_ranges[1])
    f.colorbar(im,cax=cax)
    ax.set_title('Single kernel (weighted)')

def plot_knov2d_gauss(pkl,kmu,ksig,knum,dim_ranges,fig_res=5,f=None,ax=None,cax_pos=[0.39, 0.85, 0.605, 0.05]):
    if not f and not ax: 
        f,ax = plt.subplots(figsize=(fig_res*260/90,fig_res),constrained_layout=True)
    # im = ax.imshow(pkl,cmap='Blues',extent=[dim_ranges[0][0],dim_ranges[0][1],dim_ranges[1][0],dim_ranges[1][1]],alpha=1,origin='lower',vmin=0,vmax=0.005) 
    im = ax.imshow(pkl,cmap='Blues',extent=[dim_ranges[0][0],dim_ranges[0][1],dim_ranges[1][0],dim_ranges[1][1]],alpha=1,origin='lower') 
    cax = f.add_axes(cax_pos)
    for i in range(knum**2):
        # Plot centers
        ax.plot(kmu[0][i],kmu[1][i],'ko')
        # Plot elliptic support
        # t = np.linspace(0,2*np.pi)
        # xell = kmu[0][i]+ksig[0][i]*np.cos(t)
        # yell = kmu[1][i]+ksig[1][i]*np.sin(t)
        # ax.plot(xell,yell,'k--')
    ax.set_xlim(dim_ranges[0])
    ax.set_ylim(dim_ranges[1])
    f.colorbar(im,cax=cax,orientation='horizontal')
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')