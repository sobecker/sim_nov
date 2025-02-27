import numpy as np
import models.snov.kernel_nov as knov
import models.snov.kernel_nov_2d as knov_2d

#############################################################################################################################################################################################################
# Map stimulus onto separate feature dimensions
def map_gmat_to_feature(x,dim):
    if len(x)==0:
        xi = x 
    else:
        xi = x[dim,:]
    return xi

# Choose kernel centers randomly or equidistantly for each dimension
def choose_centers(dim_ranges,knum,rng,mode='equidistant',type='torus'): #mode={'equidistant','random'}, type={'random','centered'}
    kc_list = []
    kw_list = []
    for i in range(len(dim_ranges)):
        dr = dim_ranges[i]
        if mode=='equidistant':
            kwidth = (dr[1]-dr[0])/knum
            if type=='random':  k0 = dr[0] + rng.uniform(0,kwidth)
            elif type=='centered': k0 = dr[0] + kwidth/2
            kc = [k0+j*kwidth for j in range(knum)]
        elif mode=='random':
            kwidth = None
            kc = dr[0]+rng.uniform(0,dr[1]-dr[0])
        kc_list.append(kc)
        kw_list.append(kwidth)
    return kc_list, kw_list

#############################################################################################################################################################################################################
# Compute kernel values for array of states 
def compute_kmat(k,kc,ksig,maps,x=np.array([])):
    kmat_list = []
    for i in range(len(kc)):
        xi = maps[i](x)
        kmat = knov.compute_kmat(k[i],kc[i],ksig[i],x=xi)
        kmat_list.append(kmat)
    return kmat_list

# Initialize novelty variables
def init_nov(k,kc,ksig,maps,dim_types=[],x=np.array([]),seq=np.array([]),update_means=False,update_widths=False,full_update=False): 
    # k:kernel function, kc:kernel centers, ksig:kernel widths, seq:sequence of stimuli presented
    if len(dim_types)==0: dim_types = ['single']*len(k)
    kwl = []; kmatl = []; kmat_seql = []; rksuml = []; rkmatl = []; kmumatl = []
    for i in range(len(kc)):
        if len(x)>0 :   
            xi = maps[i](x)
        else:
            xi = np.array([])
        if len(seq)>0:  
            seqi = maps[i](seq)
        else:
            seq = np.array([])
        if dim_types[i]=='single':
            kw, kmat, kmat_seq, rksum, rkmat, kmumat = knov.init_nov(k[i],kc[i],ksig[i],xi,seqi,update_means,update_widths,full_update)
        elif dim_types[i]=='shared':
            xi1,xi2 = np.meshgrid(xi[0,:],xi[1,:])
            seqi1,seqi2 = np.meshgrid(seqi[0,:],seqi[1,:])
            kw, kmat, kmat_seq, rksum, rkmat, kmumat = knov_2d.init_nov_2d(k[i],kc[i][0],kc[i][1],ksig[i][0],ksig[i][1],xi1,xi2,seqi1,seqi2,update_means,update_widths,full_update)
        kwl.append(kw); kmatl.append(kmat); kmat_seql.append(kmat_seq); rksuml.append(rksum); rkmatl.append(rkmat); kmumatl.append(kmumat)
    return kwl, kmatl, kmat_seql, rksuml, rkmatl, kmumatl

# Evaluate novelty for array of states
def comp_nov(dw,kwl,kmatl,dim_types=[]):
    if len(dim_types)==0: dim_types=['single']*len(dw)
    if not 'shared' in dim_types: 
        pkd = 1; nkd = 0
    else:
        pkd = None; nkd = None
    kkl = []; pkl = []; nkl = []
    for i in range(len(kwl)):
        kk,pk,nk = knov.comp_nov(kwl[i],kmatl[i])
        kkl.append(np.squeeze(kk)); pkl.append(np.squeeze(pk)); nkl.append(np.squeeze(nk))
        if not 'shared' in dim_types:
            pkd *= pk
            nkd += -np.log(pk)
            # pkd = dw[i]*pk
    # if not 'shared' in dim_types:
    #     nkd = -np.log(pkd)
    # else:
    #     pkd = None; nkd = None
    return kkl,pkl,nkl,pkd,nkd

#############################################################################################################################################################################################################
# Update responsibilities
def update_rk_approx(kwl,kmatl,t,dim_types=[]):
    if len(dim_types)==0: dim_types=['single']*len(kwl)
    rkl = []
    for i in range(len(kwl)): 
        if dim_types[i]=='single':
            rk = knov.update_rk_approx(kwl[i],kmatl[i],t)
        elif dim_types[i]=='shared':
            rk = knov_2d.update_rk_approx(kwl[i],kmatl[i],t)
        rkl.append(rk)
    return rkl

def update_rkmat_approx(kwl,kmatl,rkmatl,t):
    for i in range(len(kwl)): 
        rkmat = knov.update_rkmat_approx(kwl[i],kmatl[i],rkmatl[i],t)
        rkmatl[i] = rkmat.copy()
    return rkmatl

def update_rkmat_full(kwl,kmatl):
    rkmatl = []
    for i in range(len(kwl)):
        rkmat = knov.update_rkmat_full(kwl[i],kmatl[i])
        rkmatl.append(rkmat)
    return rkmatl

#############################################################################################################################################################################################################
# Update weights (incremental update, fixed learning rate)
def update_nov_approx_flr(kwl,rkl,alphl):
    for i in range(len(kwl)):
        kw = knov.update_nov_approx_flr(kwl[i],rkl[i],alphl[i])
        kwl[i] = kw.copy()
    return kwl

# Update weights (incremental update)
def update_nov_approx(kwl,t,rkl,knum,eps=[1]):
    if len(eps)==1: eps = eps[0]*np.ones(len(kwl))
    for i in range(len(kwl)):
        kw = knov.update_nov_approx(kwl[i],t,rkl[i],knum,eps[i])
        kwl[i] = kw.copy()
    return kwl

# Update weights (full update)
def update_nov_full(kwl,t,rkmatl,rkmat_oldl,knum,eps=[1]):
    if len(eps)==1: eps = eps[0]*np.ones(len(kwl))
    for i in range(len(kwl)):
        kw = knov.update_nov_full(kwl[i],t,rkmatl[i],rkmat_oldl[i],knum,eps[i])
        kwl[i] = kw.copy()
    return kwl

