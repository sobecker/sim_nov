import numpy as np
from scipy.stats import qmc
from scipy.special import binom

###########################################################################
# Define standard dimensions for Gabor images                             #
###########################################################################
# For biological justification of the feature ranges, see Niell & Stryker, 2008; Kim, Homann, Tank, Berry, 2019; Homann et al., 2022
# The width of the Gaussian is given in degrees to make it consistent with the values for spatial frequency and phase offset of the sinusoidal modulation.
range_alph  = [0,180]       # angular orientation [in degrees]                          -> torus
range_alph_rad = [0,np.pi]  # angular orientation [in radians]                          -> torus
range_phase = [-180,180]    # phase [in degrees]                                        -> torus
range_phase_rad = [-np.pi,np.pi]
range_freq  = [0.04,0.08]   # spatial frequency [in cycles per degree*]                 -> interval
range_width = [5,10]       # spatial width [in degrees*], [10,20]                       -> interval
range_x0    = [-130,130]    # center location in horizontal (x) direction [in degrees*] -> interval
range_y0    = [-20,70]      # center location in vertical (y) direction [in degrees*]   -> interval

dim_ranges = [range_alph,range_phase,range_freq,range_width,range_x0,range_y0]
dim_ranges_rad = [range_alph_rad,range_phase_rad,range_freq,range_width,range_x0,range_y0]
dim_names = ['orientation','phase','frequency','width','x-position','y-position']

###########################################################################
# Create new stimulus matrix                                              #
###########################################################################
def sample_sobol_all(dim_ranges,gabor_num,rng):
    m = int(np.ceil(np.log2(gabor_num)))
    sampler4d = qmc.Sobol(d=4, scramble=False, seed=rng)
    sampler2d = qmc.Sobol(d=2, scramble=False, seed=rng)
    df_list = []
    # Sample 4D Sobol sequence for Gabor properties
    df4d_unit = sampler4d.random_base2(m)[:gabor_num]
    for i in range(df4d_unit.shape[-1]):
        dr = dim_ranges[i]
        df = dr[0]+(dr[1]-dr[0])*df4d_unit[:,i]
        df_list.append(df)
    # Sample 2D Sobol sequence for Gabor locations
    df2d_unit = sampler2d.random_base2(m)[:gabor_num]
    dr = dim_ranges[-2]
    df2d = dr[0]+(dr[1]-dr[0])*df2d_unit[:,0]
    df_list.append(df2d)
    dr = dim_ranges[-1]
    df2d = dr[0]+(dr[1]-dr[0])*df2d_unit[:,1]
    df_list.append(df2d)
    return df_list

def sample_sobol_4d(dim_ranges,gabor_num,rng):
    m = int(np.ceil(np.log2(gabor_num)))
    sampler4d = qmc.Sobol(d=4, scramble=False, seed=rng)
    df_list = []
    df4d_unit = sampler4d.random_base2(m)[:gabor_num]
    sobol_dims = np.array([0,1,4,5])
    rand_dims  = np.array([2,3])
    for i in range(len(dim_ranges)):
        if i in sobol_dims:
            # Sample 4D Sobol sequence for orientation, phase and location 
            si = np.where(sobol_dims==i)[0][0]
            dr = dim_ranges[i]
            df = dr[0]+(dr[1]-dr[0])*df4d_unit[:,si]
            df_list.append(df)
        elif i in rand_dims:
            # Sample width and frequency uniformly
            dr = dim_ranges[i]
            if dr[0]<0: dr_shift = dr[0]
            else:       dr_shift = 0
            df = rng.uniform(low=dr[0]-dr_shift,high=dr[1]-dr_shift,size=gabor_num) + dr_shift
            df_list.append(df)
    return df_list

def sample_sobol_loc(dim_ranges,gabor_num,rng):
    m = int(np.ceil(np.log2(gabor_num)))
    sampler2d = qmc.Sobol(d=2, scramble=False, seed=rng)
    df_list = []
    # Sample Gabor properties except location uniformly
    for i in range(len(dim_ranges)-2):
        dr = dim_ranges[i]
        if dr[0]<0: dr_shift = dr[0]
        else:       dr_shift = 0
        df = rng.uniform(low=dr[0]-dr_shift,high=dr[1]-dr_shift,size=gabor_num) + dr_shift
        df_list.append(df)
    # Sample 2D Sobol sequence for Gabor locations
    df2d_unit = sampler2d.random_base2(m)[:gabor_num]
    dr = dim_ranges[-2]
    df2d = dr[0]+(dr[1]-dr[0])*df2d_unit[:,0]
    df_list.append(df2d)
    dr = dim_ranges[-1]
    df2d = dr[0]+(dr[1]-dr[0])*df2d_unit[:,1]
    df_list.append(df2d)
    return df_list

def sample_equidist(dim_ranges,gabor_num,rng,sampling='',loc=True,fixed_freq=None,fixed_width=None):
    df_list = []
    # Sample orientations equidistantly
    if 'fixed' in sampling:     o_init = 0                                                               # initial orientation equal to 0
    else:                       o_init = rng.uniform(low=dim_ranges[0][0],high=dim_ranges[0][1],size=1)  # sample initial orientation randomly
    o_num  = int(np.floor(gabor_num/2))                        # get number of intervals                                 
    o_w = (dim_ranges[0][1]-dim_ranges[0][0]) / o_num          # get interval width
    o_df = np.array([o_init + i*o_w for i in range(o_num)])    # get equidistant orientation values (from intervals)
    o_df_corr = np.array((list(o_df[np.where(o_df>=dim_ranges[0][1])]-dim_ranges[0][1]+dim_ranges[0][0]) + list(o_df[np.where(o_df<dim_ranges[0][1])]))*2) # correct for orientation values outside of range
    df_list.append(o_df_corr)
    # Assign phases (binary)
    p_df = np.array(list(-0.5*np.pi*np.ones(o_num)) + list(0.5*np.pi*np.ones(o_num)))
    df_list.append(p_df)
    # Assign widths & frequences (w/wo jittering)
    if 'jitter' in sampling:
        df_list.append(rng.uniform(low=dim_ranges[2][0],high=dim_ranges[2][1],size=2*o_num))
        df_list.append(rng.uniform(low=dim_ranges[3][0],high=dim_ranges[3][1],size=2*o_num))
    else:
        if fixed_freq is None: fixed_freq = dim_ranges[2][0]
        if fixed_width is None: fixed_width = (dim_ranges[3][0]+(dim_ranges[3][1]-dim_ranges[3][0])/2)
        df_list.append([fixed_freq]*np.ones(2*o_num))
        df_list.append([fixed_width]*np.ones(2*o_num))
    # Assign position (center)
    if loc:
        df_list.append(np.zeros(2*o_num))
        df_list.append(np.zeros(2*o_num))
    return df_list

def sample_basic(dim_ranges,gabor_num,rng,loc=True,fixed_freq=None,fixed_width=None):
    df_list = []
    if loc: num_dims = len(dim_ranges)
    else: num_dims = len(dim_ranges)-2
    for i in range(num_dims):
        if i==2 and fixed_freq:
            df = fixed_freq*np.ones(gabor_num)
        elif i==3 and fixed_width:
            df = fixed_width*np.ones(gabor_num)
        else:
            dr = dim_ranges[i]
            if dr[0]<0: dr_shift = dr[0]
            else:       dr_shift = 0
            df = rng.uniform(low=dr[0]-dr_shift,high=dr[1]-dr_shift,size=gabor_num) + dr_shift
        df_list.append(df)
    return df_list

def map_pixel_to_im(pixel,pixel_dim,image_dim):
    return image_dim[0] + (image_dim[1]-image_dim[0])/(pixel_dim[1]-pixel_dim[0])*(pixel-pixel_dim[0])

def sample_patch_loc(patches,mult_patch,rng,pixel_dim=[[0,199],[0,99]],image_dim=[[-130,130],[-20,70]],centered=False):
    df_list = []
    patch_lim = []
    for i in range(len(patches)):
        patch_i = patches[i]
        lim_i = np.array([np.min(patch_i[1].flatten()), np.max(patch_i[1].flatten()), np.min(patch_i[0].flatten()), np.max(patch_i[0].flatten())]) # left bound, right bound, lower bound, upper bound
        patch_lim.append(lim_i)
    patch_lim = np.stack(patch_lim)
    # patch_lim = np.repeat(patch_lim,mult_patch,axis=0)
    patch_lim_x0 = map_pixel_to_im(patch_lim[:,:2],pixel_dim[0],image_dim[0])
    patch_lim_y0 = map_pixel_to_im(patch_lim[:,2:],pixel_dim[1],image_dim[1])
    patch_lim_x0 = np.concatenate([patch_lim_x0]*mult_patch,axis=0)
    patch_lim_y0 = np.concatenate([patch_lim_y0]*mult_patch,axis=0)
    if centered:
        x0 = patch_lim_x0[:,0] + (patch_lim_x0[:,1]-patch_lim_x0[:,0])/2
        y0 = patch_lim_y0[:,0] + (patch_lim_y0[:,1]-patch_lim_y0[:,0])/2
    else:
        x0 = rng.uniform(low=patch_lim_x0[:,0],high=patch_lim_x0[:,1],size=len(patch_lim_x0))
        y0 = rng.uniform(low=patch_lim_y0[:,0],high=patch_lim_y0[:,1],size=len(patch_lim_y0))
    df_list.append(x0)
    df_list.append(y0)
    return df_list

# Available sampling types (specified via 'sampling' parameter):
# 'sobol_all': All Gabor parameters are sampled using Sobol sequences (4D for orientation, phase, frequency, width; 2D for location)
# 'sobol_4d': Only orientation, phase, frequency and width are sampled using Sobol sequences (4D); x- and y-location are sampled independently and uniformly
# 'sobol_loc': Only location is sampled using Sobol sequences (2D); orientation, phase, frequency and width are sampled independently and uniformly
# 'basic': All Gabor parameters are sampled independently and uniformly
# 'equidist': All Gabor parameters are sampled equidistantly (orientation: #kernels/2 equidistant values; phase: binary on/off; frequency and width: either fixed or sampled uniformly; location: fixed at center (0,0))
    # combination with 'fixed': Initial orientation is set to 0, else chosen randomly
    # combination with 'jitter': Width and frequency are chosen uniformly, else chosen fixed (frequency: minimum of range, width: middle of range)
# 'patch_loc': for each patch, one location is sampled within the patch range; in total, #kernel=#patch locations are sampled independently and uniformly; all other parameters are sampled  based on 'basic' or 'equidistant' sampling
def generate_stim(dim_ranges,gabor_num,rng,adj_w=True,adj_f=False,alph_adj=3,sampling='basic',patches=None,adj_jitter=None,fixed_freq=None,fixed_width=None): 
    if sampling=='sobol_all':
        df_list = sample_sobol_all(dim_ranges,gabor_num,rng)

    elif sampling=='sobol_4d':
        df_list = sample_sobol_4d(dim_ranges,gabor_num,rng)

    elif sampling=='sobol_loc':
        df_list = sample_sobol_loc(dim_ranges,gabor_num,rng)

    elif 'patch_loc' in sampling:
        mult_patch = int(gabor_num / len(patches))
        if (gabor_num % len(patches))!=0:
            gabor_num = mult_patch * len(patches)
            print(f'Number of kernels was set to {mult_patch} x number of patches.')
        else:
            if 'equidist' in sampling: df_list = sample_equidist(dim_ranges,gabor_num,rng,sampling=sampling,loc=False,fixed_freq=fixed_freq,fixed_width=fixed_width) # Sample all but location equidistantly
            else:                      df_list = sample_basic(dim_ranges,gabor_num,rng,loc=False,fixed_freq=fixed_freq,fixed_width=fixed_width)                      # Sample all but location uniformly
            df_loc = sample_patch_loc(patches,mult_patch,rng,centered=('center' in sampling))
            df_list.extend(df_loc) # Sample random location from each patch
    else:
        if 'equidist' in sampling: df_list = sample_equidist(dim_ranges,gabor_num,rng,sampling=sampling,loc=True,fixed_freq=fixed_freq,fixed_width=fixed_width) # Sample equidistantly
        else:                      df_list = sample_basic(dim_ranges,gabor_num,rng,loc=True,fixed_freq=fixed_freq,fixed_width=fixed_width)                      # Sample uniformly

    # Adjust Gabor width to frequency (if adj_w=True) or frequency to width (if adj_f=True)
    dfm = np.stack(df_list)
    if adj_jitter is not None:
        alph_adj = np.random.normal(loc=alph_adj,scale=adj_jitter,size=dfm[2,:].shape)
    if adj_w:
        dfm[3,:] = 1/(alph_adj*dfm[2,:])
    elif adj_f:
        dfm[2,:] = 1/(alph_adj*dfm[3,:])
    # print(dfm.shape)
    return dfm

def generate_stim_complex(dim_ranges,gabor_num,rng,adj_w=True,adj_f=False,alph_adj=3,dfreq=1.5,sampling='basic',patches=None,adj_jitter=None,ctype=[4],cvar='frequency',fixed_freq=None,fixed_width=None): # cvar specifies which parameter complex cells integrate over ('frequency', 'orientation'). In any case, they combine opposite phases of a given frequency and orientation. 
    cmax = max(2,2*round(max(ctype)/2)) # maximum complexity of complex cells has to be divisible by 2, and larger or equal to 2
    num_comp = int(cmax/2) # number of complexity components to be integrated over for cvar (e.g. 2 frequencies if num_comp=2 and cvar='frequency')
    # Generate gnum Gabors
    if 'equidist' in sampling:
        # Sample gabor_num orientations equidistantly, with on/off phase for each orientation 
        df_list = sample_equidist(dim_ranges,2*gabor_num,rng,sampling=sampling,loc=True,fixed_freq=fixed_freq,fixed_width=fixed_width) 
        # Assign simple cells to complex cell identities 
        if cvar=='orientation' and num_comp>1: 
            cell_id = []
            for i in range(int(np.floor(gabor_num/num_comp))):
                cell_id.extend([i]*num_comp)
            if len(cell_id)<gabor_num:
                cell_id.extend([np.floor(gabor_num/num_comp)]*(gabor_num-len(cell_id)))
            cell_id = np.array([cell_id]*2).flatten()
        else:
            cell_id = np.concatenate([np.arange(gabor_num)]*2)
    else:        
        # Sample gabor_num orientations randomly               
        df_list = sample_basic(dim_ranges,gabor_num,rng,loc=True)    
        # Duplicate Gabors with opposite phase           
        df_list = [np.concatenate([df_list[i]]*2) if i!=1 else np.concatenate([df_list[i],-df_list[i]]) for i in range(len(df_list))]  
        # Assign simple cells to complex cell identities                          
        cell_id = np.concatenate([np.arange(gabor_num)]*2)
    dfm = np.stack(df_list)
    
    if num_comp==1 or cvar=='orientation':
        dfm = np.vstack((dfm,cell_id))  # maximum complexity of complex cells = 2: only combine on/off polarity (single frequency)
    else:
        # Get number of frequency components to be added
        num_freq = num_comp
        # Adjust Gabor width to frequency (if adj_w=True) or frequency to width (if adj_f=True)
        if adj_jitter is not None:
            alph_adj = np.random.normal(loc=alph_adj,scale=adj_jitter,size=dfm[2,:].shape)
        if adj_w:
            dfm[3,:] = 1/(alph_adj*dfm[2,:])
        elif adj_f:
            dfm[2,:] = 1/(alph_adj*dfm[3,:])
        # Add frequency components to complex cells + adjust phases
        dfm_all = [dfm]
        for i in range(1,num_freq):
            dfm_i = dfm.copy()
            dfm_i[2,:] = dfreq**i * dfm_i[2,:]
            dfm_i[1,:] = dfm_i[1,:] + 1/(2*dfm_i[2,:]) - 1/(2*dfm[2,:])
            dfm_all.append(dfm_i)
        # Combine Gabors with different frequencies
        dfm = np.concatenate(dfm_all,axis=1)
        cell_id = np.concatenate([cell_id]*num_freq)
        dfm = np.vstack((dfm,cell_id))

    return dfm

def generate_teststim_parent(dim_ranges,gabor_num,n_orient=4,init_orient=0,fixed_phase=None,fixed_freq=None,fixed_width=None,rng=None,adj_w=True,adj_f=False,alph_adj=3,adj_jitter=None,mode='orientation',loc_sigma=(0,0),return_features=[0]): 
    # Compute locations of gabors in image
    dx = (dim_ranges[4][1] - dim_ranges[4][0])/gabor_num[0] 
    dy = (dim_ranges[5][1] - dim_ranges[5][0])/gabor_num[1]
    xloc = [dim_ranges[4][0] + dx/2 + i*dx for i in range(gabor_num[0])]
    yloc = [dim_ranges[5][0] + dy/2 + i*dy for i in range(gabor_num[1])]
    xloc_jitter = np.random.normal(loc=0,scale=loc_sigma[0],size=(1,len(xloc)*len(yloc))) if loc_sigma[0]>0 else np.zeros((1,len(xloc)*len(yloc)))
    yloc_jitter = np.random.normal(loc=0,scale=loc_sigma[1],size=(1,len(xloc)*len(yloc))) if loc_sigma[1]>0 else np.zeros((1,len(xloc)*len(yloc)))
    xyloc_jitter = np.concatenate([xloc_jitter,yloc_jitter],axis=0)
    # xloc = np.array(xloc) + xloc_jitter
    # yloc = np.array(yloc) + yloc_jitter
    xyloc = np.array(np.meshgrid(xloc,yloc)).reshape(2,-1) + xyloc_jitter

    # Sample orientations randomly from four equdistant orientations (in radians)
    if mode=='orientation':
        do = (2*dim_ranges[0][1]-dim_ranges[0][0])/n_orient
        o_all = [(init_orient + i*do)%(2*np.pi) for i in range(n_orient)] #[(init_orient + i*do)%(2*np.pi) for i in range(n_orient)]
    elif mode=='orientation_phase':
        do = (dim_ranges[0][1]-dim_ranges[0][0])/(n_orient/2)
        o_all = [(init_orient + i*do)%(2*np.pi) for i in range(n_orient/2)]*2 #[(init_orient + i*do)%(2*np.pi) for i in range(n_orient)]
    
    # Sample phases (in radians), frequency and width (fixed)
    if fixed_phase is None:
        fixed_phase = -0.5*np.pi
    
    if mode=='orientation':
        p_all = [fixed_phase]*n_orient
    elif mode=='orientation_phase':
        p_all = [fixed_phase]*int(n_orient/2) + [-fixed_phase]*int(n_orient/2) 

    if rng is None:
        rng = np.random.default_rng()

    idx_op = rng.choice(np.arange(len(o_all)),gabor_num[0]*gabor_num[1])
    oloc = np.array([o_all[i] for i in idx_op])
    ploc = np.array([p_all[i] for i in idx_op])

    if fixed_freq is None:
        fixed_freq = dim_ranges[2][0] #(dim_ranges[2][0]+dim_ranges[2][1])/2
    if fixed_width is None:
        fixed_width = (dim_ranges[3][0]+(dim_ranges[3][1]-dim_ranges[3][0])/2)
    floc = fixed_freq*np.ones(gabor_num[0]*gabor_num[1])
    wloc = fixed_width*np.ones(gabor_num[0]*gabor_num[1])

    df_list = [oloc,ploc,floc,wloc,xyloc[0],xyloc[1]]
    dfm = np.stack(df_list)

    # Adjust Gabor width to frequency (if adj_w=True) or frequency to width (if adj_f=True)
    if adj_jitter is not None:
        alph_adj = np.random.normal(loc=alph_adj,scale=adj_jitter,size=dfm[2,:].shape)
    if adj_w:
        dfm[3,:] = 1/(alph_adj*dfm[2,:])
        fixed_width = 1/(alph_adj*fixed_freq)
    elif adj_f:
        dfm[2,:] = 1/(alph_adj*dfm[3,:])
        fixed_freq = 1/(alph_adj*fixed_width)
    # print(dfm.shape)

    x_feature = np.array([-dx/4, 0, dx/4])
    y_feature = np.array([-dy/4, 0, dy/4])
    all_features = [np.array([o_all]).reshape((-1,1)),np.array([p_all]).reshape((-1,1)),fixed_freq,fixed_width,x_feature.reshape((-1,1)),y_feature.reshape((-1,1))]
    # if mode=='orientation':
    #     distinct_features = np.array([o_all]).reshape((-1,1))
    # elif mode=='orientation_phase':
    #     distinct_features = np.concatenate([np.array([o_all]).reshape((-1,1)),np.array([p_all]).reshape((-1,1))],axis=1)
    distinct_features = [all_features[i] for i in return_features]

    return dfm, distinct_features

def transform_identity(gabor):
    return gabor

def transform_rotate_left(gabor,rad=0.5*np.pi):
    gabor[0] += rad
    return gabor

def transform_rotate_right(gabor,rad=0.5*np.pi):
    gabor[0] -= rad
    return gabor

def transform_shift_left(gabor,shift_x=20.8,shift_y=0):
    gabor[4] -= shift_x
    gabor[5] -= shift_y
    return gabor

def transform_shift_right(gabor,shift_x=20.8,shift_y=0):
    gabor[4] += shift_x
    gabor[5] += shift_y
    return gabor

def generate_teststim_children(parent,num_child,fun_transform=[transform_identity,transform_rotate_left],prob_transform=[0.75,0.25],rng=None,mode='stochastic'):
    if rng is None:
        rng = np.random.default_rng() 
    num_gabor  = parent.shape[1]

    if mode=='fixed':
        choice_set = np.arange(num_gabor)

        num_transform = [int(np.round(num_gabor*pt)) for pt in prob_transform]
        num_toomany   = np.sum(num_transform)-num_gabor
        if num_toomany>0:
            for i in range(num_toomany):
                max_idx = np.argmax(num_transform)
                num_transform[max_idx] -= 1
        elif num_toomany<0:
            for i in range(-num_toomany):
                min_idx = np.argmin(num_transform)
                num_transform[min_idx] += 1
        assert np.sum(num_transform)==num_gabor, 'Number of gabors does not match number of transformations.'

        max_comb = [binom(num_gabor,num_transform[i]) for i in range(len(num_transform))]

        # Create children
        all_set_transform = []
        all_children      = []
        for i in range(num_child):
            child_i = parent.copy()
            set_i = choice_set.copy()
            set_transform_i = []
            for j, (ft, nt, mc) in enumerate(zip(fun_transform,num_transform,max_comb)):
                # Sample gabors to be transformed
                count_choice = 0
                duplicate = True
                while count_choice<mc and duplicate==True:
                    choice_option = rng.choice(set_i,nt,replace=False)
                    count_choice += 1
                    duplicate = np.array([set(choice_option)==set(all_set_transform[k][j]) for k in range(len(all_set_transform))]).any() # check if any other child has had the same transform set for this transform type
                    # overlap = np.array([np.intersect1d(choice_option,all_set_transform[k][j]).size==2
    
                set_transform_i.append(choice_option) # append to chosen gabors ('transform set') to list of transform sets for child i (one transform set per transform type)
                set_i = np.setdiff1d(set_i,set_transform_i[-1]) # remove chosen gabors ('transform set') from set of available gabors for child i
                # Apply transform
                child_i_update = ft(child_i[:,set_transform_i[-1]])
                child_i[:,set_transform_i[-1]] = child_i_update
            all_children.append(child_i)
            all_set_transform.append(set_transform_i)
    
    elif mode=='stochastic':

        # Create children
        all_set_transform = []
        all_children      = []
        for i in range(num_child):
            child_i = parent.copy()

            # Chose transformations for each gabor
            ti = np.random.choice(np.arange(len(fun_transform)),size=num_gabor,p=np.array(prob_transform)) 

            # Apply transformations
            set_transform_i = []
            for j, ft in enumerate(fun_transform):
                tij = np.where(ti==j)[0]
                if len(tij)>0:
                    child_i_update = ft(child_i[:,tij])
                    child_i[:,tij] = child_i_update
                set_transform_i.append(tij)
            
            # Save child and information about transformations
            all_children.append(child_i)
            all_set_transform.append(set_transform_i)
    
    return all_children, all_set_transform

def generate_teststim_iterative_1d(init_child_full,vals_gabor,num_child,field_transform=[0],prob_overlap=[0.1],rng=None):
    if rng is None:
        rng = np.random.default_rng() 

    assert prob_overlap[0]>=0, 'Overlap probability has to be larger or equal to 0.'
    assert prob_overlap[0]<=0.5, 'Overlap probability has to be smaller or equal to 0.5 to allow for at least one additional child.'

    # Create choice matrix
    idx_gabor   = np.arange(init_child_full.shape[1])
    init_child  = init_child_full[field_transform,:]
    choice_mesh_idx, choice_mesh_vals = np.meshgrid(idx_gabor,vals_gabor)
    
    # Reduce number of children if necessary 
    num_overlap = int(np.round(prob_overlap[0]*init_child.shape[1]))

    # Initialize recording variables
    all_children = [init_child]
    all_overlap = [[]]

    # Initialize overlap lists (choice_children) and distinct matrix (choice_mat)
    choice_children = [set(np.arange(init_child.shape[1]))]
    choice_mat      = ~(init_child==choice_mesh_vals) # number of values x number of gabors
    assert (np.sum(~choice_mat,axis=0) == 1).all(), 'Choice matrix is not correct.'
    assert np.sum(~choice_mat) == init_child.shape[1], 'Choice matrix is not correct.'
    
    # Create children sequentially
    child_failed = False
    for i in range(num_child-1):
        child_i = init_child.copy()

        # Compute preference vector for rows (prefer gabors indices that have used up most of their distinct features)
        taken = np.array([choice_mat.shape[0]-len(np.where(choice_mat[:,jj])[0]) for jj in range(choice_mat.shape[1])])
        p_taken = taken / np.sum(taken)

        # Sample prob_overlap*num_gabor gabors from previous children
        idx_j = set([])
        overlap_i = []
        try:
            for j in range(len(all_children)):

                # Compute preference distribution for overlap gabors (for child j)
                cc = np.array(list(choice_children[j].difference(idx_j)))
                if len(cc)<num_overlap:
                    child_failed = True
                    print(f'Required overlap between children is too large to sample additional children. Number of children returned: {len(all_children)}.')  
                    break
                pcc = p_taken[cc] / np.sum(p_taken[cc])     # indices relative to cc!
                chose_pcc = np.where(pcc==np.max(pcc))[0]   # indices relative to cc
                if len(chose_pcc) < num_overlap:
                    chose_pcc = np.concatenate([chose_pcc,rng.choice(np.where(pcc==np.max(pcc[~chose_pcc]))[0],num_overlap-len(chose_pcc),replace=False)]) # indices relative to cc

                # Sample overlaps of current child i with child j
                copy_j = rng.choice(cc[chose_pcc],num_overlap,replace=False,p=pcc[chose_pcc]/np.sum(pcc[chose_pcc])) 
                child_i[0,copy_j] = all_children[j][0,copy_j]                               # copy gabors to new child
                idx_j.update(set(copy_j))                                                   # add indices of copied gabors to set               
                choice_children[j].difference_update(set(copy_j))                           # remove indices of copied gabors from choice_children (to avoid overlap with more than one other child)
                if num_overlap==1:
                    copy_j = [copy_j]
                overlap_i.append(copy_j)
        except:
            child_failed = True
            print(f'Required overlap between children is too large to sample additional children. Number of children returned: {len(all_children)}.')  
            break

        if not child_failed:
            try:
                # Fill remaining gabors from choice_mat and update choice_mat
                for fill_j in set(idx_gabor).difference(idx_j):
                    copy_mat = rng.choice(list(np.where(choice_mat[:,fill_j])[0]),1,replace=False)
                    child_i[0,fill_j] = vals_gabor[copy_mat]
                    choice_mat[copy_mat,fill_j] = False
            except:
                child_failed = True
                print(f'Required overlap between children is too small to sample additional children. Number of children returned: {len(all_children)}.')
                break

        # Save child 
        if not child_failed:
            all_children.append(child_i)
            choice_children.append(set(np.arange(init_child.shape[1])).difference(set(np.concatenate(overlap_i).flatten())))
            all_overlap.append(overlap_i)
        else:
            break

    # Complete all children with invariant dimensions
    field_invariant = np.sort(np.array(list(set(np.arange(init_child_full.shape[0])).difference(set(field_transform)))))
    all_children_full = [np.concatenate([all_children[i],init_child_full[field_invariant,:]],axis=0) for i in range(len(all_children))]
        
    return all_children_full, all_overlap

def map_child_feat2idx(child, fl0, fl1): 
    pos1 = child[1,:].flatten()
    child_mapped = child.copy()
    child_mapped[1,:] = np.zeros(child_mapped.shape[1])
    child_mapped1 = np.array([list(set(list(np.where(child_mapped[0,j]==fl0)[0])).intersection(set(list(np.where(child_mapped[1,j]==fl1)[0]))))[0] for j in range(child_mapped.shape[1])]).reshape((1,-1))
    return child_mapped1, pos1

def map_child_idx2feat(child, fl0, fl1, pos1):
    child_mapped = np.concatenate([fl0[child].reshape((1,-1)), fl1[child].reshape((1,-1))],axis=0)
    child_mapped[1,:] += pos1
    return child_mapped

def generate_teststim_iterative(init_child_full,vals_gabor,num_child,field_transform=np.array([0,4]),prob_overlap=[0.1],rng=None):
    if rng is None:
        rng = np.random.default_rng() 

    assert prob_overlap[0]>=0, 'Overlap probability has to be larger or equal to 0.'
    assert prob_overlap[0]<=0.5, 'Overlap probability has to be smaller or equal to 0.5 to allow for at least one additional child.'

    # Create choice matrix
    idx_gabor   = np.arange(init_child_full.shape[1])                   # Get indices of available gabors
    features0, features1 = np.meshgrid(vals_gabor[0],vals_gabor[1])     # Get available feature combinations (for each gabor)
    fl0 = features0.flatten(); fl1 = features1.flatten()                
    vals_gabor_features = vals_gabor.copy()                             # Replace vals_gabor (features) by idx of feature combinations 
    vals_gabor = np.arange(len(fl0))
    choice_mesh_idx, choice_mesh_vals = np.meshgrid(idx_gabor,vals_gabor) # Get choice matrix as with all combinations of gabors x feature combinations

    # Map initial child to feature space
    init_child  = init_child_full[field_transform,:] # num_dimensions x num_gabors
    # init_child1 = init_child.copy()
    # init_child1[1,:] = np.zeros(init_child1.shape[1])
    init_child_mapped, pos_init = map_child_feat2idx(init_child,fl0,fl1)
    # np.array([list(set(list(np.where(init_child1[0,j]==fl0)[0])).intersection(set(list(np.where(init_child1[1,j]==fl1)[0]))))[0] for j in range(init_child1.shape[1])])
    
    # Reduce number of children if necessary 
    num_overlap = int(np.round(prob_overlap[0]*init_child_mapped.shape[1]))

    # Initialize recording variables
    all_children = [init_child_mapped]
    all_overlap = [[]]

    # Initialize overlap lists (choice_children) and distinct matrix (choice_mat)
    choice_children = [set(np.arange((init_child_mapped.shape[1])))]
    choice_mat      = ~(init_child_mapped==choice_mesh_vals) # number of values x number of gabors
    assert (np.sum(~choice_mat,axis=0) == 1).all(), 'Choice matrix is not correct.'
    assert np.sum(~choice_mat) == init_child_mapped.shape[1], 'Choice matrix is not correct.'
    
    # Create children sequentially
    child_failed = False
    for i in range(num_child-1):
        child_i = init_child_mapped.copy()

        # Compute preference vector for rows (prefer gabors indices that have used up most of their distinct features)
        taken = np.array([choice_mat.shape[0]-len(np.where(choice_mat[:,jj])[0]) for jj in range(choice_mat.shape[1])])
        p_taken = taken / np.sum(taken)

        # Sample prob_overlap*num_gabor gabors from previous children
        idx_j = set([])
        overlap_i = []
        try:
            for j in range(len(all_children)):

                # Compute preference distribution for overlap gabors (for child j)
                cc = np.array(list(choice_children[j].difference(idx_j)))
                if len(cc)<num_overlap:
                    child_failed = True
                    print(f'Required overlap between children is too large to sample additional children. Number of children returned: {len(all_children)}.')  
                    break
                pcc = p_taken[cc] / np.sum(p_taken[cc])     # indices relative to cc!
                chose_pcc = np.where(pcc==np.max(pcc))[0]   # indices relative to cc
                if len(chose_pcc) < num_overlap:
                    chose_pcc = np.concatenate([chose_pcc,rng.choice(np.where(pcc==np.max(pcc[~chose_pcc]))[0],num_overlap-len(chose_pcc),replace=False)]) # indices relative to cc

                # Sample overlaps of current child i with child j
                copy_j = rng.choice(cc[chose_pcc],num_overlap,replace=False,p=pcc[chose_pcc]/np.sum(pcc[chose_pcc])) 
                child_i[0,copy_j] = all_children[j][0,copy_j]                               # copy gabors to new child
                idx_j.update(set(copy_j))                                                   # add indices of copied gabors to set               
                choice_children[j].difference_update(set(copy_j))                           # remove indices of copied gabors from choice_children (to avoid overlap with more than one other child)
                if num_overlap==1:
                    copy_j = [copy_j]
                overlap_i.append(copy_j)
        except:
            child_failed = True
            print(f'Required overlap between children is too large to sample additional children. Number of children returned: {len(all_children)}.')  
            break

        if not child_failed:
            try:
                # Fill remaining gabors from choice_mat and update choice_mat
                for fill_j in set(idx_gabor).difference(idx_j):
                    copy_mat = rng.choice(list(np.where(choice_mat[:,fill_j])[0]),1,replace=False)
                    child_i[0,fill_j] = vals_gabor[copy_mat]
                    choice_mat[copy_mat,fill_j] = False
            except:
                child_failed = True
                print(f'Required overlap between children is too small to sample additional children. Number of children returned: {len(all_children)}.')
                break

        # Save child 
        if not child_failed:
            all_children.append(child_i)
            choice_children.append(set(np.arange(init_child_mapped.shape[1])).difference(set(np.concatenate(overlap_i).flatten())))
            all_overlap.append(overlap_i)
        else:
            break
    
    # Map children back to original space
    all_children_features = [map_child_idx2feat(all_children[i],fl0,fl1,pos_init) for i in range(len(all_children))]

    # Complete all children with invariant dimensions
    field_invariant = np.sort(np.array(list(set(np.arange(init_child_full.shape[0])).difference(set(field_transform)))))
    order_dim = np.argsort(np.concatenate([np.array(field_transform),field_invariant]))
    all_children_full = [np.concatenate([all_children_features[i],init_child_full[field_invariant,:]],axis=0)[order_dim,:] for i in range(len(all_children))]
        
    return all_children_full, all_overlap

def generate_teststim_children_exact(parent,num_child,fun_transform=[transform_rotate_left],prob_transform=[0.25],rng=None,mode='fixed'):
    if rng is None:
        rng = np.random.default_rng() 
    num_gabor  = parent.shape[1]

    if mode=='fixed':
        choice_set = np.arange(num_gabor)

        num_transform = [int(np.round(num_gabor*pt)) for pt in prob_transform]
        # num_toomany   = np.sum(num_transform)-num_gabor
        # if num_toomany>0:
        #     for i in range(num_toomany):
        #         max_idx = np.argmax(num_transform)
        #         num_transform[max_idx] -= 1
        # elif num_toomany<0:
        #     for i in range(-num_toomany):
        #         min_idx = np.argmin(num_transform)
        #         num_transform[min_idx] += 1
        # assert np.sum(num_transform)==num_gabor, 'Number of gabors does not match number of transformations.'
        assert np.sum(num_transform)<=num_gabor, 'Number of gabors does not match number of transformations.'

        # Create children
        all_children       = [parent.copy() for i in range(num_child)]
        choice_mat         = np.zeros((num_gabor,num_child))
        num_child_per_j    = []
        for j, (ft, nt) in enumerate(zip(fun_transform,num_transform)):
            stop_generating = False
            i = 0
            while not stop_generating and i<num_child:
                # Choose gabors for transformation (if enough available)
                free_gabors = np.where(np.sum(choice_mat,axis=1)==0)[0]
                if len(free_gabors)<nt:
                    stop_generating = True
                    last_child_j = i-1
                    break 
                choice_ji = rng.choice(free_gabors,nt,replace=False)
                choice_mat[choice_ji,i] = j+1

                # Apply transform
                child_i = all_children[i]
                child_i_update = ft(child_i[:,choice_ji])
                child_i[:,choice_ji] = child_i_update
                i += 1

            if not stop_generating:
                last_child_j = num_child-1
            num_child_per_j.append(last_child_j)
        
        num_child_succeeded = np.min(num_child_per_j)+1
        all_children        = all_children[:num_child_succeeded]
        all_set_transform   = []
        for i in range(num_child_succeeded):
            set_transform_i = [np.where(choice_mat[:,i]==j+1)[0] for j in range(len(fun_transform))]
            all_set_transform.append(set_transform_i)
        
        print(f'Successfully generated {num_child_succeeded} children.')

    return all_children, all_set_transform

###########################################################################
# Define spatial (2D) Gabor and Gaussian filters                          #
###########################################################################
# Compute empty image 
def get_empty(range_x0,range_y0,resolution=100,ratio_y_x=2,add_eps=1e-10):
    # Define mesh for evaluating Gabor filter
    xmesh,ymesh = np.meshgrid(np.linspace(range_x0[0],range_x0[1],int(np.round(ratio_y_x*resolution))),
                                          np.linspace(range_y0[0],range_y0[1],int(np.round(resolution))))
    # Get empty image
    g = np.zeros(xmesh.shape)
    g += add_eps
    return g

# Compute pixel image based on single Gabor parameter set
def comp_gabor_single(range_x0,range_y0,stim,resolution=100,magn=1,ratio_y_x=2,add_eps=1e-10):
    # Define mesh for evaluating Gabor filter
    ratio_y_x = (range_x0[1]-range_x0[0]) / (range_y0[1]-range_y0[0])
    xmesh,ymesh = np.meshgrid(np.linspace(range_x0[0],range_x0[1],int(np.round(ratio_y_x*resolution))),
                              np.linspace(range_y0[0],range_y0[1],int(np.round(resolution))))
    
    # Define params 
    alph = stim[0]; phase = stim[1] # orientation, phase (in units of radians, i.e. 1*np.pi etc.)
    freq = stim[2]; wx = stim[3]; wy = stim[3] # frequency, width
    x0 = stim[4]; y0 = stim[5] # x-y location
    if not magn: magn = 1 / (2 * np.pi * wx * wy) 

    # Define rotated coordinates
    xr = (xmesh - x0) * np.cos(alph) + (ymesh - y0) * np.sin(alph)    
    yr = - (xmesh - x0) * np.sin(alph) + (ymesh - y0) * np.cos(alph)

    # Compute Gabor filter
    g = np.exp(-0.5 * (xr**2 / wx**2 + yr**2 / wy**2)) * np.cos(2 * np.pi * freq * xr + phase)
    g_norm = magn * g

    # Add small epsilon activation to image
    g_norm += add_eps

    return g_norm

# Create combined pixel image based on vector of Gabor parameters
def comp_gabor(range_x0,range_y0,stim,resolution=100,magn=1,ratio_y_x=2,add_eps=1e-10):
    gmesh_list = []
    for i in range(stim.shape[1]):
        gmesh = comp_gabor_single(range_x0,range_y0,stim[:,i],resolution,magn,ratio_y_x,add_eps=add_eps)
        if i==0:
            gmesh_all = gmesh.copy()
        else:
            gmesh_all += gmesh.copy()
        gmesh_list.append(gmesh)
    return gmesh_all, gmesh_list

# Create single Gaussian filter
def comp_gauss_single(range_x0,range_y0,stim,resolution=100,magn=1,ratio_y_x=2,add_eps=0): # Note: we don't want to add zero directly to the mask, only to the masked stimuli (to have the 'non-stimulus' parts of the image at value add_eps)
    # Define mesh for evaluating Gabor filter
    xmesh,ymesh = np.meshgrid(np.linspace(range_x0[0],range_x0[1],int(np.round(ratio_y_x*resolution))),
                              np.linspace(range_y0[0],range_y0[1],int(np.round(resolution))))
    
    # Define params
    alph = stim[0]; phase = stim[1]; freq = stim[2]; wx = stim[3]; wy = stim[3]; x0 = stim[4]; y0 = stim[5]
    if not magn: magn = 1 / (2 * np.pi * wx * wy) 

    # Define rotated coordinates
    xr = (xmesh - x0) * np.cos(alph) + (ymesh - y0) * np.sin(alph)    
    yr = - (xmesh - x0) * np.sin(alph) + (ymesh - y0) * np.cos(alph)

    # Compute Gabor filter
    g = np.exp(-0.5 * (xr**2 / wx**2 + yr**2 / wy**2))
    g_norm = magn * g

    # Add small epsilon activation to image
    g_norm += add_eps

    return g_norm

# Create multiple Gaussian filters
def comp_gauss(range_x0,range_y0,stim,resolution=100,magn=1,ratio_y_x=2,add_eps=0):
    gmesh_list = []
    for i in range(stim.shape[1]):
        gmesh = comp_gauss_single(range_x0,range_y0,stim[:,i],resolution,magn,ratio_y_x,add_eps=add_eps)
        if i==0:
            gmesh_all = gmesh.copy()
        else:
            gmesh_all += gmesh.copy()
        gmesh_list.append(gmesh)
    return gmesh_all, gmesh_list


##########################################################################
# Old functions to compute Gabors (DEPRECATED)                           #
###########################################################################
def my_gabor(x,y,alph,phase,freq,wx,wy,x0,y0,magn=10): # (OUTDATED)
    xr = (x-x0)*np.cos(alph)+(y-y0)*np.sin(alph)    # rotated x-coord.
    yr = -(x-x0)*np.sin(alph)+(y-y0)*np.cos(alph)   # rotated y-coord.
    arg_real = -np.pi*((1/wx*xr)**2+(1/wy*yr)**2)
    fx = freq*np.cos(alph)
    fy = freq*np.sin(alph)
    arg_imag = 1j*(2*np.pi*(fx*x+fy*y)+phase)
    g = magn*np.exp(arg_real+arg_imag) 
    return g

# Evaluate single spatial Gabor filter on matrix of points (OUTDATED)
def eval_my_gabor(range_x0,range_y0,stim,resolution=10,magn=1,ratio_x_y=1):
    xmesh,ymesh = np.meshgrid(np.linspace(range_x0[0],range_x0[1],ratio_x_y*resolution),np.linspace(range_y0[0],range_y0[1],resolution))
    gmesh = my_gabor(xmesh,ymesh,alph=stim[0],phase=stim[1],freq=stim[2],wx=stim[3],wy=stim[3],x0=stim[4],y0=stim[5],magn=magn) 
    return gmesh

# Evaluate multiple spatial Gabor filters on matrix of points (OUTDATED)
def plot_gabor_stim(range_x0,range_y0,stim,resolution=10,magn=1):
    gmesh_list = []
    for i in range(stim.shape[1]):
        gmesh = eval_my_gabor(range_x0,range_y0,stim[:,i],resolution,magn)
        if i==0:
            gmesh_all = gmesh.copy()
        else:
            gmesh_all += gmesh.copy()
        gmesh_list.append(gmesh)
    return gmesh_all, gmesh_list