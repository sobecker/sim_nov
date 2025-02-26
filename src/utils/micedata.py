import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
# sys.path.append('/Users/sbecker/Projects/Rosenberg-2021-Repository/code')

import src.utils.visualization as vis
# from MM_Maze_Utils import *

###############################################################################################################
def get_UnrewExp_until_goal(savedata=True,excludefailed=True):
    # Ids for experimental data Rosenberg
    UnrewNames=['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
    AllNames=UnrewNames
    dir_load = '/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata'
    # dir_save = '/Users/sbecker/Library/CloudStorage/OneDrive-Personal/Studium/PhD/Python/RL_reward_novelty/ext_data/Rosenberg2021'
    dir_save = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Rosenberg2021'

    # Lists for df_stateseq
    subID_ll    = []    # subIDs (string)
    subRew_ll   = []    # water-deprived or not water deprived subID (bool)
    epi_ll      = []    # number of epi (int), will be always 0 but necessary for compatibility with sim analysis
    it_ll       = []    # step count
    s_ll        = []    # state identity
    r_ll        = []    # goal state visit rewarded (False for all non-goal state visits and goal-state visits without reward received)
    g_ll        = []

    failed_ll   = []
    gs = 116

    for subID in AllNames: # iterate over subjects
        states = []
        with open(dir_load+'/'+subID+'-tf', 'rb') as f:
            tf = pickle.load(f) # filter only exploration episode!!! - get reward for tf.?? and only go until there
            # tf has the following structure:
            # fr: start and end frame for each bout; (n,2) ndarray 
            # ce: cell number in each frame; list of ndarrays, one for each bout
            # ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
            # no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
            # re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout

        foundgoal = False
        for i in range(len(tf.no)): # iterate over bouts, i.e. path within maze (from home cage to home cage)
            # print(f"animal {subID}, bout {i}")
            nh = np.nonzero(tf.no[i][:,0]!=127)[0]  # exclude home cage (node 127) visits, since we include these manually
            if len(nh)>0: 
                # Append home cage visit
                states.append(0)                    
                # Append remaining states until goal first state visit
                tfno_nh = tf.no[i][nh,:]
                rt = np.nonzero(tfno_nh[:,0]==gs)[0]        # get indices of goal state encounter (node 116)
                if len(rt)>0:                                      # if goal state encountered during current bout:
                    foundgoal = True
                    states.extend(list(tfno_nh[:rt[0]+1,0]+1))     # append non-home cage visits up to first goal state encounter
                    break
                else:                                              # if goal state not encountered during current bout:
                    states.extend(list(tf.no[i][nh,0]+1))          # append all non-home cage visits       
        s_seq = np.array([np.arange(len(states)),states]).transpose()
        if savedata: 
            dir_save1 = dir_save+f'/{subID}_data'
            vis.make_dir_fig(dir_save1)
            with open(dir_save1+'/'+subID+'-stateseq_UnrewUntilG.pickle', 'wb') as f:
                pickle.dump(s_seq,f)   
            pd.DataFrame(s_seq).to_csv(dir_save+'/'+subID+'-stateseq_UnrewUntilG.csv',sep='\t',header=None,index=None)
        
        if not excludefailed or foundgoal:
            subID_ll.extend([subID]*len(s_seq))
            subRew_ll.extend([0]*len(s_seq)) 
            epi_ll.extend([0]*len(s_seq))
            it_ll.extend(s_seq[:,0])
            s_ll.extend(s_seq[:,1])
            g_ll.extend(s_seq[:,1]==gs+1)
            r_ll.extend([False]*s_seq[:,1])
        else:
            failed_ll.append(subID)

    # Safety check
    print(f"lengths of columns:{len(subID_ll)}, {len(r_ll)}, {len(g_ll)}")
    # Make df_stateseq dataframe
    df_stateseq = pd.DataFrame([subID_ll,subRew_ll,epi_ll,it_ll,s_ll,r_ll,g_ll])
    df_stateseq = df_stateseq.transpose()
    df_stateseq.columns = ['subID','subRew','epi','it','state','reward_received','goal_state']
    if savedata: 
        with open(dir_save+'/df_stateseq_UnrewUntilG.pickle','wb') as f:
            pickle.dump(df_stateseq,f)
        df_stateseq.to_csv(dir_save+'/df_statseq_UnrewUntilG.csv',sep='\t')
        print(f'Saved data for subID {subID}.\n')
    return df_stateseq,failed_ll


###############################################################################################################
def get_RewExp_until_goal(savedata=True,excludefailed=True):
    # Ids for experimental data Rosenberg
    RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
    AllNames=RewNames
    dir_load = '/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata'
    dir_save = '/Users/sbecker/Library/CloudStorage/OneDrive-Personal/Studium/PhD/Python/RL_reward_novelty/ext_data/Rosenberg2021'

    # Lists for df_stateseq
    subID_ll    = []    # subIDs (string)
    subRew_ll   = []    # water-deprived or not water deprived subID (bool)
    epi_ll      = []    # number of epi (int), will be always 0 but necessary for compatibility with sim analysis
    it_ll       = []    # step count
    s_ll        = []    # state identity
    r_ll        = []    # goal state visit rewarded (False for all non-goal state visits and goal-state visits without reward received)
    g_ll        = []

    failed_ll   = []
    gs = 116

    for subID in AllNames: # iterate over subjects
        states = []
        with open(dir_load+'/'+subID+'-tf', 'rb') as f:
            tf = pickle.load(f) # filter only exploration episode!!! - get reward for tf.?? and only go until there
            # tf has the following structure:
            # fr: start and end frame for each bout; (n,2) ndarray 
            # ce: cell number in each frame; list of ndarrays, one for each bout
            # ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
            # no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
            # re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout

        foundgoal = False
        for i in range(len(tf.no)): # iterate over bouts, i.e. path within maze (from home cage to home cage)
            # print(f"animal {subID}, bout {i}")
            nh = np.nonzero(tf.no[i][:,0]!=127)[0]  # exclude home cage (node 127) visits, since we include these manually
            if len(nh)>0: 
                # Append home cage visit
                states.append(0)                    
                r_ll.append(False)                    

                # Append remaining states until goal first state visit
                tfno_nh = tf.no[i][nh,:]
                rt = np.nonzero(tfno_nh[:,0]==gs)[0]        # get indices of goal state encounter (node 116)
                rt_rew = tf.re[i]                           # get start/end frame of reward (empty if no reward during bout i)
                if len(rt)>0:                                      # if goal state encountered during current bout:
                    foundgoal=True
                    states.extend(list(tfno_nh[:rt[0]+1,0]+1))     # append non-home cage visits up to first goal state encounter
                    # Determine whether goal state visit was rewarded or not
                    if len(rt_rew)>0:
                        rt_ng = np.nonzero(tfno_nh[rt[0]+1:,0]!=gs)[0]
                        if len(rt_ng)>0:
                            b = ((rt_rew[0][0]>tfno_nh[rt[0],1]) and (rt_rew[0][0]<tfno_nh[rt[0]+1+rt_ng[0],1]))  # is start_frame(rew) between start_frame(s_goal) and start_frame(next s_nongoal)?
                        else:
                            b = ((rt_rew[0][0]>tfno_nh[rt[0],1]) and (rt_rew[0][0]<tfno_nh[-1,1]))  # is start_frame(rew) between start_frame(s_goal) and end of recording? (only if bout finished with goal state)
                        if b:
                            r_ll.extend([False]*(len(list(tfno_nh[:rt[0]+1,0]+1))-1))
                            r_ll.extend([True])
                        else:
                            r_ll.extend([False]*len(list(tfno_nh[:rt[0]+1,0]+1)))
                    else:
                        r_ll.extend([False]*len(list(tfno_nh[:rt[0]+1,0]+1)))
                    break
                else:                                              # if goal state not encountered during current bout:
                    states.extend(list(tf.no[i][nh,0]+1))          # append all non-home cage visits
                    r_ll.extend([False]*len(list(tf.no[i][nh,0]+1)))
       
        s_seq = np.array([np.arange(len(states)),states]).transpose()
        #print(f"i={i}:{len(states)},{len(r_ll)}")
        if savedata: 
            dir_save1 = dir_save+f'/{subID}_data'
            vis.make_dir_fig(dir_save1)
            with open(dir_save1+'/'+subID+'-stateseq_RewUntilG.pickle', 'wb') as f:
                pickle.dump(s_seq,f)   
            pd.DataFrame(s_seq).to_csv(dir_save+'/'+subID+'-stateseq_RewUntilG.csv',sep='\t',header=None,index=None)
        
        if not excludefailed or foundgoal:
            subID_ll.extend([subID]*len(s_seq))
            subRew_ll.extend([1]*len(s_seq)) 
            epi_ll.extend([0]*len(s_seq))
            it_ll.extend(s_seq[:,0])
            s_ll.extend(s_seq[:,1])
            g_ll.extend(s_seq[:,1]==gs+1)
        else:
            failed_ll.append(subID)

    print(f"lengths of columns:{len(subID_ll)}, {len(r_ll)}, {len(g_ll)}")
    # Make df_stateseq dataframe
    df_stateseq = pd.DataFrame([subID_ll,subRew_ll,epi_ll,it_ll,s_ll,r_ll,g_ll])
    df_stateseq = df_stateseq.transpose()
    df_stateseq.columns = ['subID','subRew','epi','it','state','reward_received','goal_state']
    if savedata: 
        with open(dir_save+'/df_stateseq_RewUntilG.pickle','wb') as f:
            pickle.dump(df_stateseq,f)
        df_stateseq.to_csv(dir_save+'/df_statseq_RewUntilG.csv',sep='\t')
        print(f'Saved data for subID {subID}.\n')
    return df_stateseq, failed_ll


###############################################################################################################
def get_RewExp_until_reward(savedata=True):
    # Ids for experimental data Rosenberg
    RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
    AllNames=RewNames
    dir_load = '/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata'
    dir_save = '/Users/sbecker/Library/CloudStorage/OneDrive-Personal/Studium/PhD/Python/RL_reward_novelty/ext_data/Rosenberg2021'

    # Lists for df_stateseq
    subID_ll    = []    # subIDs (string)
    subRew_ll   = []    # water-deprived or not water deprived subID (bool)
    epi_ll      = []    # number of epi (int), will be always 0 but necessary for compatibility with sim analysis
    it_ll       = []    # step count
    s_ll        = []    # state identity
    g_ll        = []    # goal state (bool)
    r_ll        = []    # reward received 
    
    gs = 116            # goal state

    for subID in AllNames: # iterate over subjects
        states = []
        with open(dir_load+'/'+subID+'-tf', 'rb') as f:
            tf = pickle.load(f) # filter only exploration episode!!! - get reward for tf.?? and only go until there
            # tf has the following structure:
            # fr: start and end frame for each bout; (n,2) ndarray 
            # ce: cell number in each frame; list of ndarrays, one for each bout
            # ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
            # no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
            # re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout

        for i in range(len(tf.no)): # iterate over bouts, i.e. path within maze (from home cage to home cage)
            # print(f"animal {subID}, bout {i}")
            nh = np.nonzero(tf.no[i][:,0]!=127)[0]  # exclude home cage (node 127) visits, since we include these manually
            if len(nh)>0: 
                states.append(0)                    # append home cage visit
                r_ll.append(False)
                tfno_nh = tf.no[i][nh,:]
                rt = tf.re[i]                           # get start/end frame of reward (empty if no reward during bout i)
                if len(rt)>0:                                      # if goal state encountered during current bout:
                    frt = np.nonzero(tfno_nh[:,1]>=rt[0][0])[0][0] # get start frame of first reward
                    states.extend(list(tfno_nh[:frt,0]+1))     # append non-home cage visits up to first reward
                    r_ll.extend([False]*(len(list(tfno_nh[:frt,0]+1))-1))
                    r_ll.extend([True])
                    break
                else:                                              # if goal state not encountered during current bout:
                    states.extend(list(tf.no[i][nh,0]+1))          # append all non-home cage visits
                    r_ll.extend([False]*len(list(tf.no[i][nh,0]+1)))
        s_seq = np.array([np.arange(len(states)),states]).transpose()
        if savedata: 
            dir_save1 = dir_save+f'/{subID}_data'
            vis.make_dir_fig(dir_save1)
            with open(dir_save1+'/'+subID+'-stateseq_RewUntilR.pickle', 'wb') as f:
                pickle.dump(s_seq,f)   
            pd.DataFrame(s_seq).to_csv(dir_save+'/'+subID+'-stateseq_RewUntilR.csv',sep='\t',header=None,index=None)
        
        subID_ll.extend([subID]*len(s_seq))
        subRew_ll.extend([1]*len(s_seq)) 
        epi_ll.extend([0]*len(s_seq))
        it_ll.extend(s_seq[:,0])
        s_ll.extend(s_seq[:,1])
        g_ll.extend(s_seq[:,1]==gs+1)

    print(f"lengths of columns:{len(subID_ll)}, {len(r_ll)}, {len(g_ll)}")
    # Make df_stateseq dataframe
    df_stateseq = pd.DataFrame([subID_ll,subRew_ll,epi_ll,it_ll,s_ll,r_ll,g_ll])
    df_stateseq = df_stateseq.transpose()
    df_stateseq.columns = ['subID','subRew','epi','it','state','reward_received','goal_state']
    if savedata: 
        with open(dir_save+'/df_stateseq_RewUntilR.pickle','wb') as f:
            pickle.dump(df_stateseq,f)
        df_stateseq.to_csv(dir_save+'/df_statseq_RewUntilR.csv',sep='\t')
        print(f'Saved data for subID {subID}.\n')
    return df_stateseq


###############################################################################################################
def get_AllMice_until_maxit(savedata=True):
    # Ids for experimental data Rosenberg
    UnrewNames=['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
    RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
    dir_load = '/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata'
    # dir_save = '/Users/sbecker/Library/CloudStorage/OneDrive-Personal/Studium/PhD/Python/RL_reward_novelty/ext_data/Rosenberg2021'
    dir_save = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Rosenberg2021'

    # Lists for df_stateseq
    subID_ll    = []    # subIDs (string)
    subRew_ll   = []    # water-deprived or not water deprived subID (bool)
    epi_ll      = []    # number of epi (int), will be always 0 but necessary for compatibility with sim analysis
    it_ll       = []    # step count
    s_ll        = []    # state identity
    r_ll        = []    # goal state visit rewarded (False for all non-goal state visits and goal-state visits without reward received)
    g_ll        = []

    gs = 116

    for subID in RewNames: # iterate over rewarded subjects
        states = []
        rews   = []
        with open(dir_load+'/'+subID+'-tf', 'rb') as f:
            tf = pickle.load(f) # filter only exploration episode!!! - get reward for tf.?? and only go until there
            # tf has the following structure:
            # fr: start and end frame for each bout; (n,2) ndarray 
            # ce: cell number in each frame; list of ndarrays, one for each bout
            # ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
            # no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
            # re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout

        for i in range(len(tf.no)): # iterate over bouts, i.e. path within maze (from home cage to home cage)
            # print(f"animal {subID}, bout {i}")
            nh = np.nonzero(tf.no[i][:,0]!=127)[0]  # exclude home cage (node 127) visits, since we include these manually
            if len(nh)>0: 
                # Append home cage visit
                states.append(0)
                rews.append(False)                    
                # Append remaining states until goal first state visit
                tfno_nh = tf.no[i][nh,:]
                r  = np.array([False]*len(list(tf.no[i][nh,0]+1)))
                rt = tf.re[i]                                       # get start/end frame of reward (empty if no reward during bout i)
                if len(rt)>0:
                    for rr in range(len(rt[0])):                          
                        frt = np.nonzero(tfno_nh[:,1]>=rt[0][rr])[0]  # get frames of rewarded states
                        if len(frt)>0:
                            r[frt[0]-1] = True
                states.extend(list(tf.no[i][nh,0]+1))       # append all non-home cage visits 
                rews.extend(r)      
        s_seq = np.array([np.arange(len(states)),states]).transpose()
        if savedata: 
            dir_save1 = dir_save+f'/{subID}_data'
            vis.make_dir_fig(dir_save1)
            pd.DataFrame(s_seq).to_pickle(dir_save1+'/'+subID+'-stateseq_AllMiceFull.pickle')
            pd.DataFrame(s_seq).to_csv(dir_save+'/'+subID+'-stateseq_AllMiceFull.csv',sep='\t',header=None,index=None)
        subID_ll.extend([subID]*len(s_seq))
        subRew_ll.extend([1]*len(s_seq)) 
        epi_ll.extend([0]*len(s_seq))
        it_ll.extend(s_seq[:,0])
        s_ll.extend(s_seq[:,1])
        g_ll.extend(s_seq[:,1]==gs+1)
        r_ll.extend(rews)

    for subID in UnrewNames: # iterate over unrewarded subjects
        states = []
        rews   = []
        with open(dir_load+'/'+subID+'-tf', 'rb') as f:
            tf = pickle.load(f) # filter only exploration episode!!! - get reward for tf.?? and only go until there
            # tf has the following structure:
            # fr: start and end frame for each bout; (n,2) ndarray 
            # ce: cell number in each frame; list of ndarrays, one for each bout
            # ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
            # no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
            # re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout

        for i in range(len(tf.no)): # iterate over bouts, i.e. path within maze (from home cage to home cage)
            # print(f"animal {subID}, bout {i}")
            nh = np.nonzero(tf.no[i][:,0]!=127)[0]  # exclude home cage (node 127) visits, since we include these manually
            if len(nh)>0: 
                # Append home cage visit
                states.append(0)
                rews.append(False)                    
                # Append remaining states until goal first state visit
                tfno_nh = tf.no[i][nh,:]
                r  = np.array([False]*len(list(tf.no[i][nh,0]+1)))
                states.extend(list(tf.no[i][nh,0]+1))       # append all non-home cage visits 
                rews.extend(r)      
        s_seq = np.array([np.arange(len(states)),states]).transpose()
        if savedata: 
            dir_save1 = dir_save+f'/{subID}_data'
            vis.make_dir_fig(dir_save1)
            pd.DataFrame(s_seq).to_pickle(dir_save1+'/'+subID+'-stateseq_AllMiceFull.pickle')
            pd.DataFrame(s_seq).to_csv(dir_save+'/'+subID+'-stateseq_AllMiceFull.csv',sep='\t',header=None,index=None)
        subID_ll.extend([subID]*len(s_seq))
        subRew_ll.extend([0]*len(s_seq)) 
        epi_ll.extend([0]*len(s_seq))
        it_ll.extend(s_seq[:,0])
        s_ll.extend(s_seq[:,1])
        g_ll.extend(s_seq[:,1]==gs+1)
        r_ll.extend(rews)

    # Safety check
    print(f"lengths of columns:{len(subID_ll)}, {len(r_ll)}, {len(g_ll)}")
    # Make df_stateseq dataframe
    df_stateseq = pd.DataFrame([subID_ll,subRew_ll,epi_ll,it_ll,s_ll,r_ll,g_ll])
    df_stateseq = df_stateseq.transpose()
    df_stateseq.columns = ['subID','subRew','epi','it','state','reward_received','goal_state']
    if savedata: 
        df_stateseq.to_pickle(dir_save+'/df_stateseq_AllMiceFull.pickle')
        df_stateseq.to_csv(dir_save+'/df_statseq_AllMiceFull.csv',sep='\t')
        print(f'Saved data for subID {subID}.\n')
    return df_stateseq

###############################################################################################################
def get_time_per_partition(k_list):
    # Lists for df_dur 
    subID_list = []
    level_list = []
    kernel_list = []
    visit_list = []
    dur_list = []
    ksizes_list = []

    return None