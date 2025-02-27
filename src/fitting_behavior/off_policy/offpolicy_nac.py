import numpy as np
import pandas as pd
import scipy
import sys
import models.mf_agent.ac as ac

def offpolicy_nac(params,data,rec_counts=False,rec_qvals=False,rec_ll=False):
    # Create environment
    T       = (params['T'] if 'T' in params.keys() else np.array([]))
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)
    env        = ac.env(params['S'],params['P'],params['R'],T,t_deact)
    goal       = env.getGoal()
    terminal   = env.getTerminal()
    t_deact    = env.getTDeact()
    
    # Initialize trial
    flag_foundGoal = False
    for s in goal:
        env.activateGoal(s)
    g_deact  = []
    it_deact = []
        
    # Initialize agent
    agent = ac.agent(params)  
    env.setAgentLoc(data['state'].values[0])
    agent.giveStateInput(data['state'].values[0])
    agent.updateMod(data['state'].values[0])
    agent.resetTraces()  

    # Init loglikelihood
    if rec_ll: ll = 0

    # Initialize recording
    all_t           = []
    all_tt          = []
    all_s           = []
    all_s_new       = []
    all_nov_s_pre        = []
    all_nov_s_new_pre    = []
    all_nov_s_post       = []
    all_nov_s_new_post   = []
    if rec_counts:  
        all_c_s_pre      = []
        all_c_s_new_pre  = []
        all_c_s_post     = []
        all_c_s_new_post = []
    if rec_qvals:   all_qvals = []
    if rec_ll:      all_ll = []
            
    for it in range(len(data)):
        # Reactive goal states if their deactivation time is over
        it_act = (((np.array(it_deact)-t_deact)>0).nonzero()[0])
        for i in it_act:
            env.activateGoal(g_deact)
            g_deact.pop(i)
            it_deact.pop(i)
            
        # Compute loglikelihood of current action from observed data
        s_current = env.agentLoc
        agent.actors[0].giveStateInput(s_current)
        out_rates   = np.array([agent.actors[0].a[i].computeRate() if ~np.isnan(agent.actors[0].w[s_current][i]) else np.NaN for i in range(len(agent.actors[0].a))])
        not_nan     = (~np.isnan(out_rates)).nonzero()
        p_softmax   = scipy.special.softmax(out_rates[list(not_nan[0])]/params['temp'],axis=0)
        if params['temp']==0 or np.isnan(p_softmax).any():
            print(f"{out_rates}, {params['temp']}, {p_softmax}\n")

        #q_notnan = list((~np.isnan(q[s])).nonzero()[0])
        a         = int(data['action'].values[it])
        if rec_ll:
            lpa       = np.log(p_softmax[list(not_nan[0]).index(a)]) ###
            ll       += lpa  

        s_new_env, r    = env.evalAction(a,s_current)
        s_new           = data['next_state'].values[it]
        if s_new_env!=s_new:
            print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')
        #print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')

        # Record time and transition variables
        all_t.append(it)
        all_tt.append(agent.critics[0].pc.t)
        all_s.append(s_current)
        all_s_new.append(s_new_env)

        # Record novelty and counts before seeing s_new (used as learning signal)
        m_current, _, _, _ = agent.evalMod(s_current,r)
        m, _, _, _  = agent.evalMod(s_new_env,r)  
        all_nov_s_pre.append(m_current[0])
        all_nov_s_new_pre.append(m[0])
        if rec_counts:
            all_c_s_pre.append(agent.critics[0].pc.counts[s_current])
            all_c_s_new_pre.append(agent.critics[0].pc.counts[s_new])
        
        # Learn from novelty signal received
        agent.updateMod(s_new_env)       
        tds, new_wcs, new_ecs, new_was, new_eas = agent.learn(s_current,s_new_env,a,m) 

        # Record novelty and counts after seeing s_new (for comparison with nor)
        m_current, _, _, _ = agent.evalMod(s_current,r)
        m, _, _, _  = agent.evalMod(s_new_env,r)  
        all_nov_s_post.append(m_current[0])
        all_nov_s_new_post.append(m[0])
        if rec_counts:
            all_c_s_post.append(agent.critics[0].pc.counts[s_current])
            all_c_s_new_post.append(agent.critics[0].pc.counts[s_new])

        # Record q-values after learning from N(s_new)
        if rec_qvals:   all_qvals.append(np.array(new_was))
        if rec_ll:      all_ll.append(lpa)

        # Check whether agent has reached the goal / terminal state
        flag_foundGoal=False
        if s_new_env in env.R.nonzero()[0]:
            flag_foundGoal = True
                    
        if s_new_env == terminal:
            flag_foundTerminal = True
            
        # Deactivate reward for some time
        if flag_foundGoal:
            env.deactivateGoal(env.agentLoc)
            g_deact.append(env.agentLoc)
            it_deact.append(0)
        it_deact = [it_deact[i]+1 for i in range(len(it_deact))]
    
    # Format recording
    rec = {'time': all_t,
           'leaky_time': all_tt,
           'state': all_s,
           'next_state': all_s_new,
           'nov_s_pre': all_nov_s_pre,
           'nov_s_new_pre': all_nov_s_new_pre,
           'nov_s_post': all_nov_s_post,
            'nov_s_new_post': all_nov_s_new_post
           }
    if rec_counts: 
        rec['counts_s_pre'] = all_c_s_pre
        rec['counts_s_new_pre'] = all_c_s_new_pre
        rec['counts_s_post'] = all_c_s_post
        rec['counts_s_new_post'] = all_c_s_new_post
    if rec_ll: rec['LL'] = all_ll
    rec = pd.DataFrame(rec)

    if not rec_qvals: all_qvals = None
    
    return rec, all_qvals