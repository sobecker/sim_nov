import numpy as np
import scipy
import sys
import models.mf_agent.ac as ac

def ll_nac(params,data,qvals=[]):
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
    ll = 0
    if len(qvals)>0: all_q_equal = True
    else:            all_q_equal = None  
            
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
        lpa       = np.log(p_softmax[list(not_nan[0]).index(a)]) ###
        ll       += lpa  

        s_new_env, r    = env.evalAction(a,s_current)
        s_new           = data['next_state'].values[it]
        #print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')
        
        m, mh, mw, mg  = agent.evalMod(s_new_env,r)           
        agent.updateMod(s_new_env)       
        tds, new_wcs, new_ecs, new_was, new_eas = agent.learn(s_current,s_new_env,a,m) 

        if len(qvals)>0:
            qvals_notnan = (~np.isnan(new_was)).nonzero()
            # if not (np.array(new_was)[qvals_notnan]==qvals['qvals'].values[it][qvals_notnan]).all():
            if not (np.array(new_was)[qvals_notnan]==qvals[it]).all():
                all_q_equal = False

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
    
    return ll, all_q_equal