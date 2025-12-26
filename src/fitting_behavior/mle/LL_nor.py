import numpy as np
import scipy
import models.mb_agent.mb_surnor as nor
import models.mf_agent.ac as ac

def ll_nor(params,data,qvals=[],start_ll=None,stop_ll=None,rec_ll=False,rec_nov=False):

    try:
        if start_ll is None:
            start_ll = 0
        if stop_ll is None:
            stop_ll = len(data)

        # Create environment
        S       = params['S']
        A       = params['A']
        P       = params['P']
        R       = params['R']
        T       = (params['T'] if 'T' in params.keys() else np.array([]))
        t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)
        env     = ac.env(S,list(P),list(R),T,t_deact)
        sg      = env.getGoal()
        t_deact = env.getTDeact()

        # Extract params
        if not 'ntype' in params.keys():  
            params['ntype'] = 'N'
        if not 'k' in params.keys():      
            params['k']     = 0

        eps     = params['epsilon']
        lamR    = params['lambda_R']
        lamN    = params['lambda_N']
        Tps     = params['T_PS']
        beta    = params['beta_1']
        ntype   = params['ntype']
        k_leak  = params['k_leak']
        k       = params['k']

        # Init loglikelihood
        ll    = 0
        if len(qvals)>0: 
            all_q_equal = True
        else:            
            all_q_equal = None
        if rec_ll:
            ll_list = []
        if rec_nov:
            nov_list = []
        
        #### Compute sequence of Q-values ####      
        # Initialize the model variables
        betaN = params['beta_N1']
        alph  = eps*np.ones((S,A,S))
                    
        # Initialize  novelty variables
        if ntype=='hN':
            # Set hnov type and update function (only matters when using hierarchical novelty)
            # type 2: apply -log on each level of the hierachy, then compute weighted sum of novelty per level
            # type 3: compute weighted sum of familiarity per level, then apply -log on the sum
            hnov_type    = params['h']['hnov_type'] if 'hnov_type' in params['h'].keys() else params['hnov_type'] if 'hnov_type' in params.keys() else 2
            compute_hnov = eval(f'nor.compute_hnov{hnov_type}')
            
            # Set update type (fixed/variable learning rate for novelty signal)
            update_type = params['h']['update_type'] if 'update_type' in params['h'].keys() else params['update_type'] if 'update_type' in params.keys() else 'var'
            update_hnov = eval(f'nor.update_hnov_{update_type}rate')  # update_hnov_fixedrate, update_hnov_varrate, update_hnov_leakyrate
            
            # Set remaining hnov params
            w       = params['w']
            h       = params['h']       
            h_w     = h['h_w']              # kernel mixture weights
            kmat    = h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))

            if update_type=='fixed':
                h_eps       = None  
                h_alph_leak = None
                k_alph      = h['k_alph'] if 'k_alph' in h.keys() else [0.1]
                if not isinstance(k_alph, list):
                    k_alph = [k_alph]
                if len(k_alph)!=len(h_w):
                    k_alph = k_alph*len(h_w)
                    
            elif update_type=='leaky':
                h_eps       = h['eps_leak'] if 'eps_leak' in h.keys() else [1]
                if not isinstance(h_eps, list):
                    h_eps = [h_eps]
                if len(h_eps)!=len(h_w):
                    h_eps = h_eps*len(h_w)
                h_eps       = [h_eps[i] * h['eps'][i] for i in range(len(h_eps))] # scale epsilon with initial weights
                h_alph_leak = h['alph_leak'] if 'alph_leak' in h.keys() else [0] 
                if not isinstance(h_alph_leak, list):
                    h_alph_leak = [h_alph_leak]
                if len(h_alph_leak)!=len(h_w):
                    h_alph_leak = h_alph_leak*len(h_w)
                k_alph      = None
                gg          = [np.zeros(h_w[i].shape) for i in range(len(h_w))]

            elif update_type=='var':
                h_eps       = h['eps'] if 'eps' in h.keys() else [1]
                if not isinstance(h_eps, list):
                    h_eps = [h_eps]
                if len(h_eps)!=len(h_w):
                    h_eps = h_eps*len(h_w)  
                h_alph_leak = None
                k_alph      = None
            
        else:
            # Set update type (fixed/variable learning rate for novelty signal)
            update_type  = params['h']['update_type'] if 'update_type' in params['h'].keys() else 'var'
            update_nov   = eval(f'nor.update_nov_{update_type}rate')  # update_nov_fixedrate, update_nov_varrate, update_nov_leakyrate
            compute_nov  = eval(f'nor.compute_nov_{update_type}rate')
            
            # Set remaining params
            if update_type=='fixed':
                c_eps = None
                c_alph_leak = None
                k_alph = params['h']['k_alph'] if 'k_alph' in params['h'].keys() else 0.1
                # p = 1/S * np.ones(S)
                p = np.log(1/S) * np.ones(S)
            elif update_type=='leaky':
                c_eps = params['h']['eps_leak'][0] if 'eps_leak' in params['h'].keys() else 1
                c_alph_leak = params['h']['alph_leak'] if 'alph_leak' in params['h'].keys() else None
                k_alph = None
                c = np.zeros(S)
            elif update_type=='var':
                c_eps = 1
                c_alph_leak = None
                k_alph = None
                c = np.zeros(S)
                    
        # Compute initial novelty
        if ntype=='hN': 
            _, N0 = compute_hnov(h_w,kmat,k,w)
            N0 = N0/(1-lamN) 
        else: 
            N0 = np.log(S)/(1-lamN)*np.ones(S) 
        uN = N0
        qN = N0.reshape(-1,1)*np.ones(np.shape(env.P))

        # Set to nan where applicable (i.e. where no transitions)
        a_nan = np.isnan(env.P).nonzero()
        qN[a_nan] = np.NaN
        for i,j in zip(a_nan[0],a_nan[1]):
            alph[i,j,:] = np.NaN
        
        # Init agent to first state of observed data
        env.setAgentLoc(data['state'].values[0])
        s        = env.agentLoc
        t        = 1  # absolute time
        if update_type=='leaky':
            if ntype=='hN':
                tt   = [0]*len(h_w)
            else:
                tt   = 0  # leaky time integrator
        
        # Update novelty counts / weights
        if ntype=='hN':   
            if update_type=='fixed':
                h_w, _ = update_hnov(h_w,kmat,s,k_alph)  
            elif update_type=='leaky':
                h_w, gg, tt = update_hnov(h_w,kmat,s,h_eps,tt,gg,h_alph_leak)
            elif update_type=='var':
                h_w, _ = update_hnov(h_w,kmat,s,h_eps,t)
        else:   
            if update_type=='fixed':
                p = update_nov(p,s,k_alph)
            elif update_type=='leaky':
                c, tt = update_nov(c,s,tt,c_alph_leak)
            elif update_type=='var':
                c = update_nov(c,s)       

        # Compute initial novelty
        if ntype=='hN':     
            Nvec, N = compute_hnov(h_w,kmat,k,w)
        else:  
            if update_type=='fixed':
                N = compute_nov(p)
            elif update_type=='leaky':
                N = compute_nov(c,tt,S,c_eps)
            elif update_type=='var':
                N = compute_nov(c,tt,S) 

        if rec_nov:
            nov_list.append(Nvec[s,:] if ntype=='hN' else N[s])  

        # Compute components of loglikelihood
        for i in range(len(data)):
            #print(f'{i}')

            # Compute Q-values and softmax distribution
            q = betaN*qN                                                       #q = (1-betaN)*qR + betaN*qN
            if len(qvals)>0: # optional comparison with reference Q-values
                qvals_notnan = (~np.isnan(q)).nonzero()
                if (not i==0) and not (qN[qvals_notnan]==qvals['qvals'].values[i-1][qvals_notnan]).all():
                    all_q_equal = False
            q_notnan = list((~np.isnan(q[s])).nonzero()[0])
            p_softmax = scipy.special.softmax(beta*q[s][q_notnan])   
            if beta==0 or np.isnan(p_softmax).any():
                print(f"{q[s][:]}, {beta}, {p_softmax}\n")

            # Compute loglikelihood of current action from observed data
            a   = int(data['action'].values[i])
            lpa = np.log(p_softmax[q_notnan.index(a)])

            if i>=start_ll and i<stop_ll:
                ll  += lpa 
                if rec_ll:
                    ll_list.append(lpa)  

            s_new_env, _    = env.evalAction(a,s)
            s_new           = data['next_state'].values[i]
            if s_new_env!=s_new:
                print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')
            #print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')

            # Update time counter
            t  = t + 1

            # Update novelty variables and recompute novelty
            if ntype=='hN':  
                # Update novelty weights 
                if update_type=='fixed':
                    h_w, _ = update_hnov(h_w,kmat,s_new,k_alph)   #s_new??
                elif update_type=='leaky':
                    h_w, gg, tt = update_hnov(h_w,kmat,s_new,h_eps,tt,gg,h_alph_leak)
                elif update_type=='var':
                    h_w, _ = update_hnov(h_w,kmat,s_new,h_eps,t)
                # Compute novelty
                Nvec, N = compute_hnov(h_w,kmat,k,w)
            else:   
                if update_type=='fixed':
                    p = update_nov(p,s_new,k_alph)
                    N = compute_nov(p)
                elif update_type=='leaky':
                    c, tt = update_nov(c,s_new,tt,c_alph_leak)
                    N = compute_nov(c,tt,S,c_eps)
                elif update_type=='var':
                    c = update_nov(c,s_new)    
                    N = compute_nov(c,tt,S)   
            
            if rec_nov:
                nov_list.append(Nvec[s_new,:] if ntype=='hN' else N[s_new])  

            # Run mbNoR update step
            alph[s][a][:] = k_leak*alph[s][a][:] + (1-k_leak)*eps
            alph[s][a][s_new]+=1
            theta = alph/np.expand_dims(np.sum(alph,axis=2),axis=2)
            qN, uN = nor.prioritized_sweeping(qN,uN,N,lamN,theta,Tps)
                
            # Update state for next iteration
            s = s_new
    except Exception as e:
        print(f"Error occurred: {e}. Returning infinite LL.")
        ll = -np.inf
        if rec_ll:
            ll_list.append(ll)
        all_q_equal = None
    
    if rec_ll and rec_nov:
        return ll, all_q_equal, ll_list, nov_list
    elif rec_ll:
        return ll, all_q_equal, ll_list
    elif rec_nov:
        return ll, all_q_equal, nov_list
    else: 
        return ll, all_q_equal