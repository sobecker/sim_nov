# -*- coding: utf-8 -*-

import numpy as np
import random 
from scipy.special import expit, softmax


class env():
    ''' Environment
    
        Represents the environment that the agent is interacting with
        
        Attributes:
            S([int])                    : States of the maze
            P([float])                  : Transition matrix 
            R([float])                  : Reward in each state
            agentLoc([int]()            : State that the agent is currently in
            
        Methods:
            initAgentLoc()              : Randomly initializes the agent position to a non-goal state
            getAgentLoc()               : Returns the current agent position in the environment
            setAgentLoc(state)          : Sets the agent position to the state given as input
            getReward(state)            : Returns the reward associated with the state given as input
            evalAction(action, state)   : Set agent to new location based on 
                                          current location and action, return 
                                          new state and associated reward 
                                         
    '''
    
    def __init__(self,S,P,R,T=np.array([]),t_deact=0):
        self.S = list(range(S)) 
        self.P = P
        self.R = R.copy() # states with active reward
        self.G = R.copy() # goal states = states with active/inactive rewards
        self.t_deact = t_deact
        if len(T)==0:
            self.T=self.R
        else:
            self.T=T
     
    # Initialize agent at random non-goal state (goal state is the last state)
    def initAgentLoc(self):
        x = random.randint(0,len(self.S)-2) 
        self.agentLoc = x

    def getGoal(self):
        return np.nonzero(self.G)[0]
        #return self.S[-1]

    def getTerminal(self):
        return np.nonzero(self.T)[0]
       
    def getTDeact(self):
        return self.t_deact

    def setAgentLoc(self,s):
        s = int(s)
        self.agentLoc = s

    def evalAction(self,a,s):
        s = int(s); a = int(a)
        self.setAgentLoc(self.P[s][a])   
        return self.agentLoc, self.R[self.agentLoc]

    def deactivateGoal(self,s):
        self.R[s]=0
        return None

    def activateGoal(self,s):
        self.R[s] = 1
        return None

class rate_neuron():
    ''' Rate neuron
    
        Models an individual rate neuron whose membrane potential h is given as  
        
            tau_m dh/dt = -h(t) + R I(t)                                (1)
            
        and whose instantaneous output firing rate v is determined via a gain function F(h).
        
    
        Attributes:
            h(float)            : Membrane potential at time t
            tauM(float)         : Membrane time constant
            RM(float)           : Input membrane resistance
            v(float)            : Instantaneous firing rate as given by F(h)
            
        Methods:
            initNeuronState()   : Initializes neuron potential, input and rate
            gain(h)             : Gain function that gives the output firing rate as a function of the membrane potential
            updatePotential()   : Updates the membrane potential based on Eq. 1
            computeRate()       : Computes the output firing rate using the gain function
            
    '''
    
    def __init__(self,tauM,RM): # For the functionality of the rate neuron we don't need the constants
        self.tauM   = tauM
        self.RM     = RM
        self.initNeuronState()
        
        
    def initNeuronState(self):
        self.h = 0  # Think about good random initialization - or does this make sense? What roles do critic and actor neurons play and which values can they take?
        self.v = self.computeRate()
    
    
    def updatePotential(self,I): # should the timestep dt be included?
        self.h = (1 - 1/self.tauM)*self.h
        self.h += self.RM*I
        #dh = - 1/self.tauM * (self.h + self.RM * I) * dt # dt is a global variable
        #self.h = self.h + dh
        
        
    def computeRate(self):
        # Simple sigmoid activation/gain function. Note that it is 0.5 for zero input! Maybe it would be smarter to use something that doesn't process inhibitory input
        
        return 2*(expit(self.h)-0.5)
        #return 2*(1/(1 + np.exp(-self.h))-0.5) 
        #return 1/(1 + np.exp(-h))
        

class linear_unit():
    
    def __init__(self):
        self.initNeuronState()
        
    def initNeuronState(self):
        self.h = 0
        self.v = self.computeRate()
        
    def updatePotential(self,I):
        self.h = I
        
    def computeRate(self):
        return self.h

class granular_critic():

    def __init__(self,S,kv,tauM,RM,w0,gamma,alph,lam,round_prec):
        # k is the matrix specifying the weight by which each state from the state space is represented in each kernel
        # format of k: 'number of states in state space' x 'number of kernels covering the state space'
        self.round_prec = round_prec
        self.max_w      = 1.0e+300
        
        # Neuron layers: x - inputs from state space, k - kernels, c - critic neuron
        self.x          = np.zeros(S)
        self.k          = np.zeros(len(kv[0,:]))
        self.c          = linear_unit() #=rate_neuron(tauM,RM)
        # Connections between layers: v - weights from x to k, w - weights from k to c, e - e-traces of w
        self.v          = kv
        self.w          = w0*np.ones(len(self.k))
        self.e          = np.zeros(len(self.k))
        # Learning parameters + TD error
        self.TD         = 0
        self.gamma      = gamma
        self.alph       = alph
        self.lam        = lam

    def giveStateInput(self,s):
        # Set input neurons
        for i in range(len(self.x)):
            if i==s:
                self.x[i] = 1
            else:
                self.x[i] = 0
        # Propagate activity from input to kernel neurons
        Ik = np.dot(self.x,self.v)
        self.k = Ik
        # for i in range(len(self.k)):
        #     self.k[i].updatePotential(Ik[i])

        # Propagate activity from kernel to critic neurons
        Ic = np.dot(self.k,self.w)        
        self.c.updatePotential(Ic)

    def updateTraces(self,s):
        self.e    = np.around(self.lam * self.e,self.round_prec)
        ie = np.nonzero(self.k)[0]
        for i in ie:
            self.e[i] = self.e[i]+self.k[i]*(1-self.e[i])

    def updateWeights(self,TD):
        # new_w = self.w + self.alph * TD * self.e
        # exceed = (np.abs(new_w)>self.max_w).nonzero()[0]
        # if len(exceed)>0:
        #     new_w[exceed]=np.sign(new_w[exceed])*self.max_w
        # self.w      = np.around(new_w,self.round_prec)
        self.w = np.around(self.w + self.alph * TD * self.e,self.round_prec)
    
    def computeTD(self,v,v_next,M):
        TD      = M + self.gamma*v_next - v
        self.TD = TD
        return round(TD,self.round_prec)
    
    def evalCritic(self,s_next,M): 
        v = self.c.computeRate()
        self.giveStateInput(s_next)
        v_next = self.c.computeRate()
        TD      = self.computeTD(v,v_next,M)
        self.TD = TD
        
        return TD

    def learnCritic(self,TD,s,s_next):
        self.updateTraces(s)
        self.updateWeights(TD)
        return self.w, self.e
    
        
class critic():
    ''' Critic
    
        Learns the value function of the states and computes the TD error
        
        Attributes:
            TD(float)               : current value of the TD error
            gamma(float)            : discount factor of firing rate in TD error computation
            c(rate_neuron)          : critic neuron, combines external rewards and signals from state neurons 
            x([boolean])            : vector with binary firing rate of the state neurons
                                        x[i] = TRUE     if agent is in state i represented by x[i]
                                        x[i] = FALSE    else
            w([float])              : weight vector of connections between state neurons and critic neuron
            alph                    : learning rate that is used in weight update of the critic connections
            e([float])              : eligibility traces of connections between state neurons and critic neuron
            lam(float)              : decaying time constant of the eligibility traces
            
        Methods:
            setStateInput(s)        : updates state neuron inputs according to the current agent position s
            updateTraces(s)         : updates the eligibility traces of the critic weights
            updateWeights(TD)       : updates the connection weights of the critic
            computeTD(v,v_next,R)   : computes the TD error
            evalCritic(s_next,M)    : evaluate the critic, i.e. update state input, compute state value as output of critic neuron and TD error
            
    '''
    
    def __init__(self,S,tauM,RM,w0,gamma,alph,lam,round_prec):
        self.round_prec = round_prec
        self.max_w      = 1.0e+300
        
        #self.c          = rate_neuron(tauM,RM)
        self.c          = linear_unit()
        self.x          = np.zeros(S)
        self.w          = w0*np.ones(S)
        self.e          = np.zeros(S)
        self.TD         = 0
        self.gamma      = gamma
        self.alph       = alph
        self.lam        = lam
        
    # Set rates of state input neurons
    # def setStateInput(self,s):
    #     for i in range(len(self.x)):
    #         if i==s:
    #             self.x[i] = 1
    #         else:
    #             self.x[i] = 0
                
                
    def giveStateInput(self,s):
        # Set input neurons
        for i in range(len(self.x)):
            if i==s:
                self.x[i] = 1
            else:
                self.x[i] = 0
                
        # Compute input from state neurons to critic neuron
        I = np.dot(self.x,self.w).item()
        
        # Update potential of the critic neuron and extract its firing rate
        self.c.updatePotential(I)
       
                
    # Update eligibility traces
    def updateTraces(self,s):
        self.e    = np.around(self.lam * self.e,self.round_prec)
        #self.e[s] += 1 
        self.e[s] = 1
        
        
    def updateWeights(self,TD): 
        # new_w = self.w + self.alph * TD * self.e
        # exceed = (np.abs(new_w)>self.max_w).nonzero()[0]
        # if len(exceed)>0:
        #     new_w[exceed]=np.sign(new_w[exceed])*self.max_w
        # self.w      = np.around(new_w,self.round_prec)
        self.w = np.around(self.w + self.alph * TD * self.e,self.round_prec)
    
    # Compute TD error 
    # How can we do this in a biologically realistic way? 
        # 1 inh-inh circuit / recurrent network that sends signal through two lines with different delays
        # 2 replace old rate v with moving average that represents the membrane potential
        # 3 buffer population
    # In general: VTA neurons as relay station, processing probably in a different brain area - ask and read up on experimental papers. 
    def computeTD(self,v,v_next,M):
        TD      = M + self.gamma*v_next - v
        self.TD = TD
        return round(TD,self.round_prec)
    

    # Evaluate the critic based on the new state and the associated reward
    def evalCritic(self,s_next,M): 
        v = self.c.computeRate()
        
        self.giveStateInput(s_next)
        
        #self.setStateInput(s_next)
        # I = np.dot(self.x,self.w).item()
        # self.c.updatePotential(I)
        
        v_next = self.c.computeRate()
        
        # Compute TD error
        TD      = self.computeTD(v,v_next,M)
        self.TD = TD
        
        return TD
     
        
    # Learn weights of the critic
    def learnCritic(self,TD,s,s_next):
        self.updateTraces(s)
        #self.updateTraces(s_next)
        self.updateWeights(TD)
        return self.w, self.e


class granular_n_critic(granular_critic):
        
    def __init__(self,S,tauM,RM,w0,gamma,alph,lam,k,ntype,k_alph,h,w,round_prec):
        super().__init__(S,h['kmat'][0],tauM,RM,w0,gamma,alph,lam,round_prec)
        self.pc = pc(S,k,ntype,k_alph,h,w)
        
    def evalMod(self,s_next,r):
        return self.pc.computeNovelty(s_next)
    
    def evalModAll(self):
        return self.pc.computeNoveltyAll()
    
    def updateMod(self,s_next):
        self.pc.updateNovelty(s_next)


class oi_critic(critic):
   
    def __init__(self,S,tauM,RM,w0,gamma,alph,lam,round_prec):
        super().__init__(S,tauM,RM,w0,gamma,alph,lam,round_prec)


    def evalMod(self,s_next,r):
        return 0, None, None, None
    
    def updateMod(self,s_next):
        return None


class r_critic(critic):
   
    def __init__(self,S,tauM,RM,w0,gamma,alph,lam,round_prec):
        super().__init__(S,tauM,RM,w0,gamma,alph,lam,round_prec)


    def evalMod(self,s_next,r):
        return r, []
    
    def updateMod(self,s_next):
        return None

    
class pc():
    ''' Perirhinal cortex (novelty-computing) unit'''
        
    def __init__(self,S,k,ntype,k_alph=1,h=[],w=[]):
        self.S      = S
        self.k      = k
        self.ntype  = ntype
        self.counts = np.zeros(self.S)
        self.t      = 0                 # for novelty based on t
        #self.t      = np.zeros(self.S) # for novelty based on dt

        if self.ntype=='N-ktemp':
            self.kint   = np.zeros(len(self.counts))
            if len(h)==0 or not 'n_buffer' in h.keys():
                self.n_buffer = 5
            else:
                self.n_buffer = h['n_buffer']
            self.buffer = np.zeros((len(self.counts),self.n_buffer))

        if self.ntype=='N-kmix':
            self.kint   = 0
            if len(h)==0 or not 'n_buffer' in h.keys():
                self.n_buffer = 5
            else:
                self.n_buffer = h['n_buffer']
            self.buffer = np.zeros(self.n_buffer)

        # Set weights (w), kernel hierarchy (h) and kernel hierarchy weights (wh)
        if self.ntype=='hN':
            # Set hnov type
            hnov_type = h['hnov_type'] if (len(h)>0 and'hnov_type' in h.keys()) else 2
            if hnov_type==2:
                self.compute_hnov = self.compute_hnov2
            elif hnov_type==3:
                self.compute_hnov = self.compute_hnov3

        if self.ntype=='hN-k':
            hnov_type = 3
            self.compute_hnov = self.compute_hnov3_kpop

        if 'hN' in self.ntype:
            # Set update type (fixed/variable learning rate for novelty signal)
            self.update_type = h['update_type'] if (len(h)>0 and 'update_type' in h.keys()) else 'var'
            if self.update_type=='fix':
                self.update_hnov = self.update_hnov_fixedrate
            elif self.update_type=='var':
                self.update_hnov = self.update_hnov_varrate

            self.w = w
            self.mp = h['mp'] # coordinate mapping (here: from states to action vectors)
            self.h_k = h['h_k'] # kernel functions
            self.h_ki = h['h_ki'] # kernel widths
            self.h_kc = h['h_kc'] # kernel centers
            self.h_w = h['h_w'] # kernel mixture weights
            self.kmat = h['kmat'] # kernel function matrix (list of matrices |S|xlen(av))
            self.eps = h['k_alph'] if self.update_type=='fix' else h['eps'] # prior or fixed learning rate for novelty
            if self.eps==None:
                self.eps = [1/(len(self.h_w[i])**2) for i in range(len(self.h_w))]   
            self.h_g = None
        else:
            self.w = None
            self.mp = None
            self.h_k = None
            self.h_ki = None
            self.h_kc = None
            self.h_w = None
            self.kmat = None
            self.eps = k_alph # leakiness of counts; 1=not leaky (default)
            self.h_g = None

    def compute_hnov2(self):
        nov_vec = np.zeros((np.size(self.kmat[-1],axis=0),len(self.kmat)))
        nov = np.zeros(np.size(self.kmat[-1],axis=0))
        for i in range(len(self.h_w)):       # for each level i in the hierarchy
            nov_vec[:,i] = -np.log(np.sum(self.kmat[i]*self.h_w[i],axis=1))-self.k 
            nov += self.w[i]*nov_vec[:,i]      # summed novelty 
        hnov = nov_vec
        #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(np.sum(self.kmat[i]*self.h_w[i],axis=1)),4) for i in range(len(self.h_w))]}.\n")
        return hnov, nov

    def compute_hnov3(self):
        nov_vec = np.zeros((np.size(self.kmat[-1],axis=0),len(self.kmat)))
        nov = np.zeros(np.size(self.kmat[-1],axis=0))
        for i in range(len(self.h_w)):           # for each level i in the hierarchy
            nov_vec[:,i] = np.sum(self.kmat[i]*self.h_w[i],axis=1)  
            nov += self.w[i]*nov_vec[:,i]      # summed novelty (before log)
        hnov = nov_vec
        nov = -np.log(nov)-self.k
        #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(nov_vec[:,i]),4) for i in range(len(self.h_w))]}.\n")
        return hnov, nov

    def compute_hnov3_kpop(self):
        nov_vec = np.zeros((np.size(self.kmat[-1],axis=0),len(self.kmat)))
        nov = np.zeros(np.size(self.kmat[-1],axis=0))
        for i in range(len(self.h_w)):           # for each level i in the hierarchy
            nov_vec[:,i] = np.sum(self.kmat[i]*self.h_w[i],axis=1)  
            nov += self.w[i]*nov_vec[:,i]      # summed novelty (before log)
        hnov = nov_vec
        nov = -np.log(nov)
        nov = nov-np.mean(nov)
        #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(nov_vec[:,i]),4) for i in range(len(self.h_w))]}.\n")
        return hnov, nov

    def computeNovelty(self,s):
        hnov = None
        if self.ntype=='N':
            nov = -np.log((self.counts[s]+1)/(self.t+self.S))
        elif self.ntype=='-1/N':
            nov = 1/np.log((self.counts[s]+1)/(self.t+self.S))
        elif self.ntype=='N-k': 
            nov = -np.log((self.counts[s]+1)/(self.t+self.S))-self.k
        elif self.ntype=='N-ktemp':
            nov = -np.log((self.counts[s]+1)/(self.t+self.S))-self.kint[s]
        elif self.ntype=='N-kpop':
            nov_vec = -np.log((self.counts+1)/(self.t+self.S))
            nov = nov_vec[s]-np.mean(nov_vec)
        elif self.ntype=='N-kmix':
            nov = -np.log((self.counts[s]+1)/(self.t+self.S))-self.kint
        elif 'hN' in self.ntype:
            hnov_vec, nov_vec = self.compute_hnov()
            nov = nov_vec[s]
            hnov = hnov_vec[s,:]
            #h_w_temp = self.h_w.copy() # gamma_old vector
            #h_w_temp,h_g_temp = self.update_h_w(h_w_temp,s) # gamma_new vector: this should include updating the responsibilities
            # nov_vec = np.zeros(len(self.h_w)) 
            # nov = 0
            # for i in range(len(self.h_w)): # for each level i in the hierarchy
            #     if self.kmat:
            #         p_i_vec = self.h_w[i]*self.kmat[i]
            #     elif self.h_k:
            #         p_i_vec = np.zeros(len(self.h_w[i])) 
            #         for j in range(len(self.h_w[i])): # for each kernel j at hierarchy level i
            #             # Compute new novelty for state s
            #             p_i_vec[j] = (self.h_w[i][j]*self.h_k[i](s,self.h_kc[i][j],self.h_ki[i],self.mp)) 
            #     p_i = np.sum(p_i_vec) # empirical frequency for kernel i
            #     # nov_vec[i] = -np.log((p_i+1)/(self.t+len(self.h_w[i])))-self.k # this is wrong! would be ok if p_i were the counts
            #     nov_vec[i] = -np.log((p_i+1))-self.k
            #     nov += self.w[i]*nov_vec[i] # summed novelty 
            # hnov = nov_vec    
        else:
            nov = 0  
        # -1/N using dt (abandoned)
        #nov = 1/np.log((self.counts[s]+1)/(self.t[s]+self.S))
        # N-k using dt (abandoned)
        #nov = -np.log((self.counts[s]+1)/(self.t[s]+self.S))-self.k
        
        return nov, hnov, self.h_w, self.h_g
    
    def computeNoveltyAll(self):
        hnov = None
        if self.ntype=='N':
            nov = -np.log((self.counts+1)/(self.t+self.S))
        elif self.ntype=='-1/N':
            nov = 1/np.log((self.counts+1)/(self.t+self.S))
        elif self.ntype=='N-k': 
            nov = -np.log((self.counts+1)/(self.t+self.S))-self.k
        elif self.ntype=='N-ktemp':
            nov = -np.log((self.counts+1)/(self.t+self.S))-self.kint[s]
        elif self.ntype=='N-kpop':
            nov_vec = -np.log((self.counts+1)/(self.t+self.S))
            nov = nov_vec-np.mean(nov_vec)
        elif self.ntype=='N-kmix':
            nov = -np.log((self.counts+1)/(self.t+self.S))-self.kint
        elif 'hN' in self.ntype:
            hnov_vec, nov_vec = self.compute_hnov()
            nov = nov_vec
            hnov = hnov_vec
            #h_w_temp = self.h_w.copy() # gamma_old vector
            #h_w_temp,h_g_temp = self.update_h_w(h_w_temp,s) # gamma_new vector: this should include updating the responsibilities
            # nov_vec = np.zeros(len(self.h_w)) 
            # nov = 0
            # for i in range(len(self.h_w)): # for each level i in the hierarchy
            #     if self.kmat:
            #         p_i_vec = self.h_w[i]*self.kmat[i]
            #     elif self.h_k:
            #         p_i_vec = np.zeros(len(self.h_w[i])) 
            #         for j in range(len(self.h_w[i])): # for each kernel j at hierarchy level i
            #             # Compute new novelty for state s
            #             p_i_vec[j] = (self.h_w[i][j]*self.h_k[i](s,self.h_kc[i][j],self.h_ki[i],self.mp)) 
            #     p_i = np.sum(p_i_vec) # empirical frequency for kernel i
            #     # nov_vec[i] = -np.log((p_i+1)/(self.t+len(self.h_w[i])))-self.k # this is wrong! would be ok if p_i were the counts
            #     nov_vec[i] = -np.log((p_i+1))-self.k
            #     nov += self.w[i]*nov_vec[i] # summed novelty 
            # hnov = nov_vec    
        else:
            nov = 0  
        # -1/N using dt (abandoned)
        #nov = 1/np.log((self.counts[s]+1)/(self.t[s]+self.S))
        # N-k using dt (abandoned)
        #nov = -np.log((self.counts[s]+1)/(self.t[s]+self.S))-self.k
        
        return nov, hnov, self.h_w, self.h_g
    
    def update_hnov_varrate(self,h_w,s):
        h_w_new = []
        gamma_new = []
        for i in range(len(h_w)):
            # Update the responsibilities
            gamma_i_nom = h_w[i]*self.kmat[i][s,:]
            gamma_i_denom = np.sum(gamma_i_nom)
            gamma_i = gamma_i_nom/gamma_i_denom
            gamma_new.append(gamma_i.copy())
            
            # Update weights (incremental update rule with prior)
            h_w_i = h_w[i] + 1/(self.t+len(h_w[i])*self.eps[i])*(gamma_i-h_w[i])
            #h_w_i = h_w[i] + 1/(self.t+len(self.kmat[0])*self.eps[i])*(gamma_i-h_w[i])
            h_w_new.append(h_w_i)
        #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
        return h_w_new, gamma_new

    def update_hnov_fixedrate(self,h_w,s):
        h_w_new = []
        gamma_new = []
        for i in range(len(h_w)):
            # Update the responsibilities
            gamma_i_nom = h_w[i]*self.kmat[i][s,:]
            gamma_i_denom = np.sum(gamma_i_nom)
            gamma_i = gamma_i_nom/gamma_i_denom
            gamma_new.append(gamma_i.copy())
            
            # Update weights (incremental update rule with prior)
            h_w_i = h_w[i] + self.eps[i]*(gamma_i-h_w[i]) 
            h_w_new.append(h_w_i)
        #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
        return h_w_new, gamma_new

    def updateNovelty(self,s):
        # Update for novelty based on t
        if isinstance(self.eps,(list,np.ndarray)):
            self.t = self.eps[0]*self.t + 1
            self.counts *= self.eps[0]
        else:
            self.t = self.eps*self.t + 1
            self.counts *= self.eps
        self.counts[s] +=1

        if self.ntype=='N-ktemp':
            self.buffer[s,0:-1] = self.buffer[s,1:]
            self.buffer[s,-1] = -np.log((self.counts[s]+1)/(self.t+self.S))
            self.kint[s] = np.mean(self.buffer[s,:])
            #self.buffer[:,-1] = -np.log((self.counts+1)/(self.t+self.S))
            #self.kint = np.mean(self.buffer,axis=1)

        if self.ntype=='N-kmix':
            self.buffer[0:-1] = self.buffer[1:]
            self.buffer[-1] = -np.log((self.counts[s]+1)/(self.t+self.S))
            self.kint = np.mean(self.buffer)
        
        if 'hN' in self.ntype:
            self.h_w,self.h_g = self.update_hnov(self.h_w,s)
        # Update for novelty based on dt
        #self.t          += 1
        #self.counts[s]  += 1
        #self.t[s]        = 0 

class n_critic(critic):
        
    def __init__(self,S,tauM,RM,w0,gamma,alph,lam,k,ntype,k_alph,h,w,round_prec):
        super().__init__(S,tauM,RM,w0,gamma,alph,lam,round_prec)
        self.pc = pc(S,k,ntype,k_alph,h,w)
        
    def evalMod(self,s_next,r):
        return self.pc.computeNovelty(s_next)
    
    def evalModAll(self):
        return self.pc.computeNoveltyAll()
    
    def updateMod(self,s_next):
        self.pc.updateNovelty(s_next)
  
    
class actor():
    ''' Actor

        Decides on actions to be taken by the agent based on the current state

        Attributes:
            x([boolean])        : vector with binary firing rate of the state neurons
                                    x[i] = TRUE     if agent is in state i represented by x[i]
                                    x[i] = FALSE    else
            a([rate_neuron])    : vector of actor neurons, produce firing rate based on state neuron input  
            w([float])          : weight matrix of connections between state and actor neurons
            e([float])          : matrix of eligibility traces for actor connections
            alph                : actor learning rate
            temp                : temperature of the softmax action policy
            lam                 : decay factor of the actor eligibility traces
            
            
        Methods:
            setStateInput(s)        : adjust input neurons x to encode current state s
            updateTraces(s,action)  : updates the eligibility traces of the actor
            updateWeights()         : updates the connection weights of the actor
            greedyAction()          : implements greedy action policy
            eps_greedyAction(eps)   : implements epsilon-greedy action policy
            softmaxAction()         : implements softmax action policy
            evalActor(s)            : evaluate actor in current state, i.e. update state neurons and choose action based on action policy
            learnActor(TD,s,a)      : update eligibility traces and weights based on the current state-action pair and the TD error
            
    '''
    
    def __init__(self,S,a,tauM,RM,w0,alph,lam,temp,round_prec,P=None):
        self.round_prec = round_prec
        self.max_w      = 1.0e+300

        self.alph       = alph
        self.temp       = temp
        self.lam        = lam
        
        self.a          = []
        for i in range(a):
            #self.a.append(rate_neuron(tauM,RM)) 
            self.a.append(linear_unit()) 

        self.x = np.array(range(S)).reshape((S,1))
        #self.x = self.x.reshape((len(self.x),1))
        
        self.w = w0*np.ones((S,a))
        self.e = np.zeros((S,a))
        if P!=None and np.shape(P)==(S,a):
            self.w[np.isnan(np.array(P)).nonzero()] = np.NaN 
            self.e[np.isnan(np.array(P)).nonzero()] = np.NaN
              
    # Set rates of state input neurons
    # def setStateInput(self,s):
    #     for i in range(len(self.x)):
    #         if i==s:
    #             self.x[i] = 1
    #         else:
    #             self.x[i] = 0
                
    
    def giveStateInput(self,s):
        # Set input neurons
        for i in range(len(self.x)):
            if i==s:
                self.x[i] = 1
            else:
                self.x[i] = 0
                
        # Compute input from state neurons to actor neurons
        #I = np.dot(self.x,self.w)
        I = np.nansum(np.multiply(self.x,self.w),axis=0)
        
        # Update potential of the actor neurons
        for i in range(len(self.a)):
            self.a[i].updatePotential(I[i])
    
    
    # Update eligibility traces
    def updateTraces(self,s,action):
        self.e              = np.around(self.lam * self.e,self.round_prec) 
        #self.e[s,action]    += 1 #*(self.a[action] - self.w[s,action]) This only works for non-bounded e-traces
        self.e[s,action]    = 1
        
        
    # Update weights         
    def updateWeights(self,TD):
        # new_w = self.w + self.alph * TD * self.e
        # exceed = (np.abs(new_w)>self.max_w).nonzero()[0]
        # if len(exceed)>0:
        #     new_w[exceed]=np.sign(new_w[exceed])*self.max_w
        # self.w      = np.around(new_w,self.round_prec)
        self.w = np.around(self.w + self.alph * TD * self.e,self.round_prec)

    
    def greedyAction(self):
        out_rates = [i.computeRate() for i in self.a]        
        action = out_rates.index(max(out_rates))        # What is the typical approach when all are equal? Take the first one/a random action?
        return action
   
    
    def eps_greedyAction(self,eps):
        y = np.random.rand()
        if y<eps:
            action = np.random.randint(0,len(self.a))
        else:
            action = self.greedyAction()
        return action
          
      
    def softmaxAction(self,s):
        out_rates = np.array([self.a[i].computeRate() if ~np.isnan(self.w[s][i]) else np.NaN for i in range(len(self.a))])
        not_nan = (~np.isnan(out_rates)).nonzero()
        #p_softmax = np.exp(out_rates/self.temp) / np.sum(np.exp(out_rates/self.temp), axis=0)
        p_softmax = softmax(out_rates[list(not_nan[0])]/self.temp,axis=0)
        #print(p_softmax)
        if self.temp==0 or np.isnan(p_softmax).any():
            print(f"{out_rates}, {self.temp}, {p_softmax}\n")
        action = np.random.choice(np.arange(len(self.a))[list(not_nan[0])], p=p_softmax)
        #print(action)
        return action
        
    
    def evalActor(self,s):
        # # Adjust rates of state neurons to represent the new state of the agent
        # self.setStateInput(s)
        
        # # Compute input from state neurons to actor neurons
        # I = np.dot(self.x,self.w)
        
        # # Update potential of the critic neuron and extract its firing rate
        # for i in range(len(self.a)):
        #     self.a[i].updatePotential(I[i])
        
        self.giveStateInput(s)
        
        # Decide on action
        #action = self.greedyAction()
        action = self.softmaxAction(s)
        
        return action
          
    
    def learnActor(self,TD,s,a):
        self.updateTraces(s,a)
        self.updateWeights(TD)
        return self.w, self.e
    
      
    
class agent():
    ''' Agent
    
        Implements TD learning using an actor-critic network
        
        Attributes:
            critic(critic)      : a critic module as defined above
            actor(actor)        : an actor module as defined above
           
        Methods:
            act(s)              : take an action in current state s (using the actor module)
            learn(s,s_next,a,m) : compute TD error and use it to update critic and actor 
            
    '''
    
    def __init__(self,params):
        self.round_prec         = params['round_prec']
        self.types              = list(params['agent_types'])
        self.decision_weights   = params['decision_weights']
        self.critics            = []
        self.actors             = []
        
        for i,t in enumerate(self.types):
            if 'n' in t:
                if not ('h' in params.keys()):
                    params['h']=[]
                if not ('w' in params.keys()):
                    params['w']=[]
                if not 'k_alph' in params.keys():
                    params['k_alph'] = 1
                if len(params['h'])==0 or not 'kmat' in params['h'].keys():
                    t='n'

                if t=='n':
                    self.critics.append(n_critic(params['S'],params['tauM'][i],params['RM'][i],
                                             params['c_w0'][i],params['gamma'][i],params['c_alph'][i],
                                             params['c_lam'][i],params['k'],params['ntype'],params['k_alph'],params['h'],params['w'],
                                             self.round_prec))
                elif t=='gn':
                    self.critics.append(granular_n_critic(params['S'],params['tauM'][i],params['RM'][i],
                                             params['c_w0'][i],params['gamma'][i],params['c_alph'][i],
                                             params['c_lam'][i],params['k'],params['ntype'],params['k_alph'],params['h'],params['w'],
                                             self.round_prec))

            elif t=='r':
                self.critics.append(r_critic(params['S'],params['tauM'][i],params['RM'][i],
                                             params['c_w0'][i],params['gamma'][i],params['c_alph'][i],
                                             params['c_lam'][i],self.round_prec))

            elif t=='oi': 
                self.critics.append(oi_critic(params['S'],params['tauM'][i],params['RM'][i],
                                             params['c_w0'][i],params['gamma'][i],params['c_alph'][i],
                                             params['c_lam'][i],self.round_prec))                
                
            self.actors.append(actor(params['S'],params['a'],params['tauM'][i],params['RM'][i],
                                     params['a_w0'][i],params['a_alph'][i],params['a_lam'][i],
                                     params['temp'][i],self.round_prec,P=params['P']))        
    
    # def setStateInput(self,s):
    #     for i in range(len(self.types)):
    #         self.critics[i].setStateInput(s)
    #         self.actors[i].setStateInput(s)
    
    def giveStateInput(self,s):
        for i in range(len(self.types)):
            self.critics[i].giveStateInput(s)
            self.actors[i].giveStateInput(s)
        
    def act(self,s,epi):
        
        actions = [self.actors[i].evalActor(s) for i in range(len(self.actors))]
        
        # actor_ind = np.random.choice(np.arange(len(self.actors)),p=self.decision_weights[epi])
        # action = actions[actor_ind]
        action = actions[0]

        return action
    
    
    def evalMod(self,s_next,r):
        
        m_list = []
        mh_list = []
        mw_list = []
        mg_list = []
        for i in range(len(self.critics)):
            m, mh, mw, mg = self.critics[i].evalMod(s_next,r)
            m = round(m,self.round_prec)
            m_list.append(m)
            if mh:
                mh = [round(mh[j],self.round_prec) for j in range(len(mh))]
                mh_list.extend(mh)
            if mw:
                mw_list.extend(mw)
            if mg:
                mg_list.extend(mg)   
        #m = [round(self.critics[i].evalMod(s_next,r),self.round_prec) for i in range(len(self.critics))]
        
        return m_list, mh_list, mw_list, mg_list
    
    def evalModAll(self):
        
        m_list = []
        for i in range(len(self.critics)):
            m,_,_,_ = self.critics[i].evalModAll()
            m = np.round(m,self.round_prec)
            m_list.append(m)
        
        return m_list
    
    def updateMod(self,s_next):
        
        for i in range(len(self.critics)):
            self.critics[i].updateMod(s_next)
    
    def resetTraces(self):
        
        for i in range(len(self.types)):
            self.critics[i].e[:]    = 0
            self.actors[i].e[:]     = 0
    
    
    def learn(self,s,s_next,a,m):
        
        TDs = []
        new_w_critics = []
        new_e_critics = []
        new_w_actors = []
        new_e_actors = []
        
        for i in range(len(self.types)):
            
            TDs.append(self.critics[i].evalCritic(s_next,m[i]))
            
            new_w_c, new_e_c = self.critics[i].learnCritic(TDs[i],s,s_next)
            new_w_critics.extend(new_w_c)
            new_e_critics.extend(new_e_c)
            
            new_w_a, new_e_a = self.actors[i].learnActor(TDs[i],s,a)
            new_w_actors.extend(new_w_a.flatten())  # default stype of flatten(): 'C', i.e. row-major order
            new_e_actors.extend(new_e_a.flatten())
        
        #new_w_e_critics  = [self.critics[i].learnCritic(TDs[i],s,s_next) for i in range(len(self.critics))]
        #new_w_critics = new_w_e_critics[::2]
        #new_e_critics = new_w_e_critics[1::2]
        
        #new_w_e_actors  = [self.actors[i].learnActor(TD[i],s,a) for i in range(len(self.actors))]
        #new_w_actors = new_w_e_actors[::2]
        #new_e_actors = new_w_e_actors[1::2]
        
        return TDs, new_w_critics, new_e_critics, new_w_actors, new_e_actors
