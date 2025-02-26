from skopt.space import Real, Integer

# Parameters to be optimized (nAC)
dim1 = Real(name='gamma', low=0, high=0.999)      # discount of future rewards: should not be equal to 1
dim2 = Real(name='c_alph', low=0.001, high=0.5)     # critic learning rate: should not be larger than 0.5
dim3 = Real(name='a_alph', low=0.001, high=0.5)     # actor learning rate: should not be larger than 0.5
dim4 = Real(name='c_lam', low=0, high=0.999)            # decay time constant of critic e-traces
dim5 = Real(name='a_lam', low=0, high=0.999)            # decay time constant of actor e-traces
dim6 = Real(name='temp', low=0.001, high=1)       # softmax temperature: should not be equal to 0
dim7 = Real(name='c_w0', low=-100, high=100)        # initialization of critic weights: potentially increase range!
dim8 = Real(name='a_w0', low=-100, high=100)        # initialization of actor weights: potentially increase range!  
dim9 = Real(name='k', low=-10.0, high=10.0)         # factor k that can replace critic OI weight
dim10 = Integer(name='hT', low=1, high=100)          # time horizon for N-k heuristics

dim_noOI        = [dim1, dim2, dim3, dim4, dim5, dim6]
dim_OI          = [dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8]
dim_K           = [dim1, dim2, dim3, dim4, dim5, dim6,       dim8, dim9]
dim_Nkpop       = [dim1, dim2, dim3, dim4, dim5, dim6,       dim8]
dim_Nkpop_T     = [dim1, dim2, dim3, dim4, dim5, dim6,       dim8,       dim10]

dim_nac_OI      = [dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8]
dim_nac_noOI    = [dim1, dim2, dim3, dim4, dim5, dim6,       dim8]

# Parameters to be optimized (NoR)
dim1_nor = Real(name='epsilon', low=0.00001, high=0.1)
dim2_nor = Real(name='lambda_R', low=0, high=0.999)       
dim3_nor = Real(name='lambda_N', low=0, high=0.999)    # lambda_N cannot be 1 since we divide by 1-lambda_N to compute N0  
dim4_nor = Real(name='beta_1', low=0.1, high=30)
dim5_nor = Real(name='beta_N1', low=0.1, high=30)
dim6_nor = Real(name='beta_1r',low=0.1, high=0.9)   # shouldn't take value 0 or 1 (since this would correspond to 0 vs. infinite temperature)
    
dim_full        = [dim1_nor, dim2_nor, dim3_nor, dim6_nor, dim5_nor]
dim_nov         = [dim1_nor,           dim3_nor, dim6_nor]
dim_nov_noeps   = [                    dim3_nor, dim6_nor]