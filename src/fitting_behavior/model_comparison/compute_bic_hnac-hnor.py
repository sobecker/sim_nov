import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl
from src.fitting_behavior.model_comparison.compute_bic import compute_bic, plot_bic_bar

if __name__=="__main__":

    # Specify list of candidate models
    alg_types    = ['hnor_center-triangle','hnor_notrace','hnor_notrace_center-box',
                    'hnac-gn_center-triangle','hnac-gn_notrace','hnac-gn_notrace_center-box',                    
                    'hhybrid2_center-triangle','hhybrid2_notrace','hhybrid2_notrace_center-box'
                    ] #['hhybrid2','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hhybrid'] # 'hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    data_type    = 'mice' # 'mice','opt','naive'
    opt_method   = 'Nelder-Mead' # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types   = ['sep','app']
    measure_type = 'LL' # 'bic','LL'
    epss = [[50,100]]*len(alg_types)
    maxit        = [False]*len(alg_types) #True


    for i_alg in range(len(alg_types)):
        for i_comb in range(len(comb_types)):

            alg_type = alg_types[i_alg]
            comb_type = comb_types[i_comb]
            eps = epss[i_alg][i_comb]
            
            path_models = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/MLE_results/Fits/SingleRun/mle{"-maxit" if maxit[i_alg] else ""}_{alg_type}-{data_type}_{opt_method}/'
            levels      = [1,2,3,4,5,6]
            candidates  = [f'{alg_type}-l{i}' for i in levels]
            xtl         = [f'l{i}' for i in levels]

            path_save   = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
            sl.make_long_dir(path_save)
            path_save1  = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/ModelSelection/BIC/'
            sl.make_long_dir(path_save1)

            name        = f'{alg_type}'
            name_save   = f'{alg_type}'
            name_save1  = f'{alg_type}_which-alg'

            bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_save,name_save,[maxit[i_alg]]*len(candidates))
            plot_bic_bar(bic_df,comb_type,measure_type,name,name_save1,path_save1,eps=eps,xtl=xtl)

    print('done')
    
    