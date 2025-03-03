import numpy as np
import matplotlib.pyplot as plt
import utils.saveload as sl
import utils.visualization as vis

from fitting_behavior.model_comparison.plot_model_comparison import plot_scenario_schwartz_approx, plot_schwartz_approx_per_level
from fitting_behavior.model_comparison.bicmatrix_model_comparison import plot_bic_matrix
from fitting_behavior.recovery.level_recovery_model_comparison import plot_modelrecov_levels

if __name__=="__main__":

    plot_bw         = True      # Plot best and worst case BIC/LL (across similarity-based models of a given type)      - Fig. 3 E
    plot_bm         = False     # Plot BIC matrix (across algorithms and levels)                                        - Supplementary Fig. S3 A
    plot_cl         = True      # Plot BIC/LL for models of one RL type                                                 - Fig. 3 F         
    plot_cl_nor     = False     # Plot BIC/LL for models of one RL type (MB)
    plot_cl_hyb     = False     # Plot BIC/LL for models of one RL type (Hybrid)
    plot_mr_nor     = False     # Plot model recovery for models of one RL type (MB)                                    - Supplementary Fig. S3 B1
    plot_mr_hyb     = False     # Plot model recovery for models of one RL type (Hybrid)                                - Supplementary Fig. S3 B2
    
    plt.style.use(sl.get_rootpath() / 'src' / 'scripts' / 'paper.mplstyle')

    opt_method      = 'Nelder-Mead'    
    comb_type       = 'app'
    data_type       = 'mice' 
    
    save_plot       = True
    path_save       = sl.get_rootpath() / 'output' / f'figures_paper' / 'fig_model_comparison'
    sl.make_long_dir(path_save)

    path_load       = sl.get_rootpath() / 'data' / 'model_selection' / 'bic'

    ###################### Plot Fig. 3 A (best and worst case BIC/LL) #############################################
    if plot_bw:
        algs_basic      = ['nac','nor'] #,'hybrid']
        alg_types       = ['hnac-gn','hnor','hhybrid2'] #['hnac-gn-goi','hnor','hhybrid2']
        alg_labels      = ['c-MF','c-MB','s-MF*','s-MB*','s-Hyb*'] #['c-MF','c-MB','c-Hyb','s-MF*','s-MB*','s-Hyb*']
        ylim            = [-6800,-6500]
        alg_pattern     = ['']*len(algs_basic) + ['']*len(alg_types) #['//']*len(alg_types)
        
        cmap    = vis.prep_cmap_discrete('tab20c')
        col_bic = cmap[16]
        col_LL  = cmap[18]

        figshape        = (3,2.5)
        plot_scenario_schwartz_approx('best',algs_basic,alg_types,alg_labels,comb_type,path_load,path_save,save_plot=save_plot,save_name='best-schwartz',figshape=figshape,bar_pattern=alg_pattern,ax=None,ylim=ylim,plot_legend=True,col_bic=col_bic,col_LL=col_LL)

        # plot_best_worst(algs_basic,alg_types,alg_labels,opt_method,comb_type,path_load,path_save,save_plot=save_plot,save_name='best-worst',figshape=figshape_bw)

    ###################### Plot Fig. 3 B (BIC matrix: levels/algorithms) ##########################################
    if plot_bm:
        measure_type = 'bic' 
        title        = 'Model selection across algorithms and levels'
        figshape     = (3,2.2) #(6,3)
        
        save_name    = 'bic-matrix'
        alg_types    = ['hnac-gn','hnor','hhybrid2'] #'hnac-gn-goi',
        alg_labels   = ['MF','MB','Hybrid'] #'MF (adapt.)',
        plot_bic_matrix(comb_type,measure_type,alg_types,alg_labels,figshape,path_load,title,save_plot,path_save,save_name)

        save_name    = 'bic-matrix-1mf'
        alg_types    = ['hnac-gn','hnor','hhybrid2'] #'hnac-gn-goi'
        alg_labels   = ['MF','MB','Hyb']
        plot_bic_matrix(comb_type,measure_type,alg_types,alg_labels,figshape,path_load,title,save_plot,path_save,save_name)

    ###################### Plot Fig. 3 C-D (model comparison: levels) ###############################################
    if plot_cl_nor:
        figshape    = (2.5,2.2)
        alg_details = 'hhybrid2'
        title       = 'Hybrid'
        # plot_bic_bar(path_bicdata,alg_details,comb_type,path_save,save_name='bic-per-level',title=title,figshape=figshape)
        plot_schwartz_approx_per_level(path_load,alg_details,comb_type,path_save,save_plot=True,save_name='schwartz-per-level',title=title,figshape=figshape,ax=None)
    
    if plot_cl_hyb:
        figshape    = (2.5,2.2)
        alg_details = 'hnor'
        title       = 'MB'
        # plot_bic_bar(path_bicdata,alg_details,comb_type,path_save,save_name='bic-per-level',title=title,figshape=figshape)
        plot_schwartz_approx_per_level(path_load,alg_details,comb_type,path_save,save_plot=True,save_name='schwartz-per-level',title=title,figshape=figshape,ax=None)
    
    if plot_cl: 
        figshape    = (5,2.5)
        alg_details = ['hhybrid2','hnor']
        title       = ['s-Hyb','s-MB']
        save_name   = 'schwartz-per-level'
        ylim        = [-6800,-6500]
        f,ax = plt.subplots(1,2,figsize=figshape)

        cmap    = vis.prep_cmap_discrete('tab20c')
        col_bic = [cmap[16]]*6
        col_LL  = [cmap[18]]*6

        plot_schwartz_approx_per_level(path_load,alg_details[0],comb_type,path_save,save_plot=True,save_name=save_name,title=title[0],figshape=figshape,ax=ax[0],ylim=ylim,col_bic=col_bic,col_LL=col_LL)
        plot_schwartz_approx_per_level(path_load,alg_details[1],comb_type,path_save,save_plot=True,save_name=save_name,title=title[1],figshape=figshape,ax=ax[1],ylim=ylim,col_bic=col_bic,col_LL=col_LL)

        ax[1].set_yticks([])
        ax[1].set_ylabel('')
        f.tight_layout()
        save_name1 = f'{save_name}_{alg_details[0]}_{alg_details[1]}_{comb_type}'
        plt.savefig(path_save / f'{save_name1}.svg',bbox_inches='tight')
        plt.savefig(path_save / f'{save_name1}.eps',bbox_inches='tight')
    
    ###################### Plot Fig. 3 E-F (model recovery: levels) #################################################
    if plot_mr_nor:

        alg_type        = 'hnor'
        sims            = list(np.arange(30))
        title           = 'MB' #f'Recovery (MB)'
        uniparam        = True
        data_gen        = ['l4','l5','l6']
        candidates      = ['l4','l5','l6']
        measure_type    = 'bic' # 'bic','LL'
        figshape        = (3,2.2)

        f_comp          = False
        name_save       = f'modelrecov{"-uniparam" if uniparam else ""}-{alg_type}'
        plot_modelrecov_levels(alg_type,measure_type,comb_type,opt_method,data_gen,candidates,sims,f_comp,save_plot,path_save,name_save,figshape=figshape,uniparam=uniparam,title=title,show_n=False)

    if plot_mr_hyb:
        alg_type        = 'hhybrid2'
        sims            = list(np.arange(30))
        title           = 'Hybrid' #f'Recovery (Hybrid)'
        uniparam        = True
        data_gen        = ['l4','l5','l6']
        candidates      = ['l4','l5','l6']
        measure_type    = 'bic' # 'bic','LL'
        figshape        = (3,2.2)
        
        f_comp          = False
        name_save       = f'modelrecov{"-uniparam" if uniparam else ""}-{alg_type}'
        plot_modelrecov_levels(alg_type,measure_type,comb_type,opt_method,data_gen,candidates,sims,f_comp,save_plot,path_save,name_save,figshape=figshape,uniparam=uniparam,title=title,show_n=False)

    print('done')