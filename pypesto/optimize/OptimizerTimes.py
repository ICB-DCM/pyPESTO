# compare optimization times of all considered optimizers

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Times_Rosenbrock():
    # open times tsv file
    rosenbrock_df = pd.read_csv('./Rosenbrock_times_convstarts.tsv', sep='\t')
    pyswarms = rosenbrock_df['PySwarms_time']
    pyswarms_convstarts = rosenbrock_df['PySwarms_conv_starts']
    scipydiffevolopt = rosenbrock_df['Scipy_DiffEvol_time']
    scipydiffevolopt_convstarts = rosenbrock_df['Scipy_DiffEvol_conv_starts']
    cmaes = rosenbrock_df['CMA-ES_time']
    cmaes_convstarts = rosenbrock_df['CMA-ES_conv_starts']
    pyswarm = rosenbrock_df['PySwarm_time']
    pyswarm_convstarts = rosenbrock_df['PySwarm_conv_starts']
    Times(pyswarms, scipydiffevolopt, cmaes, pyswarm, len(pyswarms), 'Rosenbrock')
    ConvergedStarts_by_Times(pyswarms, scipydiffevolopt, cmaes, pyswarm,
                             pyswarms_convstarts, scipydiffevolopt_convstarts,
                             cmaes_convstarts, pyswarm_convstarts,
                             len(pyswarms), 'Rosenbrock')


def Times_Boehm():
    # open times tsv file
    boehm_df = pd.read_csv('./Boehm_times_convstarts.tsv', sep='\t')
    pyswarms = boehm_df['PySwarms_time']
    pyswarms_convstarts = boehm_df['PySwarms_conv_starts']
    scipydiffevolopt = boehm_df['Scipy_DiffEvol_time']
    scipydiffevolopt_convstarts = boehm_df['Scipy_DiffEvol_conv_starts']
    cmaes = boehm_df['CMA-ES_time']
    cmaes_convstarts = boehm_df['CMA-ES_conv_starts']
    pyswarm = boehm_df['PySwarm_time']
    pyswarm_convstarts = boehm_df['PySwarm_conv_starts']
    Times(pyswarms, scipydiffevolopt, cmaes, pyswarm, len(pyswarms), 'Boehm')
    ConvergedStarts_by_Times(pyswarms, scipydiffevolopt, cmaes, pyswarm,
                             pyswarms_convstarts, scipydiffevolopt_convstarts,
                             cmaes_convstarts, pyswarm_convstarts,
                             len(pyswarms), 'Boehm')

def Times_Crauste():
    # open times tsv file
    crauste_df = pd.read_csv('./Crauste_times_convstarts.tsv', sep='\t')
    pyswarms = crauste_df['PySwarms_time']
    pyswarms_convstarts = crauste_df['PySwarms_conv_starts']
    scipydiffevolopt = crauste_df['Scipy_DiffEvol_time']
    scipydiffevolopt_convstarts = crauste_df['Scipy_DiffEvol_conv_starts']
    cmaes = crauste_df['CMA-ES_time']
    cmaes_convstarts = crauste_df['CMA-ES_conv_starts']
    pyswarm = crauste_df['PySwarm_time']
    pyswarm_convstarts = crauste_df['PySwarm_conv_starts']
    Times(pyswarms, scipydiffevolopt, cmaes, pyswarm, len(pyswarms), 'Crauste')
    ConvergedStarts_by_Times(pyswarms, scipydiffevolopt, cmaes, pyswarm,
                             pyswarms_convstarts, scipydiffevolopt_convstarts,
                             cmaes_convstarts, pyswarm_convstarts,
                             len(pyswarms), 'Crauste')

def Times(pyswarms, scipydiffevolopt, cmaes, pyswarm, dimension, model):
    # open axes object
    ax = plt.axes()
    ax.plot(range(0, dimension), pyswarms, '-x', color=(215/255, 25/255, 28/255, 0.5), label='PySwarms')
    ax.plot(range(0, dimension), scipydiffevolopt, '-x', color=(94/255, 60/255, 153/255, 0.5), label='Scipy_DiffEvol')
    ax.plot(range(0, dimension), cmaes, '-x', color=(44/255, 123/255, 182/255, 0.5), label='CMA-ES')
    ax.plot(range(0, dimension), pyswarm, '-x', color=(255/255, 153/255, 0/255, 0.5), label='PySwarm')
    print('Median_Pyswarms: ' + str(np.median(pyswarms)))
    print('Median_Scipy: ' + str(np.median(scipydiffevolopt)))
    print('Median_CMAES: ' + str(np.median(cmaes)))
    print('Median_Pyswarm: ' + str(np.median(pyswarm)))

    # change position of the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
    # more options
    ax.set_xlabel('Repetition of 20-fold multi-start optimization', fontsize=20)
    ax.set_ylabel('Computation Time [s]', fontsize=20)
    ax.set_title(f'Computation Times of Optimization Runs - {model}', fontdict={'fontsize': 20})

    if model == 'Rosenbrock':
        ax.set_ylim([0,35])
        ax.set_xticks([0., 4., 9., 14., 19., 24., 29., 34., 39.])
        ax.set_xticklabels([1, 5, 10, 15, 20, 25, 30, 35, 40])
    elif model == 'Crauste':
        #ax.set_ylim([0, 35])
        ax.set_xticks([0., 1., 2.])
        ax.set_xticklabels([1, 2, 3])
    else:
        #ax.set_ylim([240, 660])
        ax.set_xticks([0., 4, 9, 14, 19])
        ax.set_xticklabels([1, 5, 10, 15, 20])

    plt.show()


def ConvergedStarts_by_Times(pyswarms, scipydiffevolopt, cmaes, pyswarm,
                             pyswarms_convstarts, scipydiffevolopt_convstarts,
                             cmaes_convstarts, pyswarm_convstarts,
                             dimension, model):
    # open axes object
    ax = plt.axes()
    ax.plot(range(0, dimension), [i/j for i,j in zip(pyswarms_convstarts,pyswarms)],
            '-x', color=(215 / 255, 25 / 255, 28 / 255, 0.5), label='PySwarms')
    ax.plot(range(0, dimension), [i/j for i,j in zip(scipydiffevolopt_convstarts,scipydiffevolopt)],
            '-x', color=(94 / 255, 60 / 255, 153 / 255, 0.5), label='Scipy_DiffEvol')
    ax.plot(range(0, dimension), [i/j for i,j in zip(cmaes_convstarts,cmaes)],
            '-x', color=(44 / 255, 123 / 255, 182 / 255, 0.5), label='CMA-ES')
    ax.plot(range(0, dimension), [i/j for i,j in zip(pyswarm_convstarts,pyswarm)],
            '-x', color=(255 / 255, 153 / 255, 0 / 255, 0.5), label='PySwarm')
    print('Median_Pyswarms_ConvStarts: ' + str(np.median([i/j for i,j in zip(pyswarms_convstarts,pyswarms)])))
    print('Median_Scipy_ConvStarts: ' + str(np.median([i/j for i,j in zip(scipydiffevolopt_convstarts,scipydiffevolopt)])))
    print('Median_CMAES_ConvStarts: ' + str(np.median([i/j for i,j in zip(cmaes_convstarts,cmaes)])))
    print('Median_Pyswarm_ConvStarts: ' + str(np.median([i/j for i,j in zip(pyswarm_convstarts,pyswarm)])))
    print('Median_Pyswarms_only_ConvStarts: ' + str(np.median(pyswarms_convstarts)))
    print('Median_Scipy_only_ConvStarts: ' + str(np.median(scipydiffevolopt_convstarts)))
    print('Median_CMAES_only_ConvStarts: ' + str(np.median(cmaes_convstarts)))
    print('Median_Pyswarm_only_ConvStarts: ' + str(np.median(pyswarm_convstarts)))

    # change position of the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
    # more options
    ax.set_xlabel('Repetition of 20-fold multi-start optimization', fontsize=20)
    ax.set_ylabel('Converged Starts / Computation Time [1/s]', fontsize=20)
    ax.set_title(f'Converged Starts / Computation Times  of Optimization Runs - {model}', fontdict={'fontsize': 20})

    if model == 'Rosenbrock':
        ax.set_ylim([0, 3.5])
        ax.set_xticks([0., 4., 9., 14., 19., 24., 29., 34., 39.])
        ax.set_xticklabels([1, 5, 10, 15, 20, 25, 30, 35, 40])
    elif model == 'Crauste':
        #ax.set_ylim([0, 35])
        ax.set_xticks([0., 1., 2.])
        ax.set_xticklabels([1, 2, 3])
    else:
        #ax.set_ylim([240, 660])
        ax.set_xticks([0., 4, 9, 14, 19])
        ax.set_xticklabels([1, 5, 10, 15, 20])

    plt.show()


#Times_Boehm()
Times_Crauste()
#Times_Rosenbrock()
