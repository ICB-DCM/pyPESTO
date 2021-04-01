# optimize benchmark models

import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import amici
import petab
import os
import numpy as np
import matplotlib.pyplot as plt
import libsbml
import time
import pandas as pd


def Save_Times(model, runs):
    # create data frame for time storage
    column_names = ['PySwarms_time', 'Scipy_DiffEvol_time',
                    'CMA-ES_time', 'PySwarm_time',
                    'PySwarms_conv_starts', 'Scipy_DiffEvol_conv_starts',
                    'CMA-ES_conv_starts', 'PySwarm_conv_starts']
    df = pd.DataFrame(columns=column_names, data=[])

    problem, obj, petab_problem = Boehm_Optimization(model)
    for iRuns in range(0, runs):
        print(f'Starting of run: {iRuns}')
        df = df.append({}, ignore_index=True)
        times, number_converged_starts = Actual_optimization(problem, obj, petab_problem)
        for iOptimizers in range(0, len(times)):
            df[column_names[iOptimizers]][iRuns] = times[iOptimizers]
        for iOptimizers in range(0, len(number_converged_starts)):
            df[column_names[4 + iOptimizers]][iRuns] = number_converged_starts[iOptimizers]

    if model == 'Boehm':
        df.to_csv('./Boehm_times_convstarts.tsv', sep='\t')
    else:
        df.to_csv('./Crauste_times_convstarts.tsv', sep='\t')


def Boehm_Optimization(model):

    folder_base = "../../../Benchmark-Models-PEtab/Benchmark-Models/"

    # a collection of models that can be simulated
    #model_name = "Zheng_PNAS2012"
    if model == 'Boehm':
        model_name = "Boehm_JProteomeRes2014"
        print(model_name)
    else:
        model_name = "Crauste_CellSystems2017"
        print(model_name)
    #model_name = "Fujita_SciSignal2010"
    #model_name = "Sneyd_PNAS2002"
    #model_name = "Borghans_BiophysChem1997"
    #model_name = "Elowitz_Nature2000"
    #model_name = "Crauste_CellSystems2017"
    #model_name = "Lucarelli_CellSystems2018"
    #model_name = "Schwen_PONE2014"
    #model_name = "Blasi_CellSystems2016"

    # the yaml configuration file links to all needed files
    yaml_config = os.path.join(folder_base, model_name, model_name + '.yaml')

    # create a petab problem
    petab_problem = petab.Problem.from_yaml(yaml_config)
    converter_config = libsbml.SBMLLocalParameterConverter().getDefaultProperties()
    petab_problem.sbml_document.convert(converter_config)

    # load model
    importer = pypesto.petab.PetabImporter.from_yaml(yaml_config)
    problem = importer.create_problem()
    obj = importer.create_objective()

    return problem, obj, petab_problem

def Actual_optimization(problem, obj, petab_problem):
    # define the optimizers
    optimizer_pyswarms = optimize.PyswarmsOptimizer()
    optimizer_scipydiffevolopt = optimize.ScipyDifferentialEvolutionOptimizer()
    optimizer_cmaes = optimize.CmaesOptimizer()
    optimizer_pyswarm = optimize.PyswarmOptimizer()

    # engine = pypesto.engine.SingleCoreEngine()
    engine = pypesto.engine.MultiProcessEngine()

    # set number of starts
    n_starts = 20

    # run optimizations for different optimizers
    start_pyswarms = time.time()
    result_pyswarms = optimize.minimize(problem=problem, optimizer=optimizer_pyswarms,
                                        n_starts=n_starts, engine=engine)
    end_pyswarms = time.time()

    start_scipy = time.time()
    result_scipydiffevolopt = optimize.minimize(problem=problem, optimizer=optimizer_scipydiffevolopt,
                                                n_starts=n_starts, engine=engine)
    end_scipy = time.time()

    start_cmaes = time.time()
    result_cmaes = optimize.minimize(problem=problem, optimizer=optimizer_cmaes,
                                     n_starts=n_starts, engine=engine)
    end_cmaes = time.time()

    start_pyswarm = time.time()
    result_pyswarm = optimize.minimize(problem=problem, optimizer=optimizer_pyswarm,
                                       n_starts=n_starts, engine=engine)
    end_pyswarm = time.time()


    times = [end_pyswarms - start_pyswarms, end_scipy - start_scipy,
             end_cmaes - start_cmaes, end_pyswarm - start_pyswarm]

    pyswarms_conv_starts = []
    scipydiffevol_conv_starts = []
    cmaes_conv_starts = []
    pyswarm_conv_starts = []
    for iMultistart in range(0, 20):
        print('Nominal Value: ' + str(obj(petab_problem.x_nominal_scaled)))
        fval1 = result_pyswarms.optimize_result.list[iMultistart]['fval'] - obj(petab_problem.x_nominal_scaled)
        fval2 = result_scipydiffevolopt.optimize_result.list[iMultistart]['fval'] - obj(petab_problem.x_nominal_scaled)
        fval3 = result_cmaes.optimize_result.list[iMultistart]['fval'] - obj(petab_problem.x_nominal_scaled)
        fval4 = result_pyswarm.optimize_result.list[iMultistart]['fval'] - obj(petab_problem.x_nominal_scaled)
        if fval1 < 0.1:
            pyswarms_conv_starts.append(1)
        if fval2 < 0.1:
            scipydiffevol_conv_starts.append(1)
        if fval3 < 0.1:
            cmaes_conv_starts.append(1)
        if fval4 < 0.1:
            pyswarm_conv_starts.append(1)
    number_converged_starts = [sum(pyswarms_conv_starts), sum(scipydiffevol_conv_starts),
                               sum(cmaes_conv_starts), sum(pyswarm_conv_starts)]

    return times, number_converged_starts

Save_Times('Boehm',runs=10)
Save_Times('Crauste',runs=3)


'''
# visualize
ref = visualize.create_references(
    x=petab_problem.x_nominal_scaled, fval=obj(petab_problem.x_nominal_scaled))

visz = visualize.waterfall([result_pyswarms,result_scipydiffevolopt, result_cmaes, result_pyswarm],
                    legends=['PySwarms', 'Scipy_DiffEvol', 'CMA-ES', 'PySwarm'],
                    scale_y='lin',
                    reference=ref,
                    #y_limits=(8 * 10 ** -1, 6 * 10 ** 2),
                    colors=[(215 / 255, 25 / 255, 28 / 255, 0.5), (94 / 255, 60 / 255, 153 / 255, 0.5),
                            (44 / 255, 123 / 255, 182 / 255, 0.5), (255 / 255, 153 / 255, 0 / 255, 0.5)])
# change position of the legend
box = visz.get_position()
visz.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
visz.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
# more options
visz.set_xlabel('Ordered optimizer run', fontsize=20)
visz.set_ylabel('Function value', fontsize=20)
visz.set_title('Waterfall plot', fontdict={'fontsize': 20})
visz.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
plt.show()

para = visualize.parameters(result_pyswarms,
                     reference=ref,
                     legends=['PySwarms'],
                     balance_alpha=True,
                     colors=[(215 / 255, 25 / 255, 28 / 255, 0.5)])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
# more options
para.set_xlabel('Parameter value', fontsize=20)
para.set_ylabel('Parameter', fontsize=20)
para.set_title('Estimated parameters', fontdict={'fontsize': 20})
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
plt.show()

para = visualize.parameters(result_scipydiffevolopt,
                     reference=ref,
                     legends=['Scipy_DiffEvol'],
                     balance_alpha=True,
                     colors=[(94 / 255, 60 / 255, 153 / 255, 0.5)])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
# more options
para.set_xlabel('Parameter value', fontsize=20)
para.set_ylabel('Parameter', fontsize=20)
para.set_title('Estimated parameters', fontdict={'fontsize': 20})
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
plt.show()

para = visualize.parameters(result_cmaes,
                     reference=ref,
                     legends=['CMA-ES'],
                     balance_alpha=True,
                     colors=[(44 / 255, 123 / 255, 182 / 255, 0.5)])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
# more options
para.set_xlabel('Parameter value', fontsize=20)
para.set_ylabel('Parameter', fontsize=20)
para.set_title('Estimated parameters', fontdict={'fontsize': 20})
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
plt.show()

para = visualize.parameters(result_pyswarm,
                     reference=ref,
                     legends=['PySwarm'],
                     balance_alpha=True,
                     colors=[(255 / 255, 153 / 255, 0 / 255, 0.5)])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
# more options
para.set_xlabel('Parameter value', fontsize=20)
para.set_ylabel('Parameter', fontsize=20)
para.set_title('Estimated parameters', fontdict={'fontsize': 20})
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20, fancybox=True, shadow=True, ncol=4)
plt.show()

a = 4
'''