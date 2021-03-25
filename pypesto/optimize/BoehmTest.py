# test to see if the Boehm_ProteomeRes2014 model can be correctly optimized using the pyswarms optimizer

import libsbml
import importlib
import amici
import pypesto
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pypesto.optimize as optimize
import pypesto.visualize as visualize
from benchmark_import import DataProvider

# temporarily add the simulate file
sys.path.insert(0, 'boehm_JProteomeRes2014')

# read in sbml file
#sbml_file = '../../../benchmark-models-petab/Benchmark-Models/Boehm_JProteomeRes2014/model_Boehm_JProteomeRes2014.xml'
sbml_file = '../../../pypesto/doc/example/boehm_JProteomeRes2014/boehm_JProteomeRes2014.xml'
model_name = 'boehm_JProteomeRes2014'
model_output_dir = 'tmp/' + model_name
sbml_reader = libsbml.SBMLReader()
sbml_doc = sbml_reader.readSBML(os.path.abspath(sbml_file))
sbml_model = sbml_doc.getModel()
dir(sbml_doc)

# check correctness until this point
print(os.path.abspath(sbml_file))
print('Species: ', [s.getId() for s in sbml_model.getListOfSpecies()])
print('\nReactions:')
for reaction in sbml_model.getListOfReactions():
    reactants = ' + '.join(['%s %s'%(int(r.getStoichiometry()) if r.getStoichiometry() > 1 else '', r.getSpecies()) for r in reaction.getListOfReactants()])
    products  = ' + '.join(['%s %s'%(int(r.getStoichiometry()) if r.getStoichiometry() > 1 else '', r.getSpecies()) for r in reaction.getListOfProducts()])
    reversible = '<' if reaction.getReversible() else ''
    print('%3s: %10s %1s->%10s\t\t[%s]' % (reaction.getId(),
                        reactants,
                        reversible,
                         products,
                        libsbml.formulaToL3String(reaction.getKineticLaw().getMath())))

# sbml importer & add parameters
sbml_importer = amici.SbmlImporter(sbml_file)
constantParameters = ['ratio', 'specC17']
observables = amici.assignmentRules2observables(
        sbml_importer.sbml, # the libsbml model object
        filter_function=lambda variable: variable.getId().startswith('observable_') and not variable.getId().endswith('_sigma')
    )
print(observables)

sigma_vals = ['sd_pSTAT5A_rel', 'sd_pSTAT5B_rel', 'sd_rSTAT5A_rel']
observable_names = observables.keys()
sigmas = dict(zip(list(observable_names), sigma_vals))
print(sigmas)

# generate the module
sbml_importer.sbml2amici(model_name,
                         model_output_dir,
                         verbose=False,
                         observables=observables,
                         constantParameters=constantParameters,
                         sigmas=sigmas
  )

# import and load the module
sys.path.insert(0, os.path.abspath(model_output_dir))
model_module = importlib.import_module(model_name)
model = model_module.getModel()

# check further correctness
print("Model parameters:", list(model.getParameterIds()))
print("Model outputs:   ", list(model.getObservableIds()))
print("Model states:    ", list(model.getStateIds()))

# run simulations
h5_file = './data_boehm_JProteomeRes2014.h5'
dp = DataProvider(h5_file)

# set timepoints for which we want to simulate the model
timepoints = amici.DoubleVector(dp.get_timepoints())
model.setTimepoints(timepoints)

# set fixed parameters for which we want to simulate the model
model.setFixedParameters(amici.DoubleVector(np.array([0.693, 0.107])))

# set parameters to optimal values found in the benchmark collection
model.setParameterScale(2)
model.setParameters(amici.DoubleVector(np.array([-1.568917588,
-4.999704894,
-2.209698782,
-1.786006548,
4.990114009,
4.197735488,
0.585755271,
0.818982819,
0.498684404
])))

# Create solver instance
solver = model.getSolver()

# Run simulation using model parameters from the benchmark collection and default solver options
rdata = amici.runAmiciSimulation(model, solver)

# Create edata
edata = amici.ExpData(rdata, 1.0, 0)

# set observed data
edata.setObservedData(amici.DoubleVector(dp.get_measurements()[0][:, 0]), 0)
edata.setObservedData(amici.DoubleVector(dp.get_measurements()[0][:, 1]), 1)
edata.setObservedData(amici.DoubleVector(dp.get_measurements()[0][:, 2]), 2)

# set standard deviations to optimal values found in the benchmark collection
edata.setObservedDataStdDev(amici.DoubleVector(np.array(16*[10**0.585755271])), 0)
edata.setObservedDataStdDev(amici.DoubleVector(np.array(16*[10**0.818982819])), 1)
edata.setObservedDataStdDev(amici.DoubleVector(np.array(16*[10**0.498684404])), 2)
rdata = amici.runAmiciSimulation(model, solver, edata)

print('Chi2 value reported in benchmark collection: 47.9765479')
print('chi2 value using AMICI:')
print(rdata['chi2'])


# run optimization using pypesto's pyswarms optimizer
model.requireSensitivitiesForAllParameters()
solver.setSensitivityMethod(amici.SensitivityMethod_forward)
solver.setSensitivityOrder(amici.SensitivityOrder_first)

objective = pypesto.AmiciObjective(model, solver, [edata], 1)

# create optimizer object which contains all information for doing the optimization
optimizer_pyswarms = optimize.PyswarmsOptimizer()
optimizer_scipydiffevolopt = optimize.ScipyDifferentialEvolutionOptimizer()
optimizer_cmaes = optimize.CmaesOptimizer()
optimizer_pyswarm = optimize.PyswarmOptimizer()

optimizer_pyswarms.solver = 'pyswarms'
optimizer_scipydiffevolopt.solver = 'ScipyDifferentialEvolutionOptimizer'
optimizer_cmaes.solver = 'cma'
optimizer_pyswarm.solver = 'pyswarm'

# create problem object containing all information on the problem to be solved
x_names = ['x' + str(j) for j in range(0, 9)]
problem = pypesto.Problem(objective=objective,
                          lb=-5*np.ones((9)), ub=5*np.ones((9)),
                          x_names=x_names)

# set number of starts
n_starts = 20

# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

# run optimizations for different optimizers
start_pyswarms = time.time()
result1_pyswarms = optimize.minimize(problem=problem1, optimizer=optimizer_pyswarms,
    n_starts=n_starts, history_options=history_options)
end_pyswarms = time.time()

start_scipy = time.time()
result1_scipydiffevolopt = optimize.minimize(problem=problem1, optimizer=optimizer_scipydiffevolopt,
    n_starts=n_starts, history_options=history_options)
end_scipy = time.time()

start_cmaes = time.time()
result1_cmaes = optimize.minimize(problem=problem1, optimizer=optimizer_cmaes,
    n_starts=n_starts, history_options=history_options)
end_cmaes = time.time()

start_pyswarm = time.time()
result1_pyswarm = optimize.minimize(problem=problem1, optimizer=optimizer_pyswarm,
    n_starts=n_starts, history_options=history_options)
end_pyswarm = time.time()


#### print times
print('Pyswarms: ' + '{:5.3f}s'.format(end_pyswarms - start_pyswarms))
print('Scipy: ' + '{:5.3f}s'.format(end_scipy - start_scipy))
print('Cmaes: ' + '{:5.3f}s'.format(end_cmaes - start_cmaes))
print('Pysawrm: ' + '{:5.3f}s'.format(end_pyswarm - start_pyswarm))


# Visualize waterfall
visz = visualize.waterfall([result_pyswarms, result_scipydiffevolopt, result_cmaes, result_pyswarm],
                    legends=['Pyswarms', 'Scipy_DiffEvol', 'CMA-ES', 'PySwarm'],
                    scale_y='lin',
                    colors=[(31/255, 120/255, 180/255, 0.5), (178/255, 223/255, 138/255, 0.5),
                            (51/255, 160/255, 44/255, 0.5), (166/255, 206/255, 227/255, 0.5)])
                    #colors=['#1f78b4', '#b2df8a', '#33a02c', '#a6cee3'])
# change position of the legend
box = visz.get_position()
visz.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
visz.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
#visz.axhline(y=0, xmin=0, xmax=19, color='black', linestyle='--', alpha=0.75)


# Visulaize Parameters
para = visualize.parameters([result1_pyswarms, result1_scipydiffevolopt, result1_cmaes, result1_pyswarm],
                     legends=['PySwarms'],
                     balance_alpha=True,
                     colors=[(31/255, 120/255, 180/255, 0.5)])
# change position of the legend
box = para.get_position()
para.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
para.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
'''                         
                         , (178/255, 223/255, 138/255, 0.5),
                             (51/255, 160/255, 44/255, 0.5), (166/255, 206/255, 227/255, 0.5)])
                     #colors=['#1f78b4', '#b2df8a', '#33a02c', '#a6cee3'])
'''
a = 4
