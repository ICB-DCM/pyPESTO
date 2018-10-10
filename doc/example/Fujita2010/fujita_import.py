import libsbml
import importlib
import amici
import amici.pandas
import os
import sys
import numpy as np
import pandas as pd
import pypesto

dir_name = '/home/paul/Documents/pesto/pyPESTO/doc/example/Fujita2010'

model_name = 'Fujita_SciSignal2010' # inf in name of parameter inflow

# SBML model we want to import
sbml_file = dir_name + '/SBML/model1_data1_l2v4.xml'

# import general info (e.g. for the constant parameters)
info_file = dir_name + '/General_info.xlsx'
generalInfo = pd.ExcelFile(info_file)
tmp = generalInfo.parse('Experimental conditions')


# import data (for the observables names)
data_file  = dir_name + '/Data/model1_data1.xlsx'
xlsData = pd.ExcelFile(data_file)
simData = xlsData.parse('Simulation')
tmp = list(simData)

obsData = pd.read_excel(data_file)

# remove time from the list of names
observables_list = tmp[1:] #

# get timepoints for which system was simulated in d2d (to compare)
timepoints = simData['time']

# Directory to which the generated model code is written
model_output_dir = model_name

sbml_reader = libsbml.SBMLReader()
sbml_doc = sbml_reader.readSBML(sbml_file)
sbml_model = sbml_doc.getModel()

sbml_importer = amici.SbmlImporter(sbml_file)

print(observables_list)

observables = amici.assignmentRules2observables(
        sbml_importer.sbml, # the libsbml model object
        filter_function=lambda variableId: variableId in observables_list
    )

timepoints = obsData['time']
for iObs in obsData.keys():
    if not iObs in observables_list:
        del obsData[iObs]



sbml_importer.sbml2amici(model_name,
                         model_output_dir,
                         verbose=False,
                         observables=observables,
                         constantParameters=['EGF']
    )

sys.path.insert(0, os.path.abspath(model_output_dir))
model_module = importlib.import_module(model_name)

model = model_module.getModel()
par = np.log10(np.array(model.getParameters()))

model.setParameterScale(2)
model.requireSensitivitiesForAllParameters()
model.setParameters(amici.DoubleVector(par))
model.fixedParameters = amici.DoubleVector(np.array([-1.]))

# Create solver instance
solver = model.getSolver()
solver.setSensitivityMethod(1)
solver.setSensitivityOrder(1)
solver.setAbsoluteTolerance(1e-10)
solver.setRelativeTolerance(1e-6)
solver.setMaxSteps(100000)

# Run simulation using default model parameters and solver options
simAmici = amici.runAmiciSimulation(model, solver)

eData = amici.ExpData(simAmici, 0, 0)
tp2 = amici.DoubleVector(np.array(timepoints, dtype=float))
eData.setTimepoints(tp2)
eData.setObservedData(amici.DoubleVector(obsData['pEGFR_tot']), 0)
eData.setObservedData(amici.DoubleVector(obsData['pAkt_tot']), 1)
eData.setObservedData(amici.DoubleVector(obsData['pS6_tot']), 2)
eData.fixedParameters = amici.DoubleVector(np.array([-1.]))

print(list(model.getObservableIds()))

simAmici2 = amici.runAmiciSimulation(model, solver, eData)
# print(simAmici2)

# create objective function from amici model
# pesto.AmiciObjective is derived from pesto.Objective,
# the general pesto objective function class
objective = pypesto.AmiciObjective(model, solver, [eData], 1)

true_par = np.array([-2.82985140290572,-1.56974876134701,7.57444976997831,2.27430370307951,-2.43288347364865,-2.63805026108042,-3.02849184348735,4.78508205034262,-0.363286437827988,-1.52063811194138,-5.48503965936016,-3.39952121863019,-5.26255314438994,-1.92799992805853,-3.02467768651383,-1.54499061957732,1.6167600821369,-7.2481164920359,4.89499108357191,-2.,-2.,-2.])


llh = objective(true_par)

print(llh)

raise('stop')

# create optimizer object which contains all information for doing the optimization
optimizer = pypesto.ScipyOptimizer()

optimizer.solver = 'TNC'
# if select meigo -> also set default values in solver_options
#optimizer.options = {'maxiter': 1000, 'disp': True} # = pesto.default_options_meigo()
#optimizer.n_starts = 100

# see PestoOptions.m for more required options here
# returns OptimizationResult, see parameters.MS for what to return
# list of final optim results foreach multistart, times, hess, grad,
# flags, meta information (which optimizer -> optimizer.get_repr())

# create problem object containing all information on the problem to be solved
x_names = ['x' + str(j) for j in range(0, 22)]
problem = pypesto.Problem(objective=objective,
                          lb=-8*np.ones((22)), ub=8*np.ones((22)),
                          x_names=x_names)

# maybe lb, ub = inf
# other constraints: kwargs, class pesto.Constraints
# constraints on pams, states, esp. pesto.AmiciConstraints (e.g. pam1 + pam2<= const)
# if optimizer cannot handle -> error
# maybe also scaling / transformation of parameters encoded here

# do the optimization
result = pypesto.minimize(problem=problem,
                          optimizer=optimizer,
                          n_starts=10)
# optimize is a function since it does not need an internal memory,
# just takes input and returns output in the form of a Result object
# 'result' parameter: e.g. some results from somewhere -> pick best start points

import pypesto.visualize

pypesto.visualize.waterfall(result)
pypesto.visualize.parameters(result)


