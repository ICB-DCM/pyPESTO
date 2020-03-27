#!/usr/bin/env python
# coding: utf-8

# # Model import using the Petab format

# In this notebook, we illustrate using pyPESTO together with PEtab and AMICI. We employ models from the benchmark collection, which we first download:

# In[1]:


import pypesto
import amici
import petab

import os
import numpy as np
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

# Download benchmark models - Note: 200MB :(
#get_ipython().system('git clone --depth 1 https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.git tmp/benchmark-models || (cd tmp/benchmark-models && git pull)')

folder_base = "tmp/benchmark-models/Benchmark-Models/"


# ## Manage PEtab model

# A PEtab problem comprises all the information on the model, the data and the parameters to perform parameter estimation:

# In[2]:


# a collection of models that can be simulated

#model_name = "Zheng_PNAS2012"
model_name = "Boehm_JProteomeRes2014"
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


# ## Import model to AMICI

# The model must be imported to AMICI:

# In[3]:


importer = pypesto.PetabImporter(petab_problem)

model = importer.create_model()

print("Model parameters:", list(model.getParameterIds()), '\n')
print("Model const parameters:", list(model.getFixedParameterIds()), '\n')
print("Model outputs:   ", list(model.getObservableIds()), '\n')
print("Model states:    ", list(model.getStateIds()), '\n')


# ## Create objective function

# In[4]:


import libsbml
converter_config = libsbml.SBMLLocalParameterConverter()    .getDefaultProperties()
petab_problem.sbml_document.convert(converter_config)

obj = importer.create_objective()

# for some models, hyperparamters need to be adjusted
#obj.amici_solver.setMaxSteps(10000)
#obj.amici_solver.setRelativeTolerance(1e-7)
#obj.amici_solver.setAbsoluteTolerance(1e-7)

ret = obj(petab_problem.x_nominal_scaled, sensi_orders=(0,1), return_dict=True)
print(ret)


# For debugging: There is an alternative way of computing the function, in amici directly.

# In[5]:


import libsbml
converter_config = libsbml.SBMLLocalParameterConverter()    .getDefaultProperties()
petab_problem.sbml_document.convert(converter_config)

obj2 = importer.create_objective()

obj2.use_amici_petab_simulate = True

# for some models, hyperparamters need to be adjusted
#obj2.amici_solver.setMaxSteps(int(1e8))
#obj2.amici_solver.setRelativeTolerance(1e-3)
#obj2.amici_solver.setAbsoluteTolerance(1e-3)

ret2 = obj2(petab_problem.x_nominal_scaled, sensi_orders=(0,1), return_dict=True)
print(ret2)


# A finite difference check whether the computed gradient is accurate:

# In[6]:


problem = importer.create_problem(obj)

objective = problem.objective

ret = objective(petab_problem.x_nominal_free_scaled, sensi_orders=(0,1))
print(ret)


# In[7]:


eps = 1e-4

def fd(x):
    grad = np.zeros_like(x)
    j = 0
    for i, xi in enumerate(x):
        mask = np.zeros_like(x)
        mask[i] += eps
        valinc, _ = objective(x+mask, sensi_orders=(0,1))
        valdec, _ = objective(x-mask, sensi_orders=(0,1))
        grad[j] = (valinc - valdec) / (2*eps)
        j += 1
    return grad

fdval = fd(petab_problem.x_nominal_free_scaled)
print("fd: ", fdval)
print("l2 difference: ", np.linalg.norm(ret[1] - fdval))


# ## Run optimization

# In[8]:


print(problem.x_fixed_indices, problem.x_free_indices)


# In[9]:


optimizer = pypesto.ScipyOptimizer()

engine = pypesto.SingleCoreEngine()
engine = pypesto.MultiProcessEngine()
engine = pypesto.MultiThreadEngine()

# do the optimization
result = pypesto.minimize(problem=problem, optimizer=optimizer,
                          n_starts=10, engine=engine)


# ## Visualize

# In[10]:


print(result.optimize_result.get_for_key('fval'))


# In[11]:


import pypesto.visualize

ref = pypesto.visualize.create_references(x=petab_problem.x_nominal_scaled, fval=obj(petab_problem.x_nominal_scaled))

pypesto.visualize.waterfall(result, reference=ref, scale_y='lin')
pypesto.visualize.parameters(result, reference=ref)

