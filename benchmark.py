#!/usr/bin/env python3
import libsbml
import os
import sys
#sys.path.insert(0, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(0, '/home/dweindl/src/pyPESTO')

import pypesto
import sys
import amici
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import petab
import amici

models = ['Bachmann_MSB2011', 'beer_MolBioSystems2014', 'boehm_JProteomeRes2014',
 'Borghans_BiophysChem1997', 'Brannmark_JBC2010', 'Bruno_JExpBio2016',
'Chen_MSB2009', 'Crauste_CellSystems2017', 'Elowitz_Nature2000',
'Fiedler_BMC2016', 'Fujita_SciSignal2010', 'Hass_PONE2017',
'Isensee_JCB2018', 'Lucarelli_CellSystems_2018',
'Merkle_PCB2016', 'Raia_CancerResearch2011',
'Schwen_PONE2014','Sneyd_PNAS2002', 'Sobotta_Frontiers2017',
'Swameye_PNAS2003', 'Weber_BMC2015','Zheng_PNAS2012']


#model_root = os.path.abspath(os.path.join('Benchmark-Models', 'hackathon_contributions_new_data_format'))
model_root = '/home/yannik/benchmark-models/hackathon_contributions_new_data_format/'
benchmark_model = 'Zheng_PNAS2012' # 'Zheng_PNAS2012'
#benchmark_model = "Boehm_JProteomeRes2014"
#benchmark_model = "Fujita_SciSignal2010"
condition_filename = os.path.join(model_root, benchmark_model, f'experimentalCondition_{benchmark_model}.tsv')
measurement_filename = os.path.join(model_root, benchmark_model,f'measurementData_{benchmark_model}.tsv')
parameter_filename = os.path.join(model_root, benchmark_model, f'parameters_{benchmark_model}.tsv')
sbml_model_file = os.path.join(model_root, benchmark_model, f'model_{benchmark_model}.xml')
model_name = f'model_{benchmark_model}'
model_output_dir = f'deleteme-{model_name}'
 
rebuild = False
#rebuild = True
if rebuild:
    import_sbml_model(sbml_model_file=sbml_model_file,
                      condition_file=condition_filename,
                      measurement_file=measurement_filename,
                      model_output_dir=model_output_dir,
                      model_name=model_name)


sys.path.insert(0, os.path.abspath(model_output_dir))
model_module = importlib.import_module(model_name)

model = model_module.getModel()
model.requireSensitivitiesForAllParameters()

solver = model.getSolver()
solver.setSensitivityMethod(amici.SensitivityMethod_forward)
solver.setSensitivityOrder(amici.SensitivityOrder_first)

print("Model parameters:", list(model.getParameterIds()))
print()
print("Model const parameters:", list(model.getFixedParameterIds()))
print()
print("Model outputs:   ", list(model.getObservableIds()))
print()
print("Model states:    ", list(model.getStateIds()))

from pypesto.logging import log_to_console
log_to_console()

# load nominal parameters from parameter description file
parameter_df = petab.get_parameter_df(parameter_filename)

# Create objective function instance from model and measurements

manager = petab.Manager.from_folder(model_root + benchmark_model)
manager.map_par_sim_to_par_opt()
importer = pypesto.objective.Importer(manager)
model = importer.model

print("MODEL PAMS:", list(model.getParameterIds()))
print("MODEL CONST PAMS:", list(model.getFixedParameterIds()))
print("MODEL OUTPUTS:", list(model.getObservableIds()))

model.setParameterScale(amici.ParameterScaling_log10)
obj, edatas = importer.create_objective()
x_nom = manager.parameter_df['nominalValue'].values

print(x_nom)
print("obj: ", obj(x_nom))

