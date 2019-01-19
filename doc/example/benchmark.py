#!/usr/bin/env python3

import petab
import sys

sys.path.insert(0, "/home/yannik/pypesto")
import pypesto

models = ['Bachmann_MSB2011', 'beer_MolBioSystems2014',
          'boehm_JProteomeRes2014',
          'Borghans_BiophysChem1997', 'Brannmark_JBC2010', 'Bruno_JExpBio2016',
          'Chen_MSB2009', 'Crauste_CellSystems2017', 'Elowitz_Nature2000',
          'Fiedler_BMC2016', 'Fujita_SciSignal2010', 'Hass_PONE2017',
          'Isensee_JCB2018', 'Lucarelli_CellSystems_2018',
          'Merkle_PCB2016', 'Raia_CancerResearch2011',
          'Schwen_PONE2014', 'Sneyd_PNAS2002', 'Sobotta_Frontiers2017',
          'Swameye_PNAS2003', 'Weber_BMC2015', 'Zheng_PNAS2012']

model_root = \
    '/home/yannik/benchmark-models/hackathon_contributions_new_data_format/'
benchmark_model = 'Zheng_PNAS2012'  # 'Zheng_PNAS2012'
# benchmark_model = "Boehm_JProteomeRes2014"
# benchmark_model = "Fujita_SciSignal2010"

manager = petab.Problem.from_folder(model_root + benchmark_model)
print(
    "PARAMETER MAPPING:",
    manager.get_optimization_to_simulation_parameter_mapping())
importer = pypesto.PetabImporter(manager)
model = importer.model

print("MODEL PAMS:", list(model.getParameterIds()))
print("MODEL CONST PAMS:", list(model.getFixedParameterIds()))
print("MODEL OUTPUTS:", list(model.getObservableIds()))

# model.setParameterScale(amici.ParameterScaling_log10)
obj, edatas = importer.create_objective()
x_nom = manager.parameter_df['nominalValue'].values

print(x_nom)
print("obj: ", obj(x_nom))

rdatas = obj(x_nom, return_dict=True)['rdatas']
print("rdata:   ", rdatas[0])
#print(rdatas)
df = importer.rdatas_to_measurement_df(rdatas)
print(df)

df.to_csv("tmp/simulation.csv", sep='\t')
