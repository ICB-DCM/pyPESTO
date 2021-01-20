import h5py
import numpy as np
import pandas as pd
import copy

f = h5py.File('/home/paulstapor/Dokumente/Projekte/Speedy/'
              'cell_line_optimization/petab_model/best_parameters.h5', 'r')
pars = f['parameters'][:]
par_ids = f['parameterIds'][:]
par_ids = [(str(id)[2:])[:-1] for id in par_ids]

par_df_in = pd.read_csv('/home/paulstapor/Dokumente/Projekte/Speedy/'
                        'cell_line_optimization/petab_model/Speedy_V5_1_MutDrugEE'
                        '_globDecay_r448400_mod_parameters.tsv', sep='\t',
                        index_col=0)

for ivector in range(pars.shape[0]):
    vector = pars[ivector, :]
    par_df = copy.deepcopy(par_df_in)
    for i, id in enumerate(par_ids):
        par_df.loc[id, 'nominalValue'] = np.power(10., vector[i])

    par_df.to_csv('/home/paulstapor/Dokumente/Projekte/Speedy/'
                  'cell_line_optimization/petab_model/Speedy_V5_1_MutDrugEE'
                  f'_globDecay_r448400_mod_parameters_vector_{ivector}.tsv', sep='\t')
    del par_df
