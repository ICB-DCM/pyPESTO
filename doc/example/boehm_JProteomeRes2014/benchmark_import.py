import numpy as np
import scipy as sp
import h5py
import pandas as pd

class DataProvider:

    def __init__(self, h5_file):
        self.h5_file = h5_file

    def get_edata(self):
        pass

    def get_timepoints(self):
        with h5py.File(self.h5_file, 'r') as f:
            timepoints = f['/amiciOptions/ts'][:]
        return timepoints

    def get_pscales(self):
        with h5py.File(self.h5_file, 'r') as f:
            pscale = f['/amiciOptions/pscale'][:]
        return pscale

    def get_fixed_parameters(self):
        with h5py.File(self.h5_file, 'r') as f:
            fixed_parameters = f['/fixedParameters/k'][:]
            fixed_parameters = fixed_parameters[0]
        return fixed_parameters

    def get_fixed_parameters_names(self):
        with h5py.File(self.h5_file, 'r') as f:
            fixed_parameters_names = f['/fixedParameters/parameterNames'][:]
        return fixed_parameters_names

    def get_initial_states(self):
        pass

    def get_measurements(self):
        with h5py.File(self.h5_file, 'r') as f:
            measurements = f['/measurements/y'][:]
        return measurements

    def get_ysigma(self):
        with h5py.File(self.h5_file, 'r') as f:
            ysigma = f['/measurements/ysigma'][:]
        return ysigma

    def get_observableNames(self):
        with h5py.File(self.h5_file, 'r') as f:
            observable_names = f['/measurements/observableNames']
        return observable_names

    @staticmethod
    def convert_benchmark_to_h5(model_name):
        """
        Convert the data format in the benchmark study by Loos et al. to
        hdf5.

        Parameters
        ----------

        model: str
            e.g. "Fujita_SciSignal2010", file name relative to current
            working directory

        Returns
        -------

        dp: DataProvider
            a DataProvider to the generated h5_file
        """

        # check if h5 file exists already, then error

        # observables

        # fixed parameters

        # timepoints

        # ...


#h5_file = '/home/lenoard/PycharmProjects/benchmark_import/example_fujita_SciSignal2010/data_fujita_SciSignal2010.h5'
#dp = DataProvider(h5_file)
#timepoints = dp.get_timepoints()
#fixed_parameters = dp.get_fixed_parameters()
#measurements = dp.get_measurements()
#print('timepoints:')
#print(timepoints)
#print('Fixed parameters:')
#print(fixed_parameters)
#print('Measurements:')
#print(measurements)