import h5py


class DataProvider:
    def __init__(self, h5_file):
        self.h5_file = h5_file

    def get_edata(self):
        pass

    def get_timepoints(self):
        with h5py.File(self.h5_file, "r") as f:
            timepoints = f["/amiciOptions/ts"][:]
        return timepoints

    def get_pscales(self):
        with h5py.File(self.h5_file, "r") as f:
            pscale = f["/amiciOptions/pscale"][:]
        return pscale

    def get_fixed_parameters(self):
        with h5py.File(self.h5_file, "r") as f:
            fixed_parameters = f["/fixedParameters/k"][:]
            fixed_parameters = fixed_parameters[0]
        return fixed_parameters

    def get_fixed_parameters_names(self):
        with h5py.File(self.h5_file, "r") as f:
            fixed_parameters_names = f["/fixedParameters/parameterNames"][:]
        return fixed_parameters_names

    def get_initial_states(self):
        pass

    def get_measurements(self):
        with h5py.File(self.h5_file, "r") as f:
            measurements = f["/measurements/y"][:]
        return measurements

    def get_ysigma(self):
        with h5py.File(self.h5_file, "r") as f:
            ysigma = f["/measurements/ysigma"][:]
        return ysigma

    def get_observableNames(self):
        with h5py.File(self.h5_file, "r") as f:
            observable_names = f["/measurements/observableNames"]
        return observable_names
