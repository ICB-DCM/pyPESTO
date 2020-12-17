import numpy as np
import os
import pandas as pd
import h5py
from typing import Sequence, Union, Callable, Tuple
from warnings import warn
from time import time

from .constants import MODE_FUN
from .amici import AmiciObjective


try:
    import amici
    import amici.petab_objective
    import amici.parameter_mapping
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass


class AmiciPrediction():
    """
    This class allows to perform forward simulation via an amici model.
    These simulations can either be exactly those from amici, or they can be
    post-processed after simulation.
    If a post-processing method is applied, also the sensitivities of the amici
    model must be post-processed, if a gradient of the forward simulation is
    requested. This responsibility for correctness of this sensitivity
    post-processing is EXPLICITLY left to the user.
    """
    def __init__(self,
                 amici_objective: AmiciObjective,
                 post_processing: Union[Callable, None] = None,
                 post_processing_sensi: Union[Callable, None] = None,
                 max_num_conditions: int = 0):
        """
        Constructor.

        Parameters
        ----------
        amici_objective:
            An objective object, which will be used to get model simulations
        post_processing:
            A callable function which applies postprocessing to the simulation
            results. Default are the observables of the amici model.
            This method must take a list of amici ReturnData objects as input.
        post_processing_sensi:
            A callable function which applies postprocessing to the
            sensitivities of the simulation results. Default are the
            observable sensitivities of the amici model.
            This method must take a list of amici ReturnData objects as input.
        max_num_conditions:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum number of conditions, which
            should be simulated at a time.
            Default is 0 meaning that all conditions will be simulated.
            Other values are only applicable, if an output file is specified.
        """
        # save settings and objective
        self.amici_objective = amici_objective
        self.max_num_conditions = max_num_conditions
        if post_processing is None:
            self.post_processing = self.default_post_processing
        if post_processing_sensi is None:
            self.post_processing_sensi = self.default_post_processing_sensi

    def __call__(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...] = (0, ),
            mode: str = MODE_FUN,
            output_file: str = '',
            output_format: str = 'csv',
            max_num_conditions: int = 0,
    ) -> Union[Sequence[np.ndarray], Tuple]:
        """
        Method to simulate a model for a certain prediction function.
        This method relies on the AmiciObjective, which is underlying, but
        allows the user to apply any post-processing of the results and the
        sensitivities.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the prediction function.
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.
        output_file:
            Path to an output file.
        max_num_conditions:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum number of conditions, which
            should be simulated at a time. Default is self.max_num_conditions.
        output_format:
            Either 'csv', 'h5'. If an output file is specified, this routine
            will return a csv file, created from a DataFrame, or an h5 file,
            created from a dict.

        Returns
        -------
        outputs:
            List of postprocessed outputs (ndarrays)
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays)
        """
        # sanity check for output
        if 2 in sensi_orders:
            raise Exception('Prediction simulation does currently not support '
                            'second order output.')

        # prepare for storing the results
        outputs = []
        outputs_sensi = []
        timepoints = []

        # simulate the model and get the output
        self._get_model_outputs(outputs, outputs_sensi, timepoints,
                                x, sensi_orders, mode)

        # Should the results be saved to a file?
        if output_file != '':
            # Do we want a pandas dataframe like format?
            if output_format == 'csv':
                self._write_to_csv(outputs=outputs,
                                   outputs_sensi=outputs_sensi,
                                   output_file=output_file,
                                   timepoints=timepoints)
            # Do we want an h5 file?
            elif output_format == 'h5':
                self._write_to_h5(outputs=outputs,
                                  outputs_sensi=outputs_sensi,
                                  output_file=output_file,
                                  timepoints=timepoints)
            else:
                raise Exception(f'Call to unknown format {output_format} for '
                                f'output of pyPESTO prediction.')

        # return dependent on sensitivity order
        if sensi_orders == (0, 1):
            return outputs, outputs_sensi
        elif sensi_orders == (0,):
            return outputs
        elif sensi_orders == (1,):
            return outputs_sensi
        else:
            raise Exception('Prediction simulation called with unsupported '
                            'input for sensi_orders:', sensi_orders)

    def _get_model_outputs(self,
                           outputs,
                           outputs_sensi,
                           timepoints,
                           x,
                           sensi_orders,
                           mode):
        """
        The main purpose of this function is to encapsulate the call to amici:
        This allows to use variable scoping as a mean to clean up the memory
        after calling amici, which is beneficial if large models with large
        datasets are used.

        Parameters
        ----------
        outputs:
            List of postprocessed outputs (ndarrays), to which results will
            be appended
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays),
            to which results will be appended
        timepoints:
            List of output timepoints (ndarrays), to which simulation
            timepoints will be appended
        x:
            The parameters for which to evaluate the prediction function.
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.
        """

        # Do we have a maximum number of simulations allowed?
        if self.max_num_conditions == 0:
            # simulate all conditions at once
            n_simulations = len(self.amici_objective.edatas)
        else:
            # simulate only a subset of conditions
            n_simulations = 1 + int(len(self.amici_objective.edatas) /
                                    self.max_num_conditions)

        for i_sim in range(n_simulations):
            # slice out the conditions we actually want
            ids = slice(i_sim * self.max_num_conditions,
                        max((i_sim + 1) * self.max_num_conditions,
                            n_simulations))

            # call amici
            ret = self.amici_objective(x=x, sensi_orders=sensi_orders,
                                       edatas=self.amici_objective.edatas[ids],
                                       mode=mode, return_dict=True)
            # post process
            timepoints += [rdata['t'] for rdata in ret['rdatas']]
            if 0 in sensi_orders:
                outputs += self.post_processing(ret['rdatas'])
            if 1 in sensi_orders:
                outputs_sensi += self.post_processing_sensi(ret['rdatas'])

    def _write_to_csv(self,
                      outputs: Sequence[np.ndarray],
                      outputs_sensi: Sequence[np.ndarray],
                      output_file: str,
                      timepoints: Sequence):
        """
        This method saves predictions from an amici model to a csv file.

        Parameters
        ----------
        outputs:
            List of postprocessed outputs (ndarrays), to which results will
            be appended
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays),
            to which results will be appended
        output_file:
            path to file/folder to which results will be written
        timepoints:
            List of output timepoints (ndarrays), to which simulation
            timepoints will be appended
        """
        def _prepare_csv_output(output_file):
            """
            If a csv is requested, this routine will create a folder for it,
            with a suiting name: csv's are by default 2-dimensional, but the
            output will have the format n_conditions x n_timepoints x n_outputs
            For sensitivities, we even have x n_parameters. This makes it
            necessary to create multiple files and hence, a folder of its own
            makes sense.
            """
            # allow entering with names with and without file type endings
            if '.' in output_file:
                output_file_path, output_file_suffix = output_file.split('.')
            else:
                output_file_path = output_file
                output_file_suffix = 'csv'

            # parse path
            if '/' in output_file_path:
                tmp = output_file_path.split('/')[-1]
            else:
                tmp = [output_file_path,]

            output_file_dummy = tmp[-1]
            output_path = os.path.join(*tmp)

            # create folder with files contianing the return values
            if os.path.exists(output_path):
                output_path += str(int(time() * 1000))
                warn('Output folder already existed! Changed the name of the '
                     'output folder by appending the unix timestampp to make '
                     'it unique!')
            os.mkdir(output_path)

            return output_path, output_file_dummy, output_file_suffix

        # process the name of the output file, create a folder
        output_path, output_file_dummy, output_file_suffix = \
            _prepare_csv_output(output_file)

        # get observables as column names
        observables = self.amici_objective.amici_model.getObservableIds()

        if outputs:
            # loop over conditions (i.e., amici edata objects)
            for i_out, output in enumerate(outputs):
                i_timepoints = timepoints[i_out]
                # create filename for this condition
                tmp_filename = os.path.join(output_path,
                    output_file_dummy + f'_{i_out}.' + output_file_suffix)
                # create DataFrame and write to file
                result = pd.DataFrame(index=i_timepoints, columns=observables,
                                      data=output)
                result.to_csv(tmp_filename, sep='\t')

        if outputs_sensi:
            # loop over conditions (i.e., amici edata objects)
            for i_out, output_sensi in enumerate(outputs_sensi):
                i_timepoints = timepoints[i_out]
                # loop over parameters
                for i_par in range(output_sensi[i_out].shape[1]):
                    # create filename for this condition and parameter
                    tmp_filename = os.path.join(output_path, output_file_dummy +
                                                f'_{i_out}__s{i_par}.' +
                                                output_file_suffix)
                    # create DataFrame and write to file
                    result = pd.DataFrame(index=i_timepoints,
                                          columns=observables,
                                          data=output_sensi[:,i_par,:])
                    result.to_csv(tmp_filename, sep='\t')

    def _write_to_h5(self,
                     outputs: Sequence[np.ndarray],
                     outputs_sensi: Sequence[np.ndarray],
                     output_file: str,
                     timepoints: Sequence):
        """
        This method saves predictions from an amici model to a h5 file.

        Parameters
        ----------
        outputs:
            List of postprocessed outputs (ndarrays), to which results will
            be appended
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays),
            to which results will be appended
        output_file:
            path to file/folder to which results will be written
        timepoints:
            List of output timepoints (ndarrays), to which simulation
            timepoints will be appended
        """
        observables = self.amici_objective.amici_model.getObservableIds()

        # create file
        with h5py.File(output_file, 'w') as f:
            # save observable IDs
            f.create_dataset('observableIds', data=observables)

            # loop over conditions (i.e., amici edata objects)
            n_groups = max(len(outputs), len(outputs_sensi))
            for i_out in range(n_groups):
                # each conditions gets a group of its own
                f.create_group(str(i_out))
                # save timepoints, outputs, and sensitivities of outputs
                f.create_dataset(f'{i_out}/timepoints', data=timepoints[i_out])
                if outputs:
                    f.create_dataset(f'{i_out}/sim', data=outputs[i_out])
                if outputs_sensi:
                    f.create_dataset(f'{i_out}/sim_sensi',
                                     data=outputs_sensi[i_out])

    @staticmethod
    def default_post_processing(rdatas) -> Sequence[np.ndarray]:
        """
        Default simulation output: Just amici observables
        """
        return [rdata['y'] for rdata in rdatas]

    @staticmethod
    def default_post_processing_sensi(rdatas) -> Sequence[np.ndarray]:
        """
        Default simulation output sensis: Just amici observable sensitivities
        """
        return [rdata['sy'] for rdata in rdatas]
