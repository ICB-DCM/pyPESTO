import numpy as np
import os
import pandas as pd
import h5py
from typing import Sequence, Union, Callable, Tuple
from warnings import warn
from time import time

from .constants import (MODE_FUN, OBSERVABLE_IDS, TIMEPOINTS, OUTPUT,
                        OUTPUT_SENSI, TIME, CSV, H5, T, Y, SY, RDATAS)
from .prediction import PredictionResult
from ..objective import AmiciObjective


class AmiciPredictor:
    """
    Do forward simulations (predictions) with parameter vectors,
    for an AMICI model. The user may supply post-processing methods.
    If post-processing methods are supplied, and a gradient of the prediction
    is requested, then the sensitivities of the AMICI model must also be
    post-processed. There are no checks here to ensure that the sensitivities
    are correctly post-processed, this is explicitly left to the user.
    """
    def __init__(self,
                 amici_objective: AmiciObjective,
                 post_processor: Union[Callable, None] = None,
                 post_processor_sensi: Union[Callable, None] = None,
                 post_processor_time: Union[Callable, None] = None,
                 max_num_conditions: int = 0,
                 observable_ids: Sequence[str] = None):
        """
        Constructor.

        Parameters
        ----------
        amici_objective:
            An objective object, which will be used to get model simulations
        post_processor:
            A callable function which applies postprocessing to the simulation
            results. Default are the observables of the amici model.
            This method takes a list of ndarrays (as returned in the field
            ['y'] of amici ReturnData objects) as input.
        post_processor_sensi:
            A callable function which applies postprocessing to the
            sensitivities of the simulation results. Default are the
            observable sensitivities of the amici model.
            This method takes two lists of ndarrays (as returned in the
            fields ['y'] and ['sy'] of amici ReturnData objects) as input.
        post_processor_time:
            A callable function which applies postprocessing to the timepoints
            of the simulations. Default are the timepoints of the amici model.
            This method takes a list of ndarrays (as returned in the field
            ['t'] of amici ReturnData objects) as input.
        max_num_conditions:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum number of conditions, which
            should be simulated at a time.
            Default is 0 meaning that all conditions will be simulated.
            Other values are only applicable, if an output file is specified.
        observable_ids:
            IDs of observables, if post-processing is used
        """
        # save settings and objective
        self.amici_objective = amici_objective
        self.max_num_conditions = max_num_conditions
        self.post_processor = post_processor
        self.post_processor_sensi = post_processor_sensi
        self.post_processor_time = post_processor_time

        if observable_ids is None:
            self.observable_ids = \
                amici_objective.amici_model.getObservableIds()
        else:
            self.observable_ids = observable_ids

    def __call__(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...] = (0, ),
            mode: str = MODE_FUN,
            output_file: str = '',
            output_format: str = CSV,
    ) -> PredictionResult:
        """
        Simulate a model for a certain prediction function.
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
        amici_y = []
        amici_sy = []
        amici_t = []

        # simulate the model and get the output
        self._get_model_outputs(amici_y, amici_sy, amici_t, x,
                                sensi_orders, mode)

        # postprocess
        outputs = amici_y
        outputs_sensi = amici_sy
        timepoints = amici_t
        if self.post_processor is not None:
            outputs = self.post_processor(outputs)
        if self.post_processor_sensi is not None:
            outputs_sensi = self.post_processor_sensi(amici_y, amici_sy)
        if self.post_processor_time is not None:
            timepoints = self.post_processor_time(amici_t)

        condition_results = []
        for i_cond in range(len(timepoints)):
            result = {TIMEPOINTS: timepoints[i_cond],
                      OBSERVABLE_IDS: self.observable_ids}
            if outputs:
                result[OUTPUT] = outputs[i_cond]
            if outputs_sensi:
                result[OUTPUT_SENSI] = outputs_sensi[i_cond]

            condition_results.append(result)

        results = PredictionResult(condition_results)

        # Should the results be saved to a file?
        if output_file:
            # Do we want a pandas dataframe like format?
            if output_format == CSV:
                self._write_to_csv(outputs=outputs,
                                   outputs_sensi=outputs_sensi,
                                   timepoints=timepoints,
                                   output_file=output_file)
            # Do we want an h5 file?
            elif output_format == H5:
                self._write_to_h5(outputs=outputs,
                                  outputs_sensi=outputs_sensi,
                                  timepoints=timepoints,
                                  output_file=output_file)
            else:
                raise Exception(f'Call to unknown format {output_format} for '
                                f'output of pyPESTO prediction.')

        # return dependent on sensitivity order
        return results

    def _get_model_outputs(self,
                           amici_y,
                           amici_sy,
                           amici_t,
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
        amici_y:
            List of raw outputs (ndarrays), to which results will be appended
        amici_sy:
            List of sensitivities of raw outputs (ndarrays), to which results
            will be appended
        amici_t:
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
        n_edatas = len(self.amici_objective.edatas)
        if self.max_num_conditions == 0:
            # simulate all conditions at once
            n_simulations = 1
        else:
            # simulate only a subset of conditions
            n_simulations = 1 + int(len(self.amici_objective.edatas) /
                                    self.max_num_conditions)

        for i_sim in range(n_simulations):
            # slice out the conditions we actually want
            if self.max_num_conditions == 0:
                ids = slice(0, n_edatas)
            else:
                ids = slice(i_sim * self.max_num_conditions,
                            min((i_sim + 1) * self.max_num_conditions,
                                n_edatas))

            # call amici
            ret = self.amici_objective(x=x, sensi_orders=sensi_orders,
                                       edatas=self.amici_objective.edatas[ids],
                                       mode=mode, return_dict=True)
            # post process
            amici_t += [rdata[T] for rdata in ret[RDATAS]]
            if 0 in sensi_orders:
                amici_y += [rdata[Y] for rdata in ret[RDATAS]]
            if 1 in sensi_orders:
                amici_sy += [rdata[SY] for rdata in ret[RDATAS]]

    def _write_to_csv(self,
                      outputs: Sequence[np.ndarray],
                      outputs_sensi: Sequence[np.ndarray],
                      timepoints: Sequence,
                      output_file: str):
        """
        This method saves predictions from an amici model to a csv file.

        Parameters
        ----------
        outputs:
            List of postprocessed outputs (ndarrays)
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays)
        timepoints:
            List of output timepoints (ndarrays)
        output_file:
            path to file/folder to which results will be written
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
                output_file_suffix = CSV

            # parse path
            if '/' in output_file_path:
                tmp = output_file_path.split('/')[-1]
            else:
                tmp = [output_file_path, ]

            output_file_dummy = tmp[-1]
            output_path = os.path.join(*tmp)

            # create folder with files contianing the return values
            if os.path.exists(output_path):
                output_path += '__' + str(int(time() * 1000))
                warn('Output folder already existed! Changed the name of the '
                     'output folder by appending the unix timestampp to make '
                     'it unique!')
            os.mkdir(output_path)

            return output_path, output_file_dummy, output_file_suffix

        # process the name of the output file, create a folder
        output_path, output_file_dummy, output_suffix = \
            _prepare_csv_output(output_file)

        if outputs:
            # loop over conditions (i.e., amici edata objects)
            for i_out, output in enumerate(outputs):
                i_timepoints = pd.Series(name=TIME, data=timepoints[i_out])
                # create filename for this condition
                tmp_filename = output_file_dummy + f'_{i_out}.' + output_suffix
                tmp_filename = os.path.join(output_path, tmp_filename)
                # create DataFrame and write to file
                result = pd.DataFrame(index=i_timepoints,
                                      columns=self.observable_ids,
                                      data=output)
                result.to_csv(tmp_filename, sep='\t')

        if outputs_sensi:
            # loop over conditions (i.e., amici edata objects)
            for i_out, output_sensi in enumerate(outputs_sensi):
                i_timepoints = pd.Series(name=TIME, data=timepoints[i_out])
                # loop over parameters
                for i_par in range(output_sensi[i_out].shape[0]):
                    # create filename for this condition and parameter
                    tmp_filename = output_file_dummy + f'_{i_out}__s{i_par}.' \
                                   + output_suffix
                    tmp_filename = os.path.join(output_path, tmp_filename)
                    # create DataFrame and write to file
                    result = pd.DataFrame(index=i_timepoints,
                                          columns=self.observable_ids,
                                          data=output_sensi[:, i_par, :])
                    result.to_csv(tmp_filename, sep='\t')

    def _write_to_h5(self,
                     outputs: Sequence[np.ndarray],
                     outputs_sensi: Sequence[np.ndarray],
                     timepoints: Sequence,
                     output_file: str):
        """
        This method saves predictions from an amici model to a h5 file.

        Parameters
        ----------
        outputs:
            List of postprocessed outputs (ndarrays)
        outputs_sensi:
            List of sensitivities of postprocessed outputs (ndarrays)
        timepoints:
            List of output timepoints (ndarrays)
        output_file:
            path to file/folder to which results will be written
        """
        if os.path.exists(output_file):
            tmp = output_file.split('.')
            file_name = '.'.join(tmp[:-1]) + '__' + str(int(time() * 1000))
            output_file = file_name + '.' + tmp[-1]
            warn('Output folder already existed! Changed the name of the '
                 'output folder by appending the unix timestampp to make '
                 'it unique!')

        with h5py.File(output_file, 'w') as f:
            # save observable IDs
            f.create_dataset(OBSERVABLE_IDS, data=self.observable_ids)

            # loop over conditions (i.e., amici edata objects)
            n_groups = max(len(outputs), len(outputs_sensi))
            for i_out in range(n_groups):
                # each conditions gets a group of its own
                f.create_group(str(i_out))
                # save timepoints, outputs, and sensitivities of outputs
                f.create_dataset(f'{i_out}/{TIMEPOINTS}',
                                 data=timepoints[i_out])
                if outputs:
                    f.create_dataset(f'{i_out}/{OUTPUT}', data=outputs[i_out])
                if outputs_sensi:
                    f.create_dataset(f'{i_out}/{OUTPUT_SENSI}',
                                     data=outputs_sensi[i_out])
