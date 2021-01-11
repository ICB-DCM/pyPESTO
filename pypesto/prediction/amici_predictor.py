import numpy as np
from typing import Sequence, Union, Callable, Tuple

from .constants import (MODE_FUN, OBSERVABLE_IDS, TIMEPOINTS, OUTPUT,
                        OUTPUT_SENSI, CSV, H5, T, Y, SY, RDATAS)
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
                 max_chunk_size: Union[int, None] = None,
                 observable_ids: Union[Sequence[str], None] = None):
        """
        Constructor.

        Parameters
        ----------
        amici_objective:
            An objective object, which will be used to get model simulations
        post_processor:
            A callable function which applies postprocessing to the simulation
            results and possibly defines different observables than those of
            the amici model. Default are the observables of the amici model.
            This method takes a list of dicts (with the returned fields ['t'],
            ['x'], and ['y'] of the amici ReturnData objects) as input.
        post_processor_sensi:
            A callable function which applies postprocessing to the
            sensitivities of the simulation results. Defaults to the
            observable sensitivities of the amici model.
            This method takes a list of dicts (with the returned fields ['t'],
            ['x'], ['y'], ['sx'], and ['sy'] of the amici ReturnData objects)
            as input.
        post_processor_time:
            A callable function which applies postprocessing to the timepoints
            of the simulations. Defaults to the timepoints of the amici model.
            This method takes a list of dicts (with the returned field ['t'] of
            the amici ReturnData objects) as input.
        max_chunk_size:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum chunk size of conditions,
            which should be simulated at a time.
            Defaults to None, meaning that all conditions will be simulated.
        observable_ids:
            IDs of observables, as post-processing allows the creation of
            customizable observables, which may not coincide with those from
            the amici model (defaults to amici observables).
        """
        # save settings and objective
        self.amici_objective = amici_objective
        self.max_chunk_size = max_chunk_size
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
        allows the user to apply any post-processing of the results, the
        sensitivities, and the timepoints.

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
        results:
            PredictionResult object containing timepoints, outputs, and
            output_sensitivities if requested
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
                results.write_to_csv(output_file=output_file)
            # Do we want an h5 file?
            elif output_format == H5:
                results.write_to_h5(output_file=output_file)
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
        if self.max_chunk_size is None:
            # simulate all conditions at once
            n_simulations = 1
        else:
            # simulate only a subset of conditions
            n_simulations = 1 + int(len(self.amici_objective.edatas) /
                                    self.max_chunk_size)

        for i_sim in range(n_simulations):
            # slice out the conditions we actually want
            if self.max_chunk_size is None:
                ids = slice(0, n_edatas)
            else:
                ids = slice(i_sim * self.max_chunk_size,
                            min((i_sim + 1) * self.max_chunk_size,
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
