from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..C import (
    AMICI_LLH,
    AMICI_SIGMAY,
    AMICI_STATUS,
    AMICI_SX,
    AMICI_SY,
    AMICI_T,
    AMICI_X,
    AMICI_Y,
    CSV,
    H5,
    MODE_FUN,
    OUTPUT,
    OUTPUT_IDS,
    OUTPUT_SENSI,
    OUTPUT_SIGMAY,
    OUTPUT_WEIGHT,
    PARAMETER_IDS,
    RDATAS,
    TIMEPOINTS,
    ModeType,
)

if TYPE_CHECKING:
    from ..objective import AmiciObjective

from ..result import PredictionResult

PostProcessor = Callable[[list[dict[str, np.array]]], list[np.ndarray]]


class AmiciPredictor:
    """
    Do forward simulations/predictions for an AMICI model.

    The user may supply post-processing methods.
    If post-processing methods are supplied, and a gradient of the prediction
    is requested, then the sensitivities of the AMICI model must also be
    post-processed. There are no checks here to ensure that the sensitivities
    are correctly post-processed, this is explicitly left to the user.
    There are also no safeguards if the postprocessor routines fail. This may
    happen if, e.g., a call to AMICI fails, and no timepoints, states or
    observables are returned. As the AmiciPredictor is agnostic about the
    dimension of the postprocessor and also the dimension of the postprocessed
    output, these checks are also left to the user. An example for such a check
    is provided in the default output (see :meth:`_default_output()`).
    """

    def __init__(
        self,
        amici_objective: AmiciObjective,
        amici_output_fields: Sequence[str] | None = None,
        post_processor: PostProcessor | None = None,
        post_processor_sensi: PostProcessor | None = None,
        post_processor_time: PostProcessor | None = None,
        max_chunk_size: int | None = None,
        output_ids: Sequence[str] | None = None,
        condition_ids: Sequence[str] | None = None,
    ):
        """
        Initialize predictor.

        Parameters
        ----------
        amici_objective:
            An objective object, which will be used to get model simulations
        amici_output_fields:
            keys that exist in the return data object from AMICI, which should
            be available for the post-processors
        post_processor:
            A callable function which applies postprocessing to the simulation
            results and possibly defines different outputs than those of
            the amici model. Default are the observables
            (`pypesto.C.AMICI_Y`) of the AMICI model. This
            method takes a list of dicts (with the returned
            fields `pypesto.C.AMICI_T`,
            `pypesto.C.AMICI_X`, and
            `pypesto.C.AMICI_Y` of the AMICI ReturnData
            objects) as input. Safeguards for, e.g., failure of AMICI are left
            to the user.
        post_processor_sensi:
            A callable function which applies postprocessing to the
            sensitivities of the simulation results. Defaults to the
            observable sensitivities of the AMICI model.
            This method takes a list of dicts (with the returned fields
            `pypesto.C.AMICI_T`,
            `pypesto.C.AMICI_X`,
            `pypesto.C.AMICI_Y`,
            `pypesto.C.AMICI_SX`, and
            `pypesto.C.AMICI_SY` of the AMICI ReturnData
            objects) as input. Safeguards for, e.g., failure of AMICI are left
            to the user.
        post_processor_time:
            A callable function which applies postprocessing to the timepoints
            of the simulations. Defaults to the timepoints of the amici model.
            This method takes a list of dicts (with the returned field
            `pypesto.C.AMICI_T` of the amici ReturnData
            objects) as input. Safeguards for, e.g., failure of AMICI are left
            to the user.
        max_chunk_size:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum chunk size of conditions,
            which should be simulated at a time.
            Defaults to None, meaning that all conditions will be simulated.
        output_ids:
            IDs of outputs, as post-processing allows the creation of
            customizable outputs, which may not coincide with those from
            the AMICI model (defaults to AMICI observables).
        condition_ids:
            List of identifiers for the conditions of the edata objects of the
            amici objective, will be passed to the :class:`PredictionResult` at call.
        """
        # save settings and objective
        self.amici_objective = amici_objective
        self.max_chunk_size = max_chunk_size
        self.post_processor = post_processor
        self.post_processor_sensi = post_processor_sensi
        self.post_processor_time = post_processor_time
        self.condition_ids = condition_ids

        # If the user takes care of everything we can skip default readouts
        self.skip_default_outputs = False
        if (
            post_processor is not None
            and post_processor_sensi is not None
            and post_processor_time is not None
        ):
            self.skip_default_outputs = True

        self.output_ids = output_ids
        if output_ids is None:
            self.output_ids = amici_objective.amici_model.getObservableIds()

        if amici_output_fields is None:
            amici_output_fields = [
                AMICI_STATUS,
                AMICI_T,
                AMICI_Y,
                AMICI_SY,
                AMICI_X,
                AMICI_SX,
            ]
        self.amici_output_fields = amici_output_fields

    def __call__(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...] = (0,),
        mode: ModeType = MODE_FUN,
        output_file: str = "",
        output_format: str = CSV,
        include_llh_weights: bool = False,
        include_sigmay: bool = False,
    ) -> PredictionResult:
        """
        Call the predictor.

        Simulate a model for a certain prediction function.
        This method relies on the :class:`AmiciObjective`, which is underlying, but
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
            Either ``'csv'`` or ``'h5'``. If an output file is specified, this routine
            will return a csv file, created from a DataFrame, or an h5 file,
            created from a dict.
        include_llh_weights:
            Boolean whether weights should be included in the prediction.
            Necessary for weighted means of Ensembles.
        include_sigmay:
            Boolean whether standard deviations should be included in the
            prediction output. Necessary for evaluation of weighted means
            of Ensembles.

        Returns
        -------
        PredictionResult object containing timepoints, outputs, and
        output sensitivities if requested.
        """
        # sanity check for output
        if 2 in sensi_orders:
            raise Exception(
                "Prediction simulation does currently not support "
                "second order output."
            )
        # add llh and sigmay to amici output fields if requested
        if include_llh_weights and AMICI_LLH not in self.amici_output_fields:
            self.amici_output_fields.append(AMICI_LLH)
        if include_sigmay and AMICI_SIGMAY not in self.amici_output_fields:
            self.amici_output_fields.append(AMICI_SIGMAY)

        # simulate the model and get the output
        (
            timepoints,
            outputs,
            outputs_sensi,
            outputs_weight,
            outputs_sigmay,
        ) = self._get_outputs(
            x, sensi_orders, mode, include_llh_weights, include_sigmay
        )

        # group results by condition, prepare PredictionConditionResult output
        condition_results = []
        # timepoints, outputs, outputs_sensi, outputs_sigmay and
        # outputs_weight are lists with the number of simulation conditions.
        # While everything else is optional, timepoints must exist,
        # so we use this as a dummy
        n_cond = len(timepoints)
        for i_cond in range(n_cond):
            result = {
                TIMEPOINTS: timepoints[i_cond],
                OUTPUT_IDS: self.output_ids,
                PARAMETER_IDS: self.amici_objective.x_names,
            }
            if outputs:
                result[OUTPUT] = outputs[i_cond]
            if outputs_sensi:
                result[OUTPUT_SENSI] = outputs_sensi[i_cond]
            if outputs_weight:
                result[OUTPUT_WEIGHT] = outputs_weight[i_cond]
            if outputs_sigmay:
                result[OUTPUT_SIGMAY] = outputs_sigmay[i_cond]

            condition_results.append(result)
        # create result object
        results = PredictionResult(
            condition_results, condition_ids=self.condition_ids
        )

        # Should the results be saved to a file?
        if output_file:
            # Do we want a pandas dataframe like format?
            if output_format == CSV:
                results.write_to_csv(output_file=output_file)
            # Do we want an h5 file?
            elif output_format == H5:
                results.write_to_h5(output_file=output_file)
            else:
                raise ValueError(
                    f"Call to unknown format {output_format} for "
                    f"output of pyPESTO prediction."
                )

        # return dependent on sensitivity order
        return results

    def _get_outputs(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType = MODE_FUN,
        include_llh_weights: bool = False,
        include_sigmay: bool = False,
    ) -> tuple[
        list[np.array],
        list[np.array],
        list[np.array],
        list[np.array],
        list[np.array],
    ]:
        """
        Split the calls to amici into smaller chunks.

        Too large ReturnData objects from amici including many simulations
        can be problematic in terms of memory.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the prediction function.
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.

        Returns
        -------
        timepoints:
            List of np.ndarrays, every entry includes the output timepoints of
            the respective condition
        outputs:
            List of np.ndarrays, every entry includes the postprocessed outputs
            of the respective condition
        outputs_sensi:
            List of np.ndarrays, every entry includes the postprocessed output
            sensitivities of the respective condition
        include_llh_weights:
            Boolean whether weights should be included in the prediction.
            Necessary for weighted means of Ensembles.
        include_sigmay:
            Boolean whether standard deviations should be included in the
            prediction output. Necessary for evaluation of weighted means
            of Ensembles.
        """
        # Do we have a maximum number of simulations allowed?
        n_edatas = len(self.amici_objective.edatas)
        if self.max_chunk_size is None:
            # simulate all conditions at once
            n_simulations = 1
        else:
            # simulate only a subset of conditions
            n_simulations = int(np.ceil(n_edatas / self.max_chunk_size))

        # prepare result
        amici_outputs = []

        for i_sim in range(n_simulations):
            # slice out the conditions we actually want
            if self.max_chunk_size is None:
                ids = slice(0, n_edatas)
            else:
                ids = slice(
                    i_sim * self.max_chunk_size,
                    min((i_sim + 1) * self.max_chunk_size, n_edatas),
                )

            # call amici
            self._wrap_call_to_amici(
                amici_outputs=amici_outputs,
                x=x,
                sensi_orders=sensi_orders,
                parameter_mapping=self.amici_objective.parameter_mapping[ids],
                edatas=self.amici_objective.edatas[ids],
                mode=mode,
            )

        def _default_output(
            amici_outputs: list[dict[str, np.array]],
        ) -> tuple[
            list[np.array],
            list[np.array],
            list[np.array],
            list[np.array],
            list[np.array],
        ]:
            """
            Create default output of prediction.

            Equals to observables of AMICI model. We need to check that call
            to AMICI was successful (``status == 0``), before writing the output.
            """
            amici_nt = [
                len(edata.getTimepoints())
                for edata in self.amici_objective.edatas
            ]
            amici_ny = len(self.output_ids)
            amici_np = len(self.amici_objective.x_names)

            outputs = []
            outputs_sensi = []
            outputs_weights = []
            outputs_sigmay = []
            timepoints = [
                (
                    amici_output[AMICI_T]
                    if amici_output[AMICI_STATUS] == 0
                    else np.full((amici_nt[i_condition],), np.nan)
                )
                for i_condition, amici_output in enumerate(amici_outputs)
            ]
            # add outputs and sensitivities if requested
            if 0 in sensi_orders:
                outputs = [
                    (
                        amici_output[AMICI_Y]
                        if amici_output[AMICI_STATUS] == 0
                        else np.full((amici_nt[i_condition], amici_ny), np.nan)
                    )
                    for i_condition, amici_output in enumerate(amici_outputs)
                ]
            if 1 in sensi_orders:
                outputs_sensi = [
                    (
                        amici_output[AMICI_SY]
                        if amici_output[AMICI_STATUS] == 0
                        else np.full(
                            (amici_nt[i_condition], amici_np, amici_ny), np.nan
                        )
                    )
                    for i_condition, amici_output in enumerate(amici_outputs)
                ]
            # add likelihood as weights if requested
            if include_llh_weights:
                outputs_weights = [
                    (
                        amici_output[AMICI_LLH]
                        if amici_output[AMICI_STATUS] == 0
                        else np.nan
                    )
                    for i_condition, amici_output in enumerate(amici_outputs)
                ]
            # add standard deviations if requested
            if include_sigmay:
                outputs_sigmay = [
                    (
                        amici_output[AMICI_SIGMAY]
                        if amici_output[AMICI_STATUS] == 0
                        else np.full((1, amici_ny), np.nan)
                    )
                    for i_condition, amici_output in enumerate(amici_outputs)
                ]

            return (
                timepoints,
                outputs,
                outputs_sensi,
                outputs_weights,
                outputs_sigmay,
            )

        outputs_weights = []
        outputs_sigmay = []
        # Get default output
        if not self.skip_default_outputs:
            (
                timepoints,
                outputs,
                outputs_sensi,
                outputs_weights,
                outputs_sigmay,
            ) = _default_output(amici_outputs)

        # postprocess (use original Amici outputs)
        if self.post_processor is not None:
            outputs = self.post_processor(amici_outputs)
        if self.post_processor_sensi is not None:
            outputs_sensi = self.post_processor_sensi(amici_outputs)
        if self.post_processor_time is not None:
            timepoints = self.post_processor_time(amici_outputs)

        return (
            timepoints,
            outputs,
            outputs_sensi,
            outputs_weights,
            outputs_sigmay,
        )

    def _wrap_call_to_amici(
        self,
        amici_outputs: list,
        x: np.array,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        parameter_mapping,
        edatas,
    ) -> None:
        """
        Encapsulate the call to amici.

        This allows to use variable scoping as a mean to clean up the memory
        after calling amici, which is beneficial if large models with large
        datasets are used.
        """
        chunk = self.amici_objective(
            x=x,
            sensi_orders=sensi_orders,
            mode=mode,
            parameter_mapping=parameter_mapping,
            edatas=edatas,
            return_dict=True,
        )
        for rdata in chunk[RDATAS]:
            amici_outputs.append(
                {
                    output_field: deepcopy(rdata[output_field])
                    for output_field in self.amici_output_fields
                }
            )
        del chunk
