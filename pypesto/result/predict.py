"""PredictionResult and PredictionConditionResult."""

import os
from collections.abc import Sequence
from pathlib import Path
from time import time
from typing import Union
from warnings import warn

import h5py
import numpy as np
import pandas as pd

from ..C import (
    CONDITION_IDS,
    CSV,
    OUTPUT,
    OUTPUT_IDS,
    OUTPUT_SENSI,
    OUTPUT_SIGMAY,
    OUTPUT_WEIGHT,
    PARAMETER_IDS,
    TIME,
    TIMEPOINTS,
)
from ..util import get_condition_label


class PredictionConditionResult:
    """
    Light-weight wrapper for the prediction of one simulation condition.

    It should provide a common api how amici predictions should look like in
    pyPESTO.
    """

    def __init__(
        self,
        timepoints: np.ndarray,
        output_ids: Sequence[str],
        output: np.ndarray = None,
        output_sensi: np.ndarray = None,
        output_weight: float = None,
        output_sigmay: np.ndarray = None,
        x_names: Sequence[str] = None,
    ):
        """
        Initialize PredictionConditionResult.

        Parameters
        ----------
        timepoints:
            Output timepoints for this simulation condition
        output_ids:
            IDs of outputs for this simulation condition
        output:
            Postprocessed outputs (ndarray)
        output_sensi:
            Sensitivities of postprocessed outputs (ndarray)
        output_weight:
            LLH of the simulation
        output_sigmay:
            Standard deviations of postprocessed observables
        x_names:
            IDs of model parameter w.r.t to which sensitivities were computed
        """
        self.timepoints = timepoints
        self.output_ids = output_ids
        self.output = output
        self.output_sensi = output_sensi
        self.output_weight = output_weight
        self.output_sigmay = output_sigmay
        self.x_names = x_names
        if x_names is None and output_sensi is not None:
            self.x_names = [
                f"parameter_{i_par}" for i_par in range(output_sensi.shape[1])
            ]

    def __iter__(self):
        """Allow usage like a dict."""
        yield "timepoints", self.timepoints
        yield "output_ids", self.output_ids
        yield "x_names", self.x_names
        yield "output", self.output
        yield "output_sensi", self.output_sensi
        yield "output_weight", self.output_weight
        yield "output_sigmay", self.output_sigmay

    def __eq__(self, other):
        """Check equality of two PredictionConditionResults."""

        def to_bool(expr):
            if isinstance(expr, bool):
                return expr
            return expr.any()

        if to_bool(self.timepoints != other.timepoints):
            return False
        if to_bool(self.x_names != other.x_names):
            return False
        if to_bool(self.output_ids != other.output_ids):
            return False
        if to_bool(self.output != other.output):
            return False
        if to_bool(self.output_sensi != other.output_sensi):
            return False
        if to_bool(self.output_weight != other.output_weight):
            return False
        if to_bool(self.output_sigmay != other.output_sigmay):
            return False
        return True


class PredictionResult:
    """
    Light-weight wrapper around prediction from pyPESTO made by an AMICI model.

    Its only purpose is to have fixed format/api, how prediction results
    should be stored, read, and handled: as predictions are a very flexible
    format anyway, they should at least have a common definition,
    which allows to work with them in a reasonable way.
    """

    def __init__(
        self,
        conditions: Sequence[Union[PredictionConditionResult, dict]],
        condition_ids: Sequence[str] = None,
        comment: str = None,
    ):
        """
        Initialize PredictionResult.

        Parameters
        ----------
        conditions:
            A list of PredictionConditionResult objects or dicts
        condition_ids:
            IDs or names of the simulation conditions, which belong to this
            prediction (e.g., PEtab uses tuples of preequilibration condition
            and simulation conditions)
        comment:
            An additional note, which can be attached to this prediction
        """
        # cast the result per condition
        self.conditions = [
            (
                cond
                if isinstance(cond, PredictionConditionResult)
                else PredictionConditionResult(**cond)
            )
            for cond in conditions
        ]

        self.condition_ids = condition_ids
        if self.condition_ids is None:
            self.condition_ids = [
                get_condition_label(i_cond)
                for i_cond in range(len(conditions))
            ]

        # add a comment to this prediction if available
        self.comment = comment

    def __iter__(self):
        """Allow usage like an iterator."""
        parameter_ids = None
        if self.conditions:
            parameter_ids = self.conditions[0].x_names

        yield "conditions", [dict(cond) for cond in self.conditions]
        yield "condition_ids", self.condition_ids
        yield "comment", self.comment
        yield "parameter_ids", parameter_ids

    def __eq__(self, other):
        """Check equality of two PredictionResults."""
        if not isinstance(other, PredictionResult):
            return False
        if self.comment != other.comment:
            return False
        if self.condition_ids != other.condition_ids:
            return False
        for i_cond, _ in enumerate(self.conditions):
            if self.conditions[i_cond] != other.conditions[i_cond]:
                return False
        return True

    def write_to_csv(self, output_file: str):
        """
        Save predictions to a csv file.

        Parameters
        ----------
        output_file:
            path to file/folder to which results will be written
        """

        def _prepare_csv_output(output_file):
            """
            Prepare a folder for output.

            If a csv is requested, this routine will create a folder for it,
            with a suiting name: csv's are by default 2-dimensional, but the
            output will have the format n_conditions x n_timepoints x n_outputs
            For sensitivities, we even have x n_parameters. This makes it
            necessary to create multiple files and hence, a folder of its own
            makes sense. Returns a pathlib.Path object of the output.
            """
            # allow entering with names with and without file type endings
            if "." in output_file:
                output_path, output_suffix = output_file.split(".")
            else:
                output_path = output_file
                output_suffix = CSV

            # parse path and check whether the file exists
            output_path = Path(output_path)
            output_path = self._check_existence(output_path)

            # create
            output_path.mkdir(parents=True, exist_ok=False)
            # add the suffix
            output_dummy = Path(output_path.stem).with_suffix(
                f".{output_suffix}"
            )

            return output_path, output_dummy

        # process the name of the output file, create a folder
        output_path, output_dummy = _prepare_csv_output(output_file)

        # loop over conditions (i.e., amici edata objects)
        for i_cond, cond in enumerate(self.conditions):
            timepoints = pd.Series(name=TIME, data=cond.timepoints)
            # handle outputs, if computed
            if cond.output is not None:
                # create filename for this condition
                filename = output_path.joinpath(
                    output_dummy.stem + f"_{i_cond}" + output_dummy.suffix
                )
                # create DataFrame and write to file
                result = pd.DataFrame(
                    index=timepoints, columns=cond.output_ids, data=cond.output
                )
                result.to_csv(filename, sep="\t")

            # handle output sensitivities, if computed
            if cond.output_sensi is not None:
                # loop over parameters
                for i_par in range(cond.output_sensi.shape[1]):
                    # create filename for this condition and parameter
                    filename = output_path.joinpath(
                        output_dummy.stem
                        + f"_{i_cond}__s{i_par}"
                        + output_dummy.suffix
                    )
                    # create DataFrame and write to file
                    result = pd.DataFrame(
                        index=timepoints,
                        columns=cond.output_ids,
                        data=cond.output_sensi[:, i_par, :],
                    )
                    result.to_csv(filename, sep="\t")

    def write_to_h5(self, output_file: str, base_path: str = None):
        """
        Save predictions to an h5 file.

        It appends to the file if the file already exists.

        Parameters
        ----------
        output_file:
            path to file/folder to which results will be written
        base_path:
            base path in the h5 file
        """
        # check if the file exists and append to it in case it does
        output_path = Path(output_file)
        filemode = "w"
        if os.path.exists(output_path):
            filemode = "r+"

        base = Path(".")
        if base_path is not None:
            base = Path(base_path)

        with h5py.File(output_path, filemode) as f:
            # loop over conditions (i.e., amici edata objects)
            if self.conditions and self.conditions[0].x_names is not None:
                f.create_dataset(
                    os.path.join(base, PARAMETER_IDS),
                    data=self.conditions[0].x_names,
                )
            if self.condition_ids is not None:
                f.create_dataset(
                    os.path.join(base, CONDITION_IDS), data=self.condition_ids
                )
            for i_cond, cond in enumerate(self.conditions):
                # each conditions gets a group of its own
                f.create_group(os.path.join(base, str(i_cond)))
                # save output IDs
                f.create_dataset(
                    os.path.join(base, str(i_cond), OUTPUT_IDS),
                    data=cond.output_ids,
                )
                # save timepoints, outputs, and sensitivities of outputs
                f.create_dataset(
                    os.path.join(base, str(i_cond), TIMEPOINTS),
                    data=cond.timepoints,
                )
                if cond.output is not None:
                    f.create_dataset(
                        os.path.join(base, str(i_cond), OUTPUT),
                        data=cond.output,
                    )
                if cond.output_sensi is not None:
                    f.create_dataset(
                        os.path.join(base, str(i_cond), OUTPUT_SENSI),
                        data=cond.output_sensi,
                    )
                if cond.output_weight is not None:
                    f.create_dataset(
                        os.path.join(base, str(i_cond), OUTPUT_WEIGHT),
                        data=cond.output_weight,
                    )
                if cond.output_sigmay is not None:
                    f.create_dataset(
                        os.path.join(base, str(i_cond), OUTPUT_SIGMAY),
                        data=cond.output_sigmay,
                    )

    @staticmethod
    def _check_existence(output_path):
        """
        Check whether a file or a folder already exists.

        Append a timestamp if this is the case.
        """
        output_path_out = output_path
        while output_path_out.exists():
            output_path_out = output_path_out.with_name(
                output_path_out.stem + f"_{round(time() * 1000)}"
            )
            warn(
                "Output name already existed! Changed the name of the output "
                "by appending the unix timestamp to make it unique!",
                stacklevel=3,
            )

        return output_path_out
