import numpy as np
import pandas as pd
import h5py
from warnings import warn
from time import time
from typing import Sequence, Union, Dict
from pathlib import Path
import os

from .constants import (
    get_condition_label,
    CONDITION_IDS,
    CSV,
    OUTPUT,
    OUTPUT_IDS,
    OUTPUT_SENSI,
    PARAMETER_IDS,
    TIME,
    TIMEPOINTS,
)


class PredictionConditionResult:
    """
    This class is a light-weight wrapper for the prediction of one simulation
    condition of an amici model. It should provide a common api how amici
    predictions should look like in pyPESTO.
    """

    def __init__(self,
                 timepoints: np.ndarray,
                 output_ids: Sequence[str],
                 output: np.ndarray = None,
                 output_sensi: np.ndarray = None,
                 x_names: Sequence[str] = None):
        """
        Constructor.

        Parameters
        ----------
        timepoints:
            Output timepoints for this simulation condition
        output_ids:
            IDs of outputs for this simulation condition
        outputs:
            Postprocessed outputs (ndarray)
        outputs_sensi:
            Sensitivities of postprocessed outputs (ndarray)
        x_names:
            IDs of model parameter w.r.t to which sensitivities were computed
        """
        self.timepoints = timepoints
        self.output_ids = output_ids
        self.output = output
        self.output_sensi = output_sensi
        self.x_names = x_names
        if x_names is None and output_sensi is not None:
            self.x_names = [f'parameter_{i_par}' for i_par in
                            range(output_sensi.shape[1])]

    def __iter__(self):
        yield 'timepoints', self.timepoints
        yield 'output_ids', self.output_ids
        yield 'x_names', self.x_names
        yield 'output', self.output
        yield 'output_sensi', self.output_sensi


class PredictionResult:
    """
    This class is a light-weight wrapper around predictions from pyPESTO made
    via an amici model. It's only purpose is to have fixed format/api, how
    prediction results should be stored, read, and handled: as predictions are
    a very flexible format anyway, they should at least have a common
    definition, which allows to work with them in a reasonable way.
    """

    def __init__(self,
                 conditions: Sequence[Union[PredictionConditionResult, Dict]],
                 condition_ids: Sequence[str] = None,
                 comment: str = None):
        """
        Constructor.

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
        self.conditions = [cond if isinstance(cond, PredictionConditionResult)
                           else PredictionConditionResult(**cond)
                           for cond in conditions]

        self.condition_ids = condition_ids
        if self.condition_ids is None:
            self.condition_ids = [get_condition_label(i_cond)
                                  for i_cond in range(len(conditions))]

        # add a comment to this prediction if available
        self.comment = comment

    def __iter__(self):
        parameter_ids = None
        if self.conditions:
            parameter_ids = self.conditions[0].x_names

        yield 'conditions', [dict(cond) for cond in self.conditions]
        yield 'condition_ids', self.condition_ids
        yield 'comment', self.comment
        yield 'parameter_ids', parameter_ids

    def write_to_csv(self, output_file: str):
        """
        This method saves predictions to a csv file.

        Parameters
        ----------
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
            makes sense. Returns a pathlib.Path object of the output.
            """
            # allow entering with names with and without file type endings
            if '.' in output_file:
                output_path, output_suffix = output_file.split('.')
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
                f'.{output_suffix}')

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
                    output_dummy.stem + f'_{i_cond}' + output_dummy.suffix)
                # create DataFrame and write to file
                result = pd.DataFrame(index=timepoints,
                                      columns=cond.output_ids,
                                      data=cond.output)
                result.to_csv(filename, sep='\t')

            # handle output sensitivities, if computed
            if cond.output_sensi is not None:
                # loop over parameters
                for i_par in range(cond.output_sensi.shape[1]):
                    # create filename for this condition and parameter
                    filename = output_path.joinpath(
                        output_dummy.stem + f'_{i_cond}__s{i_par}' +
                        output_dummy.suffix)
                    # create DataFrame and write to file
                    result = pd.DataFrame(index=timepoints,
                                          columns=cond.output_ids,
                                          data=cond.output_sensi[:, i_par, :])
                    result.to_csv(filename, sep='\t')

    def write_to_h5(self,
                    output_file: str,
                    base_path: str = None):
        """
        This method saves predictions to an h5 file. It appends to the file if
        the file already exists.

        Parameters
        ----------
        output_file:
            path to file/folder to which results will be written

        base_path:
            base path in the h5 file
        """
        # check if the file exists and append to it in case it does
        output_path = Path(output_file).with_suffix('.h5')
        filemode = 'w'
        if os.path.exists(output_path):
            filemode = 'r+'

        base = Path('.')
        if base_path is not None:
            base = Path(base_path)

        with h5py.File(output_path, filemode) as f:
            # loop over conditions (i.e., amici edata objects)
            if self.conditions and self.conditions[0].x_names is not None:
                f.create_dataset(os.path.join(base, PARAMETER_IDS),
                                 data=self.conditions[0].x_names)
            if self.condition_ids is not None:
                f.create_dataset(os.path.join(base, CONDITION_IDS),
                                 data=self.condition_ids)
            for i_cond, cond in enumerate(self.conditions):
                # each conditions gets a group of its own
                f.create_group(os.path.join(base, str(i_cond)))
                # save output IDs
                f.create_dataset(os.path.join(base, str(i_cond),
                                              OUTPUT_IDS),
                                 data=cond.output_ids)
                # save timepoints, outputs, and sensitivities of outputs
                f.create_dataset(os.path.join(base, str(i_cond), TIMEPOINTS),
                                 data=cond.timepoints)
                if cond.output is not None:
                    f.create_dataset(os.path.join(base, str(i_cond), OUTPUT),
                                     data=cond.output)
                if cond.output_sensi is not None:
                    f.create_dataset(os.path.join(base, str(i_cond),
                                                  OUTPUT_SENSI),
                                     data=cond.output_sensi)

    @staticmethod
    def _check_existence(output_path):
        """
        Checks whether a file or a folder already exists and appends a
        timestamp if this is the case
        """
        output_path_out = output_path
        while output_path_out.exists():
            output_path_out = output_path_out.with_name(
                output_path_out.stem + f'_{round(time() * 1000)}')
            warn('Output name already existed! Changed the name of the output '
                 'by appending the unix timestampp to make it unique!')

        return output_path_out
