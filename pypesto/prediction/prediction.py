import numpy as np
from typing import Sequence, Union, Dict


class PredictionConditionResult(dict):
    """
    This class is a light-weight wrapper for the prediction of one simulation
    condition of an amici model. It should provide a common api how amici
    predictions should look like in pyPESTO.
    """
    def __init__(self,
                 timepoints: np.ndarray,
                 observable_ids: Sequence[str],
                 output: np.ndarray = None,
                 output_sensi: np.ndarray = None,
                 x_names: Sequence[str] = None):
        """
        Constructor.

        Parameters
        ----------
        timepoints:
            Output timepoints for this simulation condition
        observable_ids:
            IDs of observables for this simulation condition
        outputs:
            Postprocessed outputs (ndarray)
        outputs_sensi:
            Sensitivities of postprocessed outputs (ndarray)
        x_names:
            IDs of model parameter w.r.t to which sensitivities were computed
        """
        self.timepoints = timepoints
        self.observable_ids = observable_ids
        self.output = output
        self.output_sensi = output_sensi
        self.x_names = x_names
        if x_names is None and output_sensi is not None:
            self.x_names = [f'parameter_{i_par}' for i_par in
                            range(output_sensi.shape[1])]

        super().__init__()


class PredictionResult(dict):
    """
    This class is a light-weight wrapper around predictions from pyPESTO made
    via an amici model. It's only purpose is to have fixed format/api, how
    prediction results should be stored, read, and handled: as predictions are
    a very flexible format anyway, they should at least have a common
    definition, which allows to work with them in a reasonable way.
    """
    def __init__(self,
                 conditions: Sequence[Union[PredictionConditionResult, Dict]],
                 condition_ids: Sequence[str] = None):
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
        """
        # cast the result per condition
        self.conditions = [cond if isinstance(cond, PredictionConditionResult)
                           else PredictionConditionResult(**cond)
                           for cond in conditions]

        self.condition_ids = condition_ids
        if self.condition_ids is None:
            self.condition_ids = [f'condition_{i_cond}'
                                  for i_cond in range(len(conditions))]

        super().__init__()
