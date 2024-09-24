from functools import partial

import numpy as np

try:
    import petab.v1 as petab
    from petab.v1.C import (
        ESTIMATE,
        NOISE_PARAMETERS,
        OBSERVABLE_ID,
    )
except ImportError:
    petab = None

from ..C import (
    CENSORED,
    CENSORING_TYPES,
    MEASUREMENT_TYPE,
    ORDINAL,
    PARAMETER_TYPE,
    RELATIVE,
    SEMIQUANTITATIVE,
    InnerParameterType,
)
from ..problem import Problem
from ..startpoint import CheckedStartpoints


def get_petab_non_quantitative_data_types(
    petab_problem: petab.Problem,
) -> set[str]:
    """
    Get the data types from the PEtab problem.

    Parameters
    ----------
    petab_problem:
        The PEtab problem.

    Returns
    -------
    data_types:
        A list of the data types.
    """
    non_quantitative_data_types = set()
    caught_observables = set()
    # For ordinal, censored and semiquantitative data, search
    # for the corresponding data types in the measurement table
    meas_df = petab_problem.measurement_df
    if MEASUREMENT_TYPE in meas_df.columns:
        petab_data_types = meas_df[MEASUREMENT_TYPE].unique()
        for data_type in [ORDINAL, SEMIQUANTITATIVE] + CENSORING_TYPES:
            if data_type in petab_data_types:
                non_quantitative_data_types.add(
                    CENSORED if data_type in CENSORING_TYPES else data_type
                )
                caught_observables.update(
                    set(
                        meas_df[meas_df[MEASUREMENT_TYPE] == data_type][
                            OBSERVABLE_ID
                        ]
                    )
                )

    # For relative data, search for parameters to estimate with
    # a scaling/offset/sigma parameter type
    if PARAMETER_TYPE in petab_problem.parameter_df.columns:
        # get the df with non-nan parameter types
        par_df = petab_problem.parameter_df[
            petab_problem.parameter_df[PARAMETER_TYPE].notna()
        ]
        for par_id, row in par_df.iterrows():
            if not row[ESTIMATE]:
                continue
            if row[PARAMETER_TYPE] in [
                InnerParameterType.SCALING,
                InnerParameterType.OFFSET,
            ]:
                non_quantitative_data_types.add(RELATIVE)

            # For sigma parameters, we need to check if they belong
            # to an observable with a non-quantitative data type
            elif row[PARAMETER_TYPE] == InnerParameterType.SIGMA:
                corresponding_observables = set(
                    meas_df[meas_df[NOISE_PARAMETERS] == par_id][OBSERVABLE_ID]
                )
                if not (corresponding_observables & caught_observables):
                    non_quantitative_data_types.add(RELATIVE)

    # TODO this can be made much shorter if the relative measurements
    # are also specified in the measurement table, but that would require
    # changing the PEtab format of a lot of benchmark models.

    if len(non_quantitative_data_types) == 0:
        return None
    return non_quantitative_data_types


class PetabStartpoints(CheckedStartpoints):
    """Startpoint method for PEtab problems.

    Samples optimization startpoints from the distributions defined in the
    provided PEtab problem. The PEtab-problem is copied.
    """

    def __init__(self, petab_problem: petab.Problem, **kwargs):
        super().__init__(**kwargs)
        self._parameter_df = petab_problem.parameter_df.copy()
        self._priors: list[tuple] | None = None
        self._free_ids: list[str] | None = None

    def _setup(
        self,
        pypesto_problem: Problem,
    ):
        """Update priors if necessary.

        Check if ``problem.x_free_indices`` changed since last call, and if so,
        get the corresponding priors from PEtab.
        """
        current_free_ids = np.asarray(pypesto_problem.x_names)[
            pypesto_problem.x_free_indices
        ]

        if (
            self._priors is not None
            and len(current_free_ids) == len(self._free_ids)
            and np.all(current_free_ids == self._free_ids)
        ):
            # no need to update
            return

        # update priors
        self._free_ids = current_free_ids
        id_to_prior = dict(
            zip(
                self._parameter_df.index[self._parameter_df[ESTIMATE] == 1],
                petab.parameters.get_priors_from_df(
                    self._parameter_df, mode=petab.INITIALIZATION
                ),
            )
        )

        self._priors = list(map(id_to_prior.__getitem__, current_free_ids))

    def __call__(
        self,
        n_starts: int,
        problem: Problem,
    ) -> np.ndarray:
        """Call the startpoint method."""
        # Update the list of priors if needed
        self._setup(pypesto_problem=problem)

        return super().__call__(n_starts, problem)

    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Actual startpoint sampling.

        Must only be called through `self.__call__` to ensure that the list of priors
        matches the currently free parameters in the :class:`pypesto.Problem`.
        """
        sampler = partial(petab.sample_from_prior, n_starts=n_starts)
        startpoints = list(map(sampler, self._priors))

        return np.array(startpoints).T
