from functools import partial

import numpy as np
import pandas as pd

try:
    import petab.v1 as petab
    from petab import v2
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


def get_petab_v2_extra_field(element, field: str):
    """Get a pyPESTO-specific extra field from a PEtab v2 table element.

    pyPESTO's hierarchical optimization annotations (e.g. ``parameterType``)
    are not part of the PEtab format. In PEtab v2, such extra columns are
    preserved as extra fields on the respective table elements.

    Parameters
    ----------
    element:
        A PEtab v2 table element, e.g. a
        :class:`petab.v2.Parameter` or :class:`petab.v2.Measurement`.
    field:
        The name of the extra field.

    Returns
    -------
    The value of the extra field, or ``None`` if it is absent or empty.
    """
    value = (element.model_extra or {}).get(field)
    # mirror `petab.is_empty`: `None`, any pandas/numpy null (float nan,
    #  `pd.NA`, ...) and the empty string all count as "not set"
    if (
        value is None
        or pd.isnull(value)
        or (isinstance(value, str) and not value.strip())
    ):
        return None
    return value


def get_petab_non_quantitative_data_types(
    petab_problem: petab.Problem | v2.Problem,
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
    if isinstance(petab_problem, v2.Problem):
        return _get_petab_v2_non_quantitative_data_types(petab_problem)

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


def _get_petab_v2_non_quantitative_data_types(
    petab_problem: "v2.Problem",
) -> set[str] | None:
    """Get the non-quantitative data types from a PEtab v2 problem.

    See :func:`get_petab_non_quantitative_data_types`.
    """
    non_quantitative_data_types = set()

    # Ordinal, censored and semiquantitative data are not supported for PEtab
    # v2 yet, but they still have to be detected here so that they are
    # rejected with a clear message rather than silently treated as
    # quantitative data.
    for measurement in petab_problem.measurements:
        data_type = get_petab_v2_extra_field(measurement, MEASUREMENT_TYPE)
        if data_type in [ORDINAL, SEMIQUANTITATIVE] + CENSORING_TYPES:
            non_quantitative_data_types.add(
                CENSORED if data_type in CENSORING_TYPES else data_type
            )

    # For relative data, search for parameters to estimate with a
    # scaling/offset/sigma parameter type. Unlike for PEtab v1, sigma
    # parameters need no special case here: they can only belong to a relative
    # observable, since semiquantitative data are not supported.
    if any(
        get_petab_v2_extra_field(parameter, PARAMETER_TYPE)
        in (
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
            InnerParameterType.SIGMA,
        )
        for parameter in petab_problem.parameters
        if parameter.estimate
    ):
        non_quantitative_data_types.add(RELATIVE)

    return non_quantitative_data_types or None


class PetabStartpoints(CheckedStartpoints):
    """Startpoint method for PEtab problems.

    Samples optimization startpoints from the distributions defined in the
    provided PEtab problem.
    """

    def __init__(self, petab_problem: petab.Problem | v2.Problem, **kwargs):
        super().__init__(**kwargs)
        self._petab_problem = petab_problem
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
        if isinstance(self._petab_problem, petab.Problem):
            parameter_df = self._petab_problem.parameter_df
            id_to_prior = dict(
                zip(
                    parameter_df.index[parameter_df[ESTIMATE] == 1],
                    petab.parameters.get_priors_from_df(
                        parameter_df, mode=petab.INITIALIZATION
                    ),
                    strict=True,
                )
            )
        else:
            # PEtab v2: keep the prior distribution object (``None`` -> sample
            #  uniformly over the current bounds); sampled directly in
            #  `sample`, since the v1 `sample_from_prior` uses different
            #  distribution names
            id_to_prior = {
                parameter.id: parameter.prior_dist
                for parameter in self._petab_problem.parameters
                if parameter.estimate
            }
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
        priors=None,
    ) -> np.ndarray:
        """Actual startpoint sampling.

        Must only be called through `self.__call__` to ensure that the list of priors
        matches the currently free parameters in the :class:`pypesto.Problem`.
        """
        if isinstance(self._petab_problem, petab.Problem):
            # PEtab v1
            sampler = partial(petab.sample_from_prior, n_starts=n_starts)
            startpoints = list(map(sampler, self._priors))
        else:
            # PEtab v2 -- sample from the parameter prior distributions,
            #  falling back to a uniform distribution over the bounds of the
            #  `pypesto.Problem`, which may be tighter than the PEtab bounds
            #  (e.g. during profiling)
            startpoints = [
                prior_dist.sample(n_starts)
                if prior_dist is not None
                else np.random.uniform(cur_lb, cur_ub, n_starts)
                for prior_dist, cur_lb, cur_ub in zip(
                    self._priors, lb, ub, strict=True
                )
            ]

        return np.array(startpoints).T
