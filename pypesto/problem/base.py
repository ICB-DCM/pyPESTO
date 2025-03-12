import copy
import logging
import sys
from collections.abc import Iterable
from typing import (
    Callable,
    Optional,
    SupportsFloat,
    SupportsInt,
    Union,
)

import numpy as np
import pandas as pd

from ..objective import ObjectiveBase
from ..objective.priors import NegLogParameterPriors
from ..startpoint import StartpointMethod, to_startpoint_method, uniform
from ..version import __version__

SupportsFloatIterableOrValue = Union[Iterable[SupportsFloat], SupportsFloat]
SupportsIntIterableOrValue = Union[Iterable[SupportsInt], SupportsInt]

logger = logging.getLogger(__name__)


class Problem:
    """
    The problem formulation.

    A problem specifies the objective function, boundaries and constraints,
    parameter guesses as well as the parameters which are to be optimized.

    Parameters
    ----------
    objective:
        The objective function for minimization. Note that a shallow copy
        is created.
    lb, ub:
        The lower and upper bounds for optimization. For unbounded directions
        set to +-inf.
    lb_init, ub_init:
        The lower and upper bounds for initialization, typically for defining
        search start points.
        If not set, set to lb, ub.
    dim_full:
        The full dimension of the problem, including fixed parameters.
    x_fixed_indices:
        Vector containing the indices (zero-based) of parameter components
        that are not to be optimized.
    x_fixed_vals:
        Vector of the same length as x_fixed_indices, containing the values
        of the fixed parameters.
    x_guesses:
        Guesses for the parameter values, shape (g, dim), where g denotes the
        number of guesses. These are used as start points in the optimization.
    x_names:
        Parameter names that can be optionally used e.g. in visualizations.
        If objective.get_x_names() is not None, those values are used,
        else the values specified here are used if not None, otherwise
        the variable names are set to ['x0', ... 'x{dim_full}']. The list
        must always be of length dim_full.
    x_scales:
        Parameter scales can be optionally given and are used e.g. in
        visualisation and prior generation. Currently the scales 'lin',
        'log`and 'log10' are supported.
    x_priors_defs:
        Definitions of priors for parameters. Types of priors, and their
        required and optional parameters, are described in the `Prior` class.
    copy_objective:
        Whethter to generate a deep copy of the objective function before
        potential modification the problem class performs on it.
    startpoint_method:
        Method for how to choose start points. ``False`` means the optimizer
        does not require start points, e.g. for the ``PyswarmOptimizer``.

    Notes
    -----
    On the fixing of parameter values:

    The number of parameters dim_full the objective takes as input must
    be known, so it must be either lb a vector of that size, or dim_full
    specified as a parameter.

    All vectors are mapped to the reduced space of dimension dim in __init__,
    regardless of whether they were in dimension dim or dim_full before. If
    the full representation is needed, the methods get_full_vector() and
    get_full_matrix() can be used.
    """

    def __init__(
        self,
        objective: ObjectiveBase,
        lb: Union[np.ndarray, list[float]],
        ub: Union[np.ndarray, list[float]],
        dim_full: Optional[int] = None,
        x_fixed_indices: Optional[SupportsIntIterableOrValue] = None,
        x_fixed_vals: Optional[SupportsFloatIterableOrValue] = None,
        x_guesses: Optional[Iterable[float]] = None,
        x_names: Optional[Iterable[str]] = None,
        x_scales: Optional[Iterable[str]] = None,
        x_priors_defs: Optional[NegLogParameterPriors] = None,
        lb_init: Union[np.ndarray, list[float], None] = None,
        ub_init: Union[np.ndarray, list[float], None] = None,
        copy_objective: bool = True,
        startpoint_method: Union[StartpointMethod, Callable, bool] = None,
    ):
        if copy_objective:
            objective = copy.deepcopy(objective)
        self.objective = objective

        self.lb_full: np.ndarray = np.array(lb).flatten()
        self.ub_full: np.ndarray = np.array(ub).flatten()
        if lb_init is None:
            lb_init = lb
        self.lb_init_full: np.ndarray = np.array(lb_init).flatten()
        if ub_init is None:
            ub_init = ub
        self.ub_init_full: np.ndarray = np.array(ub_init).flatten()

        self.dim_full: int = (
            dim_full if dim_full is not None else self.lb_full.size
        )

        if x_fixed_indices is None:
            x_fixed_indices = []
        x_fixed_indices = _make_iterable_if_value(x_fixed_indices, "int")
        self.x_fixed_indices: list[int] = [
            _type_conversion_with_check(idx, ix, "fixed indices", "int")
            for idx, ix in enumerate(x_fixed_indices)
        ]

        # We want the fixed values to be a list, since we might need to add
        # or remove values during profile computation
        if x_fixed_vals is None:
            x_fixed_vals = []
        x_fixed_vals = _make_iterable_if_value(x_fixed_vals, "float")
        self.x_fixed_vals: list[float] = [
            _type_conversion_with_check(idx, x, "fixed values", "float")
            for idx, x in enumerate(x_fixed_vals)
        ]

        if x_guesses is None:
            x_guesses = np.zeros((0, self.dim_full))
        self.x_guesses_full: np.ndarray = np.array(x_guesses)

        if x_names is None and objective.x_names is not None:
            x_names = objective.x_names
        elif x_names is None:
            x_names = [f"x{j}" for j in range(0, self.dim_full)]
        if len(set(x_names)) != len(x_names):
            raise ValueError("Parameter names x_names must be unique")
        self.x_names: list[str] = list(x_names)

        if x_scales is None:
            x_scales = ["lin"] * self.dim_full
        self.x_scales = x_scales

        self.x_priors = x_priors_defs

        self.normalize()
        self._check_x_guesses()

        # startpoint method
        if startpoint_method is None:
            startpoint_method = uniform
        # convert startpoint method to class instance
        self.startpoint_method = to_startpoint_method(startpoint_method)
        # save python and pypesto version
        self.python_version = ".".join(map(str, sys.version_info[:3]))
        self.pypesto_version = __version__

    @property
    def lb(self) -> np.ndarray:
        """Return lower bounds of free parameters."""
        return self.lb_full[self.x_free_indices]

    @property
    def ub(self) -> np.ndarray:
        """Return upper bounds of free parameters."""
        return self.ub_full[self.x_free_indices]

    @property
    def lb_init(self) -> np.ndarray:
        """Return initial lower bounds of free parameters."""
        return self.lb_init_full[self.x_free_indices]

    @property
    def ub_init(self) -> np.ndarray:
        """Return initial upper bounds of free parameters."""
        return self.ub_init_full[self.x_free_indices]

    @property
    def x_guesses(self) -> np.ndarray:
        """Return guesses of the free parameter values."""
        return self.x_guesses_full[:, self.x_free_indices]

    @property
    def dim(self) -> int:
        """Return dimension only considering non fixed parameters."""
        return self.dim_full - len(self.x_fixed_indices)

    @property
    def x_free_indices(self) -> list[int]:
        """Return non fixed parameters."""
        return sorted(set(range(0, self.dim_full)) - set(self.x_fixed_indices))

    def normalize(self) -> None:
        """
        Process vectors.

        Reduce all vectors to dimension dim and have the objective accept
        vectors of dimension dim.
        """
        for attr in ["lb_full", "lb_init_full", "ub_full", "ub_init_full"]:
            value = self.__getattribute__(attr)
            if value.size == 1:
                self.__setattr__(attr, value * np.ones(self.dim_full))
            elif value.size == self.dim:
                # in this case the bounds only holds the values of the
                # reduced bounds.
                self.__setattr__(
                    attr, self.get_full_vector(value, self.x_fixed_vals)
                )

            if self.__getattribute__(attr).size != self.dim_full:
                raise AssertionError(f"{attr} dimension invalid.")

        if self.x_guesses_full.shape[1] != self.dim_full:
            x_guesses_full = np.empty(
                (self.x_guesses_full.shape[0], self.dim_full)
            )
            x_guesses_full[:] = np.nan
            x_guesses_full[:, self.x_free_indices] = self.x_guesses_full
            self.x_guesses_full = x_guesses_full

        # make objective aware of fixed parameters
        self.objective.update_from_problem(
            dim_full=self.dim_full,
            x_free_indices=self.x_free_indices,
            x_fixed_indices=self.x_fixed_indices,
            x_fixed_vals=self.x_fixed_vals,
        )

        # make prior aware of fixed parameters (for sampling etc.)
        if self.x_priors is not None:
            self.x_priors.update_from_problem(
                dim_full=self.dim_full,
                x_free_indices=self.x_free_indices,
                x_fixed_indices=self.x_fixed_indices,
                x_fixed_vals=self.x_fixed_vals,
            )

        # sanity checks
        if len(self.x_scales) != self.dim_full:
            raise AssertionError("x_scales dimension invalid.")
        if len(self.x_names) != self.dim_full:
            raise AssertionError("x_names must be of length dim_full.")
        if len(self.x_fixed_indices) != len(self.x_fixed_vals):
            raise AssertionError(
                "x_fixed_indices and x_fixed_vals must have the same length."
            )
        if np.isnan(self.lb).any():
            raise ValueError("lb must not contain nan values")
        if np.isnan(self.ub).any():
            raise ValueError("ub must not contain nan values")
        if np.any(self.lb >= self.ub):
            raise ValueError("lb<ub not fulfilled.")

    def _check_x_guesses(self):
        """Check whether the supplied x_guesses adhere to the bounds."""
        if self.x_guesses.size == 0:
            return
        adheres_ub = self.x_guesses <= self.ub
        adheres_lb = self.x_guesses >= self.lb
        adheres_bounds = adheres_ub & adheres_lb
        # if any bounds are violated, log a warning
        if not adheres_bounds.all():
            logger.warning(
                "Some initial guesses supplied violate the bounds "
                "set for this problem."
            )

    def set_x_guesses(self, x_guesses: Iterable[float]):
        """
        Set the x_guesses of a problem.

        Parameters
        ----------
        x_guesses:
        """
        x_guesses_full = np.array(x_guesses)
        if x_guesses_full.shape[1] != self.dim_full:
            raise ValueError(
                "The dimension of individual x_guesses must be dim_full."
            )
        self.x_guesses_full = x_guesses_full
        self._check_x_guesses()

    def fix_parameters(
        self,
        parameter_indices: SupportsIntIterableOrValue,
        parameter_vals: SupportsFloatIterableOrValue,
    ) -> None:
        """Fix specified parameters to specified values."""
        parameter_indices = _make_iterable_if_value(parameter_indices, "int")
        parameter_vals = _make_iterable_if_value(parameter_vals, "float")

        # first clean to-be-fixed indices to avoid redundancies
        for iter_index, (x_index, x_value) in enumerate(
            zip(parameter_indices, parameter_vals)
        ):
            # check if parameter was already fixed, otherwise add it to the
            # fixed parameters
            index = _type_conversion_with_check(
                iter_index, x_index, "indices", "int"
            )
            val = _type_conversion_with_check(
                iter_index, x_value, "values", "float"
            )
            if index in self.x_fixed_indices:
                self.x_fixed_vals[self.x_fixed_indices.index(index)] = val
            else:
                self.x_fixed_indices.append(index)
                self.x_fixed_vals.append(val)

        self.normalize()

    def unfix_parameters(
        self, parameter_indices: SupportsIntIterableOrValue
    ) -> None:
        """Free specified parameters."""
        # check and adapt input
        parameter_indices = _make_iterable_if_value(parameter_indices, "int")

        # first clean to-be-freed indices
        for iter_index, x_index in enumerate(parameter_indices):
            index = _type_conversion_with_check(
                iter_index, x_index, "indices", "int"
            )
            if index in self.x_fixed_indices:
                fixed_x_index = self.x_fixed_indices.index(index)
                self.x_fixed_indices.pop(fixed_x_index)
                self.x_fixed_vals.pop(fixed_x_index)

        self.normalize()

    def get_full_vector(
        self,
        x: Union[np.ndarray, None],
        x_fixed_vals: Iterable[float] = None,
        x_is_grad: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Map vector from dim to dim_full. Usually used for x, grad.

        Parameters
        ----------
        x: array_like, shape=(dim,)
            The vector in dimension dim.
        x_fixed_vals: array_like, ndim=1, optional
            The values to be used for the fixed indices. If None and x_is_grad=False, problem.x_fixed_vals is used; for x_is_grad=True, nans are inserted.
        x_is_grad: bool
            If true, x is treated as gradients.
        """
        if x is None:
            return None

        # make sure it is an array
        x = np.array(x)

        if len(x) == self.dim_full:
            return x

        # Note: The funny indexing construct is to handle residual gradients,
        # where the last dimension is assumed to be the parameter one.
        x_full = np.zeros(x.shape[:-1] + (self.dim_full,))
        x_full[:] = np.nan
        x_full[..., self.x_free_indices] = x
        if x_fixed_vals is not None:
            x_full[..., self.x_fixed_indices] = x_fixed_vals
            return x_full
        if not x_is_grad:
            x_full[..., self.x_fixed_indices] = self.x_fixed_vals
        return x_full

    def get_full_matrix(
        self, x: Union[np.ndarray, None]
    ) -> Union[np.ndarray, None]:
        """
        Map matrix from dim to dim_full. Usually used for hessian.

        Parameters
        ----------
        x: array_like, shape=(dim, dim)
            The matrix in dimension dim.
        """
        if x is None:
            return None

        # make sure it is an array
        x = np.array(x)

        if len(x) == self.dim_full:
            return x

        x_full = np.zeros((self.dim_full, self.dim_full))
        x_full[:, :] = np.nan
        x_full[np.ix_(self.x_free_indices, self.x_free_indices)] = x

        return x_full

    def get_reduced_vector(
        self,
        x_full: Union[np.ndarray, None],
        x_indices: Optional[list[int]] = None,
    ) -> Union[np.ndarray, None]:
        """
        Keep only those elements, which indices are specified in x_indices.

        If x_indices is not provided, delete fixed indices.

        Parameters
        ----------
        x_full: array_like, ndim=1
            The vector in dimension dim_full.
        x_indices:
            indices of x_full that should remain
        """
        if x_full is None:
            return None

        if x_indices is None:
            x_indices = self.x_free_indices

        if len(x_full) == len(x_indices):
            return x_full

        x = [x_full[idx] for idx in x_indices]
        return np.array(x)

    def get_reduced_matrix(
        self, x_full: Union[np.ndarray, None]
    ) -> Union[np.ndarray, None]:
        """
        Map matrix from dim_full to dim, i.e. delete fixed indices.

        Parameters
        ----------
        x_full: array_like, ndim=2
            The matrix in dimension dim_full.
        """
        if x_full is None:
            return None

        if len(x_full) == self.dim:
            return x_full

        x = x_full[np.ix_(self.x_free_indices, self.x_free_indices)]

        return x

    def full_index_to_free_index(self, full_index: int):
        """
        Calculate index in reduced vector from index in full vector.

        Parameters
        ----------
        full_index: The index in the full vector.

        Returns
        -------
        free_index: The index in the free vector.
        """
        fixed_indices = np.asarray(self.x_fixed_indices)
        if full_index in fixed_indices:
            raise ValueError(
                "Cannot compute index in free vector: Index is fixed."
            )
        return full_index - sum(fixed_indices < full_index)

    def print_parameter_summary(self) -> None:
        """
        Print a summary of parameters.

        Include what parameters are being optimized and parameter boundaries.
        """
        print(  # noqa: T201 (print)
            pd.DataFrame(
                index=self.x_names,
                data={
                    "free": [
                        idx in self.x_free_indices
                        for idx in range(self.dim_full)
                    ],
                    "lb_full": [
                        self.lb_full[idx]
                        if idx in self.x_free_indices
                        else "-"
                        for idx in range(self.dim_full)
                    ],
                    "ub_full": [
                        self.ub_full[idx]
                        if idx in self.x_free_indices
                        else "-"
                        for idx in range(self.dim_full)
                    ],
                    "scale": self.x_scales,
                    "fixedVal (scaled)": [
                        self.get_full_vector(np.zeros(self.dim))[idx]
                        if idx in self.x_fixed_indices
                        else "-"
                        for idx in range(self.dim_full)
                    ],
                },
            )
        )

    def get_startpoints(self, n_starts: int) -> np.ndarray:
        """
        Sample startpoints from method.

        Parameters
        ----------
        n_starts:
            Number of start points.

        Returns
        -------
        xs:
            Start points, shape (n_starts, dim).
        """
        return self.startpoint_method(n_starts, self)


_convtypes = {
    "float": {"attr": "__float__", "conv": float},
    "int": {"attr": "__int__", "conv": int},
}


def _type_conversion_with_check(
    index: int,
    value: Union[SupportsFloat, SupportsInt],
    valuename: str,
    convtype: str,
) -> Union[float, int]:
    """
    Convert values to the requested type.

    Raises and appropriate error if not possible.
    """
    if convtype not in _convtypes:
        raise ValueError(f"Unsupported type {convtype}")

    can_convert = hasattr(value, _convtypes[convtype]["attr"])
    # this may fail for weird custom ypes that can be converted to int but
    # not float, but we probably don't want those as indiced anyways
    lossless_conversion = not convtype == "int" or (
        hasattr(value, _convtypes["float"]["attr"])
        and (float(value) - int(value) == 0.0)
    )

    if not can_convert or not lossless_conversion:
        raise ValueError(
            f"All {valuename} must support lossless conversion to {convtype}. "
            f"Found type {type(value)} at index {index}, which cannot "
            f"be converted to {convtype}."
        )

    return _convtypes[convtype]["conv"](value)


def _make_iterable_if_value(
    value: Union[SupportsFloatIterableOrValue, SupportsIntIterableOrValue],
    convtype: str,
) -> Union[Iterable[SupportsFloat], Iterable[SupportsInt]]:
    """Convert scalar values to iterables for scalar input, may update type."""
    if convtype not in _convtypes:
        raise ValueError(f"Unsupported type {convtype}")

    if not hasattr(value, "__iter__"):
        return [_type_conversion_with_check(0, value, "values", convtype)]
    else:
        return value
