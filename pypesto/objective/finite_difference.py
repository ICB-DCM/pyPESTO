"""Finite differences."""

import copy
import logging
from typing import Callable, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, MODE_FUN, MODE_RES, RES, SRES, ModeType
from .base import ObjectiveBase, ResultDict

logger = logging.getLogger(__name__)


class FDDelta:
    """Finite difference step size with automatic updating.

    Reference implementation:
    https://github.com/ICB-DCM/PESTO/blob/master/private/getStepSizeFD.m

    Parameters
    ----------
    delta:
        (Initial) step size, either a float, or a vector of size (n_par,).
        If not None, this is used as initial step size.
    test_deltas:
        Step sizes to try out in step size selection. If None, a range
        [1e-1, 1e-2, ..., 1e-8] is considered.
    update_condition:
        A "good" step size may be a local property. Thus, this class allows
        updating the step size if certain criteria are met, in the
        :func:`pypesto.objective.finite_difference.FDDelta.update`
        function.
        FDDelta.CONSTANT means that the step size is only initially selected.
        FDDelta.DISTANCE means that the step size is updated if the current
        evaluation point is sufficiently far away from the last training point.
        FDDelta.STEPS means that the step size is updated `max_steps`
        evaluations after the last update.
        FDDelta.ALWAYS mean that the step size is selected in every call.
    max_distance:
        Coefficient on the distance between current and reference point beyond
        which to update, in the `FDDelta.DISTANCE` update condition.
    max_steps:
        Number of steps after which to update in the `FDDelta.STEPS` update
        condition.
    """

    # update conditions
    CONSTANT = "constant"
    DISTANCE = "distance"
    STEPS = "steps"
    ALWAYS = "always"
    UPDATE_CONDITIONS = [CONSTANT, DISTANCE, STEPS, ALWAYS]

    def __init__(
        self,
        delta: Union[np.ndarray, float, None] = None,
        test_deltas: np.ndarray = None,
        update_condition: str = CONSTANT,
        max_distance: float = 0.5,
        max_steps: int = 30,
    ):
        if not isinstance(delta, (np.ndarray, float)) and delta is not None:
            raise ValueError(f"Unexpected type {type(delta)} for delta")
        self.delta: Union[np.ndarray, float, None] = delta

        if test_deltas is None:
            test_deltas = np.array([10 ** (-i) for i in range(1, 9)])
        self.test_deltas: np.ndarray = test_deltas

        if update_condition not in FDDelta.UPDATE_CONDITIONS:
            raise ValueError(
                f"Update condition {update_condition} must be in "
                f"{FDDelta.UPDATE_CONDITIONS}.",
            )
        self.update_condition: str = update_condition

        self.max_distance: float = max_distance
        self.max_steps: int = max_steps

        # run variables
        #  parameter where the step sizes where updated last
        self.x0: Union[np.ndarray, None] = None
        #  overall number of steps
        self.steps: int = 0
        self.updates: int = 0

    def update(
        self,
        x: np.ndarray,
        fval: Union[float, np.ndarray, None],
        fun: Callable,
        fd_method: str,
    ) -> None:
        """Update delta if update conditions are met.

        Parameters
        ----------
        x:
            Current parameter vector, shape (n_par,).
        fval:
            fun(x), to avoid re-evaluation. Scalar- or vector-valued.
        fun:
            Function whose 1st-order derivative to approximate.
            Scalar- or vector-valued.
        fd_method:
            FD method employed by
            :class:`pypesto.objective.finite_difference.FD`, see there.
        """
        # scalar to vector
        if isinstance(self.delta, float):
            self.delta: np.ndarray = self.delta * np.ones(shape=x.shape)
        elif isinstance(self.delta, np.ndarray):
            if self.delta.shape != x.shape:
                # this should not happen
                raise ValueError("Shape mismatch.")

        self.steps += 1

        # return if no update needed
        if self.delta is not None:
            if (
                (
                    self.update_condition == FDDelta.DISTANCE
                    and np.sum((x - self.x0) ** 2)
                    <= self.max_distance * np.sqrt(len(x))
                )
                or (
                    self.update_condition == FDDelta.STEPS
                    and (self.steps - 1) % self.max_steps != 0
                    and self.steps > 1
                )
                or (
                    self.update_condition == FDDelta.CONSTANT
                    and self.delta is not None
                )
            ):
                return

        # update reference point
        self.x0 = x
        # actually update
        self._update(x=x, fval=fval, fun=fun, fd_method=fd_method)

        if self.delta.shape != x.shape:
            # this should not happen
            raise ValueError("Shape mismatch.")

    def _update(
        self,
        x: np.ndarray,
        fval: Union[float, np.ndarray],
        fun: Callable,
        fd_method: str,
    ) -> None:
        """
        Actually update. Wants to be called in `update` explicitly.

        Run FDs with various deltas and pick the ones, separately for each
        parameter, with the best stability properties.

        The parameters are the same as for
        :func:`pypesto.objective.finite_difference.FDDelta.update`.
        """
        # calculate gradients for all deltas for all parameters
        nablas = []
        # iterate over deltas
        for delta in self.test_deltas:
            # calculate Jacobian with step size delta
            delta_vec = delta * np.ones_like(x)
            nabla = fd_nabla_1(
                x=x,
                fval=fval,
                f_fval=fun,
                delta_vec=delta_vec,
                fd_method=fd_method,
            )

            nablas.append(nabla)

        # shape (n_delta, n_par, ...)
        nablas = np.array(nablas)

        # The stability vector is the absolute difference of Jacobian
        #  entries towards smaller and larger deltas, thus indicating the
        #  change in the approximation when changing delta.
        # This is done separately for each parameter. Then, for each the delta
        #  with the minimal entry and thus the most stable behavior
        #  is selected.
        stab_vec = np.full(shape=nablas.shape, fill_value=np.nan)
        stab_vec[1:-1] = np.mean(
            np.abs([nablas[2:] - nablas[1:-1], nablas[1:-1] - nablas[:-2]]),
            axis=0,
        )
        # on the edge, just take the single neighbor
        stab_vec[0] = np.abs(nablas[1] - nablas[0])
        stab_vec[-1] = np.abs(nablas[-1] - nablas[-2])

        # if the function is tensor-valued, consider the maximum over all
        #  entries, to constrain the worst deviation
        if stab_vec.ndim > 2:
            # flatten all dimensions > 1
            stab_vec = stab_vec.reshape(
                stab_vec.shape[0], stab_vec.shape[1], -1
            ).max(axis=2)

        # minimum delta index for each parameter
        min_ixs = np.argmin(stab_vec, axis=0)

        # extract optimal deltas per parameter
        delta_opt = np.array([self.test_deltas[ix] for ix in min_ixs])
        self.delta = delta_opt

        # log
        logger.debug(f"Optimal FD delta: {self.delta}")
        self.updates += 1

    def get(self) -> np.ndarray:
        """Get delta vector."""
        return self.delta


def to_delta(delta: Union[FDDelta, np.ndarray, float, str]) -> FDDelta:
    """Input to step size delta.

    Input can be a vector, float, or an update method type.

    Parameters
    ----------
    delta:
        Can be a vector, float, or one of Delta.UPDATE_CONDITIONS. If a
        vector or float, a constant delta is assumed.
    """
    if isinstance(delta, FDDelta):
        return delta
    elif isinstance(delta, (np.ndarray, float)):
        return FDDelta(delta=delta, update_condition=FDDelta.CONSTANT)
    else:
        return FDDelta(delta=None, update_condition=delta)


class FD(ObjectiveBase):
    """Finite differences (FDs) for derivatives.

    Given an objective that gives function values and/or residuals, this
    class allows to flexibly obtain all derivatives calculated via FDs.

    For the parameters `grad`, `hess`, `sres`, a value of None means that the
    objective derivative is used if available, otherwise resorting to FDs.
    True means that FDs are used in any case, False means that the derivative
    is not exported.

    Note that the step sizes should be carefully chosen. They should be small
    enough to provide an accurate linear approximation, but large enough to
    be robust against numerical inaccuracies, in particular if the objective
    relies on numerical approximations, such as an ODE.

    Parameters
    ----------
    grad:
        Derivative method for the gradient (see above).
    hess:
        Derivative method for the Hessian (see above).
    sres:
        Derivative method for the residual sensitivities (see above).
    hess_via_fval:
        If the Hessian is to be calculated via finite differences:
        whether to employ 2nd order FDs via fval even if the objective can
        provide a gradient.
    delta_fun:
        FD step sizes for function values.
        Can be either a float, or a :class:`np.ndarray` of shape (n_par,)
        for different step sizes for different coordinates.
    delta_grad:
        FD step sizes for gradients, if the Hessian is calculated via 1st
        order sensitivities from the gradients. Similar to `delta_fun`.
    delta_res:
        FD step sizes for residuals. Similar to `delta_fun`.
    method:
        Method to calculate FDs. Can be any of `FD.METHODS`: central,
        forward or backward differences. The latter two require only roughly
        half as many function evaluations, are however less accurate than
        central (O(x) vs O(x**2)).
    x_names:
        Parameter names that can be optionally used in, e.g., history or
        gradient checks.

    Examples
    --------
    Define residuals and objective function, and obtain all derivatives via
    FDs:

    >>> from pypesto import Objective, FD
    >>> import numpy as np
    >>> x_obs = np.array([11, 12, 13])
    >>> res = lambda x: x - x_obs
    >>> fun = lambda x: 0.5 * sum(res(x)**2)
    >>> obj = FD(Objective(fun=fun, res=res))
    """

    # finite difference types
    CENTRAL = "central"
    FORWARD = "forward"
    BACKWARD = "backward"
    METHODS = [CENTRAL, FORWARD, BACKWARD]

    def __init__(
        self,
        obj: ObjectiveBase,
        grad: Union[bool, None] = None,
        hess: Union[bool, None] = None,
        sres: Union[bool, None] = None,
        hess_via_fval: bool = True,
        delta_fun: Union[FDDelta, np.ndarray, float, str] = 1e-6,
        delta_grad: Union[FDDelta, np.ndarray, float, str] = 1e-6,
        delta_res: Union[FDDelta, float, np.ndarray, str] = 1e-6,
        method: str = CENTRAL,
        x_names: list[str] = None,
    ):
        super().__init__(x_names=x_names)
        self.obj: ObjectiveBase = obj
        self.grad: Union[bool, None] = grad
        self.hess: Union[bool, None] = hess
        self.sres: Union[bool, None] = sres
        self.hess_via_fval: bool = hess_via_fval
        self.delta_fun: FDDelta = to_delta(delta_fun)
        self.delta_grad: FDDelta = to_delta(delta_grad)
        self.delta_res: FDDelta = to_delta(delta_res)
        self.method: str = method
        self.pre_post_processor = obj.pre_post_processor

        if method not in FD.METHODS:
            raise ValueError(
                f"Method must be one of {FD.METHODS}.",
            )

    def __deepcopy__(
        self,
        memodict: dict = None,
    ) -> "FD":
        """Create deepcopy of Objective."""
        other = self.__class__.__new__(self.__class__)
        for attr, val in self.__dict__.items():
            other.__dict__[attr] = copy.deepcopy(val)
        return other

    @property
    def has_fun(self) -> bool:
        """Check whether function is defined."""
        return self.obj.has_fun

    @property
    def has_grad(self) -> bool:
        """Check whether gradient is defined."""
        return self.grad is not False and self.obj.has_fun

    @property
    def has_hess(self) -> bool:
        """Check whether Hessian is defined."""
        return self.hess is not False and self.obj.has_fun

    @property
    def has_res(self) -> bool:
        """Check whether residuals are defined."""
        return self.obj.has_res

    @property
    def has_sres(self) -> bool:
        """Check whether residual sensitivities are defined."""
        return self.sres is not False and self.obj.has_res

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        return_dict: bool,
        **kwargs,
    ) -> ResultDict:
        """
        See `ObjectiveBase` for more documentation.

        Main method to overwrite from the base class. It handles and
        delegates the actual objective evaluation.
        """
        if mode == MODE_FUN:
            result = self._call_mode_fun(
                x=x,
                sensi_orders=sensi_orders,
                return_dict=return_dict,
                **kwargs,
            )
        elif mode == MODE_RES:
            result = self._call_mode_res(
                x=x,
                sensi_orders=sensi_orders,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            raise ValueError("This mode is not supported.")

        return result

    def _call_mode_fun(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        return_dict: bool,
        **kwargs,
    ) -> ResultDict:
        """Handle calls in function value mode.

        Delegated from `call_unprocessed`.
        """
        # get from objective what it can and should deliver
        sensi_orders_obj, result = self._call_from_obj_fun(
            x=x,
            sensi_orders=sensi_orders,
            return_dict=return_dict,
            **kwargs,
        )

        # remaining sensis via FDs

        # whether gradient and Hessian are intended as FDs
        grad_via_fd = 1 in sensi_orders and 1 not in sensi_orders_obj
        hess_via_fd = 2 in sensi_orders and 2 not in sensi_orders_obj

        if not grad_via_fd and not hess_via_fd:
            return result

        # whether the Hessian should be based on 2nd order FD from fval
        hess_via_fd_fval = hess_via_fd and (
            self.hess_via_fval or not self.obj.has_grad
        )
        hess_via_fd_grad = hess_via_fd and not hess_via_fd_fval

        def f_fval(x):
            """Short-hand to get a function value."""
            return self.obj.call_unprocessed(
                x=x,
                sensi_orders=(0,),
                mode=MODE_FUN,
                return_dict=return_dict,
                **kwargs,
            )[FVAL]

        def f_grad(x):
            """Short-hand to get a gradient value."""
            return self.obj.call_unprocessed(
                x=x,
                sensi_orders=(1,),
                mode=MODE_FUN,
                return_dict=return_dict,
                **kwargs,
            )[GRAD]

        # update delta vectors
        if grad_via_fd or hess_via_fd_fval:
            # note: we use the same delta for 1st and 2nd order approximations
            # this may be not ideal
            self.delta_fun.update(
                x=x, fval=result.get(FVAL), fun=f_fval, fd_method=self.method
            )
        if hess_via_fd_grad:
            self.delta_grad.update(
                x=x, fval=result.get(GRAD), fun=f_grad, fd_method=self.method
            )

        # calculate gradient
        if grad_via_fd:
            result[GRAD] = fd_nabla_1(
                x=x,
                fval=result.get(FVAL),
                f_fval=f_fval,
                delta_vec=self.delta_fun.get(),
                fd_method=self.method,
            )

        # calculate Hessian
        if hess_via_fd:
            if hess_via_fd_fval:
                result[HESS] = fd_nabla_2(
                    x=x,
                    fval=result.get(FVAL),
                    f_fval=f_fval,
                    delta_vec=self.delta_fun.get(),
                    fd_method=self.method,
                )
            else:
                hess = fd_nabla_1(
                    x=x,
                    fval=result.get(GRAD),
                    f_fval=f_grad,
                    delta_vec=self.delta_fun.get(),
                    fd_method=self.method,
                )
                # make it symmetric
                result[HESS] = 0.5 * (hess + hess.T)

        return result

    def _call_mode_res(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        return_dict: bool,
        **kwargs,
    ) -> ResultDict:
        """Handle calls in residual mode.

        Delegated from `call_unprocessed`.
        """
        # get from objective what it can and should deliver
        sensi_orders_obj, result = self._call_from_obj_res(
            x=x,
            sensi_orders=sensi_orders,
            return_dict=return_dict,
            **kwargs,
        )

        if sensi_orders == sensi_orders_obj:
            # all done
            return result

        def f_res(x):
            """Short-hand to get a function value."""
            return self.obj.call_unprocessed(
                x=x,
                sensi_orders=(0,),
                mode=MODE_RES,
                return_dict=return_dict,
                **kwargs,
            )[RES]

        # update delta vector
        self.delta_res.update(
            x=x, fval=result.get(RES), fun=f_res, fd_method=self.method
        )

        # sres
        sres = fd_nabla_1(
            x=x,
            fval=result.get(RES),
            f_fval=f_res,
            delta_vec=self.delta_res.get(),
            fd_method=self.method,
        )
        # sres should have shape (n_res, n_par)
        result[SRES] = sres.T

        return result

    def _call_from_obj_fun(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        return_dict: bool,
        **kwargs,
    ) -> tuple[tuple[int, ...], ResultDict]:
        """
        Call objective function for sensitivities.

        Calculate from the objective the sensitivities that are supposed to
        be calculated from the objective and not via FDs.
        """
        # define objective sensis
        sensi_orders_obj = []
        if 0 in sensi_orders:
            sensi_orders_obj.append(0)
        if 1 in sensi_orders and self.grad is None and self.obj.has_grad:
            sensi_orders_obj.append(1)
        if 2 in sensi_orders and self.hess is None and self.obj.has_hess:
            sensi_orders_obj.append(2)
        sensi_orders_obj = tuple(sensi_orders_obj)
        # call objective
        result = {}
        if sensi_orders_obj:
            result = self.obj.call_unprocessed(
                x=x,
                sensi_orders=sensi_orders_obj,
                mode=MODE_FUN,
                return_dict=return_dict,
                **kwargs,
            )
        return sensi_orders_obj, result

    def _call_from_obj_res(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        return_dict: bool,
        **kwargs,
    ) -> tuple[tuple[int, ...], ResultDict]:
        """
        Call objective function for sensitivities in residual mode.

        Calculate from the objective the sensitivities that are supposed to
        be calculated from the objective and not via FDs.
        """
        # define objective sensis
        sensi_orders_obj = []
        if 0 in sensi_orders:
            sensi_orders_obj.append(0)
        if 1 in sensi_orders and self.sres is None and self.obj.has_sres:
            sensi_orders_obj.append(1)
        sensi_orders_obj = tuple(sensi_orders_obj)
        #  call objective
        result = {}
        if sensi_orders_obj:
            result = self.obj.call_unprocessed(
                x=x,
                sensi_orders=sensi_orders_obj,
                mode=MODE_RES,
                return_dict=return_dict,
                **kwargs,
            )
        return sensi_orders_obj, result


def unit_vec(dim: int, ix: int) -> np.ndarray:
    """
    Return unit vector of dimension `dim` at coordinate `ix`.

    Parameters
    ----------
    dim: Vector dimension.
    ix: Index to contain the unit value.

    Returns
    -------
    vector: The unit vector.
    """
    vector = np.zeros(shape=dim)
    vector[ix] = 1
    return vector


def fd_nabla_1(
    x: np.ndarray,
    fval: float,
    f_fval: Callable,
    delta_vec: np.ndarray,
    fd_method: str,
) -> np.ndarray:
    """Calculate FD approximation to 1st order derivative (Jacobian/Gradient).

    Parameters
    ----------
    x: Parameter vector, shape (n_par,).
    fval: Function value at `x`, calculated if None.
    f_fval: Function returning function values. Scalar- or vector-valued.
    delta_vec: Step size vector, shape (n_par,).
    fd_method: FD method.

    Keyword arguments are passed on to the objective.

    Returns
    -------
    nabla_1:
        The FD approximation to the 1st order derivatives.
        Shape (n_par, ...) with ndim > 1 if `f_fval` is not scalar-valued.
    """
    # parameter dimension
    n_par = len(x)

    # calculate value at x only once if needed
    if fval is None and fd_method in [FD.FORWARD, FD.BACKWARD]:
        fval = f_fval(x)

    nabla_1 = []
    for ix in range(n_par):
        delta_val = delta_vec[ix]
        delta = delta_val * unit_vec(dim=n_par, ix=ix)

        if fd_method == FD.CENTRAL:
            fp = f_fval(x + delta / 2)
            fm = f_fval(x - delta / 2)
        elif fd_method == FD.FORWARD:
            fp = f_fval(x + delta)
            fm = fval
        elif fd_method == FD.BACKWARD:
            fp = fval
            fm = f_fval(x - delta)
        else:
            raise ValueError("Method not recognized.")

        nabla_1.append((fp - fm) / delta_val)

    return np.array(nabla_1)


def fd_nabla_2(
    x: np.ndarray,
    fval: float,
    f_fval: Callable,
    delta_vec: np.ndarray,
    fd_method: str,
) -> np.ndarray:
    """Calculate FD approximation to 2nd order derivatives (e.g. Hessian).

    Parameters
    ----------
    x: Parameter vector, shape (n_par,).
    fval: Function value at `x`, calculated if None. Scalar- or vector-valued.
    f_fval: Function returning function values.
    delta_vec: Step size vector, shape (n_par,).
    fd_method: FD method.

    Returns
    -------
    nabla_2:
        The FD approximation of the 2nd order derivative tensor.
        Shape (n_par, n_par, ...) with ndim > 2 if `f_fval` is not
        scalar-valued.
    """
    # parameter dimension
    n_par = len(x)

    # needed for diagonal entries at least
    if fval is None:
        fval = f_fval(x)

    # create empty matrix
    nabla_2 = []
    for _ in range(n_par):
        nabla_2.append([None] * n_par)

    # iterate over all parameter index tuples
    for ix1 in range(n_par):
        delta1_val = delta_vec[ix1]
        delta1 = delta1_val * unit_vec(dim=n_par, ix=ix1)

        # diagonal entry
        if fd_method == FD.CENTRAL:
            f2p = f_fval(x + delta1)
            fc = fval
            f2m = f_fval(x - delta1)
        elif fd_method == FD.FORWARD:
            f2p = f_fval(x + 2 * delta1)
            fc = f_fval(x + delta1)
            f2m = fval
        elif fd_method == FD.BACKWARD:
            f2p = fval
            fc = f_fval(x - delta1)
            f2m = f_fval(x - 2 * delta1)
        else:
            raise ValueError(f"Method {fd_method} not recognized.")

        nabla_2[ix1][ix1] = (f2p + f2m - 2 * fc) / delta1_val**2

        # off-diagonals
        for ix2 in range(ix1):
            delta2_val = delta_vec[ix2]
            delta2 = delta2_val * unit_vec(dim=n_par, ix=ix2)

            if fd_method == FD.CENTRAL:
                fpp = f_fval(x + delta1 / 2 + delta2 / 2)
                fpm = f_fval(x + delta1 / 2 - delta2 / 2)
                fmp = f_fval(x - delta1 / 2 + delta2 / 2)
                fmm = f_fval(x - delta1 / 2 - delta2 / 2)
            elif fd_method == FD.FORWARD:
                fpp = f_fval(x + delta1 + delta2)
                fpm = f_fval(x + delta1 + 0)
                fmp = f_fval(x + 0 + delta2)
                fmm = fval
            elif fd_method == FD.BACKWARD:
                fpp = fval
                fpm = f_fval(x + 0 - delta2)
                fmp = f_fval(x - delta1 + 0)
                fmm = f_fval(x - delta1 - delta2)
            else:
                raise ValueError(f"Method {fd_method} not recognized.")

            nabla_2[ix1][ix2] = nabla_2[ix2][ix1] = (fpp - fpm - fmp + fmm) / (
                delta1_val * delta2_val
            )

    return np.array(nabla_2)
