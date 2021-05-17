"""Finite differences."""

from typing import List, Tuple, Union
import numpy as np

from .base import ObjectiveBase, ResultDict
from .constants import (
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES,
)


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
        delta_fun: Union[float, np.ndarray, str] = 1e-6,
        delta_grad: Union[float, np.ndarray, str] = 1e-6,
        delta_res: Union[float, np.ndarray, str] = 1e-6,
        method: str = CENTRAL,
        x_names: List[str] = None,
    ):
        super().__init__(x_names=x_names)
        self.obj: ObjectiveBase = obj
        self.grad: Union[bool, None] = grad
        self.hess: Union[bool, None] = hess
        self.sres: Union[bool, None] = sres
        self.hess_via_fval: bool = hess_via_fval
        self.delta_fun: Union[float, np.ndarray] = delta_fun
        self.delta_grad: Union[float, np.ndarray] = delta_grad
        self.delta_res: Union[float, np.ndarray] = delta_res
        self.method: str = method

        if any(isinstance(delta, str)
               for delta in (delta_fun, delta_grad, delta_res)):
            raise NotImplementedError(
                "Adaptive FD step sizes are not implemented yet.",
            )
        if method not in FD.METHODS:
            raise ValueError(
                f"Method must be one of {FD.METHODS}.",
            )

    def get_delta_fun(self, par_ix: int) -> float:
        """Get function value step size delta for a given parameter index.

        Parameters
        ----------
        par_ix: Parameter index.

        Returns
        -------
        delta: Delta value.
        """
        if isinstance(self.delta_fun, np.ndarray):
            return self.delta_fun[par_ix]
        return self.delta_fun

    def get_delta_grad(self, par_ix: int) -> float:
        """Get gradient step size delta for a given parameter index.

        Parameters
        ----------
        par_ix: Parameter index.

        Returns
        -------
        delta: Delta value.
        """
        if isinstance(self.delta_fun, np.ndarray):
            return self.delta_fun[par_ix]
        return self.delta_fun

    def get_delta_res(self, par_ix: int) -> float:
        """Get residual step size delta for a given parameter index.

        Parameters
        ----------
        par_ix: Parameter index.

        Returns
        -------
        delta: Delta value.
        """
        if isinstance(self.delta_res, np.ndarray):
            return self.delta_res[par_ix]
        return self.delta_res

    @property
    def has_fun(self) -> bool:
        return self.obj.has_fun

    @property
    def has_grad(self) -> bool:
        return self.grad is not False and self.obj.has_fun

    @property
    def has_hess(self) -> bool:
        return self.hess is not False and self.obj.has_fun

    @property
    def has_res(self) -> bool:
        return self.obj.has_res

    @property
    def has_sres(self) -> bool:
        return self.sres is not False and self.obj.has_res

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
        **kwargs,
    ) -> ResultDict:
        # This is the main method to overwrite from the base class, it handles
        #  and delegates the actual objective evaluation.

        if mode == MODE_FUN:
            result = self._call_mode_fun(
                x=x, sensi_orders=sensi_orders, **kwargs)
        elif mode == MODE_RES:
            result = self._call_mode_res(
                x=x, sensi_orders=sensi_orders, **kwargs)
        else:
            raise ValueError("This mode is not supported.")

        return result

    def _call_mode_fun(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        **kwargs,
    ) -> ResultDict:
        """Handle calls in function value mode.

        Delegated from `call_unprocessed`.
        """
        # get from objective what it can and should deliver
        sensi_orders_obj, result = self._call_from_obj_fun(
            x=x, sensi_orders=sensi_orders, **kwargs,
        )

        # remaining sensis via FDs

        # whether gradient and Hessian are intended as FDs
        grad_via_fd = 1 in sensi_orders and 1 not in sensi_orders_obj
        hess_via_fd = 2 in sensi_orders and 2 not in sensi_orders_obj

        if not grad_via_fd and not hess_via_fd:
            return result

        # whether the Hessian should be based on 2nd order FD from fval
        hess_via_fd_fval = \
            hess_via_fd and (self.hess_via_fval or not self.obj.has_grad)

        if grad_via_fd:
            result[GRAD] = self._grad_via_fd(
                x=x, fval=result.get(FVAL), **kwargs)

        if hess_via_fd:
            if hess_via_fd_fval:
                result[HESS] = self._hess_via_fd_fval(
                    x=x, fval=result.get(FVAL), **kwargs)
            else:
                result[HESS] = self._hess_via_fd_grad(
                    x=x, fval=result.get(FVAL), **kwargs)

        return result

    def _call_mode_res(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        **kwargs,
    ) -> ResultDict:
        """Handle calls in residual mode.

        Delegated from `call_unprocessed`.
        """
        # get from objective what it can and should deliver
        sensi_orders_obj, result = self._call_from_obj_res(
            x=x, sensi_orders=sensi_orders, **kwargs,
        )

        if sensi_orders == sensi_orders_obj:
            # all done
            return result

        result[SRES] = self._sres_via_fd(x=x, res=result.get(RES), **kwargs)

        return result

    def _call_from_obj_fun(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        **kwargs,
    ) -> Tuple[Tuple[int, ...], ResultDict]:
        """
        Helper function that calculates from the objective the sensitivities
        that are supposed to be calculated from the objective and not via FDs.
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
                x=x, sensi_orders=sensi_orders_obj, mode=MODE_FUN, **kwargs)
        return sensi_orders_obj, result

    def _call_from_obj_res(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        **kwargs,
    ) -> Tuple[Tuple[int, ...], ResultDict]:
        """
        Helper function that calculates from the objective the sensitivities
        that are supposed to be calculated from the objective and not via FDs.
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
                x=x, sensi_orders=sensi_orders_obj, mode=MODE_RES, **kwargs)
        return sensi_orders_obj, result

    def _grad_via_fd(self, x: np.ndarray, fval: float, **kwargs) -> np.ndarray:
        """Calculate FD approximation to gradient."""
        # parameter dimension
        n_par = len(x)

        def f_fval(x):
            """Short-hand to get a function value."""
            return self.obj.call_unprocessed(
                x=x, sensi_orders=(0,), mode=MODE_FUN, **kwargs)[FVAL]

        # calculate value at x only once if needed
        if fval is None and self.method in [FD.FORWARD, FD.BACKWARD]:
            fval = f_fval(x)

        grad = np.full(shape=n_par, fill_value=np.nan)
        for ix in range(n_par):
            delta_val = self.get_delta_fun(par_ix=ix)
            delta = delta_val * unit_vec(dim=n_par, ix=ix)

            if self.method == FD.CENTRAL:
                fp = f_fval(x + delta / 2)
                fm = f_fval(x - delta / 2)
            elif self.method == FD.FORWARD:
                fp = f_fval(x + delta)
                fm = fval
            elif self.method == FD.BACKWARD:
                fp = fval
                fm = f_fval(x - delta)
            else:
                raise ValueError("Method not recognized.")

            grad[ix] = (fp - fm) / delta_val

        return grad

    def _hess_via_fd_fval(
        self, x: np.ndarray, fval: float, **kwargs,
    ) -> np.ndarray:
        """Calculate 2nd order FD approximation to Hessian."""
        # parameter dimension
        n_par = len(x)

        def f_fval(x):
            """Short-hand to get a function value."""
            return self.obj.call_unprocessed(
                x=x, sensi_orders=(0,), mode=MODE_FUN, **kwargs)[FVAL]

        hess = np.full(shape=(n_par, n_par), fill_value=np.nan)

        # needed for diagonal entries at least
        if fval is None:
            fval = f_fval(x)

        for ix1 in range(n_par):
            delta1_val = self.get_delta_fun(par_ix=ix1)
            delta1 = delta1_val * unit_vec(dim=n_par, ix=ix1)

            # diagonal entry
            if self.method == FD.CENTRAL:
                f2p = f_fval(x + delta1)
                fc = fval
                f2m = f_fval(x - delta1)
            elif self.method == FD.FORWARD:
                f2p = f_fval(x + 2 * delta1)
                fc = f_fval(x + delta1)
                f2m = fval
            elif self.method == FD.BACKWARD:
                f2p = fval
                fc = f_fval(x - delta1)
                f2m = f_fval(x - 2 * delta1)
            else:
                raise ValueError(f"Method {self.method} not recognized.")

            hess[ix1, ix1] = (f2p + f2m - 2 * fc) / delta1_val ** 2

            # off-diagonals
            for ix2 in range(ix1):
                delta2_val = self.get_delta_fun(par_ix=ix2)
                delta2 = delta2_val * unit_vec(dim=n_par, ix=ix2)

                if self.method == FD.CENTRAL:
                    fpp = f_fval(x + delta1 / 2 + delta2 / 2)
                    fpm = f_fval(x + delta1 / 2 - delta2 / 2)
                    fmp = f_fval(x - delta1 / 2 + delta2 / 2)
                    fmm = f_fval(x - delta1 / 2 - delta2 / 2)
                elif self.method == FD.FORWARD:
                    fpp = f_fval(x + delta1 + delta2)
                    fpm = f_fval(x + delta1 + 0)
                    fmp = f_fval(x + 0 + delta2)
                    fmm = fval
                elif self.method == FD.BACKWARD:
                    fpp = fval
                    fpm = f_fval(x + 0 - delta2)
                    fmp = f_fval(x - delta1 + 0)
                    fmm = f_fval(x - delta1 - delta2)
                else:
                    raise ValueError(f"Method {self.method} not recognized.")

                hess[ix1, ix2] = hess[ix2, ix1] = \
                    (fpp - fpm - fmp + fmm) / (delta1_val * delta2_val)

        return hess

    def _hess_via_fd_grad(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate 1st order FD approximation to Hessian via gradients."""
        # parameter dimension
        n_par = len(x)

        def f_grad(x):
            """Short-hand to get a gradient value."""
            return self.obj.call_unprocessed(
                x=x, sensi_orders=(1,), mode=MODE_FUN, **kwargs)[GRAD]

        # calculate value at x only once if needed
        g = None
        if self.method in [FD.FORWARD, FD.BACKWARD]:
            g = f_grad(x)

        hess = np.full(shape=(n_par, n_par), fill_value=np.nan)

        for ix in range(n_par):
            delta_val = self.get_delta_grad(par_ix=ix)
            delta = delta_val * unit_vec(dim=n_par, ix=ix)

            if self.method == FD.CENTRAL:
                gp = f_grad(x + delta / 2)
                gm = f_grad(x - delta / 2)
            elif self.method == FD.FORWARD:
                gp = f_grad(x + delta)
                gm = g
            elif self.method == FD.BACKWARD:
                gp = g
                gm = f_grad(x - delta)
            else:
                raise ValueError(f"Method {self.method} not recognized.")

            hess[:, ix] = (gp - gm) / delta_val
        # make it symmetric
        hess = 0.5 * (hess + hess.T)

        return hess

    def _sres_via_fd(
        self, x: np.ndarray, res: np.ndarray, **kwargs,
    ) -> np.ndarray:
        """Calculate FD approximation to residual sensitivities."""
        # parameter dimension
        n_par = len(x)

        def f_res(x):
            """Short-hand to get a function value."""
            return self.obj.call_unprocessed(
                x=x, sensi_orders=(0,), mode=MODE_RES, **kwargs)[RES]

        # calculate value at x only once if needed
        if res is None and self.method in [FD.FORWARD, FD.BACKWARD]:
            res = f_res(x)

        sres = []
        for ix in range(n_par):
            delta_val = self.get_delta_res(par_ix=ix)
            delta = delta_val * unit_vec(dim=n_par, ix=ix)

            if self.method == FD.CENTRAL:
                rp = f_res(x + delta / 2)
                rm = f_res(x - delta / 2)
            elif self.method == FD.FORWARD:
                rp = f_res(x + delta)
                rm = res
            elif self.method == FD.BACKWARD:
                rp = res
                rm = f_res(x - delta)
            else:
                raise ValueError(f"Method {self.method} not recognized.")

            sres.append((rp - rm) / delta_val)

        # sres should have shape (n_res, n_par)
        sres = np.array(sres).T

        return sres


def unit_vec(dim: int, ix: int) -> np.ndarray:
    """Unit vector of dimension `dim` at coordinate `ix`.

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
