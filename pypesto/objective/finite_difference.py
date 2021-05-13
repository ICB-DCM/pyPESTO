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
    objective derivative is used if possible, otherwise resorting to FDs.
    True means that FDs are used in any case, False means that the derivative
    is not exported.

    Parameters
    ----------
    grad:
        Derivative method for the gradient.
    hess:
        Derivative method for the Hessian
    sres:
        Derivative method for the residual sensitivities.
    hess_via_fval:
        If the Hessian is to be calculated via finite differences, whether
        to employ 2nd order FDs via fval even if the objective can provide one,
        or 1st order FDs from gradients if available.
    delta_fun:
        FD step sizes for gradient and Hessian.
        Can be either a float, or a :class:`np.ndarray` of shape (n_par,)
        for different step sizes for different coordinates.
    delta_res:
        FD step sizes for residual sensitivities.
        Similar to `delta_fun`.
    method:
        Method to calculate FDs. Currently, only "center" is supported.
    x_names:
        Parameter names that can be optionally used in, e.g., history or
        gradient checks.
    """

    def __init__(
        self,
        obj: ObjectiveBase,
        grad: Union[bool, None] = None,
        hess: Union[bool, None] = None,
        sres: Union[bool, None] = None,
        hess_via_fval: bool = True,
        delta_fun: Union[float, np.ndarray, str] = 1e-6,
        delta_res: Union[float, np.ndarray, str] = 1e-6,
        method: str = "center",
        x_names: List[str] = None,
    ):
        super().__init__(x_names=x_names)
        self.obj: ObjectiveBase = obj
        self.grad: Union[bool, None] = grad
        self.hess: Union[bool, None] = hess
        self.sres: Union[bool, None] = sres
        self.hess_via_fval: bool = hess_via_fval
        self.delta_fun: Union[float, np.ndarray] = delta_fun
        self.delta_res: Union[float, np.ndarray] = delta_res
        self.method: str = method

        if isinstance(delta_fun, str) or isinstance(delta_res, str):
            raise NotImplementedError(
                "Adaptive FD step sizes are not implemented yet.",
            )
        if method != 'center':
            raise NotImplementedError(
                "Currently only centered finite differences are implemented.",
            )

    def get_delta_fun(self, par_ix: int) -> float:
        """Get function value epsilon for a given parameter index.

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
        """Get residual epsilon for a given parameter index.

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

        #  define objective sensis
        sensi_orders_obj = []
        if 0 in sensi_orders:
            sensi_orders_obj.append(0)
        if 1 in sensi_orders and self.grad is None and self.obj.has_grad:
            sensi_orders_obj.append(1)
        if 2 in sensi_orders and self.hess is None and self.obj.has_hess:
            sensi_orders_obj.append(2)
        sensi_orders_obj = tuple(sensi_orders_obj)
        #  call objective
        result = {}
        if sensi_orders_obj:
            result = self.obj.call_unprocessed(
                x=x, sensi_orders=sensi_orders_obj, mode=MODE_FUN, **kwargs)

        # remaining sensis via FDs

        # indicate whether gradient and Hessian are intended as FDs
        grad_via_fd = 1 in sensi_orders and 1 not in sensi_orders_obj
        hess_via_fd = 2 in sensi_orders and 2 not in sensi_orders_obj

        if not grad_via_fd and not hess_via_fd:
            return result

        # indicate whether the Hessian should be based on 2nd order sensis
        hess_via_fd_fval = \
            hess_via_fd and (self.hess_via_fval or not self.obj.has_grad)

        # parameter dimension
        n_par = len(x)

        # calculate 1d differences
        diffs_1d = []
        if grad_via_fd or hess_via_fd:
            sensis = []
            if grad_via_fd or hess_via_fd_fval:
                sensis.append(0)
            if hess_via_fd and not hess_via_fd_fval:
                sensis.append(1)
            sensis = tuple(sensis)

            for ix in range(n_par):
                delta = self.get_delta_fun(par_ix=ix) * \
                    unit_vec(dim=n_par, ix=ix)

                ret_p = self.obj.call_unprocessed(
                    x=x + delta, sensi_orders=sensis, mode=MODE_FUN, **kwargs)
                ret_m = self.obj.call_unprocessed(
                    x=x - delta, sensi_orders=sensis, mode=MODE_FUN, **kwargs)

                fp, fm = ret_p.get(FVAL), ret_m.get(FVAL)
                gp, gm = ret_p.get(GRAD), ret_m.get(GRAD)

                diffs_1d.append(((fp, fm), (gp, gm)))

        if grad_via_fd:
            # gradient via FDs
            grad = np.nan * np.empty(shape=n_par)
            for ix, ((fp, fm), _) in enumerate(diffs_1d):
                grad[ix] = (fp - fm) / (2 * self.get_delta_fun(par_ix=ix))
            result[GRAD] = grad

        if hess_via_fd:
            # Hessian via FDs
            hess = np.nan * np.empty(shape=(n_par, n_par))
            if hess_via_fd_fval:
                # Hessian via 2nd order FDs from function values
                def f_fval(x):
                    """Short-hand to get a function value."""
                    return self.obj.call_unprocessed(
                        x=x, sensi_orders=(0,), mode=MODE_FUN, **kwargs)[FVAL]

                # needed for diagonal entries
                f = result.get(FVAL, None)
                if f is None:
                    f = f_fval(x)

                for ix1, ((fp, fm), _) in enumerate(diffs_1d):
                    delta1_val = self.get_delta_fun(par_ix=ix1)
                    delta1 = delta1_val * unit_vec(dim=n_par, ix=ix1)

                    # diagonal entry
                    hess[ix1, ix1] = (fp + fm - 2 * f) / delta1_val**2

                    # off-diagonals
                    for ix2 in range(ix1):
                        delta2_val = self.get_delta_fun(par_ix=ix2)
                        delta2 = delta2_val * unit_vec(dim=n_par, ix=ix2)

                        fpp = f_fval(x + delta1 + delta2)
                        fpm = f_fval(x + delta1 - delta2)
                        fmp = f_fval(x - delta1 + delta2)
                        fmm = f_fval(x - delta1 - delta2)

                        hess[ix1, ix2] = \
                            (fpp - fpm - fmp + fmm) / \
                            (4 * delta1_val * delta2_val)
                        hess[ix2, ix1] = hess[ix1, ix2]

            else:
                # Hessian via 1st order FDs from gradients
                for ix, (_, (gp, gm)) in enumerate(diffs_1d):
                    hess[:, ix] = \
                        (gp - gm) / (2 * self.get_delta_fun(par_ix=ix))
                # make it symmetric
                hess = 0.5 * (hess + hess.T)

            result[HESS] = hess

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

        #  define objective sensis
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

        if sensi_orders == sensi_orders_obj:
            # all done
            return result

        n_par = len(x)
        sres = []

        for ix in range(n_par):
            delta_val = self.get_delta_res(par_ix=ix)
            delta = delta_val * unit_vec(dim=n_par, ix=ix)

            rp = self.obj.call_unprocessed(
                x=x + delta, sensi_orders=(0,), mode=MODE_RES, **kwargs)[RES]
            rm = self.obj.call_unprocessed(
                x=x - delta, sensi_orders=(0,), mode=MODE_RES, **kwargs)[RES]

            sres.append((rp - rm) / (2 * delta_val))

        # sres should have shape (n_res, n_par)
        sres = np.array(sres).T
        result[SRES] = sres

        return result


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
