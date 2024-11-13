from collections.abc import Sequence
from typing import Callable, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, MODE_FUN, MODE_RES, RES, SRES, ModeType
from .base import ObjectiveBase, ResultDict


class Objective(ObjectiveBase):
    """
    Objective class.

    The objective class allows the user explicitly specify functions that
    compute the function value and/or residuals as well as respective
    derivatives.

    Denote dimensions `n` = parameters, `m` = residuals.

    Parameters
    ----------
    fun:
        The objective function to be minimized. If it only computes the
        objective function value, it should be of the form

            ``fun(x) -> float``

        where x is an 1-D array with shape (n,), and n is the parameter space
        dimension.
    grad:
        Method for computing the gradient vector. If it is a callable,
        it should be of the form

            ``grad(x) -> array_like, shape (n,).``

        If its value is True, then fun should return the gradient as a second
        output.
    hess:
        Method for computing the Hessian matrix. If it is a callable,
        it should be of the form

            ``hess(x) -> array, shape (n, n).``

        If its value is True, then fun should return the gradient as a
        second, and the Hessian as a third output, and grad should be True as
        well.
    hessp:
        Method for computing the Hessian vector product, i.e.

            ``hessp(x, v) -> array_like, shape (n,)``

        computes the product H*v of the Hessian of fun at x with v.
    res:
        Method for computing residuals, i.e.

            ``res(x) -> array_like, shape(m,).``
    sres:
        Method for computing residual sensitivities. If it is a callable,
        it should be of the form

            ``sres(x) -> array, shape (m, n).``

        If its value is True, then res should return the residual
        sensitivities as a second output.
    x_names:
        Parameter names. None if no names provided, otherwise a list of str,
        length dim_full (as in the Problem class). Can be read by the
        problem.
    """

    def __init__(
        self,
        fun: Callable = None,
        grad: Union[Callable, bool] = None,
        hess: Callable = None,
        hessp: Callable = None,
        res: Callable = None,
        sres: Union[Callable, bool] = None,
        x_names: Sequence[str] = None,
    ):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        super().__init__(x_names)

    @property
    def has_fun(self) -> bool:
        """Check whether function is defined."""
        return callable(self.fun)

    @property
    def has_grad(self) -> bool:
        """Check whether gradient is defined."""
        return callable(self.grad) or self.grad is True

    @property
    def has_hess(self) -> bool:
        """Check whether Hessian is defined."""
        return callable(self.hess) or self.hess is True

    @property
    def has_hessp(self) -> bool:
        """Check whether Hessian vector product is defined."""
        # Not supported yet
        return False

    @property
    def has_res(self) -> bool:
        """Check whether residuals are defined."""
        return callable(self.res)

    @property
    def has_sres(self) -> bool:
        """Check whether residual sensitivities are defined."""
        return callable(self.sres) or self.sres is True

    def get_config(self) -> dict:
        """Return basic information of the objective configuration."""
        info = super().get_config()
        info["x_names"] = self.x_names
        sensi_order = 0
        while self.check_sensi_orders(
            sensi_orders=(sensi_order,), mode=MODE_FUN
        ):
            sensi_order += 1
        info["sensi_order"] = sensi_order - 1
        return info

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        return_dict: bool,
        **kwargs,
    ) -> ResultDict:
        """
        Call objective function without pre- or post-processing and formatting.

        Returns
        -------
        result:
            A dict containing the results.
        """
        if mode == MODE_FUN:
            result = self._call_mode_fun(x=x, sensi_orders=sensi_orders)
        elif mode == MODE_RES:
            result = self._call_mode_res(x=x, sensi_orders=sensi_orders)
        else:
            raise ValueError("This mode is not supported.")
        return result

    def _call_mode_fun(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
    ) -> ResultDict:
        if not sensi_orders:
            result = {}

        elif sensi_orders == (0,):
            if self.grad is True:
                fval = self.fun(x)[0]
            else:
                fval = self.fun(x)
            result = {FVAL: fval}

        elif sensi_orders == (1,):
            if self.grad is True:
                grad = self.fun(x)[1]
            else:
                grad = self.grad(x)
            result = {GRAD: grad}
        elif sensi_orders == (2,):
            if self.hess is True:
                hess = self.fun(x)[2]
            else:
                hess = self.hess(x)
            result = {HESS: hess}
        elif sensi_orders == (0, 1):
            if self.grad is True:
                fval, grad = self.fun(x)[:2]
            else:
                fval = self.fun(x)
                grad = self.grad(x)
            result = {FVAL: fval, GRAD: grad}
        elif sensi_orders == (0, 2):
            if self.hess is True:
                fval, _, hess = self.fun(x)[:3]
            else:
                if self.grad is True:
                    fval = self.fun(x)[0]
                else:
                    fval = self.fun(x)
                hess = self.hess(x)
            result = {FVAL: fval, HESS: hess}
        elif sensi_orders == (1, 2):
            if self.hess is True:
                grad, hess = self.fun(x)[1:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    grad = self.fun(x)[1]
                else:
                    grad = self.grad(x)
            result = {GRAD: grad, HESS: hess}
        elif sensi_orders == (0, 1, 2):
            if self.hess is True:
                fval, grad, hess = self.fun(x)[0:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    fval, grad = self.fun(x)[0:2]
                else:
                    fval = self.fun(x)
                    grad = self.grad(x)
            result = {FVAL: fval, GRAD: grad, HESS: hess}
        else:
            raise ValueError("These sensitivity orders are not supported.")

        return result

    def _call_mode_res(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
    ) -> ResultDict:
        if not sensi_orders:
            result = {}

        elif sensi_orders == (0,):
            if self.sres is True:
                res = self.res(x)[0]
            else:
                res = self.res(x)
            result = {RES: res}

        elif sensi_orders == (1,):
            if self.sres is True:
                sres = self.res(x)[1]
            else:
                sres = self.sres(x)
            result = {SRES: sres}
        elif sensi_orders == (0, 1):
            if self.sres is True:
                res, sres = self.res(x)
            else:
                res = self.res(x)
                sres = self.sres(x)
            result = {RES: res, SRES: sres}
        else:
            raise ValueError("These sensitivity orders are not supported.")

        return result
