import numpy as np

from .base import ObjectiveBase, ResultDict
from typing import Callable, Sequence, Tuple, Union

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES


class Objective(ObjectiveBase):
    """
    The objective class allows the user explicitely specify functions that
    compute the function value and/or residuals as well as respective
    derivatives.

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

            ``hess(x) -> array, shape (n,n).``

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
        Method for computing residual sensitivities. If its is a callable,
        it should be of the form

            ``sres(x) -> array, shape (m,n).``

        If its value is True, then res should return the residual
        sensitivities as a second output.

    x_names:
        Parameter names. None if no names provided, otherwise a list of str,
        length dim_full (as in the Problem class). Can be read by the
        problem.
    """
    def __init__(self, fun: Callable = None,
                 grad: Union[Callable, bool] = None, hess: Callable = None,
                 hessp: Callable = None, res: Callable = None,
                 sres: Union[Callable, bool] = None,
                 x_names: Sequence[str] = None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        super().__init__(x_names)

    @property
    def has_fun(self) -> bool:
        return callable(self.fun)

    @property
    def has_grad(self) -> bool:
        return callable(self.grad) or self.grad is True

    @property
    def has_hess(self) -> bool:
        return callable(self.hess) or self.hess is True

    @property
    def has_hessp(self) -> bool:
        # Not supported yet
        return False

    @property
    def has_res(self) -> bool:
        return callable(self.res)

    @property
    def has_sres(self) -> bool:
        return callable(self.sres) or self.sres is True

    def check_sensi_orders(self, sensi_orders, mode):
        if (mode is MODE_FUN and
            (0 in sensi_orders and not self.has_fun
             or 1 in sensi_orders and not self.has_grad
             or 2 in sensi_orders and not self.has_hess)
            ) or (mode is MODE_RES and
                  (0 in sensi_orders and not self.has_res
                   or 1 in sensi_orders and not self.has_sres)
                  ):
            return False

        return True

    def check_mode(self, mode):
        if mode == MODE_FUN and not self.has_fun:
            return False

        if mode == MODE_RES and not self.has_res:
            return False

        return True

    def call_unprocessed(self, x, sensi_orders, mode):
        """
        Call objective function without pre- or post-processing and
        formatting.

        Returns
        -------
        result:
            A dict containing the results.
        """
        if mode == MODE_FUN:
            result = self._call_mode_fun(x, sensi_orders)
        elif mode == MODE_RES:
            result = self._call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")
        return result

    def _call_mode_fun(
            self, x: np.ndarray, sensi_orders: Tuple[int, ...]
    ) -> ResultDict:
        if sensi_orders == (0,):
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
                fval, grad = self.fun(x)[0:2]
            else:
                fval = self.fun(x)
                grad = self.grad(x)
            result = {FVAL: fval,
                      GRAD: grad}
        elif sensi_orders == (1, 2):
            if self.hess is True:
                grad, hess = self.fun(x)[1:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    grad = self.fun(x)[1]
                else:
                    grad = self.grad(x)
            result = {GRAD: grad,
                      HESS: hess}
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
            result = {FVAL: fval,
                      GRAD: grad,
                      HESS: hess}
        else:
            raise ValueError("These sensitivity orders are not supported.")

        return result

    def _call_mode_res(
            self, x: np.ndarray, sensi_orders: Tuple[int, ...]
    ) -> ResultDict:
        if sensi_orders == (0,):
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
            result = {RES: res,
                      SRES: sres}
        else:
            raise ValueError("These sensitivity orders are not supported.")

        return result
