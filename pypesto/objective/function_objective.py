import numpy as np

from .objective import Objective
from typing import Callable, Dict, Sequence, Tuple, Union

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES


class FunctionObjective(Objective):
    """
    The objective class is a simple wrapper around the objective function,
    giving a standardized way of calling. Apart from that, it manages several
    things including fixing of parameters and history.

    The objective function is assumed to be in the format of a cost function,
    log-likelihood function, or log-posterior function. These functions are
    subject to minimization. For profiling and sampling, the sign is internally
    flipped, all returned and stored values are however given as returned
    by this objective function. If maximization is to be performed, the sign
    should be flipped before creating the objective function.

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

    def _check_sensi_orders(self, sensi_orders, mode) -> None:
        """
        Check if the objective is able to compute the requested
        sensitivities. If not, throw an exception.

        Raises
        ------
        ValueError if the objective function cannot be called as
        requested.
        """
        if (mode is MODE_FUN and
            (0 in sensi_orders and not self.has_fun
             or 1 in sensi_orders and not self.has_grad
             or 2 in sensi_orders and not self.has_hess)
            ) or (mode is MODE_RES and
                  (0 in sensi_orders and not self.has_res
                   or 1 in sensi_orders and not self.has_sres)
                  ):
            raise ValueError(
                f"Objective cannot be called with sensi_orders={sensi_orders}"
                f" and mode={mode}")

    def call_unprocessed(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str
    ) -> Dict:
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
    ) -> Dict:
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
    ) -> Dict:
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

