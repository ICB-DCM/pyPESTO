import numpy as np
import copy
import pandas as pd
import logging
from typing import Callable, Dict, List, Tuple, Union

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .history import HistoryBase
from .pre_post_process import PrePostProcessor, FixedParametersProcessor

logger = logging.getLogger(__name__)


class Objective:
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

    fun_accept_sensi_orders:
        Flag indicating whether fun takes sensi_orders as an argument.
        Default: False.

    res_accept_sensi_orders:
        Flag indicating whether res takes sensi_orders as an argument.
        Default: False

    x_names:
        Parameter names. None if no names provided, otherwise a list of str,
        length dim_full (as in the Problem class). Can be read by the
        problem.

    Attributes
    ----------

    history:
        For storing the call history. Initialized by the methods, e.g. the
        optimizer, in `initialize_history()`.

    pre_post_processor:
        Preprocess input values to and postprocess output values from
        __call__. Configured in `update_from_problem()`.

    Notes
    -----

    If fun_accept_sensi_orders resp. res_accept_sensi_orders is True,
    fun resp. res can also return dictionaries instead of tuples.
    In that case, they are expected to follow the naming conventions
    in ``constants.py``. This is of interest, because when __call__ is
    called with return_dict = True, the full dictionary is returned, which
    can contain e.g. also simulation data or debugging information.
    """

    def __init__(self,
                 fun: Callable = None,
                 grad: Union[Callable, bool] = None,
                 hess: Callable = None,
                 hessp: Callable = None,
                 res: Callable = None,
                 sres: Union[Callable, bool] = None,
                 fun_accept_sensi_orders: bool = False,
                 res_accept_sensi_orders: bool = False,
                 x_names: List[str] = None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        self.fun_accept_sensi_orders = fun_accept_sensi_orders
        self.res_accept_sensi_orders = res_accept_sensi_orders

        self.x_names = x_names

        self.pre_post_processor = PrePostProcessor()
        self.history = HistoryBase()

    def __deepcopy__(self, memodict=None) -> 'Objective':
        other = Objective()
        for attr in self.__dict__:
            other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    # The following has_ properties can be used to find out what values
    # the objective supports.

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

    def check_sensi_orders(self, sensi_orders, mode) -> None:
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

    def __call__(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...] = (0, ),
            mode: str = MODE_FUN,
            return_dict: bool = False
    ) -> Union[float, np.ndarray, Tuple, Dict]:
        """
        Method to obtain arbitrary sensitivities. This is the central method
        which is always called, also by the get_* methods.

        There are different ways in which an optimizer calls the objective
        function, and in how the objective function provides information
        (e.g. derivatives via separate functions or along with the function
        values). The different calling modes increase efficiency in space
        and time and make the objective flexible.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the objective function.
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.
        return_dict:
            If False (default), the result is a Tuple of the requested values
            in the requested order. Tuples of length one are flattened.
            If True, instead a dict is returned which can carry further
            information.

        Returns
        -------
        result:
            By default, this is a tuple of the requested function values
            and derivatives in the requested order (if only 1 value, the tuple
            is flattened). If `return_dict`, then instead a dict is returned
            with function values and derivatives indicated by ids.
        """

        # check input
        self.check_sensi_orders(sensi_orders, mode)

        # pre-process
        x_full = self.pre_post_processor.preprocess(x)

        # compute result
        result = self._call_unprocessed(x_full, sensi_orders, mode)

        # post-process
        result = self.pre_post_processor.postprocess(result)

        # update history
        self.history.update(x, sensi_orders, mode, result)

        # map to output format
        if not return_dict:
            result = Objective.output_to_tuple(sensi_orders, mode, **result)

        return result

    def _call_unprocessed(
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
        """
        The method __call__ was called with mode MODE_FUN.
        """
        if self.fun_accept_sensi_orders:
            result = self.fun(x, sensi_orders)
            if not isinstance(result, dict):
                result = Objective.output_to_dict(
                    sensi_orders, MODE_FUN, result)
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
        """
        The method __call__ was called with mode MODE_RES.
        """
        if self.res_accept_sensi_orders:
            result = self.res(x, sensi_orders)
            if not isinstance(result, dict):
                result = Objective.output_to_dict(
                    sensi_orders, MODE_RES, result)
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
            result = {RES: res,
                      SRES: sres}
        else:
            raise ValueError("These sensitivity orders are not supported.")
        return result

    @staticmethod
    def output_to_dict(
            sensi_orders: Tuple[int, ...], mode: str, output_tuple: Tuple
    ) -> Dict:
        """
        Convert output tuple to dict.
        """
        output_dict = {}
        index = 0
        if not isinstance(output_tuple, tuple):
            output_tuple = (output_tuple,)
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                output_dict[FVAL] = output_tuple[index]
                index += 1
            if 1 in sensi_orders:
                output_dict[GRAD] = output_tuple[index]
                index += 1
            if 2 in sensi_orders:
                output_dict[HESS] = output_tuple[index]
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                output_dict[RES] = output_tuple[index]
                index += 1
            if 1 in sensi_orders:
                output_dict[SRES] = output_tuple[index]
        return output_dict

    @staticmethod
    def output_to_tuple(
            sensi_orders: Tuple[int, ...], mode: str, **kwargs
    ) -> Tuple:
        """
        Return values as requested by the caller, since usually only a subset
        is demanded. One output is returned as-is, more than one output are
        returned as a tuple in order (fval, grad, hess).
        """
        output = ()
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                output += (kwargs[FVAL],)
            if 1 in sensi_orders:
                output += (kwargs[GRAD],)
            if 2 in sensi_orders:
                output += (kwargs[HESS],)
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                output += (kwargs[RES],)
            if 1 in sensi_orders:
                output += (kwargs[SRES],)
        if len(output) == 1:
            output = output[0]
        return output

    # The following are convenience functions for getting specific outputs.

    def get_fval(self, x: np.ndarray) -> float:
        """
        Get the function value at x.
        """
        fval = self(x, (0,), MODE_FUN)
        return fval

    def get_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Get the gradient at x.
        """
        grad = self(x, (1,), MODE_FUN)
        return grad

    def get_hess(self, x: np.ndarray) -> np.ndarray:
        """
        Get the Hessian at x.
        """
        hess = self(x, (2,), MODE_FUN)
        return hess

    def get_res(self, x: np.ndarray) -> np.ndarray:
        """
        Get the residuals at x.
        """
        res = self(x, (0,), MODE_RES)
        return res

    def get_sres(self, x: np.ndarray) -> np.ndarray:
        """
        Get the residual sensitivities at x.
        """
        sres = self(x, (1,), MODE_RES)
        return sres

    def update_from_problem(
            self,
            dim_full: int,
            x_free_indices: List[int],
            x_fixed_indices: List[int],
            x_fixed_vals: List[int]):
        """
        Handle fixed parameters. Later, the objective will be given parameter
        vectors x of dimension dim, which have to be filled up with fixed
        parameter values to form a vector of dimension dim_full >= dim.
        This vector is then used to compute function value and derivatives.
        The derivatives must later be reduced again to dimension dim.

        This is so as to make the fixing of parameters transparent to the
        caller.

        The methods preprocess, postprocess are overwritten for the above
        functionality, respectively.

        Parameters
        ----------
        dim_full:
            Dimension of the full vector including fixed parameters.
        x_free_indices:
            Vector containing the indices (zero-based) of free parameters
            (complimentary to x_fixed_indices).
        x_fixed_indices:
            Vector containing the indices (zero-based) of parameter components
            that are not to be optimized.
        x_fixed_vals:
            Vector of the same length as x_fixed_indices, containing the values
            of the fixed parameters.
        """

        pre_post_processor = FixedParametersProcessor(
            dim_full=dim_full,
            x_free_indices=x_free_indices,
            x_fixed_indices=x_fixed_indices,
            x_fixed_vals=x_fixed_vals)

        self.pre_post_processor = pre_post_processor

    def check_grad(
            self,
            x: np.ndarray,
            x_indices: List[int] = None,
            eps: float = 1e-5,
            verbosity: int = 1,
            mode: str = MODE_FUN
    ) -> pd.DataFrame:
        """
        Compare gradient evaluation: Firstly approximate via finite
        differences, and secondly use the objective gradient.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the gradient.
        x_indices:
            List of index values for which to compute gradients. Default: all.
        eps:
            Finite differences step size. Default: 1e-5.
        verbosity:
            Level of verbosity for function output.
            * 0: no output,
            * 1: summary for all parameters,
            * 2: summary for individual parameters.
            Default: 1.
        mode:
            Residual (MODE_RES) or objective function value
            (MODE_FUN, default) computation mode.

        Returns
        ----------
        result:
            gradient, finite difference approximations and error estimates.
        """

        if x_indices is None:
            x_indices = list(range(len(x)))

        if hasattr(self.history, 'options'):
            tmp_trace_record = self.history.options.trace_record
            self.history.options.trace_record = False

        # function value and objective gradient
        fval, grad = self(x, (0, 1), mode)

        grad_list = []
        fd_f_list = []
        fd_b_list = []
        fd_c_list = []
        fd_err_list = []
        abs_err_list = []
        rel_err_list = []

        # loop over indices
        for ix in x_indices:
            # forward (plus) point
            x_p = copy.deepcopy(x)
            x_p[ix] += eps
            fval_p = self(x_p, (0,), mode)

            # backward (minus) point
            x_m = copy.deepcopy(x)
            x_m[ix] -= eps
            fval_m = self(x_m, (0,), mode)

            # finite differences
            fd_f_ix = (fval_p - fval) / eps
            fd_b_ix = (fval - fval_m) / eps
            fd_c_ix = (fval_p - fval_m) / (2 * eps)

            # gradient in direction ix
            grad_ix = grad[ix] if grad.ndim == 1 else grad[:, ix]

            # errors
            fd_err_ix = abs(fd_f_ix - fd_b_ix)
            abs_err_ix = abs(grad_ix - fd_c_ix)
            rel_err_ix = abs(abs_err_ix / (fd_c_ix + eps))

            # log for dimension ix
            if verbosity > 1:
                logger.info(
                    f'index:    {ix}\n'
                    f'grad:     {grad_ix}\n'
                    f'fd_f:     {fd_f_ix}\n'
                    f'fd_b:     {fd_b_ix}\n'
                    f'fd_c:     {fd_c_ix}\n'
                    f'fd_err:   {fd_err_ix}\n'
                    f'abs_err:  {abs_err_ix}\n'
                    f'rel_err:  {rel_err_ix}\n'
                )

            # append to lists
            grad_list.append(grad_ix)
            fd_f_list.append(fd_f_ix)
            fd_b_list.append(fd_b_ix)
            fd_c_list.append(fd_c_ix)
            fd_err_list.append(np.mean(fd_err_ix))
            abs_err_list.append(np.mean(abs_err_ix))
            rel_err_list.append(np.mean(rel_err_ix))

        # create dataframe
        result = pd.DataFrame(data={
            'grad': grad_list,
            'fd_f': fd_f_list,
            'fd_b': fd_b_list,
            'fd_c': fd_c_list,
            'fd_err': fd_err_list,
            'abs_err': abs_err_list,
            'rel_err': rel_err_list,
        })

        # log full result
        if verbosity > 0:
            logger.info(result)

        if hasattr(self.history, 'options'):
            self.history.options.trace_record = tmp_trace_record

        return result
