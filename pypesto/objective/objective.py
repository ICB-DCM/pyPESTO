import numpy as np
import copy
import pandas as pd
import logging
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .history import ObjectiveHistory
from .options import ObjectiveOptions


logger = logging.getLogger(__name__)


class Objective:
    """
    The objective class is a simple wrapper around the objective function,
    giving a standardized way of calling. Apart from that, it manages several
    things including fixing of parameters and history.

    Parameters
    ----------

    fun: callable, optional
        The objective function to be minimized. If it only computes the
        objective function value, it should be of the form
            ``fun(x) -> float``
        where x is an 1-D array with shape (n,), and n is the parameter space
        dimension.

    grad: callable, bool, optional
        Method for computing the gradient vector. If it is a callable,
        it should be of the form
            ``grad(x) -> array_like, shape (n,).``
        If its value is True, then fun should return the gradient as a second
        output.

    hess: callable, optional
        Method for computing the Hessian matrix. If it is a callable,
        it should be of the form
            ``hess(x) -> array, shape (n,n).``
        If its value is True, then fun should return the gradient as a
        second, and the Hessian as a third output, and grad should be True as
        well.

    hessp: callable, optional
        Method for computing the Hessian vector product, i.e.
            ``hessp(x, v) -> array_like, shape (n,)``
        computes the product H*v of the Hessian of fun at x with v.

    res: {callable, bool}, optional
        Method for computing residuals, i.e.
            ``res(x) -> array_like, shape(m,).``

    sres: callable, optional
        Method for computing residual sensitivities. If its is a callable,
        it should be of the form
            ``sres(x) -> array, shape (m,n).``
        If its value is True, then res should return the residual
        sensitivities as a second output.

    fun_accept_sensi_orders: bool, optional
        Flag indicating whether fun takes sensi_orders as an argument.
        Default: False.

    res_accept_sensi_orders: bool, optional
        Flag indicating whether res takes sensi_orders as an argument.
        Default: False

    options: pypesto.ObjectiveOptions, optional
        Options as specified in pypesto.ObjectiveOptions.

    Attributes
    ----------

    history: pypesto.ObjectiveHistory
        For storing the call history. Initialized by the optimizer in
        reset_history().

    x_names: list of str
        Parameter names. The base Objective class provides None.
        None if no names provided, otherwise a list of str, length dim_full.
        Can be read by the problem.

    preprocess: callable
        Preprocess input values to __call__.

    postprocess: callable
        Postprocess output values from __call__.

    sensitivity_orders: tuple
        Temporary variable to store requested sensitivity orders

    Notes
    -----

    preprocess, postprocess are configured in update_from_problem()
    and can be reset using the reset() method.
    """

    def __init__(self,
                 fun=None, grad=None, hess=None, hessp=None,
                 res=None, sres=None,
                 fun_accept_sensi_orders=False,
                 res_accept_sensi_orders=False,
                 options=None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        self.fun_accept_sensi_orders = fun_accept_sensi_orders
        self.res_accept_sensi_orders = res_accept_sensi_orders

        if options is None:
            options = ObjectiveOptions()
        self.options = ObjectiveOptions.assert_instance(options)

        self.history = ObjectiveHistory(self.options)

        def preprocess(x):
            return np.array(x)

        def postprocess(result):
            return result

        self.preprocess = preprocess
        self.postprocess = postprocess

        self.x_names = None

    # The following has_ properties can be used to find out what values
    # the objective supports.

    @property
    def has_fun(self):
        return callable(self.fun)

    @property
    def has_grad(self):
        return callable(self.grad) or self.grad is True

    @property
    def has_hess(self):
        return callable(self.hess) or self.hess is True

    @property
    def has_hessp(self):
        # Not supported yet
        return False

    @property
    def has_res(self):
        return callable(self.res)

    @property
    def has_sres(self):
        return callable(self.sres) or self.sres is True

    def __call__(self, x, sensi_orders: tuple=(0,), mode=MODE_FUN):
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

        x: array_like
            The parameters for which to evaluate the objective function.

        sensi_orders: tuple
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.

        mode: str
            Whether to compute function values or residuals.
        """

        # pre-process
        x = self.preprocess(x=x)

        # compute result
        result = self._call_unprocessed(x, sensi_orders, mode)

        # convert to ndarray
        result = Objective.as_ndarrays(result)

        # post-process
        result = self.postprocess(result=result)

        # update history
        self.history.update(x, sensi_orders, mode, result)

        # map to output format
        result = Objective.output_to_tuple(sensi_orders, mode, **result)

        return result

    def _call_unprocessed(self, x, sensi_orders, mode):
        """
        Call objective function without pre- or post-processing and
        formatting.
        """
        if mode == MODE_FUN:
            result = self._call_mode_fun(x, sensi_orders)
        elif mode == MODE_RES:
            result = self._call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")
        return result

    def _call_mode_fun(self, x, sensi_orders):
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

    def _call_mode_res(self, x, sensi_orders):
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
    def as_ndarrays(result):
        """
        Convert all array_like objects to numpy arrays. This has the advantage
        of a uniform output datatype which offers various methods to assess
        the data.
        """
        keys = [GRAD, HESS, RES, SRES]
        for key in keys:
            if key in result:
                value = result[key]
                if value is not None:
                    result[key] = np.array(value)

        return result

    @staticmethod
    def output_to_dict(sensi_orders, mode, output_tuple):
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
    def output_to_tuple(sensi_orders, mode, **kwargs):
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

    def get_fval(self, x):
        """
        Get the function value at x.
        """
        fval = self(x, (0,), MODE_FUN)
        return fval

    def get_grad(self, x):
        """
        Get the gradient at x.
        """
        grad = self(x, (1,), MODE_FUN)
        return grad

    def get_hess(self, x):
        """
        Get the Hessian at x.
        """
        hess = self(x, (2,), MODE_FUN)
        return hess

    def get_res(self, x):
        """
        Get the residuals at x.
        """
        res = self(x, (0,), MODE_RES)
        return res

    def get_sres(self, x):
        """
        Get the residual sensitivities at x.
        """
        sres = self(x, (1,), MODE_RES)
        return sres

    # The following are functions that are called by other parts in
    # pypesto to modify the objective state, e.g. set its history, or
    # make it aware of fixed parameters.

    def reset_history(self, index=None):
        """
        Reset the objective history and specify temporary saving options.

        Parameters
        ----------

        index: As in ObjectiveHistory.index.
        """
        self.history.reset(index=index)

    def finalize_history(self):
        """
        Finalize the history object.
        """
        self.history.finalize()

    def reset(self):
        """
        Completely reset the objective, i.e. undo the modifications in
        update_from_problem().
        """
        self.history = ObjectiveHistory(self.options)

        def preprocess(x):
            return np.array(x)

        def postprocess(result):
            return result

        self.preprocess = preprocess
        self.postprocess = postprocess

    def update_from_problem(self,
                            dim_full,
                            x_free_indices,
                            x_fixed_indices,
                            x_fixed_vals):
        """
        Handle fixed parameters. Later, the objective will be given parameter
        vectors x of dimension dim, which have to be filled up with fixed
        parameter values to form a vector of dimension dim_full >= dim.
        This vector is then used to compute function value and derivaties.
        The derivatives must later be reduced again to dimension dim.

        This is so as to make the fixing of parameters transparent to the
        caller.

        The methods preprocess, postprocess are overwritten for the above
        functionality, respectively.

        Parameters
        ----------

        dim_full: int
            Dimension of the full vector including fixed parameters.

        x_free_indices: array_like of int
            Vector containing the indices (zero-based) of free parameters
            (complimentary to x_fixed_indices).

        x_fixed_indices: array_like of int, optional
            Vector containing the indices (zero-based) of parameter components
            that are not to be optimized.

        x_fixed_vals: array_like, optional
            Vector of the same length as x_fixed_indices, containing the values
            of the fixed parameters.
        """

        # pre-process
        def preprocess(x):
            x_full = np.zeros(dim_full)
            x_full[x_free_indices] = x
            x_full[x_fixed_indices] = x_fixed_vals
            return x_full
        self.preprocess = preprocess

        # post-process
        def postprocess(result):
            if GRAD in result:
                grad = result[GRAD]
                if grad.size == dim_full:
                    grad = grad[x_free_indices]
                    result[GRAD] = grad
            if HESS in result:
                hess = result[HESS]
                if hess.shape[0] == dim_full:
                    hess = hess[np.ix_(x_free_indices, x_free_indices)]
                    result[HESS] = hess
            if RES in result:
                res = result[RES]
                if res.size == dim_full:
                    res = res.flatten()[x_free_indices]
                    result[RES] = res
            if SRES in result:
                sres = result[SRES]
                if sres.shape[-1] == dim_full:
                    sres = sres[..., x_free_indices]
                    result[SRES] = sres
            return result
        self.postprocess = postprocess

    def check_grad(self,
                   x,
                   x_indices=None,
                   eps=1e-5,
                   verbosity=1,
                   mode=MODE_FUN) -> pd.DataFrame:
        """
        Compare gradient evaluation: Firstly approximate via finite
        differences, and secondly use the objective gradient.

        Parameters
        ----------

        x: array_like
            The parameters for which to evaluate the gradient.

        x_indices: array_like, optional
            List of index values for which to compute gradients. Default: all.

        eps: float, optional
            Finite differences step size. Default: 1e-5.

        verbosity: int
            Level of verbosity for function output
                0: no output
                1: summary for all parameters
                2: summary for individual parameters
            Default: 1.

        mode: str
            Residual (MODE_RES) or objective function value
            (MODE_FUN, default) computation mode.

        Returns
        ----------

        result: pd.DataFrame
            gradient, finite difference approximations and error estimates.

        """

        if x_indices is None:
            x_indices = list(range(len(x)))

        tmp_trace_record = self.options.trace_record
        self.options.trace_record = False

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
                    'index:    ' + str(ix) + '\n' +
                    'grad:     ' + str(grad_ix) + '\n' +
                    'fd_f:     ' + str(fd_c_ix) + '\n' +
                    'fd_b:     ' + str(fd_f_ix) + '\n' +
                    'fd_c:     ' + str(fd_b_ix) + '\n' +
                    'fd_err:   ' + str(fd_err_ix) + '\n' +
                    'abs_err:  ' + str(abs_err_ix) + '\n' +
                    'rel_err:  ' + str(rel_err_ix) + '\n'
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

        self.options.trace_record = tmp_trace_record

        return result
