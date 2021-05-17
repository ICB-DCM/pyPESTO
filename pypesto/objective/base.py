import numpy as np
import pandas as pd
import copy
import logging
import abc
from typing import Dict, Iterable, Sequence, Tuple, Union

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .history import HistoryBase
from .pre_post_process import PrePostProcessor, FixedParametersProcessor

ResultDict = Dict[str, Union[float, np.ndarray, Dict]]

logger = logging.getLogger(__name__)


class ObjectiveBase(abc.ABC):
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
    x_names:
        Parameter names that can be optionally used in, e.g., history or
        gradient checks.

    Attributes
    ----------
    history:
        For storing the call history. Initialized by the methods, e.g. the
        optimizer, in `initialize_history()`.
    pre_post_processor:
        Preprocess input values to and postprocess output values from
        __call__. Configured in `update_from_problem()`.
    """

    def __init__(
        self,
        x_names: Sequence[str] = None,
    ):

        self.x_names = x_names

        self.pre_post_processor = PrePostProcessor()
        self.history = HistoryBase()

    def __deepcopy__(self, memodict=None) -> 'ObjectiveBase':
        other = type(self)()  # maintain type for derived classes
        for attr in self.__dict__:
            other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    # The following has_ properties can be used to find out what values
    # the objective supports.
    @property
    def has_fun(self) -> bool:
        return self.check_sensi_orders((0,), MODE_FUN)

    @property
    def has_grad(self) -> bool:
        return self.check_sensi_orders((1,), MODE_FUN)

    @property
    def has_hess(self) -> bool:
        return self.check_sensi_orders((2,), MODE_FUN)

    @property
    def has_hessp(self) -> bool:
        # Not supported yet
        return False

    @property
    def has_res(self) -> bool:
        return self.check_sensi_orders((0,), MODE_RES)

    @property
    def has_sres(self) -> bool:
        return self.check_sensi_orders((1,), MODE_RES)

    def initialize(self):
        """Initialize the objective function.
        This function is used at the beginning of an analysis, e.g.
        optimization, and can e.g. reset the objective memory.
        By default does nothing.
        """

    def __call__(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...] = (0, ),
        mode: str = MODE_FUN,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray, Tuple, ResultDict]:
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
        # copy parameter vector to prevent side effects
        x = np.array(x).copy()

        # check input
        if not self.check_mode(mode):
            raise ValueError(f"This Objective cannot be called with mode"
                             f"={mode}.")
        if not self.check_sensi_orders(sensi_orders, mode):
            raise ValueError(f"This Objective cannot be called with "
                             f"sensi_orders= {sensi_orders} and mode={mode}.")

        # pre-process
        x_full = self.pre_post_processor.preprocess(x=x)

        # compute result
        result = self.call_unprocessed(
            x=x_full, sensi_orders=sensi_orders, mode=mode, **kwargs)

        # post-process
        result = self.pre_post_processor.postprocess(result=result)

        # update history
        self.history.update(
            x=x, sensi_orders=sensi_orders, mode=mode, result=result)

        # map to output format
        if not return_dict:
            result = ObjectiveBase.output_to_tuple(
                sensi_orders=sensi_orders, mode=mode, **result)

        return result

    @abc.abstractmethod
    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
        **kwargs,
    ) -> ResultDict:
        """
        Call objective function without pre- or post-processing and
        formatting.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the objective function.
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.

        Returns
        -------
        result:
            A dict containing the results.
        """
        raise NotImplementedError()

    def check_mode(self, mode: str) -> bool:
        """
        Check if the objective is able to compute in the requested mode.

        Either `check_mode` or the `fun_...` functions
        must be overwritten in derived classes.

        Parameters
        ----------
        mode:
            Whether to compute function values or residuals.
        Returns
        -------
        flag:
            Boolean indicating whether mode is supported
        """
        if mode == MODE_FUN:
            return self.has_fun
        elif mode == MODE_RES:
            return self.has_res
        else:
            raise ValueError(f"Unknown mode {mode}.")

    def check_sensi_orders(
        self,
        sensi_orders: Tuple[int, ...],
        mode: str,
    ) -> bool:
        """
        Check if the objective is able to compute the requested
        sensitivities.

        Either `check_sensi_orders` or the `fun_...` functions
        must be overwritten in derived classes.

        Parameters
        ----------
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.

        Returns
        -------
        flag:
            Boolean indicating whether combination of sensi_orders and mode
            is supported
        """
        if (
            mode == MODE_FUN
            and (
                0 in sensi_orders and not self.has_fun
                or 1 in sensi_orders and not self.has_grad
                or 2 in sensi_orders and not self.has_hess
            )
        ) or (
            mode == MODE_RES
            and (
                0 in sensi_orders and not self.has_res
                or 1 in sensi_orders and not self.has_sres
            )
        ):
            return False

        return True

    @staticmethod
    def output_to_tuple(
        sensi_orders: Tuple[int, ...],
        mode: str,
        **kwargs: Union[float, np.ndarray],
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
        """Get the function value at x."""

        fval = self(x, (0,), MODE_FUN)
        return fval

    def get_grad(self, x: np.ndarray) -> np.ndarray:
        """Get the gradient at x."""
        grad = self(x, (1,), MODE_FUN)
        return grad

    def get_hess(self, x: np.ndarray) -> np.ndarray:
        """Get the Hessian at x."""
        hess = self(x, (2,), MODE_FUN)
        return hess

    def get_res(self, x: np.ndarray) -> np.ndarray:
        """Get the residuals at x."""
        res = self(x, (0,), MODE_RES)
        return res

    def get_sres(self, x: np.ndarray) -> np.ndarray:
        """Get the residual sensitivities at x."""
        sres = self(x, (1,), MODE_RES)
        return sres

    def update_from_problem(
        self,
        dim_full: int,
        x_free_indices: Sequence[int],
        x_fixed_indices: Sequence[int],
        x_fixed_vals: Sequence[float],
    ):
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

    def check_grad_multi_eps(
        self,
        *args,
        multi_eps: Iterable = None,
        label: str = 'rel_err',
        **kwargs,
    ):
        """
        Equivalent to the `ObjectiveBase.check_grad` method, except multiple
        finite difference step sizes are tested. The result contains the
        lowest finite difference for each parameter, and the corresponding
        finite difference step size.

        Parameters
        ----------
        All `ObjectiveBase.check_grad` method parameters.
        multi_eps:
            The finite difference step sizes to be tested.
        label:
            The label of the column that will be minimized for each parameter.
            Valid options are the column labels of the dataframe returned by
            the `ObjectiveBase.check_grad` method.
        """
        if multi_eps is None:
            multi_eps = {1e-1, 1e-3, 1e-5, 1e-7, 1e-9}

        results = {}
        for eps in multi_eps:
            results[eps] = self.check_grad(*args, **kwargs, eps=eps)

        # The combined result is, for each parameter, the gradient check from
        # the step size (`eps`) that produced the smallest error (`label`).
        combined_result = None
        for eps, result in results.items():
            result['eps'] = eps
            if combined_result is None:
                combined_result = result
                continue
            # Replace rows in `combined_result` with corresponding rows
            # in `result` that have an improved value in column `label`.
            mask_improvements = result[label] < combined_result[label]
            combined_result.loc[mask_improvements, :] = \
                result.loc[mask_improvements, :]

        return combined_result

    def check_grad(
        self,
        x: np.ndarray,
        x_indices: Sequence[int] = None,
        eps: float = 1e-5,
        verbosity: int = 1,
        mode: str = MODE_FUN,
        detailed: bool = False,
    ) -> pd.DataFrame:
        """
        Compare gradient evaluation: Firstly approximate via finite
        differences, and secondly use the objective gradient.

        Parameters
        ----------
        x:
            The parameters for which to evaluate the gradient.
        x_indices:
            Indices for which to compute gradients. Default: all.
        eps:
            Finite differences step size.
        verbosity:
            Level of verbosity for function output.
            0: no output,
            1: summary for all parameters,
            2: summary for individual parameters.
        mode:
            Residual (MODE_RES) or objective function value (MODE_FUN)
            computation mode.
        detailed:
            Toggle whether additional values are returned. Additional values
            are function values, and the central difference weighted by the
            difference in output from all methods (standard deviation and
            mean).

        Returns
        ----------
        result:
            gradient, finite difference approximations and error estimates.
        """

        if x_indices is None:
            x_indices = list(range(len(x)))

        # function value and objective gradient
        fval, grad = self(x, (0, 1), mode)

        grad_list = []
        fd_f_list = []
        fd_b_list = []
        fd_c_list = []
        fd_err_list = []
        abs_err_list = []
        rel_err_list = []

        if detailed:
            fval_p_list = []
            fval_m_list = []
            std_check_list = []
            mean_check_list = []

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

            if detailed:
                std_check_ix = (grad_ix - fd_c_ix)/np.std([
                    fd_f_ix,
                    fd_b_ix,
                    fd_c_ix
                ])
                mean_check_ix = abs(grad_ix - fd_c_ix)/np.mean([
                    abs(fd_f_ix - fd_b_ix),
                    abs(fd_f_ix - fd_c_ix),
                    abs(fd_b_ix - fd_c_ix),
                ])

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
            if detailed:
                fval_p_list.append(fval_p)
                fval_m_list.append(fval_m)
                std_check_list.append(std_check_ix)
                mean_check_list.append(mean_check_ix)

        # create data dictionary for dataframe
        data = {
            'grad': grad_list,
            'fd_f': fd_f_list,
            'fd_b': fd_b_list,
            'fd_c': fd_c_list,
            'fd_err': fd_err_list,
            'abs_err': abs_err_list,
            'rel_err': rel_err_list,
        }

        # update data dictionary if detailed output is requested
        if detailed:
            prefix_data = {
                'fval': [fval]*len(x_indices),
                'fval_p': fval_p_list,
                'fval_m': fval_m_list,

            }
            std_str = '(grad-fd_c)/std({fd_f,fd_b,fd_c})'
            mean_str = '|grad-fd_c|/mean(|fd_f-fd_b|,|fd_f-fd_c|,|fd_b-fd_c|)'
            postfix_data = {
                std_str: std_check_list,
                mean_str: mean_check_list,
            }
            data = {**prefix_data, **data, **postfix_data}

        # create dataframe
        result = pd.DataFrame(
            data=data,
            index=[
                self.x_names[ix] if self.x_names is not None
                else f'x_{ix}' for ix in x_indices
            ],
        )

        # log full result
        if verbosity > 0:
            logger.info(result)

        return result
