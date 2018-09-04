"""
Objective
---------

The objective class is a simple wrapper around the objective function,
giving a standardized way of calling.

"""


import numpy as np
import copy
import pandas as pd
import time
import abc
import pickle

try:
    import amici
except ImportError:
    amici = None


def res_to_fval(res):
    """
    We assume that the residuals res are related to an objective function
    value fval via::
        fval = 0.5 * sum(res**2),
    which is the 'Linear' formulation in scipy.
    """
    return 0.5 * np.power(res, 2).sum()
    

class ObjectiveOptions(dict):
    """
    Options for the objective that are used in optimization, profiles
    and sampling.
    
    Parameters
    ----------
    
    trace_record: bool, optional
        Flag indicating whether to record the trace of function calls.
        Default: False.
        
    trace_record_hess: bool, optional
        Flag indicating whether to record also the Hessian in the trace.
        Default: False.
        
    trace_all: bool, optional
        Flag indicating whether to record all (True, default) or only
        better (False) values.
        
    trace_save: bool, optional
        Flag indicating whether to save the trace.
        Default: False.
        
    trace_file: str, optional
        The string passed here is the file name for storing the trace.
        A contained substring {index} is converted to the multistart
        index.
        Default: "tmp_trace_{index}.dat".
        
    trace_save_iter. index, optional
        Trace is saved every tr_save_iter iterations.
        Default: 10.
    """
    
    def __init__(self,
                 trace_record=False,
                 trace_record_hess=False,
                 trace_all=True,
                 trace_save=False,
                 trace_file=None,
                 trace_save_iter=10):
 
        self.trace_record = trace_record
        self.trace_record_hess = trace_record_hess
        self.trace_all = trace_all
        self.trace_save = trace_save
        
        if trace_file is None:
            trace_file = "tmp_trace_{index}.dat"
        self.trace_file = trace_file
        
        self.trace_save_iter = trace_save_iter

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ObjectiveHistory:
    """
    Objective call history. Also handles saving of intermediate results.
    
    Parameteters
    ------------
    
    options: ObjectiveOptions, optional
        Values needed for creating a history are extracted.
    
    Attributes
    ----------
    
    n_fval, n_grad, n_hess, n_res, n_sres: int
        Counters of function values, gradients and hessians,
        residuals and residual sensitivities.
        
    tr: pd.DataFrame
        DataFrame containing a function value and parameter history if
        options.tr_record is True.
        
    start_time: float
        Reference start time.
        
    fval0, fval_min: float
        Initial and best function value found.
        
    x0, x_min: np.ndarray
        Initial and best parameters found.
        
    index: str
        Id identifying the history object when called in a multistart
        setting.
    """
    
    def __init__(self, options=None):
                 
        if options is None:
            options = ObjectiveOptions()
        self.options = options
        
        self.n_fval = None
        self.n_grad = None
        self.n_hess = None
        self.n_res = None
        self.n_sres = None
        
        self.trace = None
        self.start_time = None
        
        self.fval_min = None
        self.x_min = None
        self.fval0 = None
        self.x0 = None
        
        self.index = None
        
        self.reset()
        
    def reset(self, index=None):
        """
        Reset all counters, the trace, and start the timer.
        """
        self.n_fval = 0
        self.n_grad = 0
        self.n_hess = 0
        self.n_res = 0
        self.n_sres = 0
        
        self.trace = None
        self.start_time = time.time()

        self.fval0 = None
        self.x0 = None
        self.fval_min = np.inf
        self.x_min = None
        
        self.index = index
        
    def update(self, x, result):
        """
        Update the history.
        
        Parameters
        ----------
        x: np.ndarray
            The current parameter.
        result: dict
            The result for x.
        """
        self.update_counts(result)
        self.update_trace(x, result)
        self.update_vals(x, result)
        
    def update_counts(self, result):
        """
        Update the counters.
        """
        if Objective.FVAL in result:
            self.n_fval += 1
        if Objective.GRAD in result:
            self.n_grad += 1
        if Objective.HESS in result:
            self.n_hess += 1
        if Objective.RES in result:
            self.n_res += 1
        if Objective.SRES in result:
            self.n_sres += 1
            
    def update_trace(self, x, result):
        """
        Update and possibly store the trace.
        """

        if not self.options.trace_record:
            return

        # init trace
        if self.trace is None:
            self.trace = pd.DataFrame(
                columns=['time',
                         'n_fval', 'n_grad', 'n_hess', 
                         'fval', 'grad', 'hess',
                         'res', 'sres',
                         'x']
            )

        # extract function values
        fval = result.get(Objective.FVAL, None)
        grad = result.get(Objective.GRAD, None)
        hess = None if self.trace_record_hess \
            else result.get(Objective.HESS, None)
        res = result.get(Objective.RES, None)
        sres = result.get(Objective.SRES, None)
        
        # create table row
        values = [
            time.time() - self.start_time,
            self.n_fval,
            self.n_grad,
            self.n_hess,
            fval,
            grad,
            hess,
            res,
            sres,
            x
        ]
        
        # check whether to append to trace
        if not self.trace_all and fval >= self.fval_min:
            return

        # append to trace
        self.trace.loc[len(self.trace)] = values

        # save to file via pickle
        # format might be revised when storage is implemented
        if self.options.trace_file is not None \
                and len(self.trace > 0) \
                and len(self.trace) % self.options.trace_save_iter == 0:
            filename = self.options.trace_file
            if self.index is not None:
                filename = filename.replace("{index}", str(self.index))
            pickle.dump(self.trace, open(filename, 'wb'))

    def update_vals(self, x, result):
        """
        Update initial and best function values. Must be called after
        update_trace().
        """
        
        # update initial point
        if self.fval0 is None:
            if Objective.FVAL in result:
                self.fval0 = result[Objective.FVAL]
                self.x0 = x
            elif Objective.RES in result:
                self.fval0 = res_to_fval(result[Objective.RES])
                self.x0 = x
        
        # update best point
        fval = np.inf
        if Objective.FVAL in result:
            fval = result[Objective.FVAL]
        elif Objective.RES in result:
            fval = res_to_fval(result[Objective.RES])
        if fval < self.fval_min:
            self.fval_min = fval
            self.x_min = x


class Objective:
    """
    This class contains the objective function.

    Parameters
    ----------

    fun: callable
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
        
    options: pypesto.ObjectiveOptions, optional
        Options as specified in pypesto.ObjectiveOptions.

    Attributes
    ----------
    
    history: pypesto.ObjectiveHistory
        For storing the call history. Initialized by the optimizer in
        reset_history().

    The following variables are set by the problem in update_from_problem():
    
    preprocess: callable
        Preprocess input values to __call__.

    postprocess: callable
        Postprocess output values from __call__.
        
    dim, dim_full: int
        Dimension of the reduced and full problem.
    
    parameter_names: list of str
        Parameter names.
    """

    MODE_FUN = 'mode_fun'  # mode for function values
    MODE_RES = 'mode_res'  # mode for residuals
    FVAL = 'fval'
    GRAD = 'grad'
    HESS = 'hess'
    RES = 'res'
    SRES = 'sres'

    def __init__(self, fun,
                 grad=None, hess=None, hessp=None,
                 res=None, sres=None,
                 options=None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        
        if options is None:
            options = ObjectiveOptions()
        self.options = options
        
        self.history = ObjectiveHistory(self.options)
        
        self.preprocess = lambda x: np.array(x)
        self.postprocess = lambda result: result

    def __call__(self, x, sensi_orders: tuple=(0,), mode=MODE_FUN):
        """
        Method to get arbitrary sensitivities. This is the central method
        which is always called, also by the get_ functions.

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

        # compute function value
        result = self._call_unprocessed(x, sensi_orders, mode)
            
        # compute result
        result = self._call_unprocessed(x, sensi_orders, mode)
        
        # convert to ndarray
        result = Objective.as_ndarrays(result)

        # post-process
        result = self.postprocess(result=result)
        
        # update history
        self.history.update(x, result)

        # map to output format
        result = Objective.map_to_output(sensi_orders, mode, **result)
        
        return result
        
    def _call_unprocessed(self, x, sensi_orders, mode):
        """
        Call objective function without pre- or post-processing and
        formatting.
        """
        if mode == Objective.MODE_FUN:
            result = self._call_mode_fun(x, sensi_orders)
        elif mode == Objective.MODE_RES:
            result = self._call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")
            
        return result

    def _call_mode_fun(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_FUN.
        """
        if sensi_orders == (0,):
            if self.grad is True:
                fval = self.fun(x)[0]
            else:
                fval = self.fun(x)
            result = {Objective.FVAL: fval}
        elif sensi_orders == (1,):
            if self.grad is True:
                grad = self.fun(x)[1]
            else:
                grad = self.grad(x)
            result = {Objective.GRAD: grad}
        elif sensi_orders == (2,):
            if self.hess is True:
                hess = self.fun(x)[2]
            else:
                hess = self.hess(x)
            result = {Objective.HESS: hess}
        elif sensi_orders == (0, 1):
            if self.grad is True:
                fval, grad = self.fun(x)[0:2]
            else:
                fval = self.fun(x)
                grad = self.grad(x)
            result = {Objective.FVAL: fval,
                      Objective.GRAD: grad}
        elif sensi_orders == (1, 2):
            if self.hess is True:
                grad, hess = self.fun(x)[1:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    grad = self.fun(x)[1]
                else:
                    grad = self.grad(x)
            result = {Objective.GRAD: grad,
                      Objective.HESS: hess}
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
            result = {Objective.FVAL: fval,
                      Objective.GRAD: grad,
                      Objective.HESS: hess}
        else:
            raise ValueError("These sensitivity orders are not supported.")
        return result

    def _call_mode_res(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_RES.
        """
        if sensi_orders == (0,):
            if self.sres is True:
                res = self.res(x)[0]
            else:
                res = self.res(x)
            result = {Objective.RES: res}
        elif sensi_orders == (1,):
            if self.sres is True:
                sres = self.res(x)[1]
            else:
                sres = self.sres(x)
            result = {Objective.SRES: sres}
        elif sensi_orders == (0, 1):
            if self.sres is True:
                res, sres = self.res(x)
            else:
                res = self.res(x)
                sres = self.sres(x)
            result = {Objective.RES: res,
                      Objective.SRES: sres}
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
        keys = [Objective.GRAD, Objective.HESS, Objective.RES, Objective.SRES]
        for key in keys:
            if key in result:
                value = result[key]
                if value is not None:
                    result[key] = np.array(value)
                
        return result

    @staticmethod
    def map_to_output(sensi_orders, mode, **kwargs):
        """
        Return values as requested by the caller, since usually only a subset
        is demanded. One output is returned as-is, more than one output are
        returned as a tuple in order (fval, grad, hess).
        """
        output = ()
        if mode == Objective.MODE_FUN:
            if 0 in sensi_orders:
                output += (kwargs[Objective.FVAL],)
            if 1 in sensi_orders:
                output += (kwargs[Objective.GRAD],)
            if 2 in sensi_orders:
                output += (kwargs[Objective.HESS],)
        elif mode == Objective.MODE_RES:
            if 0 in sensi_orders:
                output += (kwargs[Objective.RES],)
            if 1 in sensi_orders:
                output += (kwargs[Objective.SRES],)
        if len(output) == 1:
            # return a single value not as tuple
            output = output[0]
        return output

    """
    The following are convenience functions for getting specific outputs.
    """

    def get_fval(self, x):
        """
        Get the function value at x.
        """
        fval = self(x, (0,), Objective.MODE_FUN)
        return fval

    def get_grad(self, x):
        """
        Get the gradient at x.
        """
        grad = self(x, (1,), Objective.MODE_FUN)
        return grad

    def get_hess(self, x):
        """
        Get the Hessian at x.
        """
        hess = self(x, (2,), Objective.MODE_FUN)
        return hess

    def get_hessp(self, x, p):
        """
        Get the product of the Hessian at x with p.
        """
        hess = self(x, (2,), Objective.MODE_FUN)
        return np.dot(hess, p)

    def get_res(self, x):
        """
        Get the residuals at x.
        """
        res = self(x, (0,), Objective.MODE_RES)
        return res

    def get_sres(self, x):
        """
        Get the residual sensitivities at x.
        """
        sres = self(x, (1,), Objective.MODE_RES)
        return sres

    def reset_history(self, index=None):
        """
        Reset the objective history and specify temporary saving options.
        
        Parameters
        ----------
        
        index: As in ObjectiveHistory.index.
        """
        self.history.reset(index=index)
        
    def reset(self):
        """
        Completely reset the objective, i.e. undo the modifications in
        update_from_problem().
        """
        self.history = ObjectiveHistory(self.options)
        self.preprocess = lambda x: np.array(x)
        self.postprocess = lambda result: result

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

        dim = len(x_free_indices)

        # pre-process
        def preprocess(x):
            x_full = np.zeros(dim_full)
            x_full[x_free_indices] = x
            x_full[x_fixed_indices] = x_fixed_vals
            return x_full
        self.preprocess = preprocess

        # post-process
        def postprocess(result):
            if Objective.GRAD in result:
                grad = result[Objective.GRAD]
                if grad.size == dim_full:
                    grad = grad[x_free_indices]
                    result[Objective.GRAD] = grad
            if Objective.HESS in result:
                hess = result[Objective.HESS]
                if hess.shape[0] == dim_full:
                    hess = hess[np.ix_(x_free_indices, x_free_indices)]
                    result[Objective.HESS] = hess
            if Objective.RES in result:
                res = result[Objective.RES]
                if res.size == dim_full:
                    res = res.flatten()[x_free_indices]
                    result[Objective.RES] = res
            if Objective.SRES in result:
                sres = result[Objective.SRES]
                if sres.shape[-1] == dim_full:
                    sres = sres[..., x_free_indices]
                    result[Objective.SRES] = sres
            return result
        self.postprocess = postprocess
        
    def get_x_names(self):
        """
        Get parameter names.
        
        Returns
        -------
        
        None if no names provided, otherwise a list of str, length dim_full.
        """
        return None
        
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
            Residual (Objective.MODE_RES) or objective function value
            (Objective.MODE_FUN, default) computation mode.

        Returns
        ----------
        
        result: pd.DataFrame
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
                print('index:    ' + str(ix) + '\n' +
                      'grad: ' + str(grad_ix) + '\n' +
                      'fd_f:  ' + str(fd_c_ix) + '\n' +
                      'fd_b:  ' + str(fd_f_ix) + '\n' +
                      'fd_c:  ' + str(fd_b_ix) + '\n' +
                      'fd_err:   '
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
            print(result)

        return result


class AmiciObjective(Objective):
    """
    This is a convenience class to compute an objective function from an
    AMICI model.

    Parameters
    ----------

    amici_model: amici.Model
        The amici model.

    amici_solver: amici.Solver
        The solver to use for the numeric integration of the model.

    edata:
        The experimental data.

    max_sensi_order: int
        Maximum sensitivity order supported by the model.
    """

    def __init__(self, amici_model, amici_solver, edata, max_sensi_order=None,
                 preprocess_edata=True):
        if amici is None:
            raise ImportError('This objective requires an installation of '
                              'amici (github.com/icb-dcm/amici. Install via '
                              'pip3 install amici.')
        super().__init__(fun=None)
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.dim = amici_model.np()
        if preprocess_edata:
            self.preequilibration_edata = dict()
            self.preprocess_edata(edata)
            self.edata = edata
        else:
            self.edata = edata
            self.preequilibration_edata = None

        self.max_sensi_order = max_sensi_order
        if self.max_sensi_order is None:
            self.max_sensi_order = 2 if amici_model.o2mode else 1

    def _call_unprocessed(
            self,
            x,
            sensi_orders: tuple=(0,),
            mode=Objective.MODE_FUN
    ):
        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed
        sensi_order = min(max(sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        """
        TODO: For large-scale models it might be bad to always reserve
        space in particular for the Hessian.
        """

        nllh = 0.0
        snllh = np.zeros(self.dim)
        ssnllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        # set parameters in model
        self.amici_model.setParameters(amici.DoubleVector(x))

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        if self.preequilibration_edata:
            for fixedParameters in self.preequilibration_edata:
                rdata = amici.runAmiciSimulation(
                    self.amici_model,
                    self.amici_solver,
                    self.preequilibration_edata[fixedParameters]['edata'])

                if rdata['status'] < 0.0:
                    return self.get_error_output(sensi_orders, mode)

                self.preequilibration_edata[fixedParameters]['x0'] = \
                    rdata['x0']
                if self.amici_solver.getSensitivityOrder() > \
                        amici.SensitivityOrder_none:
                    self.preequilibration_edata[fixedParameters]['sx0'] = \
                        rdata['sx0']

        # loop over experimental data
        for data in self.edata:

            if self.preequilibration_edata:
                original_value_dict = self.preprocess_preequilibration(data)
            else:
                original_value_dict = None

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                data)

            if self.preequilibration_edata:
                self.postprocess_preequilibration(data, original_value_dict)

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            # extract required result fields
            if mode == Objective.MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    snllh -= rdata['sllh']
                    # TODO: Compute the full Hessian, and check here
                    ssnllh -= rdata['FIM']
            elif mode == Objective.MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([sres, rdata['sres']]) \
                        if sres.size else rdata['sres']
        
        return {Objective.FVAL: nllh,
                Objective.GRAD: snllh,
                Objective.HESS: ssnllh,
                Objective.RES: res,
                Objective.SRES: sres}

    def preprocess_preequilibration(self, data):
        original_fixed_parameters_preequilibration = None
        original_initial_states = None
        original_initial_state_sensitivities = None
        if data.fixedParametersPreequilibration.size():
            original_initial_states = self.amici_model.getInitialStates()

            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                original_initial_state_sensitivities = \
                    self.amici_model.getInitialStateSensitivities()

            fixed_parameters = copy.deepcopy(
                list(data.fixedParametersPreequilibration)
            )
            data.fixedParametersPreequilibration = amici.DoubleVector([])
            original_fixed_parameters_preequilibration = fixed_parameters

            self.amici_model.setInitialStates(
                amici.DoubleVector(
                    self.preequilibration_edata[str(fixed_parameters)]['x0']
                )
            )
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                self.amici_model.setInitialStateSensitivities(
                    amici.DoubleVector(
                        self.preequilibration_edata[
                            str(fixed_parameters)
                        ]['sx0'].flatten())
                )

        return {
            'k': original_fixed_parameters_preequilibration,
            'x0': original_initial_states,
            'sx0': original_initial_state_sensitivities
        }

    def postprocess_preequilibration(self, data, original_value_dict):
        if original_value_dict['k']:
            data.fixedParametersPreequilibration = amici.DoubleVector(
                original_value_dict['k']
            )

        if original_value_dict['x0']:
            self.amici_model.setInitialStates(original_value_dict['x0'])

        if original_value_dict['sx0']:
            self.amici_model.setInitialStateSensitivities(
                original_value_dict['sx0']
            )

    def preprocess_edata(self, edata_vector):
        for edata in edata_vector:
            fixed_parameters = list(edata.fixedParametersPreequilibration)
            if str(fixed_parameters) in self.preequilibration_edata.keys() or \
               len(fixed_parameters) == 0:
                continue  # we only need to keep unique ones

            preeq_edata = amici.ExpData(self.amici_model.get())
            preeq_edata.fixedParametersPreequilibration = amici.DoubleVector(
                fixed_parameters
            )

            # only preequilibration
            preeq_edata.setTimepoints(amici.DoubleVector([]))

            self.preequilibration_edata[str(fixed_parameters)] = dict(
                edata=preeq_edata
            )

    def get_error_output(self, sensi_orders, mode):
        if not self.amici_model.nt():
            nt = sum([data.nt() for data in self.edata])
        else:
            nt = sum([data.nt() if data.nt() else self.amici_model.nt()
                      for data in self.edata])
        n_res = nt * self.amici_model.nytrue
        return Objective.map_to_output(
            sensi_orders=sensi_orders,
            mode=mode,
            fval=np.inf,
            grad=np.nan * np.ones(self.dim),
            hess=np.nan * np.ones([self.dim, self.dim]),
            res=np.nan * np.ones(n_res),
            sres=np.nan * np.ones([n_res, self.dim])
        )

    def get_x_names(self):
        """
        Extracts the parameter names from the model.
        """
        return list(self.amici_model.getParameterNames())
