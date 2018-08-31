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

try:
    import amici
except ImportError:
    amici = None
    
    
class ObjectiveState(dict):
	def __init__(self,
	             n_fval = 0,
	             n_grad = 0,
	             n_hess = 0,
	             
	             
	def init(self):
		self.n_fval = 0
		self.n_grad = 0
		self.n_hess = 0


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

    Attributes
    ----------

    preprocess: callable
        Preprocess input values to __call__.

    postprocess: callable
        Postprocess output values from __call__.
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
                 res=None, sres=None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres

<<<<<<< HEAD
        self.n_fval = 0
        self.n_grad = 0
        self.n_hess = 0

        self.temp_file = None
        self.temp_save_iter = None
        self.min_fval = float('inf')
        self.min_x = None
        self.trace = None

        """
        TODO:

        * Implement methods to compute grad via finite differences (with
        an automatic adaptation of the step size),
        and diverse approximations of the Hessian.
        """
=======
        self.preprocess = lambda x: x
        self.postprocess = lambda result: result
        
        self.state = ObjectiveState()
>>>>>>> feature_fixedpars


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

<<<<<<< HEAD
        if mode == Objective.MODE_FUN:
            result = self.call_mode_fun(x, sensi_orders)

        elif mode == Objective.MODE_RES:
            result = self.call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")

        return result

    def call_mode_fun(self, x, sensi_orders):
        self.update_eval_counts(sensi_orders)
=======
        # pre-process
        x = self.preprocess(x=x)

        # function or residue mode
        if mode == Objective.MODE_FUN:
            result = self._call_mode_fun(x, sensi_orders)
        elif mode == Objective.MODE_RES:
            result = self._call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")

        # post-process
        result = self.postprocess(result=result)

        # map to output format
        result = Objective.map_to_output(sensi_orders, mode, **result)

        return result

    def _call_mode_fun(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_FUN.
        """
>>>>>>> feature_fixedpars
        if sensi_orders == (0,):
            if self.grad is True:
                fval = self.fun(x)[0]
            else:
                fval = self.fun(x)
<<<<<<< HEAD
            if self.trace is not None:
                self.update_trace(fval, x)
            return fval
=======
            result = {Objective.FVAL: fval}
>>>>>>> feature_fixedpars
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
<<<<<<< HEAD
            if self.trace is not None:
                self.update_trace(fval, x)
            return fval, grad
=======
            result = {Objective.FVAL: fval,
                      Objective.GRAD: grad}
>>>>>>> feature_fixedpars
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
<<<<<<< HEAD
            if self.trace is not None:
                self.update_trace(fval, x)
            return fval, grad, hess
=======
            result = {Objective.FVAL: fval,
                      Objective.GRAD: grad,
                      Objective.HESS: hess}
>>>>>>> feature_fixedpars
        else:
            raise ValueError("These sensitivity orders are not supported.")
        return result

<<<<<<< HEAD
    def call_mode_res(self, x, sensi_orders):
        self.update_eval_counts(sensi_orders)
=======
    def _call_mode_res(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_RES.
        """
>>>>>>> feature_fixedpars
        if sensi_orders == (0,):
            if self.sres is True:
                res = self.res(x)[0]
            else:
                res = self.res(x)
<<<<<<< HEAD
            if self.trace is not None:
                self.update_trace(np.power(res, 2).sum(), x)
            return res
=======
            result = {Objective.RES: res}
>>>>>>> feature_fixedpars
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
<<<<<<< HEAD
            if self.trace is not None:
                self.update_trace(np.power(res, 2).sum(), x)
            return res, sres
=======
            result = {Objective.RES: res,
                      Objective.SRES: sres}
>>>>>>> feature_fixedpars
        else:
            raise ValueError("These sensitivity orders are not supported.")
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
                output += (np.array(kwargs[Objective.GRAD]),)
            if 2 in sensi_orders:
                output += (np.array(kwargs[Objective.HESS]),)
        elif mode == Objective.MODE_RES:
            if 0 in sensi_orders:
                output += (np.array(kwargs[Objective.RES]),)
            if 1 in sensi_orders:
                output += (np.array(kwargs[Objective.SRES]),)
        if len(output) == 1:
            # return a single value not as tuple
            output = output[0]
        return output

    def reset_history(self,
                      dim,
                      temp_file=None,
                      temp_save_iter=10):
        """
        Method to reset the evaluation history of the objective and specify
        temporary saving options.

        Parameters
        ----------

        dim: number of parameters

        temp_file: filename
            If specified, temporary results of traces for every optimization
            run will be saved to that file

        temp_save_iter: int
            Update interval for temporary saving.
        """

        self.n_fval = 0
        self.n_grad = 0
        self.n_hess = 0
        self.min_fval = float('Inf')
        self.start_time = time.time()
        self.min_x = None

        parameter_names = self.get_parameter_names()

        if parameter_names is None:
            parameter_names = ['x' + str(ix) for ix in range(dim)]
        else:
            if len(parameter_names) != dim:
                raise ValueError('List of parameter names must be of the same'
                                 'length as the length of the startpoint')

        if temp_file is not None:
            cols = ['time', 'n_fval', 'n_grad', 'n_hess', 'fval'] \
                + parameter_names
            self.trace = pd.DataFrame(columns=cols)
            self.temp_file = temp_file
            self.temp_save_iter = temp_save_iter
        else:
            self.trace = None
            self.temp_file = None
            self.temp_save_iter = None

    def update_eval_counts(self, sensi_orders):
        if max(sensi_orders) == 0:
            self.n_fval += 1
        if max(sensi_orders) == 1:
            self.n_grad += 1
        if max(sensi_orders) == 2:
            self.n_hess += 1

    def update_trace(self, fval, x):
        if fval < self.min_fval:
            self.min_fval = fval
            self.min_x = x

            values = [
                time.time() - self.start_time,
                self.n_fval,
                self.n_grad,
                self.n_hess,
                fval,
            ] + list(x)

            self.trace.loc[len(self.trace)] = values

            if (len(self.trace) - 1) % self.temp_save_iter == 0:
                self.trace.to_csv(self.temp_file)

    @abc.abstractmethod
    def get_parameter_names(self):
        return None

    """
    The following are convenience functions for getting specific outputs.
    """

    def get_fval(self, x):
<<<<<<< HEAD
        fval = self.call_mode_fun(x, (0,))
        return fval

    def get_grad(self, x):
        grad = self.call_mode_fun(x, (1,))
        return grad

    def get_hess(self, x):
        hess = self.call_mode_fun(x, (2,))
        return hess

    def get_hessp(self, x, p):
        hess = self.call_mode_fun(x, (2,))
        return np.dot(hess, p)

    def get_res(self, x):
        res = self.call_mode_res(x, (0,))
        return res

    def get_sres(self, x):
        sres = self.call_mode_res(x, (1,))
        return sres

    def check_grad(self,
                   x0,
                   param_indices=None,
                   eps=1e-5,
                   verbosity=1,
                   mode='MODE_FUN') -> pd.DataFrame:
        """
        Method to evaluate the gradient via finite differences and compare the
        result to the objective gradient.

        Parameters
        ----------

        x0: list
            Parameter values at which the gradient will be evaluated

        param_indices: list
            List of index values which allows computation of finite differences
            only for the specified subset of parameters

        TODO: pass param_indices to amici instead of subselecting result

        eps: float
            Step size

        verbosity: int
            Level of verbosity for function output
                0: no output
                1: summary for all parameters
                2: summary for individual parameters

        mode: str
            Computation mode can be used to switch between residual
            computation ('MODE_RES') and objective function value computation
            ('MODE_FUN')

        Returns
        ----------
        gradient, finite difference approximations and error estimates as
        DataFrame


        """

        if param_indices is None:
            param_indices = range(len(x0))

        f = self.__call__(x0, (0,), mode)
        g = self.__call__(x0, (1,), mode)

        g_list = []
        fd_f = []
        fd_b = []
        fd_c = []
        fd_error = []
        rel_error = []
        abs_error = []

        for ipar in param_indices:
            xp = copy.deepcopy(x0)
            xp[ipar] = xp[ipar] + eps

            fp = self.__call__(xp, (0,), mode)

            xm = copy.deepcopy(x0)
            xm[ipar] = xm[ipar] - eps

            fm = self.__call__(xm, (0,), mode)

            fd_f_single = (fp - f) / eps
            fd_b_single = (f - fm) / eps
            fd_c_single = (fp - fm) / (2 * eps)

            g_ipar = None
            if len(g.shape) == 1:
                g_ipar = g[ipar]
            elif len(g.shape) == 2:
                g_ipar = g[:, ipar]

            if verbosity > 1:
                print('index ' + str(ipar) + ':\n' +
                      'gradient: ' + str(g_ipar) + '\n' +
                      'cntr FDs: ' + str(fd_c_single) + '\n' +
                      'fwd  FDs: ' + str(fd_f_single) + '\n' +
                      'bwd  FDs: ' + str(fd_b_single) + '\n' +
                      'rel err: ' + str(abs((g_ipar - fd_c_single) /
                                            (fd_c_single + eps))) + '\n' +
                      'abs err: ' + str(abs((g_ipar - fd_c_single)))
                      )

            g_list.append(g_ipar)
            fd_f.append(fd_f_single)
            fd_b.append(fd_b_single)
            fd_c.append(fd_c_single)
            rel_error.append(np.mean(abs((g_ipar - fd_c_single) /
                                         (fd_c_single + eps))))
            abs_error.append(np.mean(abs((g_ipar - fd_c_single))))
            fd_error.append(np.mean(abs(fd_f_single - fd_b_single)))

        result = pd.DataFrame(data={
            'gradient': g_list,
            'FD_f': fd_f,
            'FD_b': fd_b,
            'FD_c': fd_c,
            'rel_err': rel_error,
            'abs_err': abs_error,
            'FD_err': fd_error,
        })

        if verbosity > 0:
            print(result)

        return result
=======
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

    """
    The following functions handle parameter mappings.
    """

    def handle_x_fixed(self,
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
                assert grad.size == dim
            if Objective.HESS in result:
                hess = result[Objective.HESS]
                if hess.shape[0] == dim_full:
                    hess = hess[np.ix_(x_free_indices, x_free_indices)]
                    result[Objective.HESS] = hess
                assert hess.shape == (dim, dim)
            return result
        self.postprocess = postprocess
>>>>>>> feature_fixedpars


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
        super().__init__(
            fun=lambda x: self.call_amici(
                x,
                tuple(i for i in range(max_sensi_order)),
                'MODE_FUN'
            ),
            grad=max_sensi_order > 0,
            hess=lambda x: self.call_amici(
                x,
                (2, ),
                'MODE_FUN'
            ),
            res=lambda x: self.call_amici(
                x,
                (0,),
                'MODE_RES'
            ),
            sres=lambda x: self.call_amici(
                x,
                (1,),
                'MODE_RES'
            ),
        )
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

    def call_amici(
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

        return Objective.map_to_output(
            sensi_orders=sensi_orders,
            mode=mode,
            fval=nllh, grad=snllh, hess=ssnllh,
            res=res, sres=sres)

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
<<<<<<< HEAD

    def get_parameter_names(self):
        return list(self.amici_model.getParameterNames())

    @staticmethod
    def map_to_output(sensi_orders, mode, **kwargs):
        """
        Return values as requested by the caller (sometimes only a subset of
        the outputs are demanded).
        """
        output = ()
        if mode == Objective.MODE_FUN:
            if 0 in sensi_orders:
                output += (kwargs['fval'],)
            if 1 in sensi_orders:
                output += (kwargs['grad'],)
            if 2 in sensi_orders:
                output += (kwargs['hess'],)
        elif mode == Objective.MODE_RES:
            if 0 in sensi_orders:
                output += (kwargs['res'],)
            if 1 in sensi_orders:
                output += (kwargs['sres'],)
        if len(output) == 1:
            # return a single value not as tuple
            output = output[0]
        return output
=======
>>>>>>> feature_fixedpars
