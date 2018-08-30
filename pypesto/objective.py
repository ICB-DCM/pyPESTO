"""
Objective
---------

The objective class is a simple wrapper around the objective function,
giving a standardized way of calling.

"""


import numpy as np
import sys
import copy

try:
    import amici
except ImportError:
    amici = None


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

        self.preprocess = lambda x: x
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
        if 'amici' not in sys.modules:
            print('This objective requires an installation of amici ('
                  'github.com/icb-dcm/amici. Install via pip3 install amici.')
        super().__init__(None)
        self.amici_model = amici_model
        self.amici_solver = amici_solver
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
        self.dim = amici_model.np()

    def __call__(self, x, sensi_orders: tuple=(0,), mode=Objective.MODE_FUN):
        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed
        sensi_order = max(sensi_orders)

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

        for fixedParameters in self.preequilibration_edata:
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                self.preequilibration_edata[fixedParameters]['edata'])

            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            self.preequilibration_edata[fixedParameters]['x0'] = rdata['x0']
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
