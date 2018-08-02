import numpy as np

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

    """

    MODE_FUN = 'MODE_FUN'  # mode for function values
    MODE_RES = 'MODE_RES'  # mode for residuals

    def __init__(self, fun,
                 grad=None, hess=None, hessp=None,
                 res=None, sres=None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres

        """
        TODO: 
        
        * Implement methods to compute grad via finite differences (with 
        an automatic adaptation of the step size), 
        and diverse approximations of the Hessian.
        """

    def __call__(self, x, sensi_orders: tuple=(0,), mode=MODE_FUN):
        """
        Method to get arbitrary sensitivities.

        There are different ways in which
        an optimizer calls the objective function, and in how the objective
        function provides
        information (e.g. derivatives via separate functions or along with
        the function values). The different calling modes increase efficiency
        in space and time and make the objective flexible.

        Parameters
        ----------

        x: array_like
            The parameters for which to evaluate the objective function.

        sensi_orders: tuple
            Specifying which sensitivities to compute, e.g. (0,1) -> fval, grad.

        mode: str
            Whether to compute function values or residuals.
        """
        if mode == Objective.MODE_FUN:
            return self.call_mode_fun(x, sensi_orders)
        elif mode == Objective.MODE_RES:
            return self.call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")

    def call_mode_fun(self, x, sensi_orders):
        if sensi_orders == (0,):
            if self.grad is True:
                fval = self.fun(x)[0]
            else:
                fval = self.fun(x)
            return fval
        elif sensi_orders == (1,):
            if self.grad is True:
                grad = self.fun(x)[1]
            else:
                grad = self.grad(x)
            return grad
        elif sensi_orders == (2,):
            if self.hess is True:
                hess = self.fun(x)[2]
            else:
                hess = self.hess(x)
            return hess
        elif sensi_orders == (0, 1):
            if self.grad is True:
                fval, grad = self.fun(x)[0, 1]
            else:
                fval = self.fun(x)
                grad = self.grad(x)
            return fval, grad
        elif sensi_orders == (1, 2):
            if self.hess is True:
                grad, hess = self.fun(x)[1, 2]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    grad = self.fun(x)[1]
                else:
                    grad = self.grad(x)
            return grad, hess
        elif sensi_orders == (0, 1, 2):
            if self.hess is True:
                fval, grad, hess = self.fun(x)[0, 1, 2]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    fval, grad = self.fun(x)[0, 1]
                else:
                    fval = self.fun(x)
                    grad = self.grad(x)
            return fval, grad, hess
        else:
            raise ValueError("These sensitivity orders are not supported.")

    def call_mode_res(self, x, sensi_orders):
        if sensi_orders == (0,):
            if self.sres is True:
                res = self.res(x)[0]
            else:
                res = self.res(x)
            return res
        elif sensi_orders == (1,):
            if self.sres is True:
                sres = self.res(x)[1]
            else:
                sres = self.sres(x)
            return sres
        elif sensi_orders == (0, 1):
            if self.sres is True:
                res, sres = self.res(x)
            else:
                res = self.res(x)
                sres = self.sres(x)
            return res, sres
        else:
            raise ValueError("These sensitivity orders are not supported.")

    """
    The following are convenience functions for getting specific outputs.
    """

    def get_fval(self, x):
        fval = self.__call__(x, (0,), Objective.MODE_FUN)
        return fval

    def get_grad(self, x):
        grad = self.__call__(x, (1,), Objective.MODE_FUN)
        return grad

    def get_hess(self, x):
        hess = self.__call__(x, (2,), Objective.MODE_FUN)
        return hess

    def get_hessp(self, x, p):
        hess = self.__call__(x, (2,), Objective.MODE_FUN)
        return np.dot(hess, p)

    def get_res(self, x):
        res = self.__call__(x, (0,), Objective.MODE_RES)
        return res

    def get_sres(self, x):
        sres = self.__call__(x, (1,), Objective.MODE_RES)
        return sres


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

    def __init__(self, amici_model, amici_solver, edata, max_sensi_order=None):
        super().__init__(None)
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.edata = edata
        self.max_sensi_order = max_sensi_order
        if self.max_sensi_order is None:
            self.max_sensi_order = 2 if amici_model.o2mode else 1
        self.dim = amici_model.np()

    def __call__(self, x, sensi_orders: tuple=(0,), mode=Objective.MODE_FUN):
        try:
            import amici
        except ImportError:
            print('This objective requires an installation of amici ('
                  'github.com/icb-dcm/amici. Install via pip3 install amici.')

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

        # loop over experimental data
        for data in self.edata:

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                data)

            # check if the computation failed
            if rdata['status'] < 0.0:
                # TODO: Not sure about res, sres.
                return AmiciObjective.map_to_output(
                    sensi_orders=sensi_orders,
                    fval=np.inf,
                    grad=np.nan * np.ones(self.dim),
                    hess=np.nan * np.ones([self.dim, self.dim]),
                    res=np.nan * np.ones([0]),
                    sres=np.nan * np.ones([0, self.dim])
                )

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
                    sres = np.vstack([rdata['sres'], rdata['sres']]) \
                        if sres.size else rdata['sres']

        return AmiciObjective.map_to_output(
            sensi_orders=sensi_orders,
            fval=nllh, grad=snllh, hess=ssnllh,
            res=res, sres=sres)

    @staticmethod
    def map_to_output(sensi_orders, **kwargs):
        """
        Return values as requested by the caller (sometimes only a subset of
        the outputs are demanded).
        """
        output = ()
        if mode == Objective.MODE_FUN:
            if 0 in sensi_orders:
                return kwargs['fval']
            if 1 in sensi_orders:
                return kwargs['grad']
            if 2 in sensi_orders:
                return kwargs['hess']
        elif mode == Objective.MODE_RES:
            if 0 in sensi_orders:
                return kwargs['res']
            if 1 in sensi_orders:
                return kwargs['sres']
        return output
