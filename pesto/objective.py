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

    hess: callable, optional
        Method for computing the Hessian matrix. If it is a callable,
        it should be of the form
            ``hess(x) -> array, shape (n,n).``

    hessp: callable, optional
        Method for computing the Hessian vector product, i.e.
            ``hessp(x, v) -> array_like, shape (n,)``
        computes the product H*v of the Hessian of fun at x with v.

    res: callable, optional
        Method for computing residuals, i.e.
            ``res(x) -> array_like, shape(m,).``

    sres: callable, optional
        Method for computing residual sensitivities, i.e.
            ``sres(x) -> array, shape (m,n).``

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
    
        * Also allow having residuals for least squares programing.
        """

    def __call__(self, x, sensi_order, mode=MODE_FUN):
        if mode == Objective.MODE_FUN:
            res = self.res(x)

        if sensi_order is 0:
            fval = self.fun(x)
        fval = 0
        grad = np.zeros(self.n, 1)
        hess = np.zeros(self.n, self.n)

        if sensi_order is 0:
            fval = self.fun(x)
            if self.grad is True:
                fval = fval[1]
        elif sensi_order is 1:
            fval
        return fval, grad, hess

    def get_fval(self, x):
        nllh, snllh, ssnllh = self.__call__(x, sensi_order=0,
                                            mode=Objective.MODE_FUN)
        return nllh

    def get_grad(self, x):
        nllh, snllh, ssnllh = self.__call__(x, sensi_order=1,
                                            mode=Objective.MODE_FUN)
        return snllh

    def get_hess(self, x):
        nllh, snllh, ssnllh = self.__call__(x, sensi_order=1,
                                            mode=Objective.MODE_FUN)
        return ssnllh

    def get_hessp(self, x, p):
        nllh, snllh, ssnllh = self.__call__(x, sensi_order=1,
                                            mode=Objective.MODE_FUN)
        return np.dot(ssnllh, p)

    def get_res(self, x):
        res, sres = self.__call__(x, sensi_order=0,
                                   mode=Objective.MODE_RES)
        return res

    def get_sres(self, x):
        res, sres = self.__call__(x, sensi_order=1,
                                  mode=Objective.MODE_RES)
        return sres


class AmiciObjective(Objective):

    def __init__(self, amici_model, amici_solver, edata, sensi_order):
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.edata = edata
        self.sensi_order = sensi_order
        self.dim = amici_model.np()
        super().__init__(None)

    def call(self, x, sensi_order=0, mode='fun'):
        if sensi_order > self.sensi_order:
            raise Exception("Sensitivity order not allowed.")

        nllh = 0
        snllh = np.zeros(self.dim)
        ssnllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        self.amici_model.setParameters(amici.DoubleVector(x))
        self.amici_solver.setSensitivityOrder(sensi_order)
        for data in self.edata:
            rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver, data)
            if rdata['status'] < 0.0:
                return float('inf'), np.nan*np.ones(self.dim), np.nan*np.ones([self.dim, self.dim])

            if mode == 'fun':
                nllh -= rdata['llh']
                if sensi_order > 0:
                    snllh -= rdata['sllh']
                    ssnllh += rdata['FIM']

            elif mode == 'res':
                res = np.hstack([res, rdata['res']]) if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([rdata['sres'],rdata['sres']]) if sres.size else rdata['sres']

        if mode == 'fun':
            return nllh, snllh, ssnllh
        elif mode == 'res':
            return res, sres
