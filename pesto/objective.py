import numpy as np
import amici


class Objective:
    
    def __init__(self, fun):
        # must be nll, i.e. to be minimized:
        self.fun = fun

    def get_fval_vararg(self, *par):
        nllh, snllh, ssnllh = self.call(par, sensi_order=0, mode='fun')
        return nllh

    def get_fval(self, par):
        nllh, snllh, ssnllh = self.call(par, sensi_order=0, mode='fun')
        return nllh

    def get_grad(self, par):
        nllh, snllh, ssnllh = self.call(par, sensi_order=1, mode='fun')
        return snllh

    def get_hess(self, par):
        nllh, snllh, ssnllh = self.call(par, sensi_order=1, mode='fun')
        return ssnllh

    def get_hessp(self, par, p):
        nllh, snllh, ssnllh = self.call(par, sensi_order=1, mode='fun')
        return np.dot(ssnllh, p)

    def get_res(self, par):
        res, sres  = self.call(par, sensi_order=0, mode='res')
        return res

    def get_sres(self,par):
        res, sres = self.call(par, sensi_order=1, mode='res')
        return sres

    def call(self, par, sensi_order=1, mode='fun'):
        return self.fun(par, sensi_order)


class AmiciObjective(Objective):

    def __init__(self, amici_model, amici_solver, edata, sensi_order):
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.edata = edata
        self.sensi_order = sensi_order
        self.dim = amici_model.np()
        super().__init__(None)

    def call(self, par, sensi_order=0, mode='fun'):
        if sensi_order > self.sensi_order:
            raise Exception("Sensitivity order not allowed.")

        nllh = 0
        snllh = np.zeros(self.dim)
        ssnllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        self.amici_model.setParameters(amici.DoubleVector(par))
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
