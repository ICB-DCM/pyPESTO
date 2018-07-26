import numpy as np
import amici


class Objective:
    
    def __init__(self, fun):
        # must be nll, i.e. to be minimized:
        self.fun = fun

    def get_fval(self, par):
        return self.call(par, sensi_order=0)[0]

    def get_grad(self, par):
        return self.call(par, sensi_order=1)[1]

    def call(self, par, sensi_order=1):
        return self.fun(par, sensi_order)


class AmiciObjective(Objective):

    def __init__(self, amici_model, amici_solver, edata, sensi_order):
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.edata = edata
        self.sensi_order = sensi_order
        self.dim = amici_model.np()
        super().__init__(None)

    def call(self, par, sensi_order=0):
        if sensi_order > self.sensi_order:
            raise Exception("Sensitivity order not allowed.")

        nllh = 0
        snllh = np.zeros(self.dim)

        self.amici_model.setParameters(amici.DoubleVector(par))
        self.amici_solver.setSensitivityOrder(sensi_order)
        for data in self.edata:
            rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver, data)
            if rdata['status'] < 0.0:
                return float('inf'), np.nan(self.dim)

            nllh -= rdata['llh']
            if sensi_order > 0:
                snllh -= rdata['sllh']



        return nllh, snllh
