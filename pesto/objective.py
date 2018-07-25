import amici

class Objective:
    
    def __init__(fun, dim)
        # must be nll, i.e. to be minimized:
        self.fun = fun
        self.dim = dim
        # dim rather in optimizer

    def call(par, sensi_order=0):
        return nllh

class AmiciObjective(Objective):
    
    def __init__(amici_model, amici_solver, edata, dim, sensi_order):
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.edata = edata
        self.sensi_order = sensi_order
        # dim can be read from model

    def call(par, sensi_order=0)
        if sensi_order > self.sensi_order
            raise Exception("Sensitivity order not allowed.")

        self.amici_solver.setSensitivityOrder(sensi_order)
        rdata = amici.runAmiciSimulation(self.amici_model, self.amici_solver, self.edata)

        return - rdata.llh, - rdata.sllh
