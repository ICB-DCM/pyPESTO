import numpy as np
import copy
from .objective import Objective

try:
    import amici
except ImportError:
    amici = None

import logging

logger = logging.getLogger(__name__)


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
                 preprocess_edata=True, options=None):
        if amici is None:
            raise ImportError('This objective requires an installation of '
                              'amici (github.com/icb-dcm/amici. Install via '
                              'pip3 install amici.')

        if max_sensi_order is None:
            max_sensi_order = 2 if amici_model.o2mode else 1

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        if max_sensi_order > 0:
            sres = True
        else:
            sres = None

        super().__init__(
            fun=None, grad=grad, hess=hess, hessp=None,
            res=None, sres=sres,
            options=options,
            overwritefun=False, overwriteres=False
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

        # extract parameter names from model
        self.x_names = list(self.amici_model.getParameterNames())

    def fun(self, x):
        return self.call_amici(
            x,
            Objective.MODE_FUN
        )

    def res(self, x):
        return self.call_amici(
            x,
            Objective.MODE_RES
        )

    def call_amici(
            self,
            x,
            mode
    ):
        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed

        # gradients can always be computed
        if self.sensi_orders is None:
            raise Exception('Sensitivity Orders were not specified. Please use'
                            '__call__ to evaluate the objective function.')
        sensi_order = min(max(self.sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM

        # check if sensitivities can be computed
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        # TODO: For large-scale models it might be bad to always reserve
        # space in particular for the Hessian.

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
                    return self.get_error_output(mode)

                self.preequilibration_edata[fixedParameters]['x0'] = \
                    rdata['x0']
                if self.amici_solver.getSensitivityOrder() > \
                        amici.SensitivityOrder_none:
                    self.preequilibration_edata[fixedParameters]['sx0'] = \
                        rdata['sx0']

        # loop over experimental data
        for data_index, data in enumerate(self.edata):

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

            logger.debug('=== DATASET %d ===' % data_index)
            logger.debug('status: ' + str(rdata['status']))
            logger.debug('llh: ' + str(rdata['llh']))
            logger.debug('y:\n' + str(rdata['y']))

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(mode)

            # extract required result fields
            if mode == Objective.MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    snllh -= rdata['sllh']
                    # TODO: Compute the full Hessian, and check here
                    ssnllh -= rdata['FIM']
                return nllh, snllh, ssnllh

            elif mode == Objective.MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([sres, rdata['sres']]) \
                        if sres.size else rdata['sres']
                return res, sres

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

    def get_error_output(self, mode):
        if mode == Objective.MODE_FUN:
            return \
                np.inf, \
                np.nan * np.ones(self.dim), \
                np.nan * np.ones([self.dim, self.dim])
        elif mode == Objective.MODE_RES:
            if not self.amici_model.nt():
                nt = sum([data.nt() for data in self.edata])
            else:
                nt = sum([data.nt() if data.nt() else self.amici_model.nt()
                          for data in self.edata])
            n_res = nt * self.amici_model.nytrue
            return \
                np.nan * np.ones(n_res), \
                np.nan * np.ones([n_res, self.dim])
