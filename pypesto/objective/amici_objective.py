import numpy as np
import copy
import logging
from .objective import Objective
from .constants import MODE_FUN, MODE_RES

try:
    import amici
except ImportError:
    amici = None

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

        def fun(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_FUN)

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        def res(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_RES)

        if max_sensi_order > 0:
            sres = True
        else:
            sres = None

        super().__init__(
            fun=fun, grad=grad, hess=hess, hessp=None,
            res=res, sres=sres,
            fun_accept_sensi_orders=True,
            res_accept_sensi_orders=True,
            options=options
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

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed

        # gradients can always be computed
        sensi_order = min(max(sensi_orders), 1)
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
                    return self.get_error_output(sensi_orders, mode)

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

            # logging
            logger.debug('=== DATASET %d ===' % data_index)
            logger.debug('status: ' + str(rdata['status']))
            logger.debug('llh: ' + str(rdata['llh']))
            if 't_steadystate' in rdata and rdata['t_steadystate'] != np.nan:
                logger.debug('t_steadystate: ' + str(rdata['t_steadystate']))
            logger.debug('res: ' + str(rdata['res']))

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            # extract required result fields
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    snllh -= rdata['sllh']
                    # TODO: Compute the full Hessian, and check here
                    ssnllh -= rdata['FIM']

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([sres, rdata['sres']]) \
                        if sres.size else rdata['sres']

        # map_to_output is called twice, might be prettified
        return Objective.output_to_tuple(
            sensi_orders,
            mode,
            fval=nllh, grad=snllh, hess=ssnllh,
            res=res, sres=sres
        )

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

            preeq_edata = amici.amici.ExpData(self.amici_model.get())
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
        return Objective.output_to_tuple(
            sensi_orders=sensi_orders,
            mode=mode,
            fval=np.inf,
            grad=np.nan * np.ones(self.dim),
            hess=np.nan * np.ones([self.dim, self.dim]),
            res=np.nan * np.ones(n_res),
            sres=np.nan * np.ones([n_res, self.dim])
        )
