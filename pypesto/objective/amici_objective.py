import numpy as np
import copy
import logging
import pandas as pd
import numbers
from .objective import Objective
from .constants import MODE_FUN, MODE_RES, HESS

try:
    import amici
except ImportError:
    amici = None

logger = logging.getLogger(__name__)


class AmiciObjective(Objective):
    """
    This is a convenience class to compute an objective function from an
    AMICI model.
    """

    def __init__(self,
                 amici_model, amici_solver, edata,
                 max_sensi_order=None,
                 x_ids=None, x_names=None,
                 mapping_par_opt_to_par_sim=None,
                 mapping_scale_opt_to_scale_sim=None,
                 preprocess_edata=True,
                 options=None):
        """
        Constructor

        Parameters
        ----------

        amici_model: amici.Model
            The amici model.

        amici_solver: amici.Solver
            The solver to use for the numeric integration of the model.

        edata: amici.ExpData or list of amici.ExpData
            The experimental data. If a list is passed, its entries correspond
            to multiple experimental conditions.

        max_sensi_order: int
            Maximum sensitivity order supported by the model.

        x_ids: list of str, optional
            Ids of optimization parameters. In the simplest case, this will be
            the AMICI model parameters (default). Translates

        x_names: list of str, optional
            See Objective.

        mapping_par_opt_to_par_sim: optional
            Mapping of optimization parameters to model parameters. List array
            of size n_simulation_parameters * n_conditions.
            The default is just to assume that optimization and simulation
            parameters coincide. The default is to assume equality of both.

        mapping_scale_opt_to_scale_sim: optional
            Mapping of optimization parameter scales to simulation parameter
            scales. The default is to just use the scales specified in the
            `amici_model` already.

        preprocess_edata: bool, optional
            Whether to preprocess the experimental data.

        options: pypesto.ObjectiveOptions, optional
            Further options.
        """
        if amici is None:
            raise ImportError('This objective requires an installation of '
                              'amici (github.com/icb-dcm/amici. Install via '
                              'pip3 install amici.')

        if max_sensi_order is None:
            # 2 if model was compiled with second orders,
            # otherwise 1 can be guaranteed
            max_sensi_order = 2 if amici_model.o2mode else 1

        fun = self.get_bound_fun()

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        res = self.get_bound_res()

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

        # make sure the edatas are a list of edata objects
        if not isinstance(edata, list):
            edata = [edata]

        if preprocess_edata:
            # preprocess the experimental data
            self.preequilibration_edata = []
            self.init_preequilibration_edata(edata)
        else:
            self.preequilibration_edata = None

        # set the experimental data container
        self.edata = edata

        # set the maximum sensitivity order
        self.max_sensi_order = max_sensi_order

        # optimization parameter ids
        if x_ids is None:
            # use model parameter ids as ids
            x_ids = list(self.amici_model.getParameterIds())
        self.x_ids = x_ids

        self.dim = len(self.x_ids)

        # mapping of parameters
        if mapping_par_opt_to_par_sim is None:
            # use identity mapping for each condition
            mapping_par_opt_to_par_sim = [x_ids for _ in range(len(edata))]
        self.mapping_par_opt_to_par_sim = mapping_par_opt_to_par_sim

        # mapping of parameter scales
        if mapping_scale_opt_to_scale_sim is None:
            # use scales from amici_model
            mapping_scale_opt_to_scale_sim = [
                self.amici_model.getParameterScale() for _ in range(len(edata))
            ]
        self.mapping_scale_opt_to_scale_sim = mapping_scale_opt_to_scale_sim

        # optimization parameter names
        if x_names is None:
            # use model parameter names as names
            x_names = list(self.amici_model.getParameterNames())
        self.x_names = x_names

    def get_bound_fun(self):
        """
        Generate a fun function that calls _call_amici with MODE_FUN. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def fun(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_FUN)

        return fun

    def get_bound_res(self):
        """
        Generate a res function that calls _call_amici with MODE_RES. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def res(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_RES)

        return res

    def rebind_fun(self):
        """
        Replace the current fun function with one that is bound to the current
        instance
        """
        self.fun = self.get_bound_fun()

    def rebind_res(self):
        """
        Replace the current res function with one that is bound to the current
        instance
        """
        self.res = self.get_bound_res()

    def __deepcopy__(self, memodict=None):
        model = amici.ModelPtr(self.amici_model.clone())
        solver = amici.SolverPtr(self.amici_solver.clone())
        edata = [amici.ExpData(data) for data in self.edata]
        other = AmiciObjective(model, solver, edata)
        for attr in self.__dict__:
            if attr not in ['amici_solver', 'amici_model', 'edata', 'preequilibration_edata']:
                other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    def _update_from_problem(self,
                            dim_full,
                            x_free_indices,
                            x_fixed_indices,
                            x_fixed_vals):
        """
        Handle fixed parameters. Here we implement the amici exclusive
        initialization of ParameterLists and respectively replace the
        generic postprocess function

        TODO: Currently, this method is not used. Instead, the super fallback
        Objective.update_from_problem ist used, which in particular does not
        make use of pesto's ability to compute only compute requried directiol
        derivativs. If that is inteded, the mapping between simulation and
        optimization paraemters must be accounted for.

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
        super(AmiciObjective, self).update_from_problem(
            dim_full,
            x_free_indices,
            x_fixed_indices,
            x_fixed_vals
        )

        # we subindex the existing plist in case there already is a user
        # specified plist
        plist = self.amici_model.getParameterList()
        plist = [plist[idx] for idx in x_free_indices]
        self.amici_model.setParameterList(plist)

        self.dim = len(plist)

        def postprocess(result):
            if HESS in result:
                hess = result[HESS]
                if hess.shape[0] == dim_full:
                    # see https://github.com/ICB-DCM/AMICI/issues/274
                    hess = hess[..., x_free_indices]
                    result[HESS] = hess
            return result

        self.postprocess = postprocess

        # now we need to rebind fun and res to this instance of AmiciObjective
        # for the changes to have an effect
        self.rebind_fun()
        self.rebind_res()

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        # TODO: extract function to run simulations and return list of amici.ReturnData. use this then here, as well as for simulations_to_measurement_df below

        # amici is built so that only the maximum sensitivity is required,
        # the lower orders are then automatically computed

        # gradients can always be computed
        sensi_order = min(max(sensi_orders), 0)
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

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        if self.preequilibration_edata:
            preeq_status = self.run_preequilibration(sensi_orders, mode, x)
            if preeq_status is not None:
                return preeq_status

        # loop over experimental data
        for data_ix, data in enumerate(self.edata):

            # set model parameter scale for condition index
            self.set_parameter_scale(data_ix)

            # set parameters in model, according to mapping
            self.set_par_sim_for_condition(data_ix, x)

            if self.preequilibration_edata:
                original_value_dict = self.preprocess_preequilibration(data_ix)
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
            logger.debug(f'=== DATASET {data_ix} ===')
            logger.debug(f'status: {rdata["status"]}')
            logger.debug(f'llh: {rdata["llh"]}')

            if 't_steadystate' in rdata and rdata['t_steadystate'] != np.nan:
                logger.debug(f't_steadystate: {rdata["t_steadystate"]}')
            logger.debug(f'res: {rdata["res"]}')

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            # TODO: must respect mapping matrix when assembling overall gradient
            # extract required result fields
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
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

    def init_preequilibration_edata(self, edatas):
        """
        Extract information needed for doing preequilibration.
        """
        self.preequilibration_edata = []

        for edata in edatas:
            # extract values of the fixed parameters
            fixed_parameters = list(edata.fixedParametersPreequilibration)

            # create edata object from model
            preeq_edata = amici.amici.ExpData(self.amici_model.get())

            # fill in fixed parameter values for preequilibration
            preeq_edata.fixedParametersPreequilibration = fixed_parameters

            # only preequilibration
            preeq_edata.setTimepoints([])

            # indicate whether preequilibration is required for this data set
            preequilibrate = len(fixed_parameters) > 0
            # TODO: Currently, len(fixed_parameters) == 0 is used as an
            # indicator of requiring no preequilibration. However, when there
            # are events (which amici cannot deal with yet in python), the
            # situation can occur that there are no fixed_parameters, but
            # events that are omitted in the preequilibration run.

            self.preequilibration_edata.append(dict(
                edata=preeq_edata,
                preequilibrate=preequilibrate
            ))

    def run_preequilibration(self, sensi_orders, mode, x):
        """
        Run preequilibration.
        """

        for data_ix, preeq_dict in enumerate(self.preequilibration_edata):

            if not preeq_dict['preequilibrate']:
                # no preequilibration required
                continue


            # set model parameter scales for condition index
            self.set_parameter_scale(data_ix)

            # map to simulation parameters
            self.set_par_sim_for_condition(data_ix, x)

            # TODO: Conditions might share preeq conditions and dynamic
            # parameters. In that case, we can save time here.

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                preeq_dict['edata'])

            # check if an error occurred
            if rdata['status'] < 0.0:
                return self.get_error_output(sensi_orders, mode)

            # fill state
            preeq_dict['x0'] = rdata['x0']
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                # fill state sensitivities
                # TODO check that these are always computed, i.e.
                # forward sensitivities used
                preeq_dict['sx0'] = rdata['sx0']

    def preprocess_preequilibration(self, edata_ix):
        """
        Update the model and data from the preequilibration states,
        before running the real simulation.
        """

        data = self.edata[edata_ix]

        original_fixed_parameters_preequilibration = None
        original_initial_states = None
        original_initial_state_sensitivities = None

        # if this data set needed preequilibration, adapt the states
        # according to the previously run preequilibration
        if self.preequilibration_edata[edata_ix]['preequilibrate']:

            # remember original states and sensitivities
            original_initial_states = self.amici_model.getInitialStates()
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                original_initial_state_sensitivities = \
                    self.amici_model.getInitialStateSensitivities()

            # remember original fixed parameters for preequilibration
            original_fixed_parameters_preequilibration \
                = data.fixedParametersPreequilibration
            # unset fixed preequilibration parameters in data (TODO: Why?)
            data.fixedParametersPreequilibration = []

            # set initial state from preequilibration
            self.amici_model.setInitialStates(
                self.preequilibration_edata[edata_ix]['x0']
            )

            # set initial sensitivities from preequilibration
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                self.amici_model.setInitialStateSensitivities(
                    self.preequilibration_edata[edata_ix]['sx0'].flatten()
                )

        # return the original values
        return {
            'k': original_fixed_parameters_preequilibration,
            'x0': original_initial_states,
            'sx0': original_initial_state_sensitivities
        }

    def postprocess_preequilibration(self, edata, original_value_dict):
        """
        Reset the model and edata to the true values, i.e. undo
        the temporary changes done in preprocess_preequilibration.
        """

        # reset values in edata from values in original_value_dict, if
        # the corresponding entry in original_value_dict is not None.

        if original_value_dict['k']:
            edata.fixedParametersPreequilibration = original_value_dict['k']

        if original_value_dict['x0']:
            self.amici_model.setInitialStates(original_value_dict['x0'])

        if original_value_dict['sx0']:
            self.amici_model.setInitialStateSensitivities(
                original_value_dict['sx0']
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

    def set_par_sim_for_condition(self, condition_ix, x):
        """
        Set the simulation parameters from the optimization parameters
        for the given condnition.
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        self.amici_model.setParameters(x_sim)

    def set_parameter_scale(self, condition_ix):
        scale_list = self.mapping_scale_opt_to_scale_sim[condition_ix]
        amici_scale_vector = amici.ParameterScalingVector()

        for val in scale_list:

            if val == 'lin':
                scale = amici.ParameterScaling_none
            elif val == 'log10':
                scale = amici.ParameterScaling_log10
            elif val == 'log':
                scale = amici.ParameterScaling_ln
            else:
                raise ValueError(
                    f"Parameter scaling not recognized: {val}")

            # append to scale vector
            amici_scale_vector.append(scale)

        self.amici_model.setParameterScale(amici_scale_vector)


def map_par_opt_to_par_sim(mapping_par_opt_to_par_sim, par_opt_ids, x):
    """
    From the optimization vector `x`, create the simulation vector according
    to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    x: array-like of float
        The optimization parameters vector.

    Returns
    -------

    y: array-like of float
        The simulation parameters vector corresponding to x under the
        specified mapping.
    """

    # number of simulation parameters
    n_par_sim = len(mapping_par_opt_to_par_sim)

    # prepare simulation parameter vector
    par_sim_vals = np.zeros(n_par_sim)

    # iterate over simulation parameter indices
    for j_par_sim in range(n_par_sim):
        # extract entry in mapping table for j_par_sim
        val = mapping_par_opt_to_par_sim[j_par_sim]

        if isinstance(val, numbers.Number):
            # fixed value assignment
            par_sim_vals[j_par_sim] = val
        else:
            # value is optimization parameter id
            par_sim_vals[j_par_sim] = x[par_opt_ids.index(val)]

    # return the created simulation parameter vector
    return par_sim_vals


def add_sim_grad_to_opt_grad(par_opt_ids,
                             mapping_par_opt_to_par_sim,
                             sim_grad,
                             opt_grad,
                             coefficient=1.0):
    """
    Sum simulation gradients to objective gradient according to the provided
    mapping `mapping`.

    Parameters
    ----------

    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    sim_grad: array-like of float
        Simulation gradient.
    opt_grad: array-like of float
        The optimization gradient. To which sim_grad is added.
        Will be changed in place.
    coefficient: float
        Coefficient for sim_grad when adding to opt_grad.

    Returns
    -------

    """

    for par_sim_idx, par_id in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(par_id, str):
            # This was a numeric override for which we ignore the gradient
            continue

        par_opt_idx = par_opt_ids.index(par_id)
        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]
